# -*- coding: utf-8 -*-
"""
Enhancement steps (ablation pipeline): 9 switchable operations applied to the
complex S-parameter array between loading and tensor construction.

Design contract
  * every step is a pure function  fn(C, meta, channels, params) -> (C, channels, info)
      C        (n, T, ch) complex128, frequency domain (or time domain after IFFT)
      meta     list of (subject, session, class_idx, rep_idx)
      channels list of channel names, len == ch
  * steps are listed IN ORDER under cfg["enhance"]["steps"]; the list order IS
    the execution order (order-ablation variants just reorder the list)
  * frequency-domain steps run on the complex sweep; the first time-domain
    step triggers the IFFT; with domain="freq" time steps are skipped
  * an empty step list reproduces the legacy baseline behavior exactly

Step registry (frequency domain):
  channel_select  drop dead channels by name                 (dead-channel fix)
  bootstrap_avg   per-sample bootstrap mean over its (group,class) repetitions
  diff_sub        subtract a group-level reference (additive drift b)        }
  diff_div        divide by a group-level reference (multiplicative drift g) } drift fix
  ifft_window     window the sweep before IFFT (sinc sidelobe suppression)   } informative
  delay_align     per-group linear-phase alignment (group delay drift)       } steps fix
Step registry (time domain):
  early_gate      remove the early-time antenna-coupling burst (peak-anchored)
  amp_norm        per-sample per-channel L2 amplitude normalization
  step_crop       crop near-constant time steps (informative segment only)
Not yet implemented:
  time_feats      hand-crafted time-domain features (changes tensor shape;
                  kept out of the 9-step ablation chain for now)
"""

import numpy as np

EPS = 1e-12

STEP_DOMAIN = {
    "channel_select": "freq",
    "bootstrap_avg": "freq",
    "diff_sub": "freq",
    "diff_div": "freq",
    "ifft_window": "freq",
    "delay_align": "freq",
    "early_gate": "time",
    "amp_norm": "time",
    "step_crop": "time",
}


# ------------------------------------------------------------------ helpers
def _group_indices(meta):
    """(subj, ses) -> array of sample indices; deterministic group order."""
    groups = {}
    for i, m in enumerate(meta):
        groups.setdefault((m[0], m[1]), []).append(i)
    return {g: np.asarray(v) for g, v in sorted(groups.items())}


def _group_ref(C, meta, how):
    """(subj, ses) -> reference curve (T, ch), label-free (uses whole group)."""
    ref = {}
    for g, idx in _group_indices(meta).items():
        A = C[idx]
        if how == "group_median":
            r = (np.median(A.real, axis=0) + 1j * np.median(A.imag, axis=0))
        else:                                     # group_mean
            r = A.mean(axis=0)
        ref[g] = r
    return ref


def _to_time(C):
    """frequency sweep -> time domain, same convention as the legacy loader."""
    return np.fft.fftshift(np.fft.ifft(C, axis=1), axes=1)


# ------------------------------------------------------------ freq steps
def channel_select(C, meta, channels, params):
    drop = list(params.get("drop", []))
    keep = [j for j, ch in enumerate(channels) if ch not in drop]
    if not keep:
        raise ValueError("channel_select: dropping every channel")
    info = {"dropped": [ch for ch in channels if ch in drop],
            "kept": [channels[j] for j in keep]}
    return C[:, :, keep], [channels[j] for j in keep], info


def bootstrap_avg(C, meta, channels, params):
    """Replace each sample by a bootstrap mean over its (group, class) reps.

    Sample count and meta are unchanged, so downstream splits stay valid.
    Deterministic: buckets are visited in sorted order, one shared RNG.
    """
    n_draws = int(params.get("n_draws", 10))
    seed = int(params.get("seed", 0))
    rng = np.random.default_rng(seed)
    buckets = {}
    for i, m in enumerate(meta):
        buckets.setdefault((m[0], m[1], m[2]), []).append(i)
    out = np.empty_like(C)
    for key in sorted(buckets):
        idx = np.asarray(buckets[key])
        for i in idx:
            draws = rng.choice(idx, size=n_draws, replace=True)
            out[i] = C[draws].mean(axis=0)
    n_rep = len(next(iter(buckets.values())))       # reps per (group, class)
    exp_unique = n_rep * (1.0 - (1.0 - 1.0 / n_rep) ** n_draws)
    return out, channels, {"n_draws": n_draws,
                           "expected_unique_draws": round(exp_unique, 2),
                           "noise_gain_db": round(10 * np.log10(exp_unique), 2)}


def diff_sub(C, meta, channels, params):
    """x' = x - ref(group): removes the additive common-mode drift b."""
    ref = _group_ref(C, meta, params.get("ref", "group_median"))
    out = C.copy()
    for g, idx in _group_indices(meta).items():
        out[idx] = C[idx] - ref[g][None, :, :]
    return out, channels, {"ref": params.get("ref", "group_median")}


def diff_div(C, meta, channels, params):
    """x' = x / ref(group) where |ref| is strong enough; multiplicative drift g.

    Weak reference points (below mask_db) are left untouched: dividing by
    noise would amplify noise.
    """
    mask_db = float(params.get("mask_db", -60.0))
    ref = _group_ref(C, meta, params.get("ref", "group_median"))
    out = C.copy()
    masked_frac = 0.0
    for g, idx in _group_indices(meta).items():
        r = ref[g]
        mask = 20.0 * np.log10(np.abs(r) + EPS) > mask_db    # (T, ch)
        masked_frac += float((~mask).mean()) * len(idx)
        out[idx] = np.where(mask[None, :, :], C[idx] / (r[None, :, :] + EPS),
                            C[idx])
    return out, channels, {"mask_db": mask_db,
                           "masked_frac": round(masked_frac / len(meta), 4)}


def ifft_window(C, meta, channels, params):
    """Window the frequency sweep before IFFT (suppresses sinc sidelobes)."""
    wtype = params.get("type", "kaiser")
    T = C.shape[1]
    if wtype == "none":
        return C, channels, {"type": "none"}
    if wtype == "kaiser":
        win = np.kaiser(T, float(params.get("beta", 6.0)))
    elif wtype == "hamming":
        win = np.hamming(T)
    elif wtype == "hann":
        win = np.hanning(T)
    else:
        raise ValueError(f"ifft_window: unknown type {wtype}")
    return C * win[None, :, None], channels, {"type": wtype}


def delay_align(C, meta, channels, params):
    """Align every group to the first group via linear phase (circular shift).

    Group-mean time envelopes are cross-correlated; the integer lag is
    compensated in the frequency domain with exp(+j 2 pi k lag / T).
    """
    T = C.shape[1]
    k = np.arange(T)
    gi = _group_indices(meta)

    def env(g):
        w = _to_time(C[gi[g]].mean(axis=0)[None])[0]          # (T, ch)
        return np.abs(w).mean(axis=1)                         # (T,)

    ref_env = env(next(iter(gi)))
    shifts = {}
    out = C.copy()
    for g in list(gi)[1:]:
        e = env(g)
        cc = np.correlate(e, ref_env, mode="full")
        lag = int(np.argmax(cc)) - (T - 1)                    # e lags ref by `lag`
        if lag != 0:
            out[gi[g]] *= np.exp(1j * 2 * np.pi * k[None, :, None] * lag / T)
        shifts[f"{g[0]}/{g[1]}"] = lag
    return out, channels, {"group_lags": shifts}


# ------------------------------------------------------------ time steps
def early_gate(C, meta, channels, params):
    """Remove the early-time antenna-coupling burst.

    Per sample & channel: anchor at the main peak (searched in the first
    half), walk right until the envelope falls `depth_db` below the peak.
    A single global boundary (quantile over all per-sample gates) is then
    applied to every sample so the tensor keeps a uniform length.
    """
    depth = float(params.get("depth_db", -20.0))
    mode = params.get("mode", "crop")
    q = float(params.get("quantile", 0.9))
    n, T, ch = C.shape
    thr_f = 10.0 ** (depth / 20.0)
    gates = []
    for i in range(n):
        g_i = T
        for j in range(ch):
            env = np.abs(C[i, :, j])
            t_peak = int(np.argmax(env[: T // 2 + 1]))
            thr = env[t_peak] * thr_f
            t = t_peak
            while t < T and env[t] >= thr:
                t += 1
            g_i = min(g_i, t)          # conservative: all channels clear
        gates.append(g_i)
    T0 = int(np.quantile(gates, q))
    info = {"depth_db": depth, "mode": mode, "quantile": q,
            "gate_global": T0, "gate_min": int(min(gates)),
            "gate_max": int(max(gates))}
    if T0 >= T:
        return C * 0.0 if mode == "zero" else C[:, :0, :], channels, info
    if mode == "zero":
        out = C.copy()
        out[:, :T0, :] = 0
    else:                                        # crop
        out = C[:, T0:, :]
    return out, channels, info


def amp_norm(C, meta, channels, params):
    """Per-sample per-channel L2 normalization (gain-invariant energy)."""
    if params.get("mode", "l2") != "l2":
        raise ValueError("amp_norm: only mode=l2 is implemented")
    norm = np.sqrt((np.abs(C) ** 2).sum(axis=1, keepdims=True) + EPS)
    return C / norm, channels, {"mode": "l2"}


def step_crop(C, meta, channels, params):
    """Crop leading/trailing near-constant time steps (global std rule)."""
    frac = float(params.get("frac", 0.01))
    sd = 0.5 * (C.real.std(axis=0) + C.imag.std(axis=0)).mean(axis=1)   # (T,)
    if sd.max() <= EPS:
        return C, channels, {"frac": frac, "kept": int(C.shape[1])}
    alive = sd > frac * sd.max()
    if not alive.any():
        return C, channels, {"frac": frac, "kept": int(C.shape[1])}
    first, last = int(np.argmax(alive)), int(len(alive) - 1 - np.argmax(alive[::-1]))
    kept = last - first + 1
    return C[:, first:last + 1, :], channels, {"frac": frac, "kept": kept,
                                               "first": first, "last": last}


# ---------------------------------------------------------------- registry
_STEP_FN = {
    "channel_select": channel_select,
    "bootstrap_avg": bootstrap_avg,
    "diff_sub": diff_sub,
    "diff_div": diff_div,
    "ifft_window": ifft_window,
    "delay_align": delay_align,
    "early_gate": early_gate,
    "amp_norm": amp_norm,
    "step_crop": step_crop,
}


# ------------------------------------------------------------ time_feats
def time_feats(C_time, C_freq, meta, channels, params):
    """Extract physics-driven scalar features per channel (terminal step).

    Groups (params['groups'], default all eight):
      energy      in-gate energy in dB (requires amp_norm OFF upstream, else
                  the normalization erases exactly this information)
      peak_delay  index of the strongest echo (antenna-cheek gap proxy)
      echo_width  samples within -10 dB of the peak (axial extent of echo)
      decay_slope log-envelope slope after the main peak (tissue absorption)
      cum_energy  normalized cumulative-energy curve, `cum_bins` points
                  (temporal energy distribution of the utterance)
      group_delay mean/std of the unwrapped phase slope per frequency bin
                  (propagation delay through the cheek)
      peak_seq    heights of the `n_peaks` strongest local maxima,
                  peak-normalized (layered-reflection structure)
      cepstrum    first `cep_bins` real-cepstrum coefficients of the sweep
                  (periodic multipath structure, speech-processing heritage)

    Output C (n, k, ch) complex with imag=0 so the tensor layer (Re/Im
    interleave) and every downstream consumer work unchanged.
    """
    groups = list(params.get("groups", ["energy", "peak_delay", "echo_width",
                                        "decay_slope", "cum_energy",
                                        "group_delay", "peak_seq", "cepstrum"]))
    cum_bins = int(params.get("cum_bins", 16))
    n_peaks = int(params.get("n_peaks", 5))
    cep_bins = int(params.get("cep_bins", 16))
    n, T, ch = C_time.shape

    per_ch, feat_names = [], []
    for j, ch_name in enumerate(channels):
        xt = C_time[:, :, j]                       # (n, T) time domain
        env = np.abs(xt)
        xf = C_freq[:, :, j]                       # (n, T) frequency domain
        feats, names = [], []

        if "energy" in groups:
            feats.append(10.0 * np.log10((np.abs(xt) ** 2).mean(axis=1) + EPS))
            names.append("energy_db")
        if "peak_delay" in groups:
            feats.append(np.argmax(env, axis=1).astype(float))
            names.append("peak_delay")
        if "echo_width" in groups:
            thr = 10.0 ** (-10.0 / 20.0)
            w = np.array([float((env[i] >= env[i].max() * thr).sum())
                          for i in range(n)])
            feats.append(w)
            names.append("echo_width")
        if "decay_slope" in groups:
            s = np.zeros(n)
            for i in range(n):
                e = env[i] + EPS
                pk = int(np.argmax(e))
                if pk < T - 2:
                    y = np.log(e[pk:])
                    s[i] = np.polyfit(np.arange(len(y)), y, 1)[0]
            feats.append(s)
            names.append("decay_slope")
        if "cum_energy" in groups:
            ce = np.cumsum(env ** 2, axis=1)
            ce = ce / (ce[:, -1:] + EPS)
            idx = np.linspace(0, T - 1, cum_bins).astype(int)
            feats.append(ce[:, idx])
            names += [f"cum{i:02d}" for i in range(cum_bins)]
        if "group_delay" in groups:
            ph = np.unwrap(np.angle(xf), axis=1)
            gd = -np.diff(ph, axis=1)              # phase slope per bin
            feats.append(gd.mean(axis=1))
            feats.append(gd.std(axis=1))
            names += ["gd_mean", "gd_std"]
        if "peak_seq" in groups:
            out = np.zeros((n, n_peaks))
            for i in range(n):
                e = env[i]
                loc = np.where((e[1:-1] > e[:-2]) & (e[1:-1] >= e[2:]))[0] + 1
                if len(loc):
                    top = loc[np.argsort(e[loc])[::-1]][:n_peaks]
                    out[i, :len(top)] = e[top] / (e.max() + EPS)
            feats.append(out)
            names += [f"pk{i}" for i in range(n_peaks)]
        if "cepstrum" in groups:
            spec = np.log(np.abs(xf) + 1e-6)       # floor: avoid log(0)
            cep = np.real(np.fft.ifft(spec, axis=1))[:, :cep_bins]
            feats.append(cep)
            names += [f"cep{i:02d}" for i in range(cep_bins)]

        F = np.concatenate([f[:, None] if f.ndim == 1 else f
                            for f in feats], axis=1)          # (n, k)
        per_ch.append(F)
        feat_names += [f"{ch_name}_{nm}" for nm in names]

    # assemble the feature matrix (n, k, ch); optional sub-feature selection
    # (params['select']) keeps only the named columns per channel
    select = params.get("select")
    if select:
        keep_idx, keep_names = [], []
        pos = 0
        for j in range(len(channels)):
            for local in range(per_ch[j].shape[1]):
                nm = feat_names[pos]
                if nm in select:
                    keep_idx.append((j, local))
                    keep_names.append(nm)
                pos += 1
        out = np.zeros((n, len(keep_idx), len(channels)), dtype=complex)
        for col, (j, local) in enumerate(keep_idx):
            out[:, col, j] = per_ch[j][:, local]
        feat_names = keep_names
    else:
        k = per_ch[0].shape[1]
        out = np.zeros((n, k, len(channels)), dtype=complex)
        for j, F in enumerate(per_ch):
            out[:, :, j] = F

    # mode: replace (scalar features only) | append (concatenate Re/Im curve
    # then features along the time axis, so the model sees BOTH)
    mode = params.get("mode", "append")
    if mode == "append":
        out = np.concatenate([C_time, out], axis=1)     # (n, T + k, ch)
    elif mode != "replace":
        raise ValueError(f"time_feats: unknown mode {mode}")
    return out, channels, {"groups": groups, "mode": mode,
                           "n_features": len(feat_names),
                           "feature_names": feat_names}


# ------------------------------------------------------------- dispatcher
def apply_enhance(C, meta, channels, steps, domain="time", log=None):
    """Run the ordered step list; returns (C, channels, info).

    C comes in as the frequency-domain complex array; time steps trigger the
    IFFT at their first appearance (the pre-IFFT frequency data is kept for
    time_feats, which needs both domains). With domain="freq" time steps are
    skipped (with a note), keeping the legacy frequency evaluation usable.
    """
    log = log or (lambda *_: None)
    info = {"steps": [], "skipped": []}
    in_freq = True
    feat_mode = False
    C_freq_saved = None
    for step in steps or []:
        name = step.get("name")
        if name == "time_feats":
            if domain != "time":
                info["skipped"].append(name)
                log(f"[enhance] skip {name} (domain=freq)")
                continue
            if in_freq:                      # no time step ran before it
                C_freq_saved = C.copy()
                C = _to_time(C)
                in_freq = False
                log("[enhance] IFFT -> time domain")
            C, channels, sinfo = time_feats(C, C_freq_saved, meta, channels,
                                            step.get("params", {}) or {})
            feat_mode = True                 # C is now a feature matrix
            info["steps"].append({"name": name, **sinfo})
            log(f"[enhance] {name:<15} C{C.shape} {sinfo}")
            continue
        if name not in _STEP_FN:
            raise ValueError(f"unknown enhance step: {name}")
        if STEP_DOMAIN[name] == "time":
            if domain != "time":
                info["skipped"].append(name)
                log(f"[enhance] skip {name} (domain=freq)")
                continue
            if in_freq:
                C_freq_saved = C.copy()      # keep freq data for time_feats
                C = _to_time(C)
                in_freq = False
                log("[enhance] IFFT -> time domain")
        C, channels, sinfo = _STEP_FN[name](C, meta, channels,
                                            step.get("params", {}) or {})
        info["steps"].append({"name": name, **sinfo})
        log(f"[enhance] {name:<15} C{C.shape} {sinfo}")
    if domain == "time" and in_freq and not feat_mode:
        C = _to_time(C)
    return C, channels, info
