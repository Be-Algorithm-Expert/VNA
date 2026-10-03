# -*- coding: utf-8 -*-
"""
Figures under outputs/figures/<run_name>/:

1. <run>_allclasses.png - mean time-domain response of every class (whole dataset):
   - load R01-R10 of all 50 classes for every (subject, session) in
     data.subjects x data.sessions
   - grid = 3 channel rows x <#groups> subject/session columns
   - each cell: heatmap, rows = class (Cxx + Chinese char), x = time (ns),
     color = mean magnitude (10-rep average)
   - shared color scale per channel row

2. <run>_sample_time_<sid>.png - single-sample time-domain curves |h(t)| AFTER IFFT

3. <run>_sample_freq_<sid>.png - single-sample frequency-domain curves
   |S(f)| in dB BEFORE IFFT (the stored sweep data)

The sample ("<subject>/<session>/<class>/<rep>", cfg.plot.sample, default
1st subject / 1st session / C01 / R01) is shared by figures 2 and 3.

Usage:
  python analysis/plot_compare.py --config configs/baseline_s21.yaml
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# CJK font (Windows)
for _f in ["Microsoft YaHei", "SimHei"]:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f] + plt.rcParams["font.sans-serif"]
        break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

REPS = [f"R{r:02d}" for r in range(1, 11)]
CHANNELS = ["S_avg", "S11", "S22"]


def load_label_map(root):
    m = {}
    f = root / "label_map.csv"
    if not f.exists():
        return m
    with f.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            m[(row["subject"], row["session"], row["class_id"])] = row["label"]
    return m


def load_group(base, subj, ses):
    """Return (M[50, N, 3], t_ns, df) — per-class mean |h(t)| over R01-R10."""
    means, t_ns, df = [], None, None
    for c in range(1, 51):
        cls = f"C{c:02d}"
        arrs = []
        for rep in REPS:
            f = base / f"{subj}_{ses}_{cls}_{rep}.npz"
            if not f.exists():
                print(f"[warn] missing {f.name}")
                arrs = []
                break
            d = np.load(f, allow_pickle=False)
            if df is None:
                df = float(d["df"])
                t = np.fft.fftshift(np.fft.fftfreq(d["S"].shape[0]))
                t_ns = t / (df * 1e-9)
            arrs.append(np.abs(np.fft.fftshift(np.fft.ifft(d["S"], axis=0), axes=0)))
        means.append(np.stack(arrs).mean(axis=0) if arrs
                     else np.full((len(t_ns), 3), np.nan))
    return np.stack(means), t_ns, df


def sample_npz(cfg, proc_root):
    """Resolve cfg.plot.sample "<subject>/<session>/<class>/<rep>" -> (path, ids).

    Defaults to 1st subject / 1st session / 1st class / R01. path is None if the
    file does not exist; ids = (subj, ses, cls, rep) either way.
    """
    subjects = cfg["data"]["subjects"]
    sessions = cfg["data"]["sessions"]
    default_sid = f"{subjects[0]}/{sessions[0]}/C01/R01"
    sid = (cfg.get("plot") or {}).get("sample", default_sid)
    parts = sid.split("/")
    if len(parts) == 4:
        subj, ses, cls, rep = parts
    elif len(parts) == 3:
        subj, ses, cls, rep = parts[0], parts[1], parts[2], "R01"
    else:
        print(f"[warn] bad plot.sample '{sid}', using {default_sid}")
        subj, ses, cls, rep = default_sid.split("/")

    npz = proc_root / subj / ses / "VNA_Data" / f"{subj}_{ses}_{cls}_{rep}.npz"
    return (npz if npz.exists() else None), (subj, ses, cls, rep)


def plot_sample_freq_curves(cfg, proc_root, fig_dir, run, labels, chan_note,
                            smooth_note):
    """Single-sample frequency-domain curves: 3 channels, |S(f)| in dB vs freq.

    This is the stored data BEFORE the IFFT (what the time-domain figure is
    computed from). Same sample as plot_sample_curves().
    """
    npz, ids = sample_npz(cfg, proc_root)
    if npz is None:
        print("[warn] sample npz missing, frequency-domain figure skipped")
        return
    subj, ses, cls, rep = ids

    d = np.load(npz, allow_pickle=False)
    S = d["S"]                       # (N, 3) complex, frequency domain
    df = float(d["df"])
    freq = d["freq"] / 1e9           # Hz -> GHz
    mag_db = 20.0 * np.log10(np.maximum(np.abs(S), 1e-12))

    lab = labels.get((subj, ses, cls), "")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for j, ch in enumerate(CHANNELS):
        ax = axes[j]
        ax.plot(freq, mag_db[:, j], lw=0.9, color="tab:red")
        ax.set_ylabel(f"|{ch}| (dB)")
        ax.set_title(ch + ("  (transmission)" if j == 0 else "  (reflection)"),
                     fontsize=10)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("frequency (GHz)")
    axes[-1].set_xlim(freq[0], freq[-1])
    fig.suptitle(f"Single-sample frequency-domain S-parameters (before IFFT)  "
                 f"{subj}/{ses} {cls} '{lab}' {rep}\n"
                 f"run: {run} | {chan_note} | smooth: {smooth_note} | "
                 f"{freq[0]:.1f}-{freq[-1]:.1f} GHz, N={S.shape[0]}, "
                 f"df={df/1e6:.0f} MHz",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = fig_dir / f"{run}_sample_freq_{subj}_{ses}_{cls}_{rep}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig] {out}")


def plot_sample_curves(cfg, proc_root, fig_dir, run, labels, chan_note, smooth_note):
    """Single-sample IFFT curves: 3 channels, |h(t)| vs time.

    Same sample as plot_sample_freq_curves().
    """
    npz, ids = sample_npz(cfg, proc_root)
    if npz is None:
        print("[warn] sample npz missing, time-domain figure skipped")
        return
    subj, ses, cls, rep = ids

    d = np.load(npz, allow_pickle=False)
    S = d["S"]
    df = float(d["df"])
    h = np.fft.fftshift(np.fft.ifft(S, axis=0), axes=0)
    t = np.fft.fftshift(np.fft.fftfreq(S.shape[0])) / (df * 1e-9)
    mag = np.abs(h)

    lab = labels.get((subj, ses, cls), "")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for j, ch in enumerate(CHANNELS):
        ax = axes[j]
        ax.plot(t, mag[:, j], lw=0.9, color="tab:blue")
        ax.set_ylabel(f"|{ch}(t)|")
        ax.set_title(ch + ("  (transmission)" if j == 0 else "  (reflection)"),
                     fontsize=10)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (ns)")
    axes[-1].set_xlim(t[0], t[-1])
    fig.suptitle(f"Single-sample time-domain response  {subj}/{ses} {cls} '{lab}' {rep}\n"
                 f"run: {run} | {chan_note} | smooth: {smooth_note} | "
                 f"{d['freq'][0]/1e9:.1f}-{d['freq'][-1]/1e9:.1f} GHz, "
                 f"N={S.shape[0]}, df={df/1e6:.0f} MHz",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = fig_dir / f"{run}_sample_time_{subj}_{ses}_{cls}_{rep}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    run = cfg["run_name"]
    root = Path(cfg["data"]["root"])
    proc_root = Path(cfg["data"]["processed_root"]) / run
    fig_dir = Path("outputs/figures") / run
    fig_dir.mkdir(parents=True, exist_ok=True)

    labels = load_label_map(root)
    sym = cfg["pipeline"]["symmetrize"]
    chan_note = "S_avg=S21" if sym == "s21_only" else "S_avg=(S21+S12)/2"
    smooth_note = cfg["pipeline"]["smooth"]["type"]

    groups, data, t_ns, df = [], [], None, None
    for subj in cfg["data"]["subjects"]:
        for ses in cfg["data"]["sessions"]:
            base = proc_root / subj / ses / "VNA_Data"
            if not base.is_dir():
                print(f"[warn] missing {base}, skipped")
                continue
            M, t, d = load_group(base, subj, ses)
            groups.append(f"{subj}/{ses}")
            data.append(M)
            t_ns, df = t, d

    if not data:
        sys.exit("[abort] no processed data found")

    nC = len(CHANNELS)
    nG = len(groups)

    # shared color scale per channel
    vmax = [np.nanmax(np.stack([data[k][:, :, j] for k in range(nG)])) for j in range(nC)]

    fig, axes = plt.subplots(nC, nG, figsize=(3.0 * nG, 10.5),
                             sharex=True, sharey=True, squeeze=False)
    x0, x1 = t_ns[0], t_ns[-1]

    for j in range(nC):
        for k in range(nG):
            ax = axes[j][k]
            im = ax.imshow(data[k][:, :, j], aspect="auto", origin="lower",
                           cmap="viridis", interpolation="nearest",
                           vmin=0.0, vmax=vmax[j],
                           extent=[x0, x1, -0.5, 49.5])
            if j == 0:
                ax.set_title(groups[k], fontsize=10)
            if k == 0:
                ax.set_ylabel(f"{CHANNELS[j]}\nclass")
        fig.colorbar(im, ax=axes[j].tolist(), pad=0.01, shrink=0.9,
                     label=f"|{CHANNELS[j]}(t)|")

    # sparse y ticks on the left-most column of EVERY channel row
    tick_classes = [1, 10, 20, 30, 40, 50]
    for j in range(nC):
        ax_left = axes[j][0]
        ax_left.set_yticks([c - 1 for c in tick_classes])
        ax_left.set_yticklabels([f"C{c:02d}" for c in tick_classes], fontsize=7)
        ax_left.tick_params(labelleft=True)  # force labels visible on shared axes
    for k in range(nG):
        axes[-1][k].set_xlabel("time (ns)")

    fig.suptitle(f"All-classes mean time-domain response (whole dataset)\n"
                 f"mean of R01-R10 | run: {run} | {chan_note} | smooth: {smooth_note} | "
                 f"df={df/1e6:.0f} MHz | N={data[0].shape[1]}",
                 fontsize=12)
    out = fig_dir / f"{run}_allclasses.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig] {out}")

    # ---- 2. single-sample IFFT curves ----
    plot_sample_curves(cfg, proc_root, fig_dir, run, labels, chan_note, smooth_note)

    # ---- 3. single-sample frequency-domain S-params (before IFFT) ----
    plot_sample_freq_curves(cfg, proc_root, fig_dir, run, labels, chan_note,
                            smooth_note)


if __name__ == "__main__":
    main()
