# -*- coding: utf-8 -*-
"""
L0 health check: is a processed run trustworthy before we evaluate/train on it?

Read-only. Checks the persisted npz (layer-1 output) AND the derived tensors
(layer-2 output, src/build_baseline_tensors.py) of one run:

  [1] finite          NaN/Inf in raw S
  [2] files & axis    per-group coverage, shape, monotonic freq axis, df
  [3] dynamic range   per-channel |S| dB stats, channels stuck at noise floor
  [4] sample outliers energy far from its (group, class) median; exact duplicates
  [5] channel drift   between-group shifts of channel correlations / mean curves
  [6] tensor (L2)     finite + informative-step count of the derived tensor

Levels: PASS < WARN < FAIL. Verdict: USABLE / USABLE (with warnings) / BROKEN.

Outputs per run (each run gets its own directory, nothing is overwritten):
  outputs/analysis/l0/<run_name>/report.txt           human report (also printed)
  outputs/analysis/l0/<run_name>/flagged_samples.csv   flagged samples + reason
  outputs/analysis/l0/<run_name>/checks.json           machine-readable

Exit code 2 when the verdict is BROKEN, so a pipeline can gate on it.

Usage:
  python analysis/l0/check.py --config configs/baseline_s21.yaml
  python analysis/l0/check.py --config configs/baseline_s21.yaml --domain freq
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from build_baseline_tensors import build_dataset, iter_samples  # noqa: E402

EPS = 1e-12

# ---- thresholds (single place to tune) ---------------------------------------
NOISE_FLOOR_DB = -90.0   # |S| below this counts as noise floor
DEAD_CHAN_FRAC = 0.50    # channel is dead if > 50% of its points sit at the floor
WEAK_CHAN_MEDIAN_DB = -60.0   # ... or if its median is already this weak
ENERGY_DEV_DB = 20.0     # flag a sample when its energy deviates from its
                         # (group, class, channel) median by more than this
MAX_FLAGGED_FRAC = 0.10  # more than 10% of samples flagged -> BROKEN
CORR_DRIFT = 0.15        # between-group spread of channel correlations -> WARN
MEAN_DRIFT_DB = 3.0      # group mean curve vs global mean (dB RMS) -> WARN
FLAT_STEP_FRAC = 0.50    # tensor: fewer than 50% informative steps -> WARN

STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


def db20(x):
    return 20.0 * np.log10(np.maximum(x, EPS))


# ------------------------------------------------------------------ raw scan
def scan_raw(proc_root, subjects, sessions):
    """One pass over all npz files: raw arrays + everything needed downstream."""
    acc = {
        "mags": [], "meta": [], "nonfinite": [], "axis_issues": [],
        "dup_flags": [], "energy_by_key": defaultdict(list),
        "channels": None, "ref": None, "n_files": 0,
    }
    hashes = {}
    for f, subj, ses, cls, rep in iter_samples(proc_root, subjects, sessions):
        d = np.load(f, allow_pickle=False)
        S = d["S"]                                  # (N, 3) complex
        freq, df = d["freq"], float(d["df"])
        acc["n_files"] += 1

        if acc["channels"] is None:
            acc["channels"] = [str(c) for c in d["channels"]]
            acc["ref"] = (freq, df, S.shape)
        if not np.isfinite(S).all():
            acc["nonfinite"].append((subj, ses, cls, rep))
        if not (np.all(np.diff(freq) > 0)
                and np.array_equal(freq, acc["ref"][0])
                and abs(df - acc["ref"][1]) < 1e-6
                and S.shape == acc["ref"][2]):
            acc["axis_issues"].append((subj, ses, cls, rep))

        h = hashlib.md5(np.ascontiguousarray(S).tobytes()).hexdigest()
        if h in hashes:
            acc["dup_flags"].append(((subj, ses, cls, rep), hashes[h]))
        else:
            hashes[h] = (subj, ses, cls, rep)

        acc["mags"].append(np.abs(S).astype(np.float32))
        acc["meta"].append((subj, ses, cls, rep))
        e_db = 10.0 * np.log10(np.maximum((np.abs(S) ** 2).mean(axis=0), EPS))
        acc["energy_by_key"][(subj, ses, cls)].append((rep, e_db))
    acc["mags"] = np.stack(acc["mags"]) if acc["mags"] else np.empty((0, 0, 3))
    return acc


# -------------------------------------------------------------------- checks
def check_finite(acc):
    n = len(acc["nonfinite"])
    status = "FAIL" if n else "PASS"
    return status, [f"{n} file(s) with NaN/Inf in raw S"], {"n_bad": n}


def check_files_axis(acc, subjects, sessions):
    lines, worst = [], "PASS"
    if acc["ref"] is None:
        return "FAIL", ["no npz files found"], {"n_groups": 0, "n_files": 0}
    expected = [(s, e) for s in subjects for e in sessions]
    present = {(m[0], m[1]) for m in acc["meta"]}
    missing = [f"{s}/{e}" for s, e in expected if (s, e) not in present]
    if missing:
        worst = "FAIL"
        lines.append(f"missing group dir(s): {', '.join(missing)}")
    per_group = defaultdict(list)
    for subj, ses, cls, rep in acc["meta"]:
        per_group[(subj, ses)].append((cls, rep))
    bad_cov = []
    for g in sorted(per_group):
        items = per_group[g]
        n_cls = len({c for c, _ in items})
        reps = defaultdict(int)
        for c, _ in items:
            reps[c] += 1
        rep_counts = sorted(set(reps.values()))
        if n_cls != 50 or rep_counts != [10]:
            bad_cov.append(f"{g[0]}/{g[1]}: {n_cls} classes, reps {rep_counts}")
    freq, df, shape = acc["ref"] if acc["ref"] else (None, None, None)
    lines.append(f"{len(per_group)} groups x {acc['n_files'] // max(len(per_group), 1)}"
                 f" files | shape {shape} | df={df / 1e6:.1f} MHz"
                 if df is not None else "no files found")
    if acc["axis_issues"]:
        worst = "FAIL"
        lines.append(f"{len(acc['axis_issues'])} file(s) with inconsistent "
                     f"freq axis / df / shape")
    else:
        lines.append("freq axes identical & monotonic across all files")
    if bad_cov:
        worst = max(worst, "WARN", key=STATUS_ORDER.get)
        lines.append("coverage anomalies: " + "; ".join(bad_cov))
    if acc["n_files"] == 0:
        worst = "FAIL"
    return worst, lines, {"n_groups": len(per_group), "n_files": acc["n_files"],
                          "missing_groups": missing}


def check_dynamic_range(acc):
    M = acc["mags"]
    channels = acc["channels"] or ["ch0", "ch1", "ch2"]
    lines, frag, dead = [], {}, []
    if M.size == 0:
        return "FAIL", ["no data"], {"dead_channels": []}
    for j, ch in enumerate(channels):
        d = db20(M[:, :, j].astype(np.float64))
        p1, med, p99 = np.percentile(d, [1, 50, 99])
        frac = float(np.mean(d < NOISE_FLOOR_DB))
        dead_ch = bool(frac > DEAD_CHAN_FRAC or med < WEAK_CHAN_MEDIAN_DB)
        if dead_ch:
            dead.append(ch)
        lines.append((f"      {ch:<8} {'DEAD ' if dead_ch else '     '}"
                      f"median {med:7.1f} dB | P1..P99 {p1:6.1f}..{p99:6.1f} dB"
                      f" | {frac * 100:4.0f}% pts < {NOISE_FLOOR_DB:.0f} dB"))
        frag[ch] = {"median_db": round(float(med), 1),
                    "p1_db": round(float(p1), 1), "p99_db": round(float(p99), 1),
                    "frac_below_floor": round(frac, 3), "dead": dead_ch}
    status = "FAIL" if len(dead) == len(channels) else ("WARN" if dead else "PASS")
    frag["dead_channels"] = dead
    return status, lines, frag


def check_outliers(acc):
    flags = []
    for (subj, ses, cls), lst in acc["energy_by_key"].items():
        if len(lst) < 3:
            continue                              # too few reps to judge
        arr = np.array([e for _, e in lst])
        med = np.median(arr, axis=0)
        for rep, e in lst:
            dev = e - med
            j = int(np.argmax(np.abs(dev)))
            if abs(dev[j]) > ENERGY_DEV_DB:
                flags.append((subj, ses, cls, rep, "energy",
                              f"{acc['channels'][j]}: {e[j]:.1f} dB vs median "
                              f"{med[j]:.1f} dB (dev {dev[j]:+.1f} dB)"))
    for (subj, ses, cls, rep), other in acc["dup_flags"]:
        flags.append((subj, ses, cls, rep, "duplicate",
                     f"identical to {other[0]}_{other[1]}_C{other[2] + 1:02d}"
                     f"_R{other[3] + 1:02d}"))
    n = acc["n_files"]
    frac = len(flags) / n if n else 0.0
    status = ("FAIL" if frac > MAX_FLAGGED_FRAC
              else ("WARN" if flags else "PASS"))
    lines = [f"{len(flags)} flagged sample(s) "
             f"(energy outliers: {sum(1 for f in flags if f[4] == 'energy')}, "
             f"exact duplicates: {len(acc['dup_flags'])})"]
    return status, lines, {"n_flagged": len(flags), "frac": round(frac, 4)}, flags


def check_drift(acc):
    M, meta = acc["mags"], acc["meta"]
    channels = acc["channels"] or ["ch0", "ch1", "ch2"]
    if M.size == 0:
        return "FAIL", ["no data"], {}
    groups = defaultdict(list)
    for i, (subj, ses, cls, rep) in enumerate(meta):
        groups[(subj, ses)].append(i)

    def corr_per_sample(a, b):
        x = M[:, :, a].astype(np.float64)
        y = M[:, :, b].astype(np.float64)
        xc = x - x.mean(axis=1, keepdims=True)
        yc = y - y.mean(axis=1, keepdims=True)
        den = np.sqrt((xc ** 2).sum(1) * (yc ** 2).sum(1))
        with np.errstate(divide="ignore", invalid="ignore"):
            return (xc * yc).sum(1) / den      # nan if a channel is constant

    lines, frag, worst = [], {}, "PASS"
    corr_spread = 0.0
    for a in range(M.shape[2]):
        for b in range(a + 1, M.shape[2]):
            cs = corr_per_sample(a, b)
            meds = [float(np.nanmedian(cs[groups[g]])) for g in sorted(groups)]
            spread = max(meds) - min(meds) if meds else 0.0
            corr_spread = max(corr_spread, spread)
            frag[f"corr_{channels[a]}~{channels[b]}"] = {
                "group_medians": [round(m, 3) for m in meds],
                "spread": round(spread, 3)}
            if spread > CORR_DRIFT:
                worst = "WARN"
                lines.append(f"      corr({channels[a]},{channels[b]}) group "
                             f"medians {[round(m, 2) for m in meds]} "
                             f"(spread {spread:.2f} > {CORR_DRIFT})")
    mean_dev = 0.0
    gm = {g: M[groups[g]].mean(axis=0) for g in groups}
    glob = np.mean(list(gm.values()), axis=0)
    for g in sorted(gm):
        dev = np.sqrt(np.mean((db20(gm[g]) - db20(glob)) ** 2, axis=0))
        mean_dev = max(mean_dev, float(dev.max()))
        if float(dev.max()) > MEAN_DRIFT_DB:
            worst = "WARN"
            lines.append(f"      {g[0]}/{g[1]} mean curve deviates "
                         f"{dev.max():.1f} dB (max channel) from global mean")
    lines.insert(0, f"max corr spread {corr_spread:.2f} "
                    f"(threshold {CORR_DRIFT}) | max group-mean dev "
                    f"{mean_dev:.1f} dB (threshold {MEAN_DRIFT_DB} dB)")
    frag["max_corr_spread"] = round(corr_spread, 3)
    frag["max_mean_dev_db"] = round(mean_dev, 2)
    return worst, lines, frag


def check_tensor(cfg, domain):
    """Layer-2 output: build the real training tensor and inspect it."""
    X, y, meta = build_dataset(cfg, domain)
    lines, frag = [], {}
    n_bad = int((~np.isfinite(X)).sum())
    frag["shape"] = list(X.shape)
    frag["n_nonfinite"] = n_bad
    status = "FAIL" if n_bad else "PASS"
    lines.append(f"X{X.shape} | finite: {n_bad == 0} | "
                 f"mean {X.mean():+.3e} std {X.std():.3e} "
                 f"absmax {abs(X).max():.3e}")
    sd = X.astype(np.float64).std(axis=0).mean(axis=1)     # per-step std
    alive = int(np.sum(sd > 0.01 * sd.max()))
    n_steps = X.shape[1]
    frac = alive / n_steps
    frag["informative_steps"] = alive
    frag["informative_frac"] = round(frac, 3)
    if frac < FLAT_STEP_FRAC:
        status = max(status, "WARN", key=STATUS_ORDER.get)
        lines.append(f"only {alive}/{n_steps} informative steps "
                     f"({frac * 100:.0f}% < {FLAT_STEP_FRAC * 100:.0f}%) - "
                     f"most inputs are near-constant for the model")
    else:
        lines.append(f"informative steps {alive}/{n_steps} ({frac * 100:.0f}%)")
    return status, lines, frag


# ------------------------------------------------------------------- report
def render_report(run, domain, n_files, channels, results, flagged, verdict,
                  usable, dead, secs):
    r = []
    r.append(f"==== L0 health report : {run} ====")
    r.append(f"generated : {datetime.now():%Y-%m-%d %H:%M:%S} ({secs:.0f}s scan)")
    r.append(f"dataset   : {n_files} samples | {channels} | tensor domain={domain}")
    r.append("")
    titles = ["[1] finite (raw S)    ", "[2] files & freq axis ",
              "[3] dynamic range     ", "[4] sample outliers   ",
              "[5] channel drift     ", "[6] tensor (layer 2)  "]
    for title, (status, lines, _) in zip(titles, results):
        r.append(f"{title}. {status:<4}  {lines[0]}")
        for extra in lines[1:]:
            r.append(f"      {extra}")
        r.append("")
    r.append(f"verdict         : {verdict}")
    r.append(f"usable channels : {', '.join(usable) if usable else '(none)'}")
    r.append(f"dead channels   : {', '.join(f'{c} (noise floor)' for c in dead)}"
              if dead else "dead channels   : (none)")
    r.append(f"flagged samples : {len(flagged)}"
             f"  (details: flagged_samples.csv)")
    return "\n".join(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--domain", choices=["freq", "time"], default=None,
                    help="tensor domain to check (default: cfg data.domain / time)")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    run = cfg["run_name"]
    domain = args.domain or cfg.get("data", {}).get("domain", "time")
    proc_root = Path(cfg["data"]["processed_root"]) / run
    out_dir = Path("outputs/analysis/l0") / run
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    acc = scan_raw(proc_root, cfg["data"]["subjects"], cfg["data"]["sessions"])
    secs = time.time() - t0

    outlier_res = check_outliers(acc)
    results = [
        check_finite(acc),
        check_files_axis(acc, cfg["data"]["subjects"], cfg["data"]["sessions"]),
        check_dynamic_range(acc),
        outlier_res[:3],
        check_drift(acc),
        check_tensor(cfg, domain),
    ]
    flagged = outlier_res[3]

    statuses = [s for s, _, _ in results]
    worst = max(statuses, key=STATUS_ORDER.get) if statuses else "FAIL"
    if worst == "FAIL":
        verdict = "BROKEN"
    elif worst == "WARN":
        verdict = "USABLE (with warnings)"
    else:
        verdict = "USABLE"

    dr = results[2][2]
    dead = dr.get("dead_channels", [])
    usable = [c for c in (acc["channels"] or []) if c not in dead]

    report = render_report(run, domain, acc["n_files"], acc["channels"],
                           results, flagged, verdict, usable, dead, secs)
    print(report)

    with (out_dir / "report.txt").open("w", encoding="utf-8") as fh:
        fh.write(report + "\n")

    with (out_dir / "flagged_samples.csv").open("w", newline="",
                                                 encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "session", "class", "rep", "check", "detail"])
        for subj, ses, cls, rep, check, detail in flagged:
            w.writerow([subj, ses, f"C{cls + 1:02d}", f"R{rep + 1:02d}",
                        check, detail])

    checks = {name: {"status": s, "lines": lines, "data": data}
              for (s, lines, data), name in zip(
                  results, ["finite", "files_axis", "dynamic_range",
                            "sample_outliers", "channel_drift", "tensor"])}
    payload = {
        "run_name": run, "domain": domain,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_samples": acc["n_files"], "verdict": verdict,
        "usable_channels": usable, "dead_channels": dead,
        "thresholds": {"noise_floor_db": NOISE_FLOOR_DB,
                       "dead_chan_frac": DEAD_CHAN_FRAC,
                       "weak_chan_median_db": WEAK_CHAN_MEDIAN_DB,
                       "energy_dev_db": ENERGY_DEV_DB,
                       "max_flagged_frac": MAX_FLAGGED_FRAC,
                       "corr_drift": CORR_DRIFT,
                       "mean_drift_db": MEAN_DRIFT_DB},
        "checks": checks,
    }
    with (out_dir / "checks.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"\n-> {out_dir}/report.txt, flagged_samples.csv, checks.json")
    if verdict == "BROKEN":
        sys.exit(2)


if __name__ == "__main__":
    main()
