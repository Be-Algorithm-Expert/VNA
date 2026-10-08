# -*- coding: utf-8 -*-
"""
Ablation driver: derive every variant from configs/ablation/full.yaml, run
L0 (health) + L1 (separability) on each, and collect a master summary.

Variant families (31 runs total):
  ablation_r0            everything OFF (= legacy baseline behavior)
  ablation_r1 .. r9      cumulative chain: first N steps of the full chain
  abl_only_<step>        single step ON (one-at-a-time)
  abl_no_<step>          full chain minus one step (reverse ablation)
  abl_swap_<pair>        full chain with two adjacent-in-role steps swapped
                           swap_subdiv    diff_sub <-> diff_div
                           swap_divwin    diff_div <-> ifft_window
                           swap_gatenorm  early_gate <-> amp_norm

Seed repeats for L2 training are NOT handled here (L0/L1 are deterministic).

Usage:
  python analysis/ablation/run_ablation.py --list
  python analysis/ablation/run_ablation.py --only ablation_r0,ablation_r1
  python analysis/ablation/run_ablation.py               # all 31 (slow)
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
FULL_CFG = ROOT / "configs/ablation/full.yaml"
CFG_OUT_DIR = ROOT / "outputs/analysis/ablation_configs"
L0_DIR = ROOT / "outputs/analysis/l0"
L1_DIR = ROOT / "outputs/analysis/l1"

SWAP_PAIRS = {
    "swap_subdiv": ("diff_sub", "diff_div"),
    "swap_divwin": ("diff_div", "ifft_window"),
    "swap_gatenorm": ("early_gate", "amp_norm"),
}


# ---------------------------------------------------------------- variants
def build_variants(full_steps):
    """Return ordered list of (run_name, kind, steps)."""
    names = [s["name"] for s in full_steps]
    variants = []

    # cumulative chain: r0 empty, r1..rN prefixes, rN == full
    variants.append(("ablation_r0", "baseline", []))
    for i in range(1, len(full_steps) + 1):
        variants.append((f"ablation_r{i}", "cumulative", deepcopy(full_steps[:i])))

    # single-step ON
    for s in full_steps:
        variants.append((f"abl_only_{s['name']}", "single", [deepcopy(s)]))

    # reverse ablation: full minus one
    for s in full_steps:
        rest = [deepcopy(t) for t in full_steps if t["name"] != s["name"]]
        variants.append((f"abl_no_{s['name']}", "removed", rest))

    # order swaps
    for tag, (a, b) in SWAP_PAIRS.items():
        steps = deepcopy(full_steps)
        ia, ib = names.index(a), names.index(b)
        steps[ia], steps[ib] = steps[ib], steps[ia]
        variants.append((f"abl_{tag}", "swap", steps))

    return variants


# ---------------------------------------------------------------- runner
def run_one(run_name, kind, steps, base_cfg, force=False):
    """Write the variant config, then run L0 and L1 as subprocesses."""
    cfg_path = CFG_OUT_DIR / f"{run_name}.yaml"
    CFG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = deepcopy(base_cfg)
    cfg["run_name"] = run_name
    cfg["enhance"] = {"steps": steps}
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")

    l0_report = L0_DIR / run_name / "report.txt"
    l1_metrics = L1_DIR / f"{run_name}__time_reim" / "metrics.json"
    if force or not (l0_report.exists() and l1_metrics.exists()):
        for script in ("analysis/l0/check.py", "analysis/l1/eval_pipeline.py"):
            cmd = [sys.executable, str(ROOT / script), "--config", str(cfg_path)]
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if r.returncode not in (0, 2):        # 2 = BROKEN verdict, recorded
                print(f"    [error] {script} exit {r.returncode}\n{r.stderr[-2000:]}")
                return None
    return collect(run_name, kind, l0_report, l1_metrics)


def collect(run_name, kind, l0_report, l1_metrics):
    """Pull the headline numbers out of the L0/L1 outputs."""
    row = {"run_name": run_name, "kind": kind}
    if l1_metrics.exists():
        m = json.loads(l1_metrics.read_text(encoding="utf-8"))
        for key in ("lda_logo", "lda_pooled", "knn_logo", "knn_pooled",
                    "drift_ratio", "fisher_mean"):
            for ch, v in (m.get(key) or {}).items():
                row[f"{key}_{ch}"] = v
        row["pca_pc12"] = m.get("pca_pc12_variance")
    checks = L0_DIR / run_name / "checks.json"
    if checks.exists():
        c = json.loads(checks.read_text(encoding="utf-8"))
        row["verdict"] = c.get("verdict")
        row["n_warn"] = sum(1 for v in c.get("checks", {}).values()
                            if v.get("status") == "WARN")
        ten = c.get("checks", {}).get("tensor", {}).get("data", {})
        row["informative_frac"] = ten.get("informative_frac")
        row["tensor_shape"] = "x".join(map(str, ten.get("shape", [])))
    if not l1_metrics.exists():
        row["verdict"] = row.get("verdict", "BROKEN")
    return row


# ---------------------------------------------------------------- summary
SUMMARY_PATH = ROOT / "outputs/analysis/ablation_summary.csv"

FIELDS = ["run_name", "kind", "verdict", "n_warn", "informative_frac",
          "tensor_shape", "lda_logo_S11", "lda_logo_S22", "knn_logo_S11",
          "knn_logo_S22", "lda_pooled_S11", "lda_pooled_S22",
          "drift_ratio_S11", "drift_ratio_S22", "fisher_mean_S11",
          "fisher_mean_S22", "pca_pc12"]


def write_summary(rows):
    if not rows:
        print("no results collected")
        return
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    extra = sorted({k for r in rows for k in r} - set(FIELDS))
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS + extra, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"-> {SUMMARY_PATH} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated run_names")
    ap.add_argument("--list", action="store_true", help="list variants, run nothing")
    ap.add_argument("--force", action="store_true",
                    help="rerun even if outputs already exist")
    args = ap.parse_args()

    base_cfg = yaml.safe_load(FULL_CFG.read_text(encoding="utf-8"))
    full_steps = base_cfg.get("enhance", {}).get("steps", [])
    variants = build_variants(full_steps)

    if args.list:
        for name, kind, steps in variants:
            on = ",".join(s["name"] for s in steps) or "(none)"
            print(f"{name:<32} {kind:<10} {on}")
        print(f"\n{len(variants)} variants")
        return

    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    todo = [v for v in variants if not wanted or v[0] in wanted]
    print(f"{len(todo)} variant(s) to run")

    rows = []
    t0 = time.time()
    for i, (name, kind, steps) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {name} "
              f"({'|'.join(s['name'] for s in steps) or 'no steps'})")
        row = run_one(name, kind, steps, base_cfg, force=args.force)
        if row is None:
            row = {"run_name": name, "kind": kind, "verdict": "ERROR"}
        rows.append(row)
        print(f"    verdict={row.get('verdict')} "
              f"lda_logo S11/S22={row.get('lda_logo_S11')}/"
              f"{row.get('lda_logo_S22')} drift={row.get('drift_ratio_S11')}")
    write_summary(rows)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
