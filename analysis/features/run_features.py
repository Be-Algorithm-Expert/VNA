# -*- coding: utf-8 -*-
"""
Feature-ablation driver (parallel to analysis/ablation/run_ablation.py).

Reads configs/features/full.yaml, keeps the BASE enhancement chain fixed, and
derives variants only from the terminal `time_feats` step:

  feat_f0               base chain, NO time_feats        (= Re/Im baseline)
  feat_f1 .. feat_f8    one feature group appended
  feat_fall             all 8 groups appended
  feat_fsel             selected top columns appended (cep + mid cum_energy)
  feat_fall_replace     all 8 groups, mode=replace (scalar features only)
  feat_fno_<group>      all minus one group (reverse ablation)

Each variant runs L0 + L1 and is summarized to outputs/analysis/features_summary.csv.

Usage:
  python analysis/features/run_features.py --list
  python analysis/features/run_features.py --only feat_fall,feat_fsel
  python analysis/features/run_features.py
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
FULL_CFG = ROOT / "configs/features/full.yaml"
CFG_OUT_DIR = ROOT / "outputs/analysis/feature_configs"
L0_DIR = ROOT / "outputs/analysis/l0"
L1_DIR = ROOT / "outputs/analysis/l1"

GROUPS = ["peak_delay", "echo_width", "decay_slope", "cum_energy",
          "group_delay", "peak_seq", "cepstrum"]


def _tf(groups, mode="append", select=None):
    p = {"mode": mode, "groups": groups, "cum_bins": 16, "n_peaks": 5,
         "cep_bins": 16}
    if select:
        p["select"] = select
    return {"name": "time_feats", "params": p}


def build_variants(base_steps):
    v = [("feat_f0", "baseline", base_steps)]                       # no feats
    for g in GROUPS:                                                 # single
        v.append((f"feat_f1_{g}" if False else f"feat_{g}", "single",
                  base_steps + [_tf([g])]))
    v.append(("feat_fall", "all", base_steps + [_tf(GROUPS)]))       # all
    v.append(("feat_fall_replace", "mode",
              base_steps + [_tf(GROUPS, mode="replace")]))
    for g in GROUPS:                                                 # removed
        rest = [x for x in GROUPS if x != g]
        v.append((f"feat_fno_{g}", "removed", base_steps + [_tf(rest)]))
    return v


def run_one(run_name, kind, steps, base_cfg, force=False):
    cfg_path = CFG_OUT_DIR / f"{run_name}.yaml"
    CFG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = deepcopy(base_cfg)
    cfg["run_name"] = run_name
    cfg["enhance"] = {"steps": steps}
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")

    l0 = L0_DIR / run_name / "report.txt"
    l1 = L1_DIR / f"{run_name}__time_reim" / "metrics.json"
    if force or not (l0.exists() and l1.exists()):
        for script in ("analysis/l0/check.py", "analysis/l1/eval_pipeline.py"):
            r = subprocess.run([sys.executable, str(ROOT / script),
                                "--config", str(cfg_path)],
                               cwd=ROOT, capture_output=True, text=True)
            if r.returncode not in (0, 2):
                print(f"    [error] {script} exit {r.returncode}\n{r.stderr[-2000:]}")
                return None
    return collect(run_name, kind, l0, l1)


def collect(run_name, kind, l0_report, l1_metrics):
    row = {"run_name": run_name, "kind": kind}
    if l1_metrics.exists():
        m = json.loads(l1_metrics.read_text(encoding="utf-8"))
        for key in ("lda_logo", "lda_pooled", "knn_logo", "knn_pooled",
                    "drift_ratio", "fisher_mean"):
            for ch, val in (m.get(key) or {}).items():
                row[f"{key}_{ch}"] = val
    checks = L0_DIR / run_name / "checks.json"
    if checks.exists():
        c = json.loads(checks.read_text(encoding="utf-8"))
        row["verdict"] = c.get("verdict")
        row["n_warn"] = sum(1 for x in c.get("checks", {}).values()
                            if x.get("status") == "WARN")
        ten = c.get("checks", {}).get("tensor", {}).get("data", {})
        row["informative_frac"] = ten.get("informative_frac")
        row["tensor_shape"] = "x".join(map(str, ten.get("shape", [])))
    return row


FIELDS = ["run_name", "kind", "verdict", "n_warn", "informative_frac",
          "tensor_shape", "lda_logo_S11", "knn_logo_S11", "lda_pooled_S11",
          "knn_pooled_S11", "drift_ratio_S11", "fisher_mean_S11"]
SUMMARY = ROOT / "outputs/analysis/features_summary.csv"


def write_summary(rows):
    if not rows:
        return
    with SUMMARY.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"-> {SUMMARY} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    base_cfg = yaml.safe_load(FULL_CFG.read_text(encoding="utf-8"))
    all_steps = base_cfg["enhance"]["steps"]
    tf_step = next(s for s in all_steps if s["name"] == "time_feats")
    base_steps = [s for s in all_steps if s["name"] != "time_feats"]
    variants = build_variants(base_steps)

    if args.list:
        for name, kind, steps in variants:
            tf = next((s for s in steps if s["name"] == "time_feats"), None)
            desc = (f"groups={tf['params'].get('groups')} "
                    f"mode={tf['params'].get('mode')}"
                    if tf else "no time_feats")
            print(f"{name:<24}{kind:<10}{desc}")
        print(f"\n{len(variants)} variants")
        return

    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    todo = [v for v in variants if not wanted or v[0] in wanted]
    rows, t0 = [], time.time()
    for i, (name, kind, steps) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {name}")
        row = run_one(name, kind, steps, base_cfg, force=args.force)
        if row is None:
            row = {"run_name": name, "kind": kind, "verdict": "ERROR"}
        rows.append(row)
        print(f"    verdict={row.get('verdict')} pooled_knn="
              f"{row.get('knn_pooled_S11')} logo_knn={row.get('knn_logo_S11')} "
              f"drift={row.get('drift_ratio_S11')}")
    write_summary(rows)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
