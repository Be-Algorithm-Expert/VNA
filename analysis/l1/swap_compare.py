# -*- coding: utf-8 -*-
"""S11-only: r9 order vs swap (diff_div before diff_sub) — L1 comparison."""
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import (LeaveOneGroupOut, StratifiedKFold,
                                     cross_val_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from build_baseline_tensors import build_dataset   # noqa: E402

EPS = 1e-12
BASE = {
    "channel_select": {"drop": ["S_avg", "S22"]},
    "bootstrap_avg": {"n_draws": 10, "seed": 0},
    "diff_sub": {"ref": "group_median"},
    "diff_div": {"ref": "group_median", "mask_db": -60.0},
    "ifft_window": {"type": "kaiser", "beta": 6.0},
    "delay_align": {},
    "early_gate": {"depth_db": -20.0, "mode": "crop", "quantile": 0.9},
    "amp_norm": {"mode": "l2"},
    "step_crop": {"frac": 0.01},
}
ORDER_R9 = ["channel_select", "bootstrap_avg", "diff_sub", "diff_div",
            "ifft_window", "delay_align", "early_gate", "amp_norm", "step_crop"]
ORDER_SWAP = ["channel_select", "bootstrap_avg", "diff_div", "diff_sub",
              "ifft_window", "delay_align", "early_gate", "amp_norm", "step_crop"]


def make_cfg(order):
    cfg = {
        "run_name": "tmp",
        "data": {
            "root": "f:/H坏盘备份/VNA_Cls_2026/dataset_3_2_50_10",
            "processed_root": "f:/H坏盘备份/VNA_Cls_2026/processed",
            "run_name": "baseline_s21",
            "subjects": ["S001", "S002", "S003"],
            "sessions": ["SES01", "SES02"],
        },
        "enhance": {"steps": [{"name": n, "params": BASE[n]} for n in order]},
    }
    return cfg


def drift_ratio(db, meta):
    classes = np.unique([m[2] for m in meta])
    groups = sorted({(m[0], m[1]) for m in meta})
    mu = {}
    for g in groups:
        idx = [i for i, m in enumerate(meta) if (m[0], m[1]) == g]
        for c in classes:
            sel = [i for i in idx if meta[i][2] == c]
            mu[(g, c)] = db[sel].mean(axis=0)
    within = []
    for g in groups:
        cent = np.mean([mu[(g, c)] for c in classes], axis=0)
        within.append(np.mean([np.sqrt(np.mean((mu[(g, c)] - cent) ** 2, axis=0))
                               for c in classes], axis=0))
    drift = []
    for c in classes:
        cent = np.mean([mu[(g, c)] for g in groups], axis=0)
        drift.append(np.mean([np.sqrt(np.mean((mu[(g, c)] - cent) ** 2, axis=0))
                              for g in groups], axis=0))
    return float(np.mean(drift) / (np.mean(within) + EPS))


def evaluate(cfg):
    X, y, meta = build_dataset(cfg, "time")
    C = X[:, :, 0::2] + 1j * X[:, :, 1::2]          # complex (n, T, ch)
    db = 20 * np.log10(np.abs(C[:, :, 0]) + EPS)    # S11 magnitude dB
    Xf = X.reshape(X.shape[0], -1)
    gid = np.array([f"{m[0]}/{m[1]}" for m in meta])
    pooled, logo = [], []
    for seed in (0, 1, 2):
        clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
        p = cross_val_score(clf, Xf, y, cv=StratifiedKFold(5, shuffle=True,
                                                           random_state=seed))
        l = cross_val_score(clf, Xf, y, groups=gid, cv=LeaveOneGroupOut())
        pooled.append(p.mean()); logo.append(l.mean())
    return (X.shape, float(np.mean(pooled)), float(np.std(pooled)),
            float(np.mean(logo)), float(np.std(logo)), drift_ratio(db, meta))


for name, order in [("r9 (先减后除)", ORDER_R9), ("swap (先除后减)", ORDER_SWAP)]:
    shape, pm, ps, lm, ls, dr = evaluate(make_cfg(order))
    print(f"{name:16s} shape={shape} pooled_kNN={pm:.3f}±{ps:.3f} "
          f"LOGO_kNN={lm:.3f}±{ls:.3f} drift={dr:.3f}")
