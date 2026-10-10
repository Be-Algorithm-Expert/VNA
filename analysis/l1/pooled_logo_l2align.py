# -*- coding: utf-8 -*-
"""Pooled & LOGO separability, two inputs:
   (a) plain Re/Im  (97 steps, no time_feats)  — the pooled-optimal input
   (b) Re/Im + cum_energy (113 steps)          — the LSTM config input
Pooled = stratified 5-fold over all samples; LOGO = 6-fold leave-one-group-out.
Classifiers: LDA / kNN with internal StandardScaler. Multi-seed on pooled.
"""
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import (LeaveOneGroupOut, StratifiedKFold,
                                     cross_val_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from build_baseline_tensors import build_dataset   # noqa: E402

CFG = yaml.safe_load((ROOT / "training/protocol1/config_S11_only_r9_cum.yaml")
                     .read_text(encoding="utf-8"))


def run(cfg, tag):
    X, y, meta = build_dataset(cfg, "time")
    Xf = X.reshape(X.shape[0], -1)
    gid = np.array([f"{m[0]}/{m[1]}" for m in meta])
    print(f"\n== {tag}  X{X.shape} -> flat {Xf.shape} ==")
    for name, mk in [("LDA", lambda: make_pipeline(StandardScaler(),
                                                   LinearDiscriminantAnalysis())),
                     ("kNN", lambda: make_pipeline(StandardScaler(),
                                                   KNeighborsClassifier(n_neighbors=5)))]:
        pooled = [cross_val_score(mk(), Xf, y,
                                  cv=StratifiedKFold(5, shuffle=True,
                                                     random_state=s)).mean()
                  for s in (0, 1, 2)]
        logo = cross_val_score(mk(), Xf, y, groups=gid, cv=LeaveOneGroupOut())
        print(f"{name}: pooled {np.mean(pooled):.3f}±{np.std(pooled):.3f} "
              f"| LOGO {logo.mean():.3f}")
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    scores = cross_val_score(knn, Xf, y, groups=gid, cv=LeaveOneGroupOut())
    folds = sorted(set(gid))
    print("  LOGO folds (kNN): " +
          "  ".join(f"{g.split('/')[0][-1]}{g.split('/')[1][-2:]}:{s:.3f}"
                    for g, s in zip(folds, scores)))


# (a) plain Re/Im: drop the time_feats step
cfg_a = yaml.safe_load(yaml.safe_dump(CFG))
cfg_a["enhance"]["steps"] = [s for s in cfg_a["enhance"]["steps"]
                             if s.get("name") != "time_feats"]
run(cfg_a, "plain Re/Im (97 steps)")

# (b) with cum_energy (the LSTM config)
run(CFG, "Re/Im + cum_energy (113 steps)")
