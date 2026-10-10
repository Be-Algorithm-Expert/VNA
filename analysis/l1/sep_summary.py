# -*- coding: utf-8 -*-
"""97-dim (plain Re/Im, no time_feats) separability summary:
   per-group (protocol-1 split), pooled, LOGO — one clean table.
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
CFG["enhance"]["steps"] = [s for s in CFG["enhance"]["steps"]
                           if s.get("name") != "time_feats"]   # drop cum -> 97 steps
X, y, meta = build_dataset(CFG, "time")
Xf = X.reshape(X.shape[0], -1)
gid = np.array([f"{m[0]}/{m[1]}" for m in meta])
print(f"tensor X{X.shape} -> flat {Xf.shape}\n")

def knn(): return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
def lda(): return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())

# ---- per-group (protocol-1 split: R01-07 train / R09-10 test)
print("== per-group (protocol-1 split) ==")
print(f"{'group':<12}{'kNN_tr':>8}{'kNN_te':>8}{'LDA_tr':>8}{'LDA_te':>8}")
k, l = [], []
for subj in ["S001", "S002", "S003"]:
    for ses in ["SES01", "SES02"]:
        tr = [i for i, m in enumerate(meta) if m[0] == subj and m[1] == ses and m[3] + 1 <= 7]
        te = [i for i, m in enumerate(meta) if m[0] == subj and m[1] == ses and m[3] + 1 >= 9]
        kk = knn().fit(Xf[tr], y[tr]); ll = lda().fit(Xf[tr], y[tr])
        kte, lte = kk.score(Xf[te], y[te]), ll.score(Xf[te], y[te])
        k.append(kte); l.append(lte)
        print(f"{subj}/{ses:<7}{kk.score(Xf[tr], y[tr]):8.3f}{kte:8.3f}"
              f"{ll.score(Xf[tr], y[tr]):8.3f}{lte:8.3f}")
print(f"{'mean':<12}{'':8}{np.mean(k):8.3f}{'':8}{np.mean(l):8.3f}"
      f"   (kNN std {np.std(k):.3f}, LDA std {np.std(l):.3f})")

# ---- pooled (multi-seed)
print("\n== pooled (stratified 5-fold, 3 seeds) ==")
for name, mk in [("LDA", lda), ("kNN", knn)]:
    p = [cross_val_score(mk(), Xf, y, cv=StratifiedKFold(5, shuffle=True,
                                                        random_state=s)).mean()
         for s in (0, 1, 2)]
    print(f"{name}: {np.mean(p):.3f} ± {np.std(p):.3f}")

# ---- LOGO
print("\n== LOGO (leave-one-group-out) ==")
for name, mk in [("LDA", lda), ("kNN", knn)]:
    s = cross_val_score(mk(), Xf, y, groups=gid, cv=LeaveOneGroupOut())
    print(f"{name}: {s.mean():.3f}   folds=" +
          " ".join(f"{g.split('/')[0][-1]}{g.split('/')[1][-2:]}:{v:.3f}"
                   for g, v in zip(sorted(set(gid)), s)))
