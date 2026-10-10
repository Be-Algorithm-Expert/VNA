# -*- coding: utf-8 -*-
"""Per-group separability: LDA / kNN on each (subject, session) group's own
protocol-1 split (R01-07 train / R08 val / R09-10 test), same tensor as the
LSTM sees (S11-only + r9 chain + cum_energy append, z-scored with train
stats). Also the within-group kNN on plain Re/Im (no features) for reference.
"""
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from build_baseline_tensors import build_dataset   # noqa: E402


def per_group_table(cfg, tag):
    X, y, meta = build_dataset(cfg, "time")
    Xf = X.reshape(X.shape[0], -1)
    print(f"\n== {tag}  X{X.shape} ==")
    print(f"{'group':<12}{'kNN_tr':>8}{'kNN_te':>8}{'LDA_tr':>8}{'LDA_te':>8}")
    tes_k, tes_l = [], []
    for subj in ["S001", "S002", "S003"]:
        for ses in ["SES01", "SES02"]:
            tr = [i for i, m in enumerate(meta) if m[0] == subj and m[1] == ses and m[3] + 1 <= 7]
            te = [i for i, m in enumerate(meta) if m[0] == subj and m[1] == ses and m[3] + 1 >= 9]
            knn = make_pipeline(StandardScaler(),
                                KNeighborsClassifier(n_neighbors=5)).fit(Xf[tr], y[tr])
            lda = make_pipeline(StandardScaler(),
                                LinearDiscriminantAnalysis()).fit(Xf[tr], y[tr])
            ktr, kte = knn.score(Xf[tr], y[tr]), knn.score(Xf[te], y[te])
            ltr, lte = lda.score(Xf[tr], y[tr]), lda.score(Xf[te], y[te])
            tes_k.append(kte); tes_l.append(lte)
            print(f"{subj}/{ses:<7}{ktr:8.3f}{kte:8.3f}{ltr:8.3f}{lte:8.3f}")
    print(f"{'mean':<12}{'':8}{np.mean(tes_k):8.3f}{'':8}{np.mean(tes_l):8.3f}"
          f"   (kNN std {np.std(tes_k):.3f}, LDA std {np.std(tes_l):.3f})")


cfg = yaml.safe_load((ROOT / "training/protocol1/config_S11_only_r9_cum.yaml")
                     .read_text(encoding="utf-8"))

# with cum_energy features (what the LSTM eats)
per_group_table(cfg, "S11-only + r9 + cum_energy (LSTM input, 108x2)")

# plain Re/Im reference: drop the time_feats step
cfg2 = yaml.safe_load(yaml.safe_dump(cfg))
cfg2["enhance"]["steps"] = [s for s in cfg["enhance"]["steps"]
                            if s.get("name") != "time_feats"]
per_group_table(cfg2, "S11-only + r9, plain Re/Im (92x2)")
