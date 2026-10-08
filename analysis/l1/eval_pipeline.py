# -*- coding: utf-8 -*-
"""
L1 separability evaluation: how separable is a processed run, before any DL.

Read-only. Uses the SAME tensor construction as training (src/build_baseline_tensors)
so L1 sees exactly what L2 would see.

Feature = Re/Im of each channel (the model input): each sample is (201, 6) =
[Re_Savg, Im_Savg, Re_S11, Im_S11, Re_S22, Im_S22]; per channel that is 402 dims.

fisher ratio and drift ratio are magnitude-domain descriptors and are always
computed on dB |S| per channel (they are per-frequency-point scalar measures;
magnitude is the meaningful scalar there).

Metrics:
  fisher ratio  - per frequency point, between-class / within-class variance
  LDA / kNN     - cross-validated accuracy, two splits:
                    pooled : stratified 5-fold over all samples (ceiling)
                    logo   : leave-one-group-out (train 5 groups, test 1)
  drift ratio   - cross-session drift / within-session separation (>1 = drift dominates)
  PCA           - 2D scatter colored by class and by group
  distance      - class-mean distance matrix, reordered by hierarchical clustering

Outputs per (run, domain), never overwritten:
  outputs/analysis/l1/<run>__<domain>_reim/{report.txt, metrics.json,
    figures/fisher_curves.png, figures/pca_scatter.png, figures/class_distance.png,
    class_distance.csv}
  and one row appended to outputs/analysis/pipeline_comparison.csv

Usage:
  python analysis/l1/eval_pipeline.py --config configs/baseline_s21.yaml
  python analysis/l1/eval_pipeline.py --config configs/baseline_s21.yaml --domain freq
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import (LeaveOneGroupOut, StratifiedKFold,
                                     cross_val_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from build_baseline_tensors import build_dataset, iter_samples  # noqa: E402

# CJK font for the class-label ticks (Windows)
for _f in ["Microsoft YaHei", "SimHei"]:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f] + plt.rcParams["font.sans-serif"]
        break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

EPS = 1e-12
K_NEIGHBORS = 5
N_FOLDS = 5
SEED = 0


# --------------------------------------------------------------- feature load
def load_features(cfg, domain):
    """Return complex C (n, 201, 3), y, meta, freq (GHz), channel names."""
    X, y, meta = build_dataset(cfg, domain)
    C = X[:, :, 0::2] + 1j * X[:, :, 1::2]          # back to complex (n, 201, 3)
    run = cfg["run_name"]
    # data.run_name (optional) selects the PROCESSED directory; run_name names
    # the outputs. Ablation variants share one processed cache this way.
    data_run = cfg.get("data", {}).get("run_name", run)
    proc_root = Path(cfg["data"]["processed_root"]) / data_run
    first = next(iter_samples(proc_root, cfg["data"]["subjects"],
                              cfg["data"]["sessions"]), None)
    if first is not None:
        d = np.load(first[0], allow_pickle=False)
        freq = d["freq"] / 1e9
        channels = [str(c) for c in d["channels"]]
    else:
        freq = np.arange(C.shape[1])
        channels = ["ch0", "ch1", "ch2"]
    # keep channel names in sync with the enhancement chain (channel_select
    # may drop channels, so the tensor can have fewer than the npz lists)
    for s in (cfg.get("enhance") or {}).get("steps") or []:
        if s.get("name") == "channel_select":
            drop = (s.get("params") or {}).get("drop", [])
            channels = [c for c in channels if c not in drop]
    if len(channels) != C.shape[2]:
        channels = [f"ch{j}" for j in range(C.shape[2])]
    return C, y, meta, freq, channels


def db_mag(C):
    return 20.0 * np.log10(np.abs(C) + EPS)


def reim_matrices(C):
    """Per-channel Re/Im matrices F_j (n, 402) - exactly the model input."""
    n_ch = C.shape[2]
    return [np.concatenate([C.real[:, :, j], C.imag[:, :, j]], axis=1)
            for j in range(n_ch)]


def _load_class_labels(cfg):
    """class index (0-based) -> Chinese label, from data root/label_map.csv."""
    labels = {}
    f = Path(cfg["data"]["root"]) / "label_map.csv"
    if not f.exists():
        return labels
    with f.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            labels.setdefault(int(row["class_id"][1:]) - 1, row["label"])
    return labels


# ---------------------------------------------------------------- metrics
def fisher_ratio(db, y):
    """Per frequency point, per channel: between-class / within-class variance."""
    n = len(y)
    classes = np.unique(y)
    means = np.stack([db[y == c].mean(axis=0) for c in classes])     # (C,201,3)
    counts = np.array([(y == c).sum() for c in classes])
    grand = db.mean(axis=0)
    between = ((means - grand[None]) ** 2 * counts[:, None, None]).sum(0) / n
    within = ((db - means[y]) ** 2).sum(0) / (n - len(classes))
    return between / (within + EPS)                                   # (201,3)


def _rms_dist(a, b):
    return np.sqrt(np.mean((a - b) ** 2, axis=0))                      # per channel


def drift_ratio(db, meta):
    """(cross-session drift of class means) / (within-session class separation)."""
    classes = np.unique([m[2] for m in meta])
    groups = sorted({(m[0], m[1]) for m in meta})
    mu = {}
    for g in groups:
        idx = [i for i, m in enumerate(meta) if (m[0], m[1]) == g]
        for c in classes:
            sel = [i for i in idx if meta[i][2] == c]
            mu[(g, c)] = db[sel].mean(axis=0)                          # (201,3)

    within = []
    for g in groups:
        cent = np.mean([mu[(g, c)] for c in classes], axis=0)
        within.append(np.mean([_rms_dist(mu[(g, c)], cent)
                               for c in classes], axis=0))
    drift = []
    for c in classes:
        cent = np.mean([mu[(g, c)] for g in groups], axis=0)
        drift.append(np.mean([_rms_dist(mu[(g, c)], cent)
                              for g in groups], axis=0))
    d_within = np.mean(within, axis=0)
    d_drift = np.mean(drift, axis=0)
    return d_drift / (d_within + EPS)                                  # (3,)


# ------------------------------------------------------------- classifiers
def _make_lda():
    return make_pipeline(StandardScaler(),
                         LinearDiscriminantAnalysis(solver="svd"))


def _make_knn():
    return make_pipeline(StandardScaler(),
                         KNeighborsClassifier(n_neighbors=K_NEIGHBORS))


def _cv_score(estimator, Xch, y, meta, mode):
    """Cross-validated accuracy: pooled stratified 5-fold, or leave-one-group-out."""
    if mode == "pooled":
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        return float(cross_val_score(estimator, Xch, y, cv=cv).mean())
    gid = np.array([f"{m[0]}/{m[1]}" for m in meta])
    groups = np.unique(gid, return_inverse=True)[1]          # integer codes
    cv = LeaveOneGroupOut()
    return float(cross_val_score(estimator, Xch, y, groups=groups, cv=cv).mean())


# -------------------------------------------------------- PCA & distance
def _flat_feature(Fs, channels):
    """S11+S22 Re/Im (per-sample centered) -> (n, 804)."""
    use = [j for j, ch in enumerate(channels) if ch in ("S11", "S22")]
    if not use:
        use = list(range(len(Fs)))
    F = np.concatenate([Fs[j] for j in use], axis=1)
    return F - F.mean(axis=1, keepdims=True)          # remove per-sample level


def pca_scatter(Fs, y, meta, channels, fig_dir, run, domain):
    """2D PCA scatter colored by class AND by group; returns PC1+PC2 share."""
    F = StandardScaler().fit_transform(_flat_feature(Fs, channels))
    pca = PCA(n_components=2).fit(F)
    Z = pca.transform(F)
    evr = float(pca.explained_variance_ratio_.sum())
    gid = np.array([f"{m[0]}/{m[1]}" for m in meta])
    gcode = np.unique(gid, return_inverse=True)[1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    sc0 = axes[0].scatter(Z[:, 0], Z[:, 1], c=y, cmap="turbo", s=7, alpha=0.7)
    axes[0].set_title("colored by class (50)")
    sc1 = axes[1].scatter(Z[:, 0], Z[:, 1], c=gcode, cmap="tab10", s=7, alpha=0.7)
    axes[1].set_title("colored by group (subject/session)")
    for ax in axes:
        ax.set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    cb0 = fig.colorbar(sc0, ax=axes[0])
    cb0.set_label("class")
    cb1 = fig.colorbar(sc1, ax=axes[1])
    cb1.set_label("group")
    fig.suptitle(f"PCA (Re/Im, S11+S22) - {run} ({domain})   "
                 f"PC1+PC2 = {evr * 100:.1f}%")
    fig.tight_layout()
    fig.savefig(fig_dir / "pca_scatter.png", dpi=150)
    plt.close(fig)
    return evr


def class_distance(Fs, y, channels, labels, fig_dir, run, domain):
    """Class-mean distance matrix, reordered by hierarchical clustering.

    Returns (mean off-diagonal distance, cluster order).
    """
    F = StandardScaler().fit_transform(_flat_feature(Fs, channels))
    classes = np.unique(y)
    means = np.stack([F[y == c].mean(0) for c in classes])          # (C, d)
    D = np.sqrt(((means[:, None, :] - means[None, :, :]) ** 2).sum(2))
    order = leaves_list(linkage(squareform(D, checks=False), method="average"))
    D2 = D[np.ix_(order, order)]
    ids = [int(c) for c in order]
    ticks = np.arange(len(classes))
    tick = [f"C{i + 1:02d}" for i in ids]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(D2, cmap="magma")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(tick, fontsize=5, rotation=90)
    ax.set_yticklabels(tick, fontsize=5)
    ax.set_title(f"Class-mean distance matrix - {run} ({domain}, Re/Im)\n"
                 f"(S11+S22, reordered by average linkage)")
    fig.colorbar(im, ax=ax, label="euclidean distance")
    fig.tight_layout()
    fig.savefig(fig_dir / "class_distance.png", dpi=150)
    plt.close(fig)

    # self-describing CSV: first column/row carry the class id + label
    header = ["cls"] + [f"C{i + 1:02d}" for i in ids]
    rows = [[f"C{ids[r] + 1:02d}{labels.get(ids[r], '')}"]
            + [f"{v:.3f}" for v in D2[r]] for r in range(len(classes))]
    with (fig_dir.parent / "class_distance.csv").open(
            "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    off = D[np.triu_indices(len(classes), k=1)]
    return float(off.mean()), order


# ------------------------------------------------------------------ report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--domain", choices=["freq", "time"], default=None)
    ap.add_argument("--no-append", action="store_true",
                    help="do not append to pipeline_comparison.csv")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    run = cfg["run_name"]
    domain = args.domain or cfg.get("data", {}).get("domain", "time")
    C, y, meta, freq, channels = load_features(cfg, domain)
    db = db_mag(C)
    Fs = reim_matrices(C)
    labels = _load_class_labels(cfg)

    spec = f"{domain}_reim"
    out_dir = Path("outputs/analysis/l1") / f"{run}__{spec}"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fisher = fisher_ratio(db, y)                       # (201,3) magnitude-domain
    drift = drift_ratio(db, meta)                      # (3,)
    metrics = {"run_name": run, "domain": domain, "feature": "reim",
               "n_samples": int(len(y)), "n_classes": int(y.max() + 1),
               "fisher_mean": {}, "lda_pooled": {}, "lda_logo": {},
               "knn_pooled": {}, "knn_logo": {}, "drift_ratio": {}}
    for j, ch in enumerate(channels):
        metrics["fisher_mean"][ch] = round(float(fisher[:, j].mean()), 3)
        metrics["drift_ratio"][ch] = round(float(drift[j]), 3)
        metrics["lda_pooled"][ch] = round(
            _cv_score(_make_lda(), Fs[j], y, meta, "pooled"), 4)
        metrics["lda_logo"][ch] = round(
            _cv_score(_make_lda(), Fs[j], y, meta, "logo"), 4)
        metrics["knn_pooled"][ch] = round(
            _cv_score(_make_knn(), Fs[j], y, meta, "pooled"), 4)
        metrics["knn_logo"][ch] = round(
            _cv_score(_make_knn(), Fs[j], y, meta, "logo"), 4)

    # ---- fisher figure (magnitude-domain descriptor)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if len(freq) != fisher.shape[0]:
        # enhancement chain cropped / reshaped the axis (e.g. early_gate,
        # step_crop): fall back to a plain index axis
        xaxis = np.arange(fisher.shape[0])
        xlabel = "axis index (enhanced/cropped)"
    else:
        xaxis, xlabel = freq, "frequency (GHz)"
    for j, ch in enumerate(channels):
        ax.plot(xaxis, fisher[:, j], lw=1.0, label=ch)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fisher ratio  (between / within)")
    ax.set_title(f"Fisher ratio per point - {run} ({domain}, dB |S|)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fisher_curves.png", dpi=150)
    plt.close(fig)

    evr = pca_scatter(Fs, y, meta, channels, fig_dir, run, domain)
    dist_mean, _ = class_distance(Fs, y, channels, labels, fig_dir, run, domain)
    metrics["pca_pc12_variance"] = round(evr, 3)
    metrics["class_distance_mean"] = round(dist_mean, 3)

    # ---- report
    L = []
    L.append(f"==== L1 separability report : {run} ====")
    L.append(f"generated : {datetime.now():%Y-%m-%d %H:%M:%S}")
    L.append(f"feature   : {C.shape[1]} pts x {C.shape[2]} ch (Re/Im interleaved)"
             f" | domain={domain}")
    L.append(f"dataset   : n={len(y)}, classes={y.max() + 1}, "
             f"groups={len({(m[0], m[1]) for m in meta})}, chance={1 / (y.max() + 1):.3f}")
    L.append("")
    L.append("fisher ratio (dB |S|, mean over 201 freq bins; magnitude domain)")
    L.append("  " + " | ".join(f"{ch} {metrics['fisher_mean'][ch]:.2f}"
                              for ch in channels))
    L.append("")
    L.append("LDA (Re/Im) 5-fold pooled (within-distribution ceiling)")
    L.append("  " + " | ".join(f"{ch} {metrics['lda_pooled'][ch]:.3f}"
                              for ch in channels))
    L.append("LDA (Re/Im) leave-one-group-out (cross-session)")
    L.append("  " + " | ".join(f"{ch} {metrics['lda_logo'][ch]:.3f}"
                              for ch in channels))
    L.append("")
    L.append(f"kNN k={K_NEIGHBORS} (Re/Im) 5-fold pooled")
    L.append("  " + " | ".join(f"{ch} {metrics['knn_pooled'][ch]:.3f}"
                              for ch in channels))
    L.append(f"kNN k={K_NEIGHBORS} (Re/Im) leave-one-group-out")
    L.append("  " + " | ".join(f"{ch} {metrics['knn_logo'][ch]:.3f}"
                              for ch in channels))
    L.append("")
    L.append("drift ratio (dB |S|; cross-session drift / within-session sep, "
             ">1 = drift dominates)")
    L.append("  " + " | ".join(f"{ch} {metrics['drift_ratio'][ch]:.2f}"
                              for ch in channels))
    L.append("")
    L.append("PCA (Re/Im, S11+S22, per-sample centered)")
    L.append(f"  PC1+PC2 explain {metrics['pca_pc12_variance'] * 100:.1f}% "
             f"variance (figures/pca_scatter.png)")
    L.append("")
    L.append("class-mean distance matrix (Re/Im, S11+S22)")
    L.append(f"  mean pairwise distance {metrics['class_distance_mean']:.3f} "
             f"(figures/class_distance.png, class_distance.csv)")
    L.append("")
    L.append(f"figures  : {fig_dir}/fisher_curves.png, pca_scatter.png, "
             f"class_distance.png")
    report = "\n".join(L)
    print(report)

    with (out_dir / "report.txt").open("w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    # ---- scorecard row
    csv_path = Path("outputs/analysis/pipeline_comparison.csv")
    columns = ["run_name", "domain", "feature", "n_samples"]
    columns += [f"fisher_{ch}" for ch in channels]
    columns += [f"lda_pooled_{ch}" for ch in ["S11", "S22"]]
    columns += [f"lda_logo_{ch}" for ch in ["S11", "S22"]]
    columns += [f"knn_pooled_{ch}" for ch in ["S11", "S22"]]
    columns += [f"knn_logo_{ch}" for ch in ["S11", "S22"]]
    columns += [f"drift_{ch}" for ch in channels]
    columns += ["timestamp"]
    row = {c: "" for c in columns}
    row.update({"run_name": run, "domain": domain, "feature": "reim",
                "n_samples": int(len(y)), "timestamp":
                datetime.now().isoformat(timespec="seconds")})
    for ch in channels:
        row[f"fisher_{ch}"] = metrics["fisher_mean"][ch]
        row[f"drift_{ch}"] = metrics["drift_ratio"][ch]
    for ch in [c for c in ["S11", "S22"] if c in metrics["lda_pooled"]]:
        row[f"lda_pooled_{ch}"] = metrics["lda_pooled"][ch]
        row[f"lda_logo_{ch}"] = metrics["lda_logo"][ch]
        row[f"knn_pooled_{ch}"] = metrics["knn_pooled"][ch]
        row[f"knn_logo_{ch}"] = metrics["knn_logo"][ch]

    if not args.no_append:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            if new_file:
                w.writerow(columns)
            w.writerow([row[c] for c in columns])
        print(f"-> appended row to {csv_path}")
    else:
        print(f"-> (--no-append) row NOT written to {csv_path}")
    print(f"-> {out_dir}/report.txt, metrics.json, figures/")


if __name__ == "__main__":
    main()
