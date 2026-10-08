# -*- coding: utf-8 -*-
"""
Baseline tensor builder: processed npz -> model-ready tensors.

Each sample: [201, 3] complex -> [201, 6] float (Re, Im interleaved per channel).
Default domain is time: IFFT (fftshifted) is applied to the stored sweep data.
Pass domain="freq" to keep the raw frequency sweep instead.
Labels: class id Cxx -> 0..49, parsed from filename.

Usage (verification):
  python src/build_baseline_tensors.py --config configs/baseline_s21.yaml
  python src/build_baseline_tensors.py --config configs/baseline_s21.yaml --domain freq
"""

import argparse
from pathlib import Path

import numpy as np
import yaml


def load_sample(npz_path, domain="time"):
    """Return x [201, 6] float32 and metadata.

    domain="time" (default): IFFT of the stored sweep -> time domain.
    domain="freq": the stored frequency sweep, unchanged.
    """
    d = np.load(npz_path, allow_pickle=False)
    S = d["S"]                      # (201, 3) complex128 (frequency domain)
    if domain == "time":
        S = np.fft.fftshift(np.fft.ifft(S, axis=0), axes=0)  # (201, 3) complex, time domain
    x = np.empty((S.shape[0], S.shape[1] * 2), dtype=np.float32)
    x[:, 0::2] = S.real             # Re of channel 0,1,2
    x[:, 1::2] = S.imag             # Im of channel 0,1,2
    return x, d["channels"], d["freq"], float(d["df"])


def iter_samples(proc_root, subjects, sessions):
    """Yield (path, subject, session, class_idx (0-based), rep_idx (0-based))."""
    for subj in subjects:
        for ses in sessions:
            base = proc_root / subj / ses / "VNA_Data"
            for f in sorted(base.glob("*.npz")):
                stem = f.stem                       # S001_SES01_C01_R01
                parts = stem.split("_")
                cls = int(parts[2][1:]) - 1         # C01 -> 0
                rep = int(parts[3][1:]) - 1         # R01 -> 0
                yield f, subj, ses, cls, rep


def build_dataset(cfg, domain="time"):
    """Return X [n, T, 2*ch], y [n], meta list. Loads everything into memory.

    Two-phase construction (group-aware enhancement needs cross-sample state):
      phase 1  load every sample as complex frequency data  C (n, T, ch)
      phase 2  cfg["enhance"]["steps"] (ordered; empty = legacy behavior),
               then IFFT when domain="time" -> Re/Im interleaved float tensor

    run_name resolution: cfg["data"]["run_name"] (if present) selects the
    PROCESSED data directory; cfg["run_name"] names outputs. This lets many
    ablation variants share one processed cache.
    """
    run_name = cfg.get("run_name") or cfg["data"]["run_name"]
    data_run = cfg.get("data", {}).get("run_name", run_name)
    proc_root = Path(cfg["data"]["processed_root"]) / data_run

    Cs, ys, meta, channels = [], [], [], None
    for f, subj, ses, cls, rep in iter_samples(proc_root, cfg["data"]["subjects"],
                                                cfg["data"]["sessions"]):
        d = np.load(f, allow_pickle=False)
        if channels is None:
            channels = [str(c) for c in d["channels"]]
        Cs.append(d["S"])
        ys.append(cls)
        meta.append((subj, ses, cls, rep))
    if not Cs:
        sys.exit(f"[abort] no npz found under {proc_root}")
    C = np.stack(Cs)                                        # (n, T, ch) complex

    from enhance import apply_enhance                       # local: avoid cycle
    steps = (cfg.get("enhance") or {}).get("steps") or []
    C, channels, _ = apply_enhance(C, meta, channels, steps, domain)

    x = np.empty((C.shape[0], C.shape[1], C.shape[2] * 2), dtype=np.float32)
    x[:, :, 0::2] = C.real
    x[:, :, 1::2] = C.imag
    return x, np.asarray(ys, dtype=np.int64), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--domain", choices=["freq", "time"], default="time")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    X, y, meta = build_dataset(cfg, args.domain)

    print("=" * 60)
    print(f"run_name : {cfg['run_name']}")
    print(f"domain   : {args.domain}")
    print(f"X        : shape={X.shape}, dtype={X.dtype}")
    print(f"y        : shape={y.shape}, dtype={y.dtype}, "
          f"classes={y.min()}..{y.max()}, n_classes={len(np.unique(y))}")
    print(f"samples  : {len(meta)}  "
          f"(subjects={len({m[0] for m in meta})}, "
          f"sessions={len({(m[0], m[1]) for m in meta})})")
    print("-" * 60)
    # per-sample demonstration
    f0, subj, ses, cls, rep = next(iter_samples(
        Path(cfg["data"]["processed_root"]) / cfg["run_name"],
        cfg["data"]["subjects"], cfg["data"]["sessions"]))
    x, chans, freq, df = load_sample(f0, args.domain)
    print(f"sample   : {f0.name}")
    print(f"  tensor : {x.shape} {x.dtype}   (length=201, features=6)")
    print(f"  columns: [Re_Savg, Im_Savg, Re_S11, Im_S11, Re_S22, Im_S22]")
    print(f"  label  : C{cls+1:02d} -> y={cls}, rep R{rep+1:02d}")
    print(f"  x[0,:] : {np.array2string(x[0], precision=4, floatmode='fixed')}")
    print(f"  x[100,:]: {np.array2string(x[100], precision=4, floatmode='fixed')}")
    print(f"  finite : {np.isfinite(X).all()}, "
          f"mean={X.mean():.4e}, std={X.std():.4e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
