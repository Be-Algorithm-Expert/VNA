# -*- coding: utf-8 -*-
"""
Layer 1 - preprocessing: VNA csv -> complex npz (cached per pipeline config).

Pipeline:
  1. parse Ceyear 3674H csv -> frequency axis + 4 complex S-params (dB/deg -> complex)
  2. symmetrize transmission: s21_only | complex_avg
  3. optional smoothing on Re/Im of each channel: ma | hamming | gaussian | kaiser | savgol
  4. save per-file .npz into processed/<run_name>/<subject>/<session>/VNA_Data/

Usable as a library (run_preprocess(cfg)) or from CLI:
  python src/preprocess.py --config configs/baseline_s21.yaml
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

OUT_CHANNELS = ["S_avg", "S11", "S22"]  # S_avg = symmetrized transmission


# ---------------------------------------------------------------- csv parsing
def parse_vna_csv(path):
    """Return (freq[N], S[N,4]) with S columns ordered [S11, S21, S12, S22]."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip().rstrip(",")
            if not line or line.startswith("!") or line.startswith("BEGIN") \
                    or line.startswith("END") or line.startswith("Frequency"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 9:
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 8 or arr.shape[1] != 9:
        raise ValueError(f"Bad VNA csv: {path} shape={getattr(arr, 'shape', None)}")
    freq = arr[:, 0]

    def to_c(db, deg):
        return np.power(10.0, db / 20.0) * np.exp(1j * np.deg2rad(deg))

    S = np.column_stack([
        to_c(arr[:, 1], arr[:, 2]),   # S11
        to_c(arr[:, 3], arr[:, 4]),   # S21
        to_c(arr[:, 5], arr[:, 6]),   # S12
        to_c(arr[:, 7], arr[:, 8]),   # S22
    ])
    return freq, S


# ------------------------------------------------------------------ smoothing
def _window_vector(wtype, n, beta):
    if wtype == "ma":
        return np.ones(n) / n
    if wtype == "hamming":
        win = np.hamming(n)
    elif wtype == "gaussian":
        x = np.arange(n) - (n - 1) / 2.0
        sigma = (n - 1) / 6.0
        win = np.exp(-0.5 * (x / sigma) ** 2)
    elif wtype == "kaiser":
        win = np.kaiser(n, beta)
    else:
        raise ValueError(f"unknown smooth type: {wtype}")
    return win / win.sum()


def smooth_complex(S, wtype, window, beta, polyorder):
    if wtype == "none":
        return S
    if wtype == "savgol":
        from scipy.signal import savgol_filter
        re = savgol_filter(S.real, window, polyorder, axis=0)
        im = savgol_filter(S.imag, window, polyorder, axis=0)
        return re + 1j * im
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be odd and >= 3")
    win = _window_vector(wtype, window, beta)

    def conv(x):
        return np.apply_along_axis(lambda c: np.convolve(c, win, mode="same"), 0, x)

    return conv(S.real) + 1j * conv(S.imag)


# ------------------------------------------------------------------- pipeline
def build_channels(S, cfg):
    mode = cfg["pipeline"]["symmetrize"]
    if mode == "s21_only":
        S_t = S[:, 1]
    elif mode == "complex_avg":
        S_t = 0.5 * (S[:, 1] + S[:, 2])
    else:
        raise ValueError(f"unknown symmetrize mode: {mode}")
    out = np.column_stack([S_t, S[:, 0], S[:, 3]])  # (N, 3)
    sc = cfg["pipeline"]["smooth"]
    return smooth_complex(out, sc["type"], sc.get("window", 5),
                          sc.get("beta", 4.0), sc.get("polyorder", 2))


def run_preprocess(cfg, snapshot_text=None, log=print):
    """Run the preprocessing pipeline described by cfg (library entry point).

    cfg: {run_name, data:{root,processed_root,subjects,sessions}, pipeline:{...}}
    Returns the output directory. Skips work if it already exists with the
    same config snapshot, aborts if the snapshot differs.
    """
    run = cfg["run_name"]
    root = Path(cfg["data"]["root"])
    out_dir = Path(cfg["data"]["processed_root"]) / run
    snapshot_text = snapshot_text or yaml.safe_dump(cfg, allow_unicode=True,
                                                    sort_keys=False)
    snap = out_dir / "config_snapshot.yaml"

    if out_dir.exists() and any(out_dir.rglob("*.npz")):
        old = None
        if snap.exists():
            try:
                old = yaml.safe_load(snap.read_text(encoding="utf-8"))
            except Exception:
                old = None
        same = (isinstance(old, dict)
                and old.get("pipeline") == cfg["pipeline"]
                and old.get("data", {}).get("root") == cfg["data"]["root"])
        if not same:
            sys.exit(f"[abort] run '{run}' already exists with a DIFFERENT pipeline.\n"
                     f"        pick a new run_name or delete {out_dir}")
        log(f"[skip] processed data for '{run}' already exists: {out_dir}")
        return out_dir

    n_files, n_bad = 0, 0
    for subj in cfg["data"]["subjects"]:
        for ses in cfg["data"]["sessions"]:
            src_dir = root / subj / ses / "VNA_Data"
            dst_dir = out_dir / subj / ses / "VNA_Data"
            dst_dir.mkdir(parents=True, exist_ok=True)
            files = sorted(src_dir.glob("*.csv"))
            if not files:
                sys.exit(f"[abort] no csv found in {src_dir}")
            for f in files:
                try:
                    freq, S = parse_vna_csv(f)
                    out = build_channels(S, cfg)
                    np.savez_compressed(
                        dst_dir / (f.stem + ".npz"),
                        freq=freq, S=out,
                        channels=np.array(OUT_CHANNELS),
                        n_points=out.shape[0], df=float(freq[1] - freq[0]),
                    )
                    n_files += 1
                except Exception as e:
                    n_bad += 1
                    log(f"[warn] failed {f}: {e}")

    out_dir.mkdir(parents=True, exist_ok=True)
    snap.write_text(snapshot_text, encoding="utf-8")
    log(f"[preprocess] run={run}  files={n_files}  bad={n_bad}  -> {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    text = args.config.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    run_preprocess(cfg, snapshot_text=text)


if __name__ == "__main__":
    main()
