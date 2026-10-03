# -*- coding: utf-8 -*-
"""
Protocol 1: per-group (subject x session) repetition split 7/1/2, N+5 scheme.

Groups G = {(S001,SES01), (S001,SES02), (S002,SES01), (S002,SES02),
            (S003,SES01), (S003,SES02)}. Within each group:
    R01-R07 train / R08 val / R09-R10 test   (350/50/100 samples).

N+5:
  1. random hyperparameter search (search.n_trials) on the FIRST group only
     (data.first_group), selected by val accuracy;
  2. the other 5 groups are trained with the selected hyperparameters;
  3. report 6 per-group test accuracies as mean ± std.

Data / model:
  processed/<data.run_name>/<subj>/<ses>/VNA_Data/*.npz  (already preprocessed)
  -> X [n, 201, 6] float32 via src/build_baseline_tensors.py (time-domain Re/Im)
  model: config['model'] (name -> MODEL_REGISTRY, e.g. RsLstm -> models/RsLstmModel.py)

Standalone:
  python training/protocol1/train.py

Outputs (output.dir): train_log.txt, hp_search.csv, best_hp.yaml,
best_checkpoint.pt, ckpt/<group>.pt, summary.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "models"))

from build_baseline_tensors import build_dataset   # noqa: E402  (src/)
from RsLstmModel import RsLstmModel                # noqa: E402  (models/)

# Tensors are time-domain (IFFT) Re/Im interleaved, no extra normalization:
# see load_sample() in src/build_baseline_tensors.py.
DEFAULT_FEATURES = {"name": "freq_ri", "normalize": "none"}

# config `model.name` -> model class. Add new architectures (models/*.py) here.
MODEL_REGISTRY = {"RsLstm": RsLstmModel}


def build_model(cfg):
    """Resolve cfg['model']['name'] through MODEL_REGISTRY."""
    name = cfg["model"].get("name")
    if name not in MODEL_REGISTRY:
        sys.exit(f"[abort] model '{name}' not in MODEL_REGISTRY "
                 f"(available: {', '.join(MODEL_REGISTRY)})")
    return MODEL_REGISTRY[name]

# ------------------------------------------------------------------ splitting
def group_split(meta, cfg, log):
    tr = set(cfg["split"]["train_reps"])
    va = set(cfg["split"]["val_reps"])
    te = set(cfg["split"]["test_reps"])
    groups = {}
    for i, (subj, ses, cls, rep) in enumerate(meta):      # rep 0-based
        r = rep + 1
        buckets = groups.setdefault((subj, ses), ([], [], []))
        if r in tr:
            buckets[0].append(i)
        elif r in va:
            buckets[1].append(i)
        elif r in te:
            buckets[2].append(i)
        else:
            sys.exit(f"[abort] R{r:02d} not covered by split config")
    for key, (a, b, c) in groups.items():
        for name, idx in (("train", a), ("val", b), ("test", c)):
            if {meta[i][2] for i in idx} != set(range(50)):
                sys.exit(f"[abort] group {key} split '{name}' missing classes")
    return groups


# ------------------------------------------------------------------- logging
class Logger:
    """print() to stdout and mirror every line into a log file."""

    def __init__(self, path):
        self.path = Path(path)
        self.fh = self.path.open("w", encoding="utf-8")

    def __call__(self, message=""):
        print(message)
        self.fh.write(f"{message}\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


# -------------------------------------------------------------- hyperparameters
def fmt_params(params):
    """'num_hidden_units=224 num_lstm_layers=2 dropout_prob=0.02' (insertion order)."""
    return " ".join(f"{k}={v}" for k, v in params.items())


def sample_params(cfg, rng):
    """Draw (params, learning_rate) from the search-space specs in cfg['search'].

    Every spec is {low, high} plus optional step (int grid) or log (log-uniform).
    learning_rate is kept out of `params` so it can be logged as 'lr=...'.
    """
    params, lr = {}, None
    for name, spec in cfg["search"].items():
        if not isinstance(spec, dict) or "low" not in spec:
            continue                        # skip n_trials / seed / verbose_epochs
        low, high = spec["low"], spec["high"]
        if spec.get("log"):
            value = 10.0 ** rng.uniform(np.log10(low), np.log10(high))
        elif spec.get("step"):
            n = int(round((high - low) / spec["step"]))
            value = low + int(rng.integers(n + 1)) * spec["step"]
        elif isinstance(low, int) and isinstance(high, int):
            value = int(rng.integers(low, high + 1))
        else:
            value = float(rng.uniform(low, high))
        if name == "learning_rate":
            lr = float(value)
        elif isinstance(value, (int, np.integer)):
            params[name] = int(value)       # counts / sizes stay integral
        else:
            params[name] = float(value)
    if lr is None:
        lr = float(cfg["train"].get("learning_rate", 1e-3))
    return params, lr


def model_hp(params, cfg):
    """Sampled params + fixed model options -> model hyperparameters."""
    mm = cfg["model"]
    n_features = cfg.get("_tensor_features") or mm.get("input_size") or 6
    n_classes = cfg.get("_n_classes") or mm.get("output_size") or 50
    return {
        "input_size": int(n_features),
        "output_size": int(n_classes),
        "num_hidden_units": int(params["num_hidden_units"]),
        "num_lstm_layers": int(params["num_lstm_layers"]),
        "dropout_prob": float(params["dropout_prob"]),
        "is_bidirectional": bool(mm.get("is_bidirectional", False)),
        "batch_size": int(cfg["train"]["batch_size"]),
        "num_epochs": int(cfg["train"]["num_epochs"]),
    }


# ------------------------------------------------------------------- plumbing
def make_loader(X, y, idx, batch_size, shuffle):
    """Batches of [n, 201, 6] float tensors + int64 labels for one split."""
    x = torch.from_numpy(np.ascontiguousarray(X[idx], dtype=np.float32))
    t = torch.from_numpy(np.asarray(y[idx], dtype=np.int64))
    return DataLoader(TensorDataset(x, t), batch_size=batch_size, shuffle=shuffle)


def forward_model(model, x_batch, device):
    """RsLstm forward with this batch's sequence lengths."""
    x_batch = x_batch.to(device)
    lengths = torch.full((x_batch.size(0),), x_batch.size(1), dtype=torch.long)
    # RsLstmModel sizes h_0/c_0 with the batch_size it was built with, so keep it
    # in sync with the real batch (a trailing batch may be smaller).
    model.batch_size = x_batch.size(0)
    return model(x_batch, lengths)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            pred = forward_model(model, x_batch, device).argmax(dim=1).cpu()
            correct += int((pred == y_batch).sum())
            total += y_batch.numel()
    return correct / max(total, 1)


def train_group(cfg, params, lr, loaders, device, log, verbose=True):
    """Train one group, early-stopping on val accuracy.

    Returns (best_epoch, best_val_acc, test_acc, best_state_dict).
    """
    tr_loader, va_loader, te_loader = loaders
    tcfg = cfg["train"]
    torch.manual_seed(tcfg["seed"])

    model = build_model(cfg)(model_hp(params, cfg), device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.NLLLoss()          # RsLstm ends in log_softmax

    best_val, best_epoch, best_state, stale = -1.0, 0, None, 0
    for epoch in range(tcfg["num_epochs"]):
        model.train()
        loss_sum, seen = 0.0, 0
        for x_batch, y_batch in tr_loader:
            optimizer.zero_grad()
            loss = criterion(forward_model(model, x_batch, device),
                             y_batch.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * y_batch.numel()
            seen += y_batch.numel()

        val_acc = evaluate(model, va_loader, device)
        improved = val_acc > best_val
        if improved:
            best_val, best_epoch, stale = val_acc, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            stale += 1
        if verbose:
            log(f"    epoch {epoch:3d} | loss {loss_sum / max(seen, 1):.4f} "
                f"| val {val_acc:.4f} | best {best_val:.4f}"
                + ("  *" if improved else ""))
        if stale >= tcfg["patience"]:
            if verbose:
                log(f"    early stop at epoch {epoch} "
                    f"(no val gain for {stale} epochs)")
            break

    model.load_state_dict(best_state)
    test_acc = evaluate(model, te_loader, device)
    return best_epoch, best_val, test_acc, best_state


def random_search(cfg, fg_loaders, device, out_dir, log, n_trials):
    """Random search on the first group; returns (best_params, lr, best_record).

    Writes hp_search.csv incrementally and refreshes best_checkpoint.pt /
    best_hp.yaml whenever the val accuracy improves.
    """
    rng = np.random.default_rng(cfg["search"]["seed"])
    verbose = cfg["search"].get("verbose_epochs", True)
    best = None
    with (out_dir / "hp_search.csv").open("w", newline="",
                                          encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["trial", "hidden", "layers", "lr", "dropout",
                         "best_epoch", "val_acc", "test_acc", "time_s"])
        for trial in range(1, n_trials + 1):
            params, lr = sample_params(cfg, rng)
            log(f"\n[trial {trial:3d}/{n_trials}] {fmt_params(params)} "
                f"lr={lr:.3e}")
            t0 = time.time()
            ep, va, te, state = train_group(cfg, params, lr, fg_loaders, device,
                                            log, verbose)
            secs = int(round(time.time() - t0))
            log(f"  -> best_epoch {ep} | val {va:.4f} | test {te:.4f} | {secs}s")
            writer.writerow([trial, params["num_hidden_units"],
                             params["num_lstm_layers"], f"{lr:.3e}",
                             f"{params['dropout_prob']:.3f}", ep, f"{va:.4f}",
                             f"{te:.4f}", secs])
            fh.flush()
            if best is None or va > best[2]["val_acc"]:
                best = (params, lr, {"trial": trial, "val_acc": va,
                                     "test_acc": te, "best_epoch": ep,
                                     "time_s": secs})
                log(f"  ** new best val {va:.4f} -> saving best_checkpoint.pt")
                torch.save({"state_dict": state, "model": cfg["model"]["name"],
                            "params": params, "learning_rate": lr,
                            "val_acc": va, "test_acc": te, "trial": trial,
                            "features": cfg["features"],
                            "run_name": cfg["run_name"]},
                           out_dir / "best_checkpoint.pt")

    if best is None:                            # n_trials <= 0
        sys.exit("[abort] random search ran no trial")
    params, lr, rec = best
    with (out_dir / "best_hp.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump({"trial": rec["trial"], "model": cfg["model"]["name"],
                        "features": cfg["features"], "run_name": cfg["run_name"],
                        "params": params, "learning_rate": lr,
                        "val_acc": rec["val_acc"], "test_acc": rec["test_acc"],
                        "best_epoch": rec["best_epoch"]}, fh,
                       sort_keys=False, allow_unicode=True)
    return params, lr, rec


# ----------------------------------------------------------------------- main
def run_protocol1(cfg, X, y, meta, out_dir, n_trials=None, log=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ckpt").mkdir(exist_ok=True)
    closed = False
    if log is None:
        log = Logger(out_dir / "train_log.txt")
        closed = True

    cfg["_tensor_features"] = X.shape[2]
    cfg["_tensor_length"] = X.shape[1]
    cfg["_n_classes"] = int(y.max()) + 1
    declared = cfg["model"]                      # config shapes must match the data
    for key, actual in (("input_size", cfg["_tensor_features"]),
                        ("output_size", cfg["_n_classes"])):
        if declared.get(key) and int(declared[key]) != actual:
            sys.exit(f"[abort] model.{key}={declared[key]} but tensors are "
                     f"{actual}")
    n_trials = n_trials or cfg["search"]["n_trials"]

    log(f"device   : {device}")
    log(f"run_name : {cfg['run_name']} | model={cfg['model']['name']} "
        f"| domain={cfg.get('_domain', 'time')} | features={cfg['features']}")
    log(f"tensors  : X{X.shape} y{y.shape}")
    groups = group_split(meta, cfg, log)
    log(f"groups   : {len(groups)} | per group train/val/test = 350/50/100 "
        f"(50 classes verified in each)")

    bs = cfg["train"]["batch_size"]
    fg = tuple(cfg["data"]["first_group"])
    if fg not in groups:
        sys.exit(f"[abort] first_group {fg} not found in data")
    idx = groups[fg]
    fg_loaders = (make_loader(X, y, idx[0], bs, True),
                  make_loader(X, y, idx[1], bs, False),
                  make_loader(X, y, idx[2], bs, False))

    log(f"\n=== random search: {n_trials} trials on {fg[0]}/{fg[1]} (N of N+5) ===")
    params, lr, rec = random_search(cfg, fg_loaders, device, out_dir, log,
                                    n_trials)
    log(f"\n=== best HP (trial {rec['trial']}): {fmt_params(params)} "
        f"lr={lr:.3e} | val {rec['val_acc']:.4f} test {rec['test_acc']:.4f} ===")

    # ---------------- apply best HP to all groups ----------------
    log(f"\n=== train all {len(groups)} groups with best HP (N+5) ===")
    rows = []
    for i, key in enumerate(sorted(groups), 1):
        idx = groups[key]
        loaders = (make_loader(X, y, idx[0], bs, True),
                   make_loader(X, y, idx[1], bs, False),
                   make_loader(X, y, idx[2], bs, False))
        log(f"\n[{i}/{len(groups)}] {key[0]}/{key[1]}")
        ep, va, te, state = train_group(cfg, params, lr, loaders, device,
                                        log, cfg["search"].get("verbose_epochs", True))
        torch.save({"state_dict": state, "model": cfg["model"]["name"],
                    "params": params, "learning_rate": lr, "val_acc": va,
                    "test_acc": te, "best_epoch": ep,
                    "group": f"{key[0]}/{key[1]}", "features": cfg["features"],
                    "run_name": cfg["run_name"]},
                   out_dir / "ckpt" / f"{key[0]}_{key[1]}.pt")
        rows.append((f"{key[0]}/{key[1]}", ep, f"{va:.4f}", f"{te:.4f}"))
        log(f"  -> best_epoch {ep} | val {va:.4f} | test {te:.4f}")

    with (out_dir / "summary.csv").open("w", newline="",
                                        encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "best_epoch", "val_acc", "test_acc"])
        w.writerows(rows)

    accs = np.array([float(r[3]) for r in rows])
    log("=" * 60)
    log(f"protocol 1 | run_name={cfg['run_name']} | model={cfg['model']['name']} "
        f"| features={cfg['features']['name']} "
        f"| normalize={cfg['features'].get('normalize', 'none')}")
    log(f"best HP: {fmt_params(params)} lr={lr:.3e}")
    log(f"test accuracy over {len(accs)} groups: "
        f"{accs.mean()*100:.2f}% ± {accs.std()*100:.2f}%")
    log(f"outputs -> {out_dir}")
    if closed:
        log.close()
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--trials", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if cfg.get("protocol") != 1:
        sys.exit(f"[abort] protocol {cfg['protocol']} not handled by this runner")
    if "features" not in cfg:                    # standalone: what build_dataset emits
        cfg["features"] = dict(DEFAULT_FEATURES)
    if "model" not in cfg:                       # required: name + shapes
        sys.exit("[abort] no model section in config (need model.name)")
    build_model(cfg)                             # fail fast on unknown model name
    cfg["run_name"] = cfg.get("run_name") or cfg["data"]["run_name"]
    cfg["_domain"] = cfg["data"].get("domain", "time")   # echoed in the log

    X, y, meta = build_dataset(cfg, cfg["_domain"])
    out_dir = ROOT / cfg["output"]["dir"]
    run_protocol1(cfg, X, y, meta, out_dir, n_trials=args.trials)


if __name__ == "__main__":
    main()
