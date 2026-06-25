"""
Euclidean Alignment + EEGNet compressed cross-session experiment.

Protocol:
  Per subject, train sessions 1-4 and test session 5.

Setting:
  EA uses unlabeled train/test EEG samples to estimate covariance alignment.
  Test labels are never used for alignment, validation, training, or model
  selection. Report as unsupervised target alignment.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import linalg
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NPZ_PATH = PROJECT_ROOT / "data" / "processed" / "deep_learning" / "shu_mi_25subjects_5sessions.npz"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
RESULTS_LOGS = PROJECT_ROOT / "results" / "logs"
RESULTS_DL = PROJECT_ROOT / "results" / "dl" / "ea_eegnet_compressed"

spec = importlib.util.spec_from_file_location("eegnet_compressed", PROJECT_ROOT / "experiments" / "10_eegnet_smoke.py")
eeg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = eeg
spec.loader.exec_module(eeg)


@dataclass
class SubjectMetric:
    subject: str
    protocol: str
    method: str
    alignment: str
    seed: int
    epochs_ran: int
    best_epoch: int
    best_val_balanced_accuracy: float
    train_sessions: str
    val_strategy: str
    test_session: str
    n_train: int
    n_val: int
    n_test: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    kappa: float
    train_seconds: float


def normalized_covariance(trial: np.ndarray) -> np.ndarray:
    cov = trial @ trial.T
    trace = np.trace(cov)
    return cov / trace if trace > 0 else cov


def mean_covariance(x: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    cov = np.mean([normalized_covariance(trial.astype(np.float64, copy=False)) for trial in x], axis=0)
    return cov + reg * np.eye(cov.shape[0])


def inv_sqrtm_spd(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    eigvals, eigvecs = linalg.eigh(cov)
    eigvals = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def euclidean_align(x: np.ndarray) -> np.ndarray:
    whitening = inv_sqrtm_spd(mean_covariance(x)).astype(np.float32)
    return np.asarray([whitening @ trial for trial in x], dtype=np.float32)


def train_only_standardize(x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray):
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = np.maximum(x_train.std(axis=(0, 2), keepdims=True), 1e-6)
    return (
        ((x_train - mean) / std).astype(np.float32, copy=False),
        ((x_val - mean) / std).astype(np.float32, copy=False),
        ((x_test - mean) / std).astype(np.float32, copy=False),
    )


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def parse_subjects(args: argparse.Namespace, available: np.ndarray) -> list[int]:
    if args.all_subjects:
        return sorted(int(x) for x in np.unique(available))
    out = []
    for item in args.subjects:
        if str(item).startswith("sub-"):
            out.append(int(str(item).replace("sub-", "")))
        else:
            out.append(int(item))
    return out


def run_subject(subject: int, arrays: dict[str, np.ndarray], args: argparse.Namespace, device: torch.device, log_writer: csv.DictWriter) -> SubjectMetric:
    X, y, subj, sess = arrays["X"], arrays["y"], arrays["subject"], arrays["session"]
    train_pool = (subj == subject) & np.isin(sess, [1, 2, 3, 4])
    test_mask = (subj == subject) & (sess == 5)
    x_pool = euclidean_align(X[train_pool])
    y_pool = y[train_pool]
    session_pool = sess[train_pool]
    x_test = euclidean_align(X[test_mask])
    y_test = y[test_mask]

    stratify_key = np.asarray([f"{int(label)}_{int(session)}" for label, session in zip(y_pool, session_pool)])
    train_idx, val_idx = train_test_split(
        np.arange(len(y_pool)),
        test_size=args.val_fraction,
        random_state=args.seed,
        stratify=stratify_key,
    )
    x_train, y_train = x_pool[train_idx], y_pool[train_idx]
    x_val, y_val = x_pool[val_idx], y_pool[val_idx]
    x_train, x_val, x_test = train_only_standardize(x_train, x_val, x_test)

    train_loader = eeg.make_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader = eeg.make_loader(x_val, y_val, args.batch_size, shuffle=False)
    test_loader = eeg.make_loader(x_test, y_test, args.batch_size, shuffle=False)

    model = eeg.EEGNet(dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_state = None
    best_val_bacc = -1.0
    best_epoch = 0
    epochs_without_improve = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))

        y_val_true, y_val_pred, val_loss = eeg.predict(model, val_loader, device)
        val_metrics = metric_dict(y_val_true, y_val_pred)
        improved = val_metrics["balanced_accuracy"] > best_val_bacc + args.min_delta
        if improved:
            best_val_bacc = val_metrics["balanced_accuracy"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        log_writer.writerow({
            "subject": f"sub-{subject:03d}",
            "seed": args.seed,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_kappa": val_metrics["kappa"],
            "best_epoch": best_epoch,
            "best_val_balanced_accuracy": best_val_bacc,
        })

        if args.patience > 0 and epochs_without_improve >= args.patience:
            break

    epochs_ran = epoch
    if best_state is not None:
        model.load_state_dict(best_state)
    y_test_true, y_test_pred, _ = eeg.predict(model, test_loader, device)
    metrics = metric_dict(y_test_true, y_test_pred)
    return SubjectMetric(
        subject=f"sub-{subject:03d}",
        protocol="cross_session_train_1-4_test_5",
        method="EA+EEGNet-compressed",
        alignment="euclidean_alignment_unsupervised_target",
        seed=args.seed,
        epochs_ran=int(epochs_ran),
        best_epoch=int(best_epoch),
        best_val_balanced_accuracy=float(best_val_bacc),
        train_sessions="1,2,3,4",
        val_strategy=f"stratified_train_split_{args.val_fraction:.2f}",
        test_session="5",
        n_train=int(len(y_train)),
        n_val=int(len(y_val)),
        n_test=int(len(y_test)),
        accuracy=metrics["accuracy"],
        balanced_accuracy=metrics["balanced_accuracy"],
        macro_f1=metrics["macro_f1"],
        kappa=metrics["kappa"],
        train_seconds=float(time.time() - start),
    )


def summarize(rows: list[SubjectMetric], elapsed: float, args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    out: dict[str, object] = {
        "family": "alignment_deep_learning",
        "method": "EA+EEGNet-compressed",
        "protocol": "cross_session_train_1-4_test_5",
        "alignment": "euclidean_alignment_unsupervised_target",
        "seed": args.seed,
        "n_subjects": len(rows),
        "elapsed_seconds": elapsed,
        "device": str(device),
        "args": vars(args),
    }
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "kappa", "epochs_ran", "train_seconds"]:
        values = np.asarray([getattr(row, metric) for row in rows], dtype=float)
        out[f"{metric}_mean"] = float(values.mean())
        out[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=["1"])
    parser.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--tag", default="compressed50")
    parser.set_defaults(amp=True)
    args = parser.parse_args()

    eeg.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    RESULTS_LOGS.mkdir(parents=True, exist_ok=True)
    RESULTS_DL.mkdir(parents=True, exist_ok=True)

    data = np.load(NPZ_PATH)
    arrays = {"X": data["X"], "y": data["y"], "subject": data["subject"], "session": data["session"]}
    subjects = parse_subjects(args, arrays["subject"])

    suffix = f"{args.tag}_seed_{args.seed}_{len(subjects)}subjects"
    metrics_path = RESULTS_TABLES / f"ea_eegnet_{suffix}_subject_metrics.csv"
    log_path = RESULTS_DL / f"training_log_{suffix}.csv"
    summary_json_path = RESULTS_LOGS / f"ea_eegnet_{suffix}_summary.json"
    summary_csv_path = RESULTS_TABLES / f"ea_eegnet_{suffix}_summary.csv"

    log_fields = ["subject", "seed", "epoch", "train_loss", "val_loss", "val_accuracy", "val_balanced_accuracy", "val_macro_f1", "val_kappa", "best_epoch", "best_val_balanced_accuracy"]
    rows = []
    start = time.time()
    with log_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()
        for subject in tqdm(subjects, desc="EA+EEGNet subjects"):
            row = run_subject(subject, arrays, args, device, writer)
            rows.append(row)
            f.flush()
            print(f"{row.subject}: bacc={row.balanced_accuracy:.4f}, acc={row.accuracy:.4f}, best_epoch={row.best_epoch}, epochs={row.epochs_ran}", flush=True)

    elapsed = time.time() - start
    with metrics_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = list(asdict(rows[0]).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    summary = summarize(rows, elapsed, args, device)
    with summary_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader(); writer.writerow(summary)
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved summary: {summary_json_path}")


if __name__ == "__main__":
    main()
