"""
Compressed EEGNet experiments for SHU motor imagery EEG.

Protocol:
  Per subject, train sessions 1-4 and test session 5.

This script is intentionally self-contained and reads the project NPZ directly.
It is used for the 3-hour deep-learning add-on experiment:
  - smoke: one subject, a few epochs
  - compressed full: all 25 subjects, one seed, 50 epochs with early stopping
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NPZ_PATH = PROJECT_ROOT / "data" / "processed" / "deep_learning" / "shu_mi_25subjects_5sessions.npz"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
RESULTS_LOGS = PROJECT_ROOT / "results" / "logs"
RESULTS_DL = PROJECT_ROOT / "results" / "dl" / "eegnet_compressed"


@dataclass
class SubjectMetric:
    subject: str
    protocol: str
    method: str
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


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.weight.renorm_(p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 32,
        n_times: int = 1000,
        n_classes: int = 2,
        dropout: float = 0.5,
        f1: int = 8,
        d: int = 2,
        kernel_length: int = 125,
    ) -> None:
        super().__init__()
        f2 = f1 * d
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(f1),
            Conv2dWithConstraint(f1, f1 * d, (n_channels, 1), groups=f1, bias=False, max_norm=1.0),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(f1 * d, f1 * d, (1, 22), padding=(0, 11), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=(1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            flat_dim = int(np.prod(self.features(dummy).shape[1:]))
        self.classifier = nn.Linear(flat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def parse_subjects(args: argparse.Namespace, available: np.ndarray) -> list[int]:
    if args.all_subjects:
        return sorted(int(x) for x in np.unique(available))
    out = []
    for item in args.subjects:
        if isinstance(item, int):
            out.append(item)
        elif str(item).startswith("sub-"):
            out.append(int(str(item).replace("sub-", "")))
        else:
            out.append(int(item))
    return out


def train_only_standardize(
    x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-6)
    return (
        ((x_train - mean) / std).astype(np.float32, copy=False),
        ((x_val - mean) / std).astype(np.float32, copy=False),
        ((x_test - mean) / std).astype(np.float32, copy=False),
    )


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_t = torch.from_numpy(x[:, None, :, :].astype(np.float32, copy=False))
    y_t = torch.from_numpy(y.astype(np.int64, copy=False))
    return DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    losses: list[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(float(loss.item()))
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            targets.append(yb.cpu().numpy())
    return np.concatenate(targets), np.concatenate(preds), float(np.mean(losses))


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def run_subject(
    subject: int,
    arrays: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    log_writer: csv.DictWriter,
) -> SubjectMetric:
    X = arrays["X"]
    y = arrays["y"]
    subj = arrays["subject"]
    sess = arrays["session"]

    train_pool = (subj == subject) & np.isin(sess, [1, 2, 3, 4])
    test_mask = (subj == subject) & (sess == 5)
    x_pool = X[train_pool]
    y_pool = y[train_pool]
    session_pool = sess[train_pool]
    x_test = X[test_mask]
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

    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False)

    model = EEGNet(dropout=args.dropout).to(device)
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
        train_losses: list[float] = []
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
            train_losses.append(float(loss.item()))

        y_val_true, y_val_pred, val_loss = predict(model, val_loader, device)
        val_metrics = metric_dict(y_val_true, y_val_pred)
        train_loss = float(np.mean(train_losses))
        improved = val_metrics["balanced_accuracy"] > best_val_bacc + args.min_delta
        if improved:
            best_val_bacc = val_metrics["balanced_accuracy"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        log_writer.writerow(
            {
                "subject": f"sub-{subject:03d}",
                "seed": args.seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_kappa": val_metrics["kappa"],
                "best_epoch": best_epoch,
                "best_val_balanced_accuracy": best_val_bacc,
            }
        )

        if args.patience > 0 and epochs_without_improve >= args.patience:
            break

    epochs_ran = epoch
    if best_state is not None:
        model.load_state_dict(best_state)

    y_test_true, y_test_pred, _ = predict(model, test_loader, device)
    metrics = metric_dict(y_test_true, y_test_pred)
    train_seconds = time.time() - start

    if args.save_checkpoints:
        ckpt_dir = RESULTS_DL / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "subject": subject,
                "seed": args.seed,
                "best_epoch": best_epoch,
                "args": vars(args),
            },
            ckpt_dir / f"eegnet_sub-{subject:03d}_seed-{args.seed}.pt",
        )

    return SubjectMetric(
        subject=f"sub-{subject:03d}",
        protocol="cross_session_train_1-4_test_5",
        method="EEGNet-compressed",
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
        train_seconds=float(train_seconds),
    )


def summarize(rows: list[SubjectMetric], elapsed: float, args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    metric_names = ["accuracy", "balanced_accuracy", "macro_f1", "kappa", "epochs_ran", "train_seconds"]
    summary: dict[str, object] = {
        "method": "EEGNet-compressed",
        "protocol": "cross_session_train_1-4_test_5",
        "seed": args.seed,
        "n_subjects": len(rows),
        "elapsed_seconds": elapsed,
        "device": str(device),
        "args": vars(args),
    }
    for name in metric_names:
        values = np.asarray([getattr(row, name) for row in rows], dtype=float)
        summary[f"{name}_mean"] = float(values.mean())
        summary[f"{name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return summary


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
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--tag", default="compressed")
    parser.set_defaults(amp=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    RESULTS_LOGS.mkdir(parents=True, exist_ok=True)
    RESULTS_DL.mkdir(parents=True, exist_ok=True)

    data = np.load(NPZ_PATH)
    arrays = {"X": data["X"], "y": data["y"], "subject": data["subject"], "session": data["session"]}
    subjects = parse_subjects(args, arrays["subject"])

    suffix = f"{args.tag}_seed_{args.seed}_{len(subjects)}subjects"
    metrics_path = RESULTS_TABLES / f"dl_eegnet_{suffix}_subject_metrics.csv"
    training_log_path = RESULTS_DL / f"training_log_{suffix}.csv"
    summary_json_path = RESULTS_LOGS / f"dl_eegnet_{suffix}_summary.json"
    summary_csv_path = RESULTS_TABLES / f"dl_eegnet_{suffix}_summary.csv"

    log_fields = [
        "subject",
        "seed",
        "epoch",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_macro_f1",
        "val_kappa",
        "best_epoch",
        "best_val_balanced_accuracy",
    ]

    rows: list[SubjectMetric] = []
    start = time.time()
    with training_log_path.open("w", encoding="utf-8-sig", newline="") as log_f:
        log_writer = csv.DictWriter(log_f, fieldnames=log_fields)
        log_writer.writeheader()
        for subject in tqdm(subjects, desc="EEGNet subjects"):
            row = run_subject(subject, arrays, args, device, log_writer)
            rows.append(row)
            log_f.flush()
            print(
                f"{row.subject}: bacc={row.balanced_accuracy:.4f}, "
                f"acc={row.accuracy:.4f}, best_epoch={row.best_epoch}, "
                f"epochs={row.epochs_ran}, seconds={row.train_seconds:.1f}",
                flush=True,
            )

    elapsed = time.time() - start

    metric_fields = list(asdict(rows[0]).keys()) if rows else []
    with metrics_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    summary = summarize(rows, elapsed, args, device)
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with summary_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = list(summary.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved training log: {training_log_path}")
    print(f"Saved summary: {summary_json_path}")


if __name__ == "__main__":
    main()
