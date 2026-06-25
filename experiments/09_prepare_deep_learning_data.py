"""
Prepare a compact NPZ dataset for future deep-learning experiments.

The current CPU environment intentionally does not install PyTorch. This script
therefore exports NumPy arrays only:

  X: float32, shape (trials, channels, time)
  y: int64, shape (trials,), labels in {0, 1}
  subject: int64 subject id, 1-25
  session: int64 session id, 1-5

The exported file supports future EEGNet/FBCNet experiments while keeping the
same protocol as the classical experiments:
  train: sessions 1-4
  test:  session 5
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAT_DIR = PROJECT_ROOT / "data" / "raw" / "mat"
METADATA_DIR = PROJECT_ROOT / "data" / "raw" / "metadata"
RESULTS_LOGS = PROJECT_ROOT / "results" / "logs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "deep_learning"


def read_channel_names() -> list[str]:
    path = METADATA_DIR / "task-motorimagery_channels.tsv"
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [row["name"] for row in reader]


def iter_mat_paths(subjects: list[int]) -> list[tuple[int, int, Path]]:
    out = []
    for subject in subjects:
        for session in range(1, 6):
            path = MAT_DIR / f"sub-{subject:03d}_ses-{session:02d}_task_motorimagery_eeg.mat"
            if not path.exists():
                raise FileNotFoundError(path)
            out.append((subject, session, path))
    return out


def load_session(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mat = loadmat(path)
    x = np.asarray(mat["data"], dtype=np.float32)
    y_raw = np.asarray(mat["labels"]).reshape(-1).astype(np.int64)
    classes = sorted(np.unique(y_raw).tolist())
    if classes != [1, 2]:
        raise ValueError(f"{path.name}: expected labels [1, 2], got {classes}")
    y = y_raw - 1
    x = x - x.mean(axis=-1, keepdims=True)
    return x, y


def parse_subjects(args: argparse.Namespace) -> list[int]:
    if args.all_subjects:
        return list(range(1, 26))
    subjects = []
    for item in args.subjects:
        if item.startswith("sub-"):
            subjects.append(int(item.replace("sub-", "")))
        else:
            subjects.append(int(item))
    return subjects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=["1"])
    parser.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--output", default="shu_mi_25subjects_5sessions.npz")
    args = parser.parse_args()

    subjects = parse_subjects(args)
    paths = iter_mat_paths(subjects)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    subject_ids: list[np.ndarray] = []
    session_ids: list[np.ndarray] = []

    for subject, session, path in tqdm(paths, desc="Export DL NPZ"):
        x, y = load_session(path)
        xs.append(x)
        ys.append(y)
        subject_ids.append(np.full(len(y), subject, dtype=np.int64))
        session_ids.append(np.full(len(y), session, dtype=np.int64))

    X = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(ys, axis=0).astype(np.int64, copy=False)
    subject_arr = np.concatenate(subject_ids, axis=0)
    session_arr = np.concatenate(session_ids, axis=0)
    channel_names = np.asarray(read_channel_names())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / args.output
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        subject=subject_arr,
        session=session_arr,
        channel_names=channel_names,
        sampling_frequency=np.asarray([250.0], dtype=np.float32),
        protocol_note=np.asarray(["cross-session default: train sessions 1-4, test session 5"]),
    )

    log = {
        "output": str(output_path),
        "subjects": subjects,
        "shape": list(X.shape),
        "n_trials": int(len(y)),
        "label_counts": {
            "left_0": int(np.sum(y == 0)),
            "right_1": int(np.sum(y == 1)),
        },
        "session_counts": {
            str(session): int(np.sum(session_arr == session))
            for session in sorted(np.unique(session_arr))
        },
        "dtype": str(X.dtype),
        "channel_names": channel_names.tolist(),
        "sampling_frequency": 250.0,
        "recommended_protocol": "per-subject train sessions 1-4, test session 5",
    }

    RESULTS_LOGS.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_LOGS / "prepare_deep_learning_data.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_path}")
    print(f"Saved: {log_path}")
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
