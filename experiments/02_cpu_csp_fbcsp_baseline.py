"""
CPU baseline experiments for SHU motor imagery EEG.

This file is intentionally independent from the teacher-provided scripts because
the original CSP/FBCSP code depends on old APIs (`np.float`, `np.int`) and MNE.

Implemented baselines:
  - CSP + shrinkage LDA
  - FBCSP + shrinkage LDA

Protocols:
  - smoke: sub-001 only, within-session ses-01 5-fold + cross-session 1-4 -> 5
  - full: selected/all subjects, within-session all sessions + cross-session 1-4 -> 5

No-leakage rule:
  CSP, StandardScaler and LDA are fitted on training data only in each fold/split.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import linalg, signal
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAT_DIR = PROJECT_ROOT / "data" / "raw" / "mat"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
RESULTS_LOGS = PROJECT_ROOT / "results" / "logs"


@dataclass
class ResultRow:
    subject: str
    protocol: str
    method: str
    train_sessions: str
    test_session: str
    n_train: int
    n_test: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    kappa: float


def load_session(subject: str, session: int) -> tuple[np.ndarray, np.ndarray]:
    path = MAT_DIR / f"{subject}_ses-{session:02d}_task_motorimagery_eeg.mat"
    if not path.exists():
        raise FileNotFoundError(path)

    mat = loadmat(path)
    x = np.asarray(mat["data"], dtype=np.float64)
    y_raw = np.asarray(mat["labels"]).reshape(-1)

    classes = sorted(np.unique(y_raw).tolist())
    if len(classes) != 2:
        raise ValueError(f"{path.name}: expected 2 classes, got {classes}")

    label_map = {classes[0]: 0, classes[1]: 1}
    y = np.asarray([label_map[value] for value in y_raw], dtype=np.int64)

    # Fit-free per-trial DC removal.
    x = x - x.mean(axis=-1, keepdims=True)
    return x, y


def load_sessions(subject: str, sessions: list[int]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for session in sessions:
        x, y = load_session(subject, session)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def normalized_covariance(trial: np.ndarray) -> np.ndarray:
    cov = trial @ trial.T
    trace = np.trace(cov)
    if trace <= 0:
        return cov
    return cov / trace


class CSP:
    def __init__(self, n_components: int = 4, reg: float = 1e-6):
        if n_components % 2 != 0:
            raise ValueError("n_components must be even.")
        self.n_components = n_components
        self.reg = reg
        self.filters_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "CSP":
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"CSP expects binary classes, got {classes}")

        covs = []
        for cls in classes:
            covs.append(np.mean([normalized_covariance(trial) for trial in x[y == cls]], axis=0))

        c0, c1 = covs
        eye = np.eye(c0.shape[0])
        c0 = c0 + self.reg * eye
        composite = c0 + c1 + self.reg * eye

        eigvals, eigvecs = linalg.eigh(c0, composite)
        order = np.argsort(eigvals)
        half = self.n_components // 2
        selected = np.r_[order[:half], order[-half:]]
        self.filters_ = eigvecs[:, selected]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.filters_ is None:
            raise RuntimeError("CSP has not been fitted.")
        projected = np.asarray([self.filters_.T @ trial for trial in x])
        variances = np.var(projected, axis=-1)
        variances = variances / np.sum(variances, axis=1, keepdims=True)
        return np.log(np.maximum(variances, 1e-12))


class CSPLDA:
    def __init__(self, n_components: int = 4):
        self.csp = CSP(n_components=n_components)
        self.scaler = StandardScaler()
        self.clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    def fit(self, x: np.ndarray, y: np.ndarray) -> "CSPLDA":
        features = self.csp.fit(x, y).transform(x)
        features = self.scaler.fit_transform(features)
        self.clf.fit(features, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        features = self.scaler.transform(self.csp.transform(x))
        return self.clf.predict(features)


def bandpass_sos(low: float, high: float, fs: float = 250.0, order: int = 4) -> np.ndarray:
    return signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")


def safe_filter(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    return signal.sosfiltfilt(sos, x, axis=-1)


class FBCSPLDA:
    def __init__(self, bands: list[tuple[float, float]], n_components: int = 4):
        self.bands = bands
        self.sos_filters = [bandpass_sos(low, high) for low, high in bands]
        self.csps = [CSP(n_components=n_components) for _ in bands]
        self.scaler = StandardScaler()
        self.clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FBCSPLDA":
        features = []
        for sos, csp in zip(self.sos_filters, self.csps):
            xf = safe_filter(x, sos)
            features.append(csp.fit(xf, y).transform(xf))
        features_all = np.concatenate(features, axis=1)
        features_all = self.scaler.fit_transform(features_all)
        self.clf.fit(features_all, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        features = []
        for sos, csp in zip(self.sos_filters, self.csps):
            xf = safe_filter(x, sos)
            features.append(csp.transform(xf))
        features_all = np.concatenate(features, axis=1)
        features_all = self.scaler.transform(features_all)
        return self.clf.predict(features_all)


def make_result(
    subject: str,
    protocol: str,
    method: str,
    train_sessions: str,
    test_session: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> ResultRow:
    return ResultRow(
        subject=subject,
        protocol=protocol,
        method=method,
        train_sessions=train_sessions,
        test_session=test_session,
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
        accuracy=float(accuracy_score(y_test, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
        macro_f1=float(f1_score(y_test, y_pred, average="macro")),
        kappa=float(cohen_kappa_score(y_test, y_pred)),
    )


def evaluate_within_session(
    subject: str,
    session: int,
    method_name: str,
    model_factory: Callable[[], object],
    folds: int,
    seed: int,
) -> ResultRow:
    x, y = load_session(subject, session)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    true_parts = []
    pred_parts = []

    for train_idx, test_idx in cv.split(x, y):
        model = model_factory()
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[test_idx])
        true_parts.append(y[test_idx])
        pred_parts.append(pred)

    y_true = np.concatenate(true_parts)
    y_pred = np.concatenate(pred_parts)
    return make_result(
        subject=subject,
        protocol=f"within_session_{folds}fold",
        method=method_name,
        train_sessions=f"ses-{session:02d}",
        test_session=f"ses-{session:02d}",
        y_train=y,
        y_test=y_true,
        y_pred=y_pred,
    )


def evaluate_cross_session(
    subject: str,
    train_sessions: list[int],
    test_session: int,
    method_name: str,
    model_factory: Callable[[], object],
) -> ResultRow:
    x_train, y_train = load_sessions(subject, train_sessions)
    x_test, y_test = load_session(subject, test_session)
    model = model_factory()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return make_result(
        subject=subject,
        protocol="cross_session_train_1-4_test_5",
        method=method_name,
        train_sessions=",".join(f"ses-{s:02d}" for s in train_sessions),
        test_session=f"ses-{test_session:02d}",
        y_train=y_train,
        y_test=y_test,
        y_pred=pred,
    )


def write_outputs(rows: list[ResultRow], tag: str) -> None:
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    RESULTS_LOGS.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_TABLES / f"cpu_csp_fbcsp_{tag}.csv"
    json_path = RESULTS_LOGS / f"cpu_csp_fbcsp_{tag}.json"

    fieldnames = list(ResultRow.__dataclass_fields__.keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(row) for row in rows], f, ensure_ascii=False, indent=2)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


def parse_subjects(args: argparse.Namespace) -> list[str]:
    if args.all_subjects:
        return [f"sub-{idx:03d}" for idx in range(1, 26)]
    return args.subjects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=["sub-001"])
    parser.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    bands = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)]
    methods: list[tuple[str, Callable[[], object]]] = [
        ("CSP-LDA", lambda: CSPLDA(n_components=4)),
        ("FBCSP-LDA", lambda: FBCSPLDA(bands=bands, n_components=4)),
    ]

    subjects = parse_subjects(args)
    sessions_for_within = [1] if args.mode == "smoke" else [1, 2, 3, 4, 5]

    rows: list[ResultRow] = []
    jobs = [(subject, method_name, factory) for subject in subjects for method_name, factory in methods]
    for subject, method_name, factory in tqdm(jobs, desc="CPU baselines"):
        for session in sessions_for_within:
            rows.append(
                evaluate_within_session(
                    subject=subject,
                    session=session,
                    method_name=method_name,
                    model_factory=factory,
                    folds=args.folds,
                    seed=args.seed,
                )
            )

        rows.append(
            evaluate_cross_session(
                subject=subject,
                train_sessions=[1, 2, 3, 4],
                test_session=5,
                method_name=method_name,
                model_factory=factory,
            )
        )

    tag_subject = "all_subjects" if args.all_subjects else "_".join(subjects)
    tag = f"{args.mode}_{tag_subject}"
    write_outputs(rows, tag)

    for row in rows:
        print(
            f"{row.subject} | {row.protocol} | {row.method} | "
            f"acc={row.accuracy:.4f} | bacc={row.balanced_accuracy:.4f} | "
            f"f1={row.macro_f1:.4f} | kappa={row.kappa:.4f}"
        )


if __name__ == "__main__":
    main()
