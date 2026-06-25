"""
Euclidean Alignment + CSP-LDA cross-session experiment.

Protocol:
  Per subject, train sessions 1-4 and test session 5.

Important setting:
  EA uses unlabeled trials from each domain/session split to estimate reference
  covariance matrices. For the test session, labels are never used, but test EEG
  samples are used without labels for alignment. Report this as unsupervised
  target alignment, not as a strictly target-free baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
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
    alignment: str
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
    mat = loadmat(path)
    x = np.asarray(mat["data"], dtype=np.float64)
    y_raw = np.asarray(mat["labels"]).reshape(-1)
    classes = sorted(np.unique(y_raw).tolist())
    if classes != [1, 2]:
        raise ValueError(f"{path.name}: expected labels [1, 2], got {classes}")
    y = y_raw.astype(np.int64) - 1
    x = x - x.mean(axis=-1, keepdims=True)
    return x, y


def load_sessions(subject: str, sessions: list[int]) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for session in sessions:
        x, y = load_session(subject, session)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def normalized_covariance(trial: np.ndarray) -> np.ndarray:
    cov = trial @ trial.T
    trace = np.trace(cov)
    return cov / trace if trace > 0 else cov


def mean_covariance(x: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    cov = np.mean([normalized_covariance(trial) for trial in x], axis=0)
    return cov + reg * np.eye(cov.shape[0])


def inv_sqrtm_spd(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    eigvals, eigvecs = linalg.eigh(cov)
    eigvals = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def euclidean_align(x: np.ndarray) -> np.ndarray:
    ref = mean_covariance(x)
    whitening = inv_sqrtm_spd(ref)
    return np.asarray([whitening @ trial for trial in x])


class CSP:
    def __init__(self, n_components: int = 4, reg: float = 1e-6):
        self.n_components = n_components
        self.reg = reg
        self.filters_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "CSP":
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"CSP expects binary classes, got {classes}")
        covs = []
        for cls in classes:
            covs.append(mean_covariance(x[y == cls], reg=self.reg))
        c0, c1 = covs
        composite = c0 + c1 + self.reg * np.eye(c0.shape[0])
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


def make_result(subject: str, y_train: np.ndarray, y_test: np.ndarray, pred: np.ndarray) -> ResultRow:
    return ResultRow(
        subject=subject,
        protocol="cross_session_train_1-4_test_5",
        method="EA+CSP-LDA",
        alignment="euclidean_alignment_unsupervised_target",
        train_sessions="1,2,3,4",
        test_session="5",
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
        accuracy=float(accuracy_score(y_test, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, pred)),
        macro_f1=float(f1_score(y_test, pred, average="macro")),
        kappa=float(cohen_kappa_score(y_test, pred)),
    )


def run_subject(subject: str, n_components: int) -> ResultRow:
    x_train, y_train = load_sessions(subject, [1, 2, 3, 4])
    x_test, y_test = load_session(subject, 5)
    x_train_ea = euclidean_align(x_train)
    x_test_ea = euclidean_align(x_test)
    model = CSPLDA(n_components=n_components)
    model.fit(x_train_ea, y_train)
    pred = model.predict(x_test_ea)
    return make_result(subject, y_train, y_test, pred)


def summarize(rows: list[ResultRow]) -> dict[str, object]:
    summary: dict[str, object] = {
        "family": "alignment",
        "method": "EA+CSP-LDA",
        "protocol": "cross_session_train_1-4_test_5",
        "alignment": "euclidean_alignment_unsupervised_target",
        "n_rows": len(rows),
    }
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "kappa"]:
        values = np.asarray([getattr(row, metric) for row in rows], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=[f"sub-{i:03d}" for i in range(1, 26)])
    parser.add_argument("--n-components", type=int, default=4)
    args = parser.parse_args()

    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    RESULTS_LOGS.mkdir(parents=True, exist_ok=True)

    rows = []
    for subject in tqdm(args.subjects, desc="EA+CSP"):
        row = run_subject(subject, args.n_components)
        rows.append(row)
        print(f"{row.subject}: bacc={row.balanced_accuracy:.4f}, acc={row.accuracy:.4f}", flush=True)

    table_path = RESULTS_TABLES / "ea_csp_full_all_subjects.csv"
    summary_csv_path = RESULTS_TABLES / "ea_csp_full_summary.csv"
    summary_json_path = RESULTS_LOGS / "ea_csp_full_summary.json"

    with table_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    summary = summarize(rows)
    with summary_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {table_path}")
    print(f"Saved: {summary_csv_path}")


if __name__ == "__main__":
    main()
