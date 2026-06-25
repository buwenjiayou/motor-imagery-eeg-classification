"""
Stable-FBCSP experiment for SHU motor imagery EEG.

This script focuses on cross-session generalization:
  train: sessions 1-4
  test:  session 5

Compared variants:
  - FBCSP-equal: equal-weight band-probability fusion
  - FBCSP-discriminative-weighted: band weights from training-session CV score
  - Stable-FBCSP: band weights from mean training-session CV score minus
    lambda * cross-session score variability

No-leakage rule:
  Band scores, band weights, CSP, scaler and LDA are learned from training
  sessions only. The held-out test session is touched only once for final
  evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import linalg, signal
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
    lambda_stability: float
    train_sessions: str
    test_session: str
    n_train: int
    n_test: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    kappa: float


@dataclass
class WeightRow:
    subject: str
    method: str
    lambda_stability: float
    band: str
    mean_score: float
    session_std: float
    weight: float


def load_session(subject: str, session: int) -> tuple[np.ndarray, np.ndarray]:
    path = MAT_DIR / f"{subject}_ses-{session:02d}_task_motorimagery_eeg.mat"
    mat = loadmat(path)
    x = np.asarray(mat["data"], dtype=np.float64)
    y_raw = np.asarray(mat["labels"]).reshape(-1)

    classes = sorted(np.unique(y_raw).tolist())
    if len(classes) != 2:
        raise ValueError(f"{path.name}: expected two classes, got {classes}")
    label_map = {classes[0]: 0, classes[1]: 1}
    y = np.asarray([label_map[value] for value in y_raw], dtype=np.int64)

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


def bandpass_sos(low: float, high: float, fs: float = 250.0, order: int = 4) -> np.ndarray:
    return signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")


def apply_filter(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    return signal.sosfiltfilt(sos, x, axis=-1)


def zscore(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    if std < 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


class BandCSPLDA:
    def __init__(self, sos: np.ndarray, n_components: int = 4):
        self.sos = sos
        self.csp = CSP(n_components=n_components)
        self.scaler = StandardScaler()
        self.clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BandCSPLDA":
        xf = apply_filter(x, self.sos)
        features = self.csp.fit(xf, y).transform(xf)
        features = self.scaler.fit_transform(features)
        self.clf.fit(features, y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        xf = apply_filter(x, self.sos)
        features = self.csp.transform(xf)
        features = self.scaler.transform(features)
        return self.clf.predict_proba(features)


class WeightedBandFBCSPLDA:
    def __init__(
        self,
        bands: list[tuple[float, float]],
        weights: np.ndarray,
        n_components: int = 4,
    ):
        self.bands = bands
        self.weights = np.asarray(weights, dtype=np.float64)
        self.band_models = [BandCSPLDA(bandpass_sos(low, high), n_components=n_components) for low, high in bands]

    def fit(self, x: np.ndarray, y: np.ndarray) -> "WeightedBandFBCSPLDA":
        for model in self.band_models:
            model.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        weights = self.weights / self.weights.sum()
        probas = np.asarray([model.predict_proba(x) for model in self.band_models])
        fused = np.tensordot(weights, probas, axes=(0, 0))
        return np.argmax(fused, axis=1)


def band_score_for_session(x: np.ndarray, y: np.ndarray, sos: np.ndarray, n_components: int) -> float:
    """Estimate band discriminability inside one training session only.

    A deterministic half-split is used as a lightweight inner validation. This
    keeps the score independent from the held-out test session.
    """

    x_filtered = apply_filter(x, sos)
    idx_class0 = np.where(y == 0)[0]
    idx_class1 = np.where(y == 1)[0]
    split0 = max(1, len(idx_class0) // 2)
    split1 = max(1, len(idx_class1) // 2)

    train_idx = np.r_[idx_class0[:split0], idx_class1[:split1]]
    val_idx = np.r_[idx_class0[split0:], idx_class1[split1:]]

    if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[val_idx])) < 2:
        return 0.5

    csp = CSP(n_components=n_components)
    scaler = StandardScaler()
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    train_features = csp.fit(x_filtered[train_idx], y[train_idx]).transform(x_filtered[train_idx])
    train_features = scaler.fit_transform(train_features)
    clf.fit(train_features, y[train_idx])

    val_features = csp.transform(x_filtered[val_idx])
    val_features = scaler.transform(val_features)
    pred = clf.predict(val_features)
    return float(balanced_accuracy_score(y[val_idx], pred))


def compute_training_band_statistics(
    subject: str,
    train_sessions: list[int],
    bands: list[tuple[float, float]],
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    sos_filters = [bandpass_sos(low, high) for low, high in bands]
    scores = np.zeros((len(train_sessions), len(bands)), dtype=np.float64)

    for session_index, session in enumerate(train_sessions):
        x, y = load_session(subject, session)
        for band_index, sos in enumerate(sos_filters):
            scores[session_index, band_index] = band_score_for_session(x, y, sos, n_components)

    return scores.mean(axis=0), scores.std(axis=0)


def make_weights(mean_scores: np.ndarray, session_stds: np.ndarray, variant: str, lambda_stability: float) -> np.ndarray:
    n_bands = len(mean_scores)
    if variant == "equal":
        return np.ones(n_bands, dtype=np.float64)
    if variant == "discriminative":
        return softmax(zscore(mean_scores))
    if variant == "stable":
        return softmax(zscore(mean_scores) - lambda_stability * zscore(session_stds))
    raise ValueError(f"Unknown variant: {variant}")


def evaluate_subject(
    subject: str,
    variant: str,
    method_name: str,
    lambda_stability: float,
    train_sessions: list[int],
    test_session: int,
    bands: list[tuple[float, float]],
    n_components: int,
) -> tuple[ResultRow, list[WeightRow]]:
    mean_scores, session_stds = compute_training_band_statistics(subject, train_sessions, bands, n_components)
    weights = make_weights(mean_scores, session_stds, variant, lambda_stability)

    x_train, y_train = load_sessions(subject, train_sessions)
    x_test, y_test = load_session(subject, test_session)

    model = WeightedBandFBCSPLDA(bands=bands, weights=weights, n_components=n_components)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    result = ResultRow(
        subject=subject,
        protocol="cross_session_train_1-4_test_5",
        method=method_name,
        lambda_stability=float(lambda_stability),
        train_sessions=",".join(f"ses-{session:02d}" for session in train_sessions),
        test_session=f"ses-{test_session:02d}",
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
        accuracy=float(accuracy_score(y_test, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, pred)),
        macro_f1=float(f1_score(y_test, pred, average="macro")),
        kappa=float(cohen_kappa_score(y_test, pred)),
    )

    weight_rows = []
    for (low, high), mean_score, session_std, weight in zip(bands, mean_scores, session_stds, weights):
        weight_rows.append(
            WeightRow(
                subject=subject,
                method=method_name,
                lambda_stability=float(lambda_stability),
                band=f"{int(low)}-{int(high)}Hz",
                mean_score=float(mean_score),
                session_std=float(session_std),
                weight=float(weight),
            )
        )

    return result, weight_rows


def write_csv(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    fieldnames = list(rows[0].__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_subjects(args: argparse.Namespace) -> list[str]:
    if args.all_subjects:
        return [f"sub-{idx:03d}" for idx in range(1, 26)]
    return args.subjects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=["sub-001"])
    parser.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--lambda-stability", type=float, default=1.0)
    parser.add_argument("--n-components", type=int, default=4)
    args = parser.parse_args()

    bands = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)]
    train_sessions = [1, 2, 3, 4]
    test_session = 5
    subjects = parse_subjects(args)

    variants = [
        ("equal", "FBCSP-equal"),
        ("discriminative", "FBCSP-discriminative-weighted"),
        ("stable", "Stable-FBCSP"),
    ]

    result_rows: list[ResultRow] = []
    weight_rows: list[WeightRow] = []

    jobs = [(subject, variant, method) for subject in subjects for variant, method in variants]
    for subject, variant, method_name in tqdm(jobs, desc="Stable-FBCSP"):
        result, weights = evaluate_subject(
            subject=subject,
            variant=variant,
            method_name=method_name,
            lambda_stability=args.lambda_stability,
            train_sessions=train_sessions,
            test_session=test_session,
            bands=bands,
            n_components=args.n_components,
        )
        result_rows.append(result)
        weight_rows.extend(weights)

    tag_subject = "all_subjects" if args.all_subjects else "_".join(subjects)
    tag = f"{args.mode}_{tag_subject}_lambda_{args.lambda_stability:g}".replace(".", "p")

    result_csv = RESULTS_TABLES / f"stable_fbcsp_{tag}.csv"
    weight_csv = RESULTS_TABLES / f"stable_fbcsp_weights_{tag}.csv"
    result_json = RESULTS_LOGS / f"stable_fbcsp_{tag}.json"

    write_csv(result_csv, result_rows)
    write_csv(weight_csv, weight_rows)
    result_json.parent.mkdir(parents=True, exist_ok=True)
    with result_json.open("w", encoding="utf-8") as f:
        json.dump([asdict(row) for row in result_rows], f, ensure_ascii=False, indent=2)

    print(f"Saved result CSV: {result_csv}")
    print(f"Saved weight CSV: {weight_csv}")
    print(f"Saved JSON: {result_json}")

    for row in result_rows:
        print(
            f"{row.subject} | {row.method} | lambda={row.lambda_stability:g} | "
            f"acc={row.accuracy:.4f} | bacc={row.balanced_accuracy:.4f} | "
            f"f1={row.macro_f1:.4f} | kappa={row.kappa:.4f}"
        )


if __name__ == "__main__":
    main()
