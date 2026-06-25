"""
Stable-FBCSP V2: inner leave-one-session-out band weighting.

Target protocol:
  train: sessions 1-4
  test:  session 5

The key difference from 04_stable_fbcsp.py is the band-score estimator.
V1 estimated each band's discriminability inside each single training session
with a deterministic half split. V2 estimates each band's cross-session
generalization inside the training data:

  for validation session in {1,2,3,4}:
      train a band-specific CSP-LDA on the other three sessions
      validate on the held-out training session

The final test session is not used for band scoring, weighting, CSP fitting,
scaling, classifier fitting, or hyperparameter selection.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
RESULTS_LOGS = PROJECT_ROOT / "results" / "logs"


def load_base_module():
    base_path = Path(__file__).resolve().with_name("04_stable_fbcsp.py")
    spec = importlib.util.spec_from_file_location("stable_fbcsp_base", base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base implementation from {base_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


def compute_inner_loso_band_statistics(
    subject: str,
    train_sessions: list[int],
    bands: list[tuple[float, float]],
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate band scores with leave-one-training-session-out validation."""

    session_data = {session: BASE.load_session(subject, session) for session in train_sessions}
    sos_filters = [BASE.bandpass_sos(low, high) for low, high in bands]
    scores = np.zeros((len(train_sessions), len(bands)), dtype=np.float64)

    for val_index, val_session in enumerate(train_sessions):
        inner_train_sessions = [session for session in train_sessions if session != val_session]
        x_train = np.concatenate([session_data[session][0] for session in inner_train_sessions], axis=0)
        y_train = np.concatenate([session_data[session][1] for session in inner_train_sessions], axis=0)
        x_val, y_val = session_data[val_session]

        for band_index, sos in enumerate(sos_filters):
            model = BASE.BandCSPLDA(sos=sos, n_components=n_components)
            model.fit(x_train, y_train)
            pred = np.argmax(model.predict_proba(x_val), axis=1)
            scores[val_index, band_index] = float(balanced_accuracy_score(y_val, pred))

    return scores.mean(axis=0), scores.std(axis=0)


def make_weights(
    mean_scores: np.ndarray,
    session_stds: np.ndarray,
    variant: str,
    lambda_stability: float,
) -> np.ndarray:
    n_bands = len(mean_scores)
    if variant == "equal":
        return np.ones(n_bands, dtype=np.float64)
    if variant == "inner_loso_discriminative":
        return BASE.softmax(BASE.zscore(mean_scores))
    if variant == "stable_v2":
        return BASE.softmax(BASE.zscore(mean_scores) - lambda_stability * BASE.zscore(session_stds))
    raise ValueError(f"Unknown variant: {variant}")


def evaluate_subject(
    subject: str,
    variants: list[tuple[str, str]],
    lambda_stability: float,
    train_sessions: list[int],
    test_session: int,
    bands: list[tuple[float, float]],
    n_components: int,
) -> tuple[list[object], list[object]]:
    mean_scores, session_stds = compute_inner_loso_band_statistics(
        subject=subject,
        train_sessions=train_sessions,
        bands=bands,
        n_components=n_components,
    )

    x_train, y_train = BASE.load_sessions(subject, train_sessions)
    x_test, y_test = BASE.load_session(subject, test_session)

    result_rows = []
    weight_rows = []
    for variant, method_name in variants:
        weights = make_weights(mean_scores, session_stds, variant, lambda_stability)

        model = BASE.WeightedBandFBCSPLDA(
            bands=bands,
            weights=weights,
            n_components=n_components,
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        result_rows.append(
            BASE.ResultRow(
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
        )

        for (low, high), mean_score, session_std, weight in zip(bands, mean_scores, session_stds, weights):
            weight_rows.append(
                BASE.WeightRow(
                    subject=subject,
                    method=method_name,
                    lambda_stability=float(lambda_stability),
                    band=f"{int(low)}-{int(high)}Hz",
                    mean_score=float(mean_score),
                    session_std=float(session_std),
                    weight=float(weight),
                )
            )

    return result_rows, weight_rows


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
    parser.add_argument("--lambda-stability", type=float, default=0.5)
    parser.add_argument("--n-components", type=int, default=4)
    args = parser.parse_args()

    bands = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)]
    train_sessions = [1, 2, 3, 4]
    test_session = 5
    subjects = parse_subjects(args)
    variants = [
        ("equal", "FBCSP-equal"),
        ("inner_loso_discriminative", "FBCSP-innerLOSO-weighted"),
        ("stable_v2", "Stable-FBCSP-V2"),
    ]

    result_rows = []
    weight_rows = []
    for subject in tqdm(subjects, desc="Stable-FBCSP-V2"):
        subject_results, subject_weights = evaluate_subject(
            subject=subject,
            variants=variants,
            lambda_stability=args.lambda_stability,
            train_sessions=train_sessions,
            test_session=test_session,
            bands=bands,
            n_components=args.n_components,
        )
        result_rows.extend(subject_results)
        weight_rows.extend(subject_weights)

    tag_subject = "all_subjects" if args.all_subjects else "_".join(subjects)
    tag = f"{args.mode}_{tag_subject}_lambda_{args.lambda_stability:g}".replace(".", "p")

    result_csv = RESULTS_TABLES / f"stable_fbcsp_v2_{tag}.csv"
    weight_csv = RESULTS_TABLES / f"stable_fbcsp_v2_weights_{tag}.csv"
    result_json = RESULTS_LOGS / f"stable_fbcsp_v2_{tag}.json"

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
