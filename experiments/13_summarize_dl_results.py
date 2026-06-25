"""Summarize compressed EEGNet deep-learning results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
RESULTS_LOGS = PROJECT_ROOT / "results" / "logs"


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else float("nan")


def stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_group(rows: list[dict[str, str]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "kappa", "epochs_ran", "train_seconds"]:
        values = [float(row[metric]) for row in rows]
        out[f"{metric}_mean"] = mean(values)
        out[f"{metric}_std"] = stdev(values)
    return out


def load_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open("r", encoding="utf-8-sig")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="dl_eegnet_compressed50_seed_*_25subjects_subject_metrics.csv")
    parser.add_argument("--tag", default="compressed50_3seeds")
    args = parser.parse_args()

    input_paths = sorted(RESULTS_TABLES.glob(args.pattern))
    if not input_paths:
        raise FileNotFoundError(f"No files matched {args.pattern}")

    rows: list[dict[str, str]] = []
    for path in input_paths:
        rows.extend(load_csv(path))

    merged_path = RESULTS_TABLES / f"dl_eegnet_{args.tag}_subject_seed_metrics.csv"
    write_rows(merged_path, rows)

    seed_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    subject_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        seed_groups[row["seed"]].append(row)
        subject_groups[row["subject"]].append(row)

    by_seed = []
    for seed, group in sorted(seed_groups.items(), key=lambda item: int(item[0])):
        out: dict[str, object] = {
            "method": "EEGNet-compressed",
            "protocol": "cross_session_train_1-4_test_5",
            "seed": int(seed),
            "n_subjects": len(group),
        }
        out.update(summarize_group(group))
        by_seed.append(out)

    by_subject = []
    for subject, group in sorted(subject_groups.items()):
        out = {
            "subject": subject,
            "method": "EEGNet-compressed",
            "protocol": "cross_session_train_1-4_test_5",
            "n_seeds": len(group),
        }
        out.update(summarize_group(group))
        by_subject.append(out)

    overall: dict[str, object] = {
        "method": "EEGNet-compressed",
        "protocol": "cross_session_train_1-4_test_5",
        "n_rows": len(rows),
        "n_subjects": len(subject_groups),
        "n_seeds": len(seed_groups),
        "seeds": ",".join(sorted(seed_groups.keys(), key=int)),
    }
    overall.update(summarize_group(rows))

    # Comparison table uses existing classical cross-session summary if present.
    comparison = []
    classical_path = RESULTS_TABLES / "cross_session_method_comparison.csv"
    if classical_path.exists():
        for row in load_csv(classical_path):
            comparison.append(
                {
                    "family": row.get("family", ""),
                    "method": row["method"],
                    "protocol": row["protocol"],
                    "n_rows": row["n_rows"],
                    "accuracy_mean": row["accuracy_mean"],
                    "accuracy_std": row["accuracy_std"],
                    "balanced_accuracy_mean": row["balanced_accuracy_mean"],
                    "balanced_accuracy_std": row["balanced_accuracy_std"],
                    "macro_f1_mean": row["macro_f1_mean"],
                    "macro_f1_std": row["macro_f1_std"],
                    "kappa_mean": row["kappa_mean"],
                    "kappa_std": row["kappa_std"],
                }
            )
    comparison.append(
        {
            "family": "deep_learning",
            "method": "EEGNet-compressed 3 seeds",
            "protocol": "cross_session_train_1-4_test_5",
            "n_rows": overall["n_rows"],
            "accuracy_mean": overall["accuracy_mean"],
            "accuracy_std": overall["accuracy_std"],
            "balanced_accuracy_mean": overall["balanced_accuracy_mean"],
            "balanced_accuracy_std": overall["balanced_accuracy_std"],
            "macro_f1_mean": overall["macro_f1_mean"],
            "macro_f1_std": overall["macro_f1_std"],
            "kappa_mean": overall["kappa_mean"],
            "kappa_std": overall["kappa_std"],
        }
    )

    seed_path = RESULTS_TABLES / f"dl_eegnet_{args.tag}_by_seed_summary.csv"
    subject_path = RESULTS_TABLES / f"dl_eegnet_{args.tag}_by_subject_summary.csv"
    overall_csv_path = RESULTS_TABLES / f"dl_eegnet_{args.tag}_overall_summary.csv"
    overall_json_path = RESULTS_LOGS / f"dl_eegnet_{args.tag}_overall_summary.json"
    comparison_path = RESULTS_TABLES / f"classical_vs_dl_eegnet_{args.tag}.csv"

    write_rows(seed_path, by_seed)
    write_rows(subject_path, by_subject)
    write_rows(overall_csv_path, [overall])
    write_rows(comparison_path, comparison)
    with overall_json_path.open("w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"Merged: {merged_path}")
    print(f"By seed: {seed_path}")
    print(f"By subject: {subject_path}")
    print(f"Comparison: {comparison_path}")


if __name__ == "__main__":
    main()
