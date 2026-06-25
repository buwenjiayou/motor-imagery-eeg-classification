"""
Merge cross-session baseline and Stable-FBCSP summaries into one comparison table.
"""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    sources = [
        ("baseline", "", RESULTS_TABLES / "cpu_csp_fbcsp_full_summary.csv"),
        ("stable_fbcsp", "1.0", RESULTS_TABLES / "stable_fbcsp_full_summary_lambda_1.csv"),
        ("stable_fbcsp", "0.5", RESULTS_TABLES / "stable_fbcsp_full_summary_lambda_0p5.csv"),
        ("stable_fbcsp_v2", "0.5", RESULTS_TABLES / "stable_fbcsp_v2_full_summary_lambda_0p5.csv"),
    ]

    merged: list[dict[str, str]] = []
    for family, lambda_value, path in sources:
        if not path.exists():
            print(f"Skip missing summary: {path}")
            continue
        for row in read_rows(path):
            if row["protocol"] != "cross_session_train_1-4_test_5":
                continue
            if family == "stable_fbcsp_v2" and row["method"] == "FBCSP-equal":
                continue
            out = {"family": family, "lambda_stability": lambda_value}
            out.update(row)
            merged.append(out)

    output = RESULTS_TABLES / "cross_session_method_comparison.csv"
    fieldnames = [
        "family",
        "lambda_stability",
        "protocol",
        "method",
        "n_rows",
        "accuracy_mean",
        "accuracy_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "kappa_mean",
        "kappa_std",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    for row in merged:
        print(
            f"{row['family']} | lambda={row['lambda_stability'] or '-'} | "
            f"{row['method']} | acc={float(row['accuracy_mean']):.4f}±{float(row['accuracy_std']):.4f} | "
            f"bacc={float(row['balanced_accuracy_mean']):.4f}±{float(row['balanced_accuracy_std']):.4f}"
        )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
