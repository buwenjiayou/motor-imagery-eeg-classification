"""
Summarize experiment CSV files by protocol and method.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else float("nan")


def stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(input_path.open("r", encoding="utf-8-sig")))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["protocol"], row["method"])].append(row)

    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "kappa"]
    out_rows = []
    for (protocol, method), group_rows in sorted(groups.items()):
        out = {
            "protocol": protocol,
            "method": method,
            "n_rows": len(group_rows),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group_rows]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = stdev(values)
        out_rows.append(out)

    fieldnames = ["protocol", "method", "n_rows"]
    for metric in metrics:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    for row in out_rows:
        print(
            f"{row['protocol']} | {row['method']} | n={row['n_rows']} | "
            f"acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f} | "
            f"bacc={row['balanced_accuracy_mean']:.4f}±{row['balanced_accuracy_std']:.4f} | "
            f"f1={row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f} | "
            f"kappa={row['kappa_mean']:.4f}±{row['kappa_std']:.4f}"
        )


if __name__ == "__main__":
    main()
