"""
Generate report-ready figures from the completed CPU experiments.

The script intentionally depends only on the Python standard library and
matplotlib, so it can run in the lightweight CPU conda environment.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required table: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def save_current_figure(filename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")
    return output


def plot_cross_session_method_comparison() -> None:
    rows = read_csv(TABLES_DIR / "cross_session_method_comparison.csv")

    selected: list[dict[str, str]] = []
    for row in rows:
        method = row["method"]
        lambda_value = row["lambda_stability"]
        family = row["family"]

        if family == "baseline":
            selected.append(row)
        elif family == "stable_fbcsp" and lambda_value == "0.5" and method in {
            "FBCSP-equal",
            "FBCSP-discriminative-weighted",
        }:
            selected.append(row)
        elif family == "stable_fbcsp" and method == "Stable-FBCSP" and lambda_value in {"1.0", "0.5"}:
            selected.append(row)
        elif family == "stable_fbcsp_v2" and lambda_value == "0.5" and method in {
            "FBCSP-innerLOSO-weighted",
            "Stable-FBCSP-V2",
        }:
            selected.append(row)

    label_map = {
        ("baseline", "", "CSP-LDA"): "CSP-LDA",
        ("baseline", "", "FBCSP-LDA"): "FBCSP-LDA",
        ("stable_fbcsp", "0.5", "FBCSP-equal"): "FBCSP-equal\n(lambda=0.5)",
        (
            "stable_fbcsp",
            "0.5",
            "FBCSP-discriminative-weighted",
        ): "Disc-weighted\n(lambda=0.5)",
        ("stable_fbcsp", "1.0", "Stable-FBCSP"): "Stable-FBCSP\n(lambda=1.0)",
        ("stable_fbcsp", "0.5", "Stable-FBCSP"): "Stable-FBCSP\n(lambda=0.5)",
        (
            "stable_fbcsp_v2",
            "0.5",
            "FBCSP-innerLOSO-weighted",
        ): "Inner-LOSO\nweighted",
        ("stable_fbcsp_v2", "0.5", "Stable-FBCSP-V2"): "Stable-FBCSP\nV2",
    }

    labels = [
        label_map[(row["family"], row["lambda_stability"], row["method"])]
        for row in selected
    ]
    means = [to_float(row["balanced_accuracy_mean"]) for row in selected]
    stds = [to_float(row["balanced_accuracy_std"]) for row in selected]

    plt.figure(figsize=(9.2, 4.8))
    colors = [
        "#4C78A8",
        "#72B7B2",
        "#F58518",
        "#ECA82C",
        "#B279A2",
        "#9D755D",
        "#54A24B",
        "#FF9DA6",
    ]
    plt.bar(range(len(labels)), means, yerr=stds, capsize=4, color=colors[: len(labels)])
    plt.axhline(0.5, color="#D62728", linestyle="--", linewidth=1.2, label="Chance level")
    plt.ylabel("Balanced accuracy")
    plt.ylim(0.40, max(0.66, max(means) + max(stds) + 0.04))
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
    plt.title("Cross-session classification: train sessions 1-4, test session 5")
    plt.legend(frameon=False, loc="upper right")
    save_current_figure("cross_session_method_comparison.png")


def plot_within_vs_cross_session() -> None:
    rows = read_csv(TABLES_DIR / "cpu_csp_fbcsp_full_summary.csv")
    protocols = ["within_session_5fold", "cross_session_train_1-4_test_5"]
    methods = ["CSP-LDA", "FBCSP-LDA"]
    data = {
        (row["protocol"], row["method"]): row
        for row in rows
        if row["protocol"] in protocols and row["method"] in methods
    }

    x = list(range(len(methods)))
    width = 0.34
    offsets = [-width / 2, width / 2]
    colors = ["#4C78A8", "#F58518"]
    protocol_labels = {
        "within_session_5fold": "Within-session 5-fold",
        "cross_session_train_1-4_test_5": "Cross-session 1-4 -> 5",
    }

    plt.figure(figsize=(7.2, 4.5))
    for idx, protocol in enumerate(protocols):
        means = [
            to_float(data[(protocol, method)]["balanced_accuracy_mean"])
            for method in methods
        ]
        stds = [
            to_float(data[(protocol, method)]["balanced_accuracy_std"])
            for method in methods
        ]
        xpos = [value + offsets[idx] for value in x]
        plt.bar(
            xpos,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            color=colors[idx],
            label=protocol_labels[protocol],
        )

    plt.axhline(0.5, color="#D62728", linestyle="--", linewidth=1.2, label="Chance level")
    plt.ylabel("Balanced accuracy")
    plt.ylim(0.40, 0.82)
    plt.xticks(x, methods)
    plt.title("Within-session vs cross-session baseline performance")
    plt.legend(frameon=False)
    save_current_figure("within_vs_cross_session.png")


def plot_subject_cross_session_distribution() -> None:
    baseline_rows = read_csv(TABLES_DIR / "cpu_csp_fbcsp_full_all_subjects.csv")
    stable_rows = read_csv(TABLES_DIR / "stable_fbcsp_full_all_subjects_lambda_0p5.csv")

    values_by_method: dict[str, list[float]] = defaultdict(list)
    for row in baseline_rows:
        if row["protocol"] == "cross_session_train_1-4_test_5" and row["method"] in {
            "CSP-LDA",
            "FBCSP-LDA",
        }:
            values_by_method[row["method"]].append(to_float(row["balanced_accuracy"]))

    for row in stable_rows:
        if row["method"] in {"FBCSP-equal", "Stable-FBCSP"}:
            values_by_method[row["method"]].append(to_float(row["balanced_accuracy"]))

    methods = ["CSP-LDA", "FBCSP-LDA", "FBCSP-equal", "Stable-FBCSP"]
    values = [values_by_method[method] for method in methods]

    plt.figure(figsize=(7.8, 4.8))
    box = plt.boxplot(values, tick_labels=methods, patch_artist=True, showmeans=True)
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#B279A2"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.70)
    plt.axhline(0.5, color="#D62728", linestyle="--", linewidth=1.2, label="Chance level")
    plt.ylabel("Balanced accuracy")
    plt.ylim(0.34, 0.76)
    plt.title("Subject-level cross-session performance distribution")
    plt.legend(frameon=False, loc="upper right")
    save_current_figure("subject_cross_session_distribution.png")


def plot_stable_fbcsp_subject_delta() -> None:
    rows = read_csv(TABLES_DIR / "stable_fbcsp_full_all_subjects_lambda_0p5.csv")
    by_subject: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_subject[row["subject"]][row["method"]] = to_float(row["balanced_accuracy"])

    deltas: list[tuple[str, float, float]] = []
    for subject, method_values in by_subject.items():
        if {
            "Stable-FBCSP",
            "FBCSP-equal",
            "FBCSP-discriminative-weighted",
        }.issubset(method_values):
            delta_equal = method_values["Stable-FBCSP"] - method_values["FBCSP-equal"]
            delta_disc = (
                method_values["Stable-FBCSP"]
                - method_values["FBCSP-discriminative-weighted"]
            )
            deltas.append((subject, delta_equal, delta_disc))

    deltas.sort(key=lambda item: item[1])
    subjects = [item[0].replace("sub-", "S") for item in deltas]
    delta_equal = [item[1] for item in deltas]
    delta_disc = [item[2] for item in deltas]

    x = list(range(len(subjects)))
    width = 0.38

    plt.figure(figsize=(11.5, 4.8))
    plt.bar(
        [value - width / 2 for value in x],
        delta_equal,
        width=width,
        color="#B279A2",
        label="Stable - Equal",
    )
    plt.bar(
        [value + width / 2 for value in x],
        delta_disc,
        width=width,
        color="#E45756",
        label="Stable - Disc-weighted",
    )
    plt.axhline(0, color="black", linewidth=1.0)
    plt.ylabel("Balanced accuracy delta")
    plt.xticks(x, subjects, rotation=45, ha="right")
    plt.title("Subject-level delta of Stable-FBCSP (lambda=0.5)")
    plt.legend(frameon=False)
    save_current_figure("stable_fbcsp_subject_delta_lambda_0p5.png")


def plot_stable_fbcsp_v2_subject_delta() -> None:
    path = TABLES_DIR / "stable_fbcsp_v2_full_all_subjects_lambda_0p5.csv"
    if not path.exists():
        print(f"Skip V2 delta figure because table is missing: {path}")
        return

    rows = read_csv(path)
    by_subject: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_subject[row["subject"]][row["method"]] = to_float(row["balanced_accuracy"])

    deltas: list[tuple[str, float, float]] = []
    for subject, method_values in by_subject.items():
        if {
            "Stable-FBCSP-V2",
            "FBCSP-equal",
            "FBCSP-innerLOSO-weighted",
        }.issubset(method_values):
            delta_equal = method_values["Stable-FBCSP-V2"] - method_values["FBCSP-equal"]
            delta_inner = (
                method_values["Stable-FBCSP-V2"]
                - method_values["FBCSP-innerLOSO-weighted"]
            )
            deltas.append((subject, delta_equal, delta_inner))

    deltas.sort(key=lambda item: item[1])
    subjects = [item[0].replace("sub-", "S") for item in deltas]
    delta_equal = [item[1] for item in deltas]
    delta_inner = [item[2] for item in deltas]

    x = list(range(len(subjects)))
    width = 0.38

    plt.figure(figsize=(11.5, 4.8))
    plt.bar(
        [value - width / 2 for value in x],
        delta_equal,
        width=width,
        color="#54A24B",
        label="V2 - Equal",
    )
    plt.bar(
        [value + width / 2 for value in x],
        delta_inner,
        width=width,
        color="#FF9DA6",
        label="V2 - Inner-LOSO weighted",
    )
    plt.axhline(0, color="black", linewidth=1.0)
    plt.ylabel("Balanced accuracy delta")
    plt.xticks(x, subjects, rotation=45, ha="right")
    plt.title("Subject-level delta of Stable-FBCSP V2 (lambda=0.5)")
    plt.legend(frameon=False)
    save_current_figure("stable_fbcsp_v2_subject_delta_lambda_0p5.png")


def plot_stable_fbcsp_weight_heatmap() -> None:
    rows = read_csv(TABLES_DIR / "stable_fbcsp_weights_full_all_subjects_lambda_0p5.csv")
    stable_rows = [row for row in rows if row["method"] == "Stable-FBCSP"]

    subjects = sorted({row["subject"] for row in stable_rows})
    bands = []
    for row in stable_rows:
        if row["band"] not in bands:
            bands.append(row["band"])

    weights: dict[tuple[str, str], float] = {}
    for row in stable_rows:
        weights[(row["subject"], row["band"])] = to_float(row["weight"])

    matrix = [
        [weights.get((subject, band), float("nan")) for band in bands]
        for subject in subjects
    ]

    plt.figure(figsize=(10.0, 7.8))
    image = plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar(image, fraction=0.025, pad=0.02, label="Band fusion weight")
    plt.xticks(range(len(bands)), bands, rotation=45, ha="right")
    plt.yticks(range(len(subjects)), [subject.replace("sub-", "S") for subject in subjects])
    plt.xlabel("Frequency band")
    plt.ylabel("Subject")
    plt.title("Stable-FBCSP band weights (lambda=0.5)")
    save_current_figure("stable_fbcsp_weight_heatmap_lambda_0p5.png")


def plot_stable_fbcsp_v2_weight_heatmap() -> None:
    path = TABLES_DIR / "stable_fbcsp_v2_weights_full_all_subjects_lambda_0p5.csv"
    if not path.exists():
        print(f"Skip V2 weight heatmap because table is missing: {path}")
        return

    rows = read_csv(path)
    stable_rows = [row for row in rows if row["method"] == "Stable-FBCSP-V2"]

    subjects = sorted({row["subject"] for row in stable_rows})
    bands = []
    for row in stable_rows:
        if row["band"] not in bands:
            bands.append(row["band"])

    weights: dict[tuple[str, str], float] = {}
    for row in stable_rows:
        weights[(row["subject"], row["band"])] = to_float(row["weight"])

    matrix = [
        [weights.get((subject, band), float("nan")) for band in bands]
        for subject in subjects
    ]

    plt.figure(figsize=(10.0, 7.8))
    image = plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar(image, fraction=0.025, pad=0.02, label="Band fusion weight")
    plt.xticks(range(len(bands)), bands, rotation=45, ha="right")
    plt.yticks(range(len(subjects)), [subject.replace("sub-", "S") for subject in subjects])
    plt.xlabel("Frequency band")
    plt.ylabel("Subject")
    plt.title("Stable-FBCSP V2 band weights (lambda=0.5)")
    save_current_figure("stable_fbcsp_v2_weight_heatmap_lambda_0p5.png")


def main() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    plot_cross_session_method_comparison()
    plot_within_vs_cross_session()
    plot_subject_cross_session_distribution()
    plot_stable_fbcsp_subject_delta()
    plot_stable_fbcsp_v2_subject_delta()
    plot_stable_fbcsp_weight_heatmap()
    plot_stable_fbcsp_v2_weight_heatmap()


if __name__ == "__main__":
    main()
