"""Summarize EA+EEGNet compressed results and compare with EEGNet."""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"


def load(path: Path):
    return list(csv.DictReader(path.open("r", encoding="utf-8-sig")))


def write(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def mean(xs): return float(statistics.mean(xs))
def stdev(xs): return float(statistics.stdev(xs)) if len(xs) > 1 else 0.0


def summarize(rows):
    out = {}
    for m in ["accuracy", "balanced_accuracy", "macro_f1", "kappa", "epochs_ran", "train_seconds"]:
        vals = [float(r[m]) for r in rows]
        out[f"{m}_mean"] = mean(vals)
        out[f"{m}_std"] = stdev(vals)
    return out


def main():
    paths = sorted(TABLES.glob("ea_eegnet_compressed50_seed_*_25subjects_subject_metrics.csv"))
    rows = []
    for p in paths:
        rows.extend(load(p))
    if not rows:
        raise FileNotFoundError("No EA+EEGNet subject metrics found")

    write(TABLES / "ea_eegnet_compressed50_3seeds_subject_seed_metrics.csv", rows)

    by_seed = defaultdict(list)
    by_subject = defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append(r)
        by_subject[r["subject"]].append(r)

    seed_rows = []
    for seed, group in sorted(by_seed.items(), key=lambda x: int(x[0])):
        row = {"family":"alignment_deep_learning","method":"EA+EEGNet-compressed","protocol":"cross_session_train_1-4_test_5","alignment":"euclidean_alignment_unsupervised_target","seed":int(seed),"n_subjects":len(group)}
        row.update(summarize(group)); seed_rows.append(row)
    write(TABLES / "ea_eegnet_compressed50_3seeds_by_seed_summary.csv", seed_rows)

    subject_rows = []
    for subject, group in sorted(by_subject.items()):
        row = {"subject":subject,"method":"EA+EEGNet-compressed","protocol":"cross_session_train_1-4_test_5","alignment":"euclidean_alignment_unsupervised_target","n_seeds":len(group)}
        row.update(summarize(group)); subject_rows.append(row)
    write(TABLES / "ea_eegnet_compressed50_3seeds_by_subject_summary.csv", subject_rows)

    overall = {"family":"alignment_deep_learning","method":"EA+EEGNet-compressed","protocol":"cross_session_train_1-4_test_5","alignment":"euclidean_alignment_unsupervised_target","n_rows":len(rows),"n_subjects":len(by_subject),"n_seeds":len(by_seed),"seeds":",".join(sorted(by_seed.keys(), key=int))}
    overall.update(summarize(rows))
    write(TABLES / "ea_eegnet_compressed50_3seeds_overall_summary.csv", [overall])
    with (LOGS / "ea_eegnet_compressed50_3seeds_overall_summary.json").open("w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    # Same-seed paired comparison against non-EA EEGNet.
    base_paths = sorted(TABLES.glob("dl_eegnet_compressed50_seed_*_25subjects_subject_metrics.csv"))
    base = []
    for p in base_paths:
        base.extend(load(p))
    base_by = {(r["subject"], r["seed"]): r for r in base}
    deltas = []
    for r in rows:
        b = base_by[(r["subject"], r["seed"])]
        deltas.append({
            "subject": r["subject"],
            "seed": int(r["seed"]),
            "eegnet_bacc": float(b["balanced_accuracy"]),
            "ea_eegnet_bacc": float(r["balanced_accuracy"]),
            "delta_bacc": float(r["balanced_accuracy"]) - float(b["balanced_accuracy"]),
        })
    write(TABLES / "ea_eegnet_vs_eegnet_3seeds_subject_seed_delta.csv", deltas)
    vals = np.array([r["delta_bacc"] for r in deltas], dtype=float)
    delta_summary = {
        "comparison": "EA+EEGNet minus EEGNet, same subject-seed pairs",
        "n_pairs": len(deltas),
        "delta_balanced_accuracy_mean": float(vals.mean()),
        "delta_balanced_accuracy_std": float(vals.std(ddof=1)),
        "n_improved": int((vals > 1e-12).sum()),
        "n_tied": int((np.abs(vals) <= 1e-12).sum()),
        "n_decreased": int((vals < -1e-12).sum()),
        "best_delta": float(vals.max()),
        "worst_delta": float(vals.min()),
    }
    write(TABLES / "ea_eegnet_vs_eegnet_3seeds_delta_summary.csv", [delta_summary])

    # Update key comparison table.
    compact = []
    key_path = TABLES / "key_cross_session_alignment_dl_comparison.csv"
    if key_path.exists():
        compact = load(key_path)
        compact = [r for r in compact if r.get("method") != "EA+EEGNet-compressed seed 42"]
    compact.append({
        "family":"alignment_deep_learning",
        "method":"EA+EEGNet-compressed 3 seeds",
        "protocol":overall["protocol"],
        "n_rows":overall["n_rows"],
        "balanced_accuracy_mean":overall["balanced_accuracy_mean"],
        "balanced_accuracy_std":overall["balanced_accuracy_std"],
        "accuracy_mean":overall["accuracy_mean"],
        "accuracy_std":overall["accuracy_std"],
    })
    write(key_path, compact)

    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print(json.dumps(delta_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
