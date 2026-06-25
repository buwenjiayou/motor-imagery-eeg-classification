"""
E0 data audit for the SHU motor imagery EEG course project.

This script intentionally uses only the Python standard library so it can run
before the scientific Python environment is fully prepared. It checks file
coverage and event-label distribution across MAT/EDF/events representations.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MAT_DIR = RAW_DIR / "mat"
EDF_DIR = RAW_DIR / "edf"
EVENTS_DIR = RAW_DIR / "events"
METADATA_DIR = RAW_DIR / "metadata"

RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
RESULTS_LOGS = PROJECT_ROOT / "results" / "logs"

SESSION_RE = re.compile(r"sub-(?P<subject>\d{3})_ses-(?P<session>\d{2})_task_motorimagery")


def parse_session_key(path: Path) -> tuple[str, str]:
    match = SESSION_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse subject/session from filename: {path.name}")
    return f"sub-{match.group('subject')}", f"ses-{match.group('session')}"


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({key: (value or "").strip() for key, value in row.items()})
        return rows


def read_event_rows(path: Path) -> list[dict[str, str]]:
    """Read an events.tsv file.

    Most event files contain a BIDS-like header. One observed file
    (`sub-001_ses-04_task_motorimagery_events.tsv`) is headerless, so this
    function falls back to the expected column order instead of silently
    dropping `trial_type`.
    """

    expected_columns = ["onset", "duration", "trial_type", "response_time", "sample", "value"]
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        lines = [line for line in f.readlines() if line.strip()]

    if not lines:
        return []

    first_fields = [field.strip() for field in lines[0].rstrip("\n").split("\t")]
    has_header = "trial_type" in first_fields

    data_lines = lines[1:] if has_header else lines
    rows: list[dict[str, str]] = []
    for line in data_lines:
        fields = [field.strip() for field in line.rstrip("\n").split("\t")]
        row = {
            column: fields[index] if index < len(fields) else ""
            for index, column in enumerate(expected_columns)
        }
        rows.append(row)
    return rows


def count_events(path: Path) -> Counter:
    rows = read_event_rows(path)
    counts: Counter = Counter()
    for row in rows:
        trial_type = row.get("trial_type", "").lower().strip()
        if trial_type:
            counts[trial_type] += 1
        else:
            counts["missing_trial_type"] += 1
    return counts


def count_metadata_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return len(read_tsv_rows(path))


def size_mb(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return f"{path.stat().st_size / 1024 / 1024:.3f}"


def collect_files(directory: Path, pattern: str) -> dict[tuple[str, str], Path]:
    files: dict[tuple[str, str], Path] = {}
    for path in sorted(directory.glob(pattern)):
        files[parse_session_key(path)] = path
    return files


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    RESULTS_LOGS.mkdir(parents=True, exist_ok=True)

    mat_files = collect_files(MAT_DIR, "*.mat")
    edf_files = collect_files(EDF_DIR, "*.edf")
    event_files = collect_files(EVENTS_DIR, "*.tsv")

    all_keys = sorted(set(mat_files) | set(edf_files) | set(event_files))

    session_rows: list[dict[str, object]] = []
    global_event_counts: Counter = Counter()
    missing_pairs: list[str] = []

    for subject, session in all_keys:
        key = (subject, session)
        event_counts = Counter()
        if key in event_files:
            event_counts = count_events(event_files[key])
            global_event_counts.update(event_counts)

        has_mat = key in mat_files
        has_edf = key in edf_files
        has_events = key in event_files

        if not (has_mat and has_edf and has_events):
            missing_pairs.append(
                f"{subject}_{session}: mat={has_mat}, edf={has_edf}, events={has_events}"
            )

        left_trials = event_counts.get("left", 0)
        right_trials = event_counts.get("right", 0)
        event_trials = sum(event_counts.values())
        other_trials = event_trials - left_trials - right_trials

        session_rows.append(
            {
                "subject": subject,
                "session": session,
                "has_mat": int(has_mat),
                "has_edf": int(has_edf),
                "has_events": int(has_events),
                "mat_size_mb": size_mb(mat_files.get(key)),
                "edf_size_mb": size_mb(edf_files.get(key)),
                "event_trials": event_trials,
                "left_trials": left_trials,
                "right_trials": right_trials,
                "other_trials": other_trials,
            }
        )

    participants = count_metadata_rows(METADATA_DIR / "participants.tsv")
    channels = count_metadata_rows(METADATA_DIR / "task-motorimagery_channels.tsv")

    summary = {
        "project_root": str(PROJECT_ROOT),
        "mat_files": len(mat_files),
        "edf_files": len(edf_files),
        "event_files": len(event_files),
        "unique_sessions": len(all_keys),
        "participants_rows": participants,
        "channel_rows": channels,
        "event_trials_total": sum(global_event_counts.values()),
        "event_left_trials": global_event_counts.get("left", 0),
        "event_right_trials": global_event_counts.get("right", 0),
        "event_other_trials": sum(global_event_counts.values())
        - global_event_counts.get("left", 0)
        - global_event_counts.get("right", 0),
        "incomplete_session_records": len(missing_pairs),
    }

    write_csv(
        RESULTS_TABLES / "data_audit_by_session.csv",
        session_rows,
        [
            "subject",
            "session",
            "has_mat",
            "has_edf",
            "has_events",
            "mat_size_mb",
            "edf_size_mb",
            "event_trials",
            "left_trials",
            "right_trials",
            "other_trials",
        ],
    )

    write_csv(
        RESULTS_TABLES / "data_audit_summary.csv",
        [{"metric": key, "value": value} for key, value in summary.items()],
        ["metric", "value"],
    )

    with (RESULTS_LOGS / "data_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "event_label_counts": dict(sorted(global_event_counts.items())),
                "missing_pairs": missing_pairs,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if missing_pairs:
        print("Incomplete records:")
        for item in missing_pairs:
            print(f"- {item}")


if __name__ == "__main__":
    main()
