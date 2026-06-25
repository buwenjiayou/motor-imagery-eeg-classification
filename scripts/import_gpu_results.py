"""Import downloaded GPU experiment files into the main project tree.

This utility is intentionally conservative:

- it copies files instead of moving them, so the downloaded package remains intact;
- if a target file already exists and differs, it is backed up before overwrite;
- it writes a JSON import report for reproducibility.

Run from the course-design workspace root or from the project root:

    python scripts/import_gpu_results.py --execute

By default it performs a dry run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


WORKSPACE_NAME = "课程设计"
PROJECT_NAME = "motor-imagery-eeg-classification"
GPU_PACKAGE_NAME = "gpu_experiment_results_20260624"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def find_workspace_root(start: Path) -> Path:
    current = start.resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if candidate.name == PROJECT_NAME:
            return candidate.parent
        if (candidate / PROJECT_NAME).is_dir() and (candidate / GPU_PACKAGE_NAME).is_dir():
            return candidate
    raise FileNotFoundError(
        f"Cannot locate workspace root containing {PROJECT_NAME!r} and {GPU_PACKAGE_NAME!r} from {start}"
    )


def copy_with_backup(src_root: Path, dst_root: Path, workspace_root: Path, execute: bool) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = dst_root / "import_backups" / f"gpu_20260624_{timestamp}"

    source_files = sorted(path for path in src_root.rglob("*") if path.is_file())
    report: dict = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "execute": execute,
        "workspace_root": str(workspace_root),
        "source_root": str(src_root),
        "target_root": str(dst_root),
        "backup_root": str(backup_root),
        "counts": {
            "source_files": len(source_files),
            "new_files": 0,
            "overwritten_files": 0,
            "identical_skipped": 0,
        },
        "new_files": [],
        "overwritten_files": [],
        "identical_skipped": [],
    }

    for src in source_files:
        relative_path = src.relative_to(src_root)
        dst = dst_root / relative_path
        src_hash = sha256(src)

        if dst.exists():
            dst_hash = sha256(dst)
            if src_hash == dst_hash:
                report["counts"]["identical_skipped"] += 1
                report["identical_skipped"].append(str(relative_path))
                continue

            backup_path = backup_root / relative_path
            report["counts"]["overwritten_files"] += 1
            report["overwritten_files"].append(
                {
                    "path": str(relative_path),
                    "backup_path": str(backup_path.relative_to(dst_root)),
                    "source_sha256": src_hash,
                    "previous_sha256": dst_hash,
                }
            )
            if execute:
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dst, backup_path)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        else:
            report["counts"]["new_files"] += 1
            report["new_files"].append(str(relative_path))
            if execute:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    if execute:
        log_dir = dst_root / "results" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        report_path = log_dir / f"gpu_import_20260624_{timestamp}.json"
        report["report_path"] = str(report_path.relative_to(dst_root))
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Import GPU experiment result package.")
    parser.add_argument("--execute", action="store_true", help="Actually copy files. Omit for dry run.")
    args = parser.parse_args()

    workspace_root = find_workspace_root(Path.cwd()).resolve()
    project_root = (workspace_root / PROJECT_NAME).resolve()
    package_project_root = (workspace_root / GPU_PACKAGE_NAME / PROJECT_NAME).resolve()

    for path in [workspace_root, project_root, package_project_root]:
        if not path.exists():
            raise FileNotFoundError(path)

    if not is_relative_to(project_root, workspace_root):
        raise RuntimeError(f"Target project is outside workspace: {project_root}")
    if not is_relative_to(package_project_root, workspace_root):
        raise RuntimeError(f"Source package is outside workspace: {package_project_root}")

    report = copy_with_backup(package_project_root, project_root, workspace_root, args.execute)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
