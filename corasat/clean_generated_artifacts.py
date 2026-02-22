"""Clean generated Corasat artifacts (dry-run by default)."""
from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path
import shutil
from typing import Iterable, List


CORASAT_ROOT = Path(__file__).resolve().parent


def _iter_matches(base: Path, patterns: Iterable[str]) -> List[Path]:
    matches: List[Path] = []
    for pattern in patterns:
        for path in base.rglob("*"):
            rel = path.relative_to(base).as_posix()
            if fnmatch.fnmatch(rel, pattern):
                matches.append(path)
    dedup = sorted(set(matches), key=lambda p: str(p))
    return dedup


def _remove_path(path: Path) -> None:
    if path.is_file() or path.is_symlink():
        path.unlink(missing_ok=True)
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean generated Corasat artifacts.")
    parser.add_argument("--apply", action="store_true", help="Actually delete matching paths.")
    parser.add_argument(
        "--keep-results",
        action="store_true",
        help="Keep canonical results.csv and lab_results.csv.",
    )
    args = parser.parse_args()

    patterns = [
        "campaign_runs/*",
        "logs/*.log",
        "logs/*.err",
        "review_best_mid_worst_runs/*",
        "lora/output_*",
        "lora/**/checkpoint-*",
        "optuna_runs_*.csv",
        ".lab_state/*",
    ]
    if not args.keep_results:
        patterns.extend(["results.csv", "lab_results.csv", "results.csv.bak*"])

    matches = _iter_matches(CORASAT_ROOT, patterns)
    if not matches:
        print("No generated artifacts matched.")
        return 0

    print("Matched paths:")
    for path in matches:
        rel = path.relative_to(CORASAT_ROOT).as_posix()
        print(f"- {rel}")

    if not args.apply:
        print("Dry run only. Re-run with --apply to delete.")
        return 0

    # Remove deepest paths first so nested directories are handled cleanly.
    for path in sorted(matches, key=lambda p: len(str(p)), reverse=True):
        _remove_path(path)

    print("Cleanup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
