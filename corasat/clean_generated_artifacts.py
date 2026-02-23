"""Clean generated Corasat artifacts (dry-run by default)."""
from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path
import shutil
from typing import Iterable, List


CORASAT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CORASAT_ROOT.parent.parent
OVERLEAF_ROOT = PROJECT_ROOT / "Document" / "Overleaf"


def _iter_matches(base: Path, patterns: Iterable[str]) -> List[Path]:
    matches: List[Path] = []
    for pattern in patterns:
        for path in base.rglob("*"):
            rel = path.relative_to(base).as_posix()
            if fnmatch.fnmatch(rel, pattern):
                matches.append(path)
    dedup = sorted(set(matches), key=lambda p: str(p))
    return dedup


def _remove_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), key=lambda p: len(str(p)), reverse=True):
        if not path.is_dir():
            continue
        try:
            next(path.iterdir())
        except StopIteration:
            path.rmdir()
        except Exception:
            continue


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
    parser.add_argument(
        "--include-overleaf",
        action="store_true",
        help="Also remove generated MT artifacts from Document/Overleaf.",
    )
    args = parser.parse_args()

    patterns = [
        "campaign_runs/*",
        "logs/*.log",
        "logs/*.err",
        "review_best_mid_worst_runs/*",
        "lora/output_*",
        "lora/**/checkpoint-*",
        "lora/*_train.jsonl",
        "lora/*_val.jsonl",
        "lora/Modelfile*",
        "optuna_runs_*.csv",
        ".lab_state/*",
    ]
    if not args.keep_results:
        patterns.extend(["results.csv", "lab_results.csv", "results.csv.bak*"])

    matches = _iter_matches(CORASAT_ROOT, patterns)

    overleaf_matches: List[Path] = []
    if args.include_overleaf:
        overleaf_patterns = [
            "tex/generated/*.tex",
            "figures/generated/*.png",
            "review/generated_data/*",
        ]
        overleaf_matches = _iter_matches(OVERLEAF_ROOT, overleaf_patterns)
        matches.extend(overleaf_matches)

    if not matches:
        print("No generated artifacts matched.")
        return 0

    print("Matched paths:")
    for path in matches:
        if str(path).startswith(str(CORASAT_ROOT)):
            rel = path.relative_to(CORASAT_ROOT).as_posix()
            rel = f"corasat/{rel}"
        elif str(path).startswith(str(OVERLEAF_ROOT)):
            rel = path.relative_to(OVERLEAF_ROOT).as_posix()
            rel = f"overleaf/{rel}"
        else:
            rel = str(path)
        print(f"- {rel}")

    if not args.apply:
        print("Dry run only. Re-run with --apply to delete.")
        return 0

    # Remove deepest paths first so nested directories are handled cleanly.
    for path in sorted(matches, key=lambda p: len(str(p)), reverse=True):
        _remove_path(path)

    _remove_empty_dirs(CORASAT_ROOT / "campaign_runs")
    _remove_empty_dirs(CORASAT_ROOT / "logs")
    _remove_empty_dirs(CORASAT_ROOT / ".lab_state")
    if args.include_overleaf:
        _remove_empty_dirs(OVERLEAF_ROOT / "review" / "generated_data")
        _remove_empty_dirs(OVERLEAF_ROOT / "figures" / "generated")
        _remove_empty_dirs(OVERLEAF_ROOT / "tex" / "generated")

    print("Cleanup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
