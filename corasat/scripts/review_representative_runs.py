#!/usr/bin/env python3
"""Reproduce worst/average/best runs per lab and capture logs/screenshots."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

COMPARE_FIELDS = [
    "norm_score",
    "mission_score",
    "correct_edges",
    "false_edges",
    "rounds",
    "broadcasts",
    "coverage",
    "total_gt_edges",
]
FLOAT_FIELDS = {"norm_score", "mission_score", "coverage"}
FLOAT_TOLERANCE = 1e-5
LAB_ID_RE = re.compile(r"^(L\d+)", re.IGNORECASE)


def _resolve_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _normalize_path(value: str) -> str:
    if not value:
        return ""
    cleaned = value.replace("\\\\", "\\")
    return os.path.normpath(cleaned).lower()


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _load_lab_rows(lab_results_path: Path) -> List[Dict[str, str]]:
    rows = _read_csv(lab_results_path)
    return [row for row in rows if (row.get("label") or "").strip()]


def _rows_for_logfile(results_rows: List[Dict[str, str]], logfile: str) -> List[Dict[str, str]]:
    target = _normalize_path(logfile)
    by_seed: Dict[int, Dict[str, str]] = {}
    for row in results_rows:
        if _normalize_path(row.get("logfile", "")) != target:
            continue
        seed = _parse_int(row.get("seed"))
        if seed is None:
            continue
        by_seed[seed] = row
    return list(by_seed.values())


def _select_representatives(rows: List[Dict[str, str]]) -> Tuple[Dict[str, Dict[str, str]], float]:
    scored: List[Tuple[float, int, Dict[str, str]]] = []
    for row in rows:
        score = _parse_float(row.get("norm_score"))
        seed = _parse_int(row.get("seed"))
        if score is None or seed is None:
            continue
        scored.append((score, seed, row))
    if not scored:
        return {}, 0.0

    scores = [item[0] for item in scored]
    mean_score = mean(scores)

    worst = min(scored, key=lambda item: (item[0], item[1]))
    best = max(scored, key=lambda item: (item[0], -item[1]))
    average = min(scored, key=lambda item: (abs(item[0] - mean_score), item[1]))

    return {
        "worst": worst[2],
        "average": average[2],
        "best": best[2],
    }, mean_score


def _safe_slug(value: str) -> str:
    cleaned = []
    for ch in value.strip():
        if ch.isalnum():
            cleaned.append(ch)
        elif ch in {" ", "-", "_"}:
            cleaned.append("_")
        else:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "lab"


def _lab_id_from_label(label: str) -> str:
    match = LAB_ID_RE.match(label.strip())
    if match:
        return match.group(1).upper()
    return ""


def _copy_file(src: Path, dest: Path) -> None:
    if not src.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _write_config_for_seed(snapshot_config: Path, dest_config: Path, seed: int) -> None:
    with snapshot_config.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    sim_cfg = cfg.setdefault("simulation", {})
    sim_cfg["seed_list"] = [seed]
    sim_cfg["use_gui"] = True
    sim_cfg["create_py_export"] = False
    with dest_config.open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def _compare_metrics(
    original: Dict[str, str],
    reproduced: Dict[str, str],
) -> Tuple[bool, Dict[str, Optional[float]]]:
    diffs: Dict[str, Optional[float]] = {}
    match = True
    for field in COMPARE_FIELDS:
        orig_val = _parse_float(original.get(field))
        repro_val = _parse_float(reproduced.get(field))
        if orig_val is None or repro_val is None:
            diffs[field] = None
            if orig_val != repro_val:
                match = False
            continue
        diff = repro_val - orig_val
        diffs[field] = diff
        tol = FLOAT_TOLERANCE if field in FLOAT_FIELDS else 0.0
        if abs(diff) > tol:
            match = False
    return match, diffs


def _resolve_lab_config(base_dir: Path, lab_config_path: str) -> Path:
    lab_config_path = lab_config_path.replace("\\\\", "\\")
    path = Path(lab_config_path)
    if path.is_absolute():
        return path
    return base_dir.parent / path


def _detect_new_log(before: Iterable[Path], after: Iterable[Path]) -> Optional[Path]:
    before_set = {path.resolve() for path in before}
    candidates = [path for path in after if path.resolve() not in before_set]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_new_results(results_path: Path, before_count: int) -> List[Dict[str, str]]:
    rows = _read_csv(results_path)
    if before_count >= len(rows):
        return []
    return rows[before_count:]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce worst/average/best runs and capture logs/screenshots."
    )
    parser.add_argument(
        "--output-root",
        default="review_worst_runs",
        help="Root folder to store review runs (relative to corasat).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional subfolder under output-root (reuse to resume).",
    )
    parser.add_argument(
        "--lab-results",
        default="lab_results.csv",
        help="Path to lab_results.csv (relative to corasat).",
    )
    parser.add_argument(
        "--results",
        default="results.csv",
        help="Path to results.csv (relative to corasat).",
    )
    parser.add_argument(
        "--labs",
        default="",
        help="Comma-separated lab IDs to run (e.g., L1,L3).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already have a log file in the review folder.",
    )
    args = parser.parse_args()

    base_dir = _resolve_root()
    lab_results_path = (base_dir / args.lab_results).resolve()
    results_path = (base_dir / args.results).resolve()

    output_root = base_dir / args.output_root
    if args.output_dir:
        review_root = output_root / args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        review_root = output_root / timestamp
    review_root.mkdir(parents=True, exist_ok=True)
    backup_root = review_root / "_backup"
    backup_root.mkdir(parents=True, exist_ok=True)

    config_path = base_dir / "config.json"
    rules_path = base_dir / "rules.txt"
    drone_support_path = base_dir / "classes" / "Drone_Support.py"
    screenshot_path = base_dir / "screenshots" / "last_run.png"
    logs_dir = base_dir / "logs"

    backups = [
        config_path,
        rules_path,
        drone_support_path,
        results_path,
        screenshot_path,
    ]
    for item in backups:
        if item.exists():
            _copy_file(item, backup_root / item.name)

    created_logs: List[Path] = []
    manifest_path = review_root / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            manifest = {}
    else:
        manifest = {}

    manifest.setdefault("created_at", datetime.now().astimezone().isoformat())
    manifest["updated_at"] = datetime.now().astimezone().isoformat()
    manifest["review_root"] = os.path.relpath(str(review_root), start=str(base_dir.parent))
    manifest.setdefault("runs", [])
    new_runs: List[Dict[str, object]] = []

    results_rows = _read_csv(results_path)
    lab_rows = _load_lab_rows(lab_results_path)

    try:
        lab_filter = {
            item.strip().upper()
            for item in args.labs.split(",")
            if item.strip()
        }

        for lab in lab_rows:
            label = (lab.get("label") or "").strip()
            logfile = (lab.get("logfile") or "").strip()
            lab_config_path = (lab.get("lab_config_path") or "").strip()
            if not label or not logfile or not lab_config_path:
                continue
            lab_id = _lab_id_from_label(label)
            if lab_filter and lab_id not in lab_filter:
                continue

            lab_results_rows = _rows_for_logfile(results_rows, logfile)
            reps, mean_score = _select_representatives(lab_results_rows)
            if not reps:
                continue

            lab_dir = review_root / _safe_slug(label)
            lab_dir.mkdir(parents=True, exist_ok=True)

            for kind in ("worst", "average", "best"):
                original_row = reps.get(kind)
                if not original_row:
                    continue
                seed = _parse_int(original_row.get("seed"))
                if seed is None:
                    continue

                run_dir = lab_dir / f"{kind}_seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                if args.skip_existing and list(run_dir.glob("*.log")):
                    continue

                snapshot_dir = _resolve_lab_config(base_dir, lab_config_path)
                snapshot_config = snapshot_dir / "config.json"
                snapshot_rules = snapshot_dir / "rules.txt"
                snapshot_drone = snapshot_dir / "classes" / "Drone_Support.py"

                _write_config_for_seed(snapshot_config, config_path, seed)
                _copy_file(snapshot_rules, rules_path)
                _copy_file(snapshot_drone, drone_support_path)

                before_logs = list(logs_dir.glob("simulation_*.log"))
                before_count = len(_read_csv(results_path))

                run_error = ""
                try:
                    subprocess.run(
                        [sys.executable, "main.py"],
                        cwd=str(base_dir),
                        check=True,
                    )
                except subprocess.CalledProcessError as exc:
                    run_error = f"run failed (exit {exc.returncode})"
                except Exception as exc:
                    run_error = f"run failed ({exc})"

                after_logs = list(logs_dir.glob("simulation_*.log"))
                new_log = _detect_new_log(before_logs, after_logs)
                log_copy = ""
                if new_log:
                    _copy_file(new_log, run_dir / new_log.name)
                    log_copy = new_log.name
                    created_logs.append(new_log)

                screenshot_copy = ""
                if screenshot_path.exists():
                    screenshot_copy = "screenshot.png"
                    _copy_file(screenshot_path, run_dir / screenshot_copy)

                new_rows = _read_new_results(results_path, before_count)
                reproduced_row: Dict[str, str] = {}
                for row in new_rows:
                    if _parse_int(row.get("seed")) == seed:
                        reproduced_row = row
                        break
                if not reproduced_row and new_rows:
                    reproduced_row = new_rows[-1]

                match = False
                diffs: Dict[str, Optional[float]] = {}
                if original_row and reproduced_row:
                    match, diffs = _compare_metrics(original_row, reproduced_row)

                run_entry = {
                    "lab_id": lab_id,
                    "lab_label": label,
                    "lab_config_path": lab_config_path,
                    "lab_logfile": logfile,
                    "kind": kind,
                    "seed": seed,
                    "mean_norm_score": round(mean_score, 5),
                    "original_metrics": {field: original_row.get(field) for field in COMPARE_FIELDS},
                    "reproduced_metrics": {field: reproduced_row.get(field) for field in COMPARE_FIELDS},
                    "diffs": diffs,
                    "match": match,
                    "run_error": run_error,
                    "review_dir": os.path.relpath(str(run_dir), start=str(base_dir.parent)),
                    "logfile_copy": log_copy,
                    "screenshot_copy": screenshot_copy,
                }
                new_runs.append(run_entry)
    finally:
        if (backup_root / "results.csv").exists():
            _copy_file(backup_root / "results.csv", results_path)
        if (backup_root / "config.json").exists():
            _copy_file(backup_root / "config.json", config_path)
        if (backup_root / "rules.txt").exists():
            _copy_file(backup_root / "rules.txt", rules_path)
        if (backup_root / "Drone_Support.py").exists():
            _copy_file(backup_root / "Drone_Support.py", drone_support_path)
        if (backup_root / "last_run.png").exists():
            _copy_file(backup_root / "last_run.png", screenshot_path)
        else:
            if screenshot_path.exists():
                try:
                    screenshot_path.unlink()
                except Exception:
                    pass

        for log_path in created_logs:
            try:
                if log_path.exists():
                    log_path.unlink()
            except Exception:
                pass

    combined: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    for entry in manifest.get("runs", []):
        key = (str(entry.get("lab_label")), str(entry.get("kind")), int(entry.get("seed", 0)))
        combined[key] = entry
    for entry in new_runs:
        key = (str(entry.get("lab_label")), str(entry.get("kind")), int(entry.get("seed", 0)))
        combined[key] = entry
    manifest["runs"] = list(combined.values())

    manifest_path = review_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    summary_rows = []
    for entry in manifest["runs"]:
        summary_rows.append(
            {
                "lab_label": str(entry.get("lab_label") or ""),
                "kind": str(entry.get("kind") or ""),
                "seed": str(entry.get("seed") or ""),
                "original_norm_score": str((entry.get("original_metrics") or {}).get("norm_score") or ""),
                "reproduced_norm_score": str((entry.get("reproduced_metrics") or {}).get("norm_score") or ""),
                "match": str(entry.get("match")),
                "run_error": str(entry.get("run_error") or ""),
                "review_dir": str(entry.get("review_dir") or ""),
            }
        )
    summary_path = review_root / "summary.csv"
    _write_csv(
        summary_path,
        ["lab_label", "kind", "seed", "original_norm_score", "reproduced_norm_score", "match", "run_error", "review_dir"],
        summary_rows,
    )

    print(f"Review folder: {review_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
