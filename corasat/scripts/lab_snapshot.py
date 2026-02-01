"""Create reproducible lab config snapshots and optionally update lab_results.csv."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
from typing import Any, Dict, List, Optional, Tuple


LAB_RESULTS_FIELDS = [
    "timestamp",
    "label",
    "mean",
    "stddev",
    "mean_plus_minus",
    "seed_range",
    "seed_count",
    "lab_config_hash_SHA-256",
    "lab_config_path",
    "logfile",
    "notes",
]

GLOBAL_DOCS = [
    "decision_support_parameters.md",
]

LAB_CONFIG_HASH_SHORT_LEN = 16
LAB_CONFIG_HASH_IGNORE_NAMES = {"metadata.json"}
LAB_CONFIG_HASH_IGNORE_DIRS = {"__pycache__"}


def _global_items(base_dir: Path) -> List[str]:
    items: List[str] = []
    main_path = base_dir / "main.py"
    if main_path.exists():
        items.append(os.path.relpath(str(main_path), start=str(base_dir)))
    classes_dir = base_dir / "classes"
    if classes_dir.exists():
        for path in sorted(classes_dir.glob("*.py")):
            if path.name == "Drone_Support.py":
                continue
            items.append(os.path.relpath(str(path), start=str(base_dir)))
    return items


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _iter_lab_hash_files(base_dir: Path) -> List[Path]:
    files: List[Path] = []
    for path in sorted(base_dir.rglob("*")):
        if path.is_dir():
            if path.name in LAB_CONFIG_HASH_IGNORE_DIRS:
                continue
            continue
        if path.name in LAB_CONFIG_HASH_IGNORE_NAMES:
            continue
        files.append(path)
    return files


def _lab_config_hash_from_payloads(payloads: List[Tuple[str, bytes]]) -> str:
    hasher = hashlib.sha256()
    for rel_path, data in sorted(payloads, key=lambda item: item[0]):
        hasher.update(rel_path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(len(data)).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(data)
    return hasher.hexdigest()[:LAB_CONFIG_HASH_SHORT_LEN]


def _lab_config_hash_from_dir(lab_dir: Path) -> str:
    payloads: List[Tuple[str, bytes]] = []
    for path in _iter_lab_hash_files(lab_dir):
        rel_path = path.relative_to(lab_dir).as_posix()
        payloads.append((rel_path, path.read_bytes()))
    return _lab_config_hash_from_payloads(payloads)


def _normalize_path(value: str) -> str:
    if not value:
        return ""
    return os.path.normpath(value).lower()


def _match_logfile_row(row: Dict[str, str], target: str) -> bool:
    if not target:
        return False
    row_value = row.get("logfile", "")
    return _normalize_path(row_value) == _normalize_path(target)


def _stats_from_results(results_csv: Path, logfile: str) -> Tuple[Optional[float], Optional[float], int, str]:
    if not results_csv.exists() or not logfile:
        return None, None, 0, ""
    scores: List[float] = []
    seeds: List[int] = []
    with open(results_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not _match_logfile_row(row, logfile):
                continue
            try:
                scores.append(float(row.get("norm_score") or 0.0))
            except Exception:
                continue
            try:
                seeds.append(int(row.get("seed") or 0))
            except Exception:
                pass
    if not scores:
        return None, None, 0, ""
    mean = statistics.mean(scores)
    stddev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    seed_range = ""
    if seeds:
        seed_range = f"{min(seeds)}-{max(seeds)}"
    return mean, stddev, len(scores), seed_range


def _ensure_lab_results_columns(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return rows
        for row in reader:
            rows.append(row)
    # Normalize to the new schema.
    normalized: List[Dict[str, str]] = []
    for row in rows:
        updated = {field: row.get(field, "") for field in LAB_RESULTS_FIELDS}
        normalized.append(updated)
    return normalized


def _write_lab_results(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LAB_RESULTS_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_snapshot_dir(base_dir: Path, label: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_label = "".join(ch for ch in label if ch.isalnum() or ch in ("-", "_")).strip("_-")
    suffix = timestamp
    if safe_label:
        suffix = f"{suffix}_{safe_label[:32]}"
    return base_dir / "lab_configs" / suffix


def _copy_if_exists(src: Path, dest: Path, missing: List[str]) -> None:
    if not src.exists():
        missing.append(str(src))
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _snapshot_configs(
    base_dir: Path,
    cfg_path: Path,
    rules_path: Path,
    drone_support_path: Path,
    label: str,
    logfile: str,
    seed_range: str,
    seed_count: int,
    rules_enabled: str,
    mean: Optional[float],
    stddev: Optional[float],
) -> Path:
    snapshot_dir = _build_snapshot_dir(base_dir, label)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    missing: List[str] = []
    copied: List[str] = []
    missing: List[str] = []
    config_items: List[str] = []
    for item in [cfg_path, rules_path, drone_support_path]:
        rel_path = (
            os.path.relpath(str(item), start=str(base_dir))
            if str(item).startswith(str(base_dir))
            else item.name
        )
        config_items.append(rel_path)
        dest = snapshot_dir / rel_path
        if item.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)
            copied.append(rel_path)
        else:
            missing.append(rel_path)

    metadata = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "label": label,
        "logfile": logfile,
        "seed_range": seed_range,
        "seed_count": seed_count,
        "rules_used": rules_enabled,
        "mean": round(mean, 5) if isinstance(mean, (int, float)) else None,
        "stddev": round(stddev, 5) if isinstance(stddev, (int, float)) else None,
        "config_items": config_items,
        "files_copied": copied,
        "missing_files": missing,
        "global_items": _global_items(base_dir),
        "global_docs": GLOBAL_DOCS,
    }
    with open(snapshot_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)
    return snapshot_dir


def _update_lab_results(
    path: Path,
    mode: str,
    label: str,
    logfile: str,
    seed_range: str,
    seed_count: int,
    mean: Optional[float],
    stddev: Optional[float],
    lab_config_hash: str,
    lab_config_path: str,
    notes: str,
) -> None:
    rows = _ensure_lab_results_columns(path)
    timestamp = datetime.now().astimezone().isoformat()
    mean_val = round(mean, 5) if isinstance(mean, (int, float)) else ""
    std_val = round(stddev, 5) if isinstance(stddev, (int, float)) else ""
    mean_pm = f"{mean_val:.5f} +/- {std_val:.5f}" if isinstance(mean, (int, float)) else ""
    row_payload = {
        "timestamp": timestamp,
        "label": label,
        "mean": mean_val,
        "stddev": std_val,
        "mean_plus_minus": mean_pm,
        "seed_range": seed_range,
        "seed_count": seed_count if seed_count else "",
        "lab_config_hash_SHA-256": lab_config_hash,
        "lab_config_path": lab_config_path,
        "logfile": logfile,
        "notes": notes,
    }

    if mode == "skip":
        return

    if mode == "update":
        updated = False
        for row in rows:
            if row.get("label") == label and _match_logfile_row(row, logfile):
                row.update({k: v for k, v in row_payload.items() if v != ""})
                updated = True
        if not updated:
            rows.append(row_payload)
    else:
        rows.append(row_payload)

    _write_lab_results(path, rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Snapshot lab configs and optionally update lab_results.csv.")
    parser.add_argument("--label", required=True, help="Lab result label for lab_results.csv.")
    parser.add_argument("--logfile", default="", help="Logfile entry from results.csv for aggregation.")
    parser.add_argument("--seed-range", default="", help="Seed range string, e.g. 1-100.")
    parser.add_argument("--seed-count", type=int, default=0, help="Number of seeds in the run.")
    parser.add_argument("--rules-enabled", default="", help="true/false to record rules usage.")
    parser.add_argument("--notes", default="", help="Free-form notes for lab_results.csv.")
    parser.add_argument("--mean", type=float, default=None, help="Optional mean override.")
    parser.add_argument("--stddev", type=float, default=None, help="Optional stddev override.")
    parser.add_argument("--mode", choices=("append", "update", "skip"), default="append")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--results-csv", default="results.csv")
    parser.add_argument("--lab-results-csv", default="lab_results.csv")

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    cfg_path = (base_dir / args.config).resolve()
    cfg = _load_json(cfg_path)

    rules_path = cfg.get("rules_path") if isinstance(cfg, dict) else ""
    rules_path = rules_path or "rules.txt"
    rules_candidate = Path(rules_path)
    if rules_candidate.is_absolute():
        rules_path = rules_candidate
    else:
        config_relative = (cfg_path.parent / rules_candidate).resolve()
        if config_relative.exists():
            rules_path = config_relative
        else:
            rules_path = (base_dir / rules_candidate).resolve()
    drone_support_path = (base_dir / "classes" / "Drone_Support.py").resolve()

    results_csv = (base_dir / args.results_csv).resolve()
    mean = args.mean
    stddev = args.stddev
    seed_count = args.seed_count
    seed_range = args.seed_range
    if args.logfile and (mean is None or stddev is None or not seed_count or not seed_range):
        calc_mean, calc_std, calc_count, calc_range = _stats_from_results(results_csv, args.logfile)
        if mean is None:
            mean = calc_mean
        if stddev is None:
            stddev = calc_std
        if not seed_count and calc_count:
            seed_count = calc_count
        if not seed_range and calc_range:
            seed_range = calc_range

    snapshot_dir = _snapshot_configs(
        base_dir,
        cfg_path,
        rules_path,
        drone_support_path,
        args.label,
        args.logfile,
        seed_range,
        seed_count,
        args.rules_enabled,
        mean,
        stddev,
    )
    lab_config_hash = _lab_config_hash_from_dir(snapshot_dir)

    lab_results_csv = (base_dir / args.lab_results_csv).resolve()
    lab_config_path = os.path.relpath(str(snapshot_dir), start=str(base_dir.parent))
    _update_lab_results(
        lab_results_csv,
        args.mode,
        args.label,
        args.logfile,
        seed_range,
        seed_count,
        mean,
        stddev,
        lab_config_hash,
        lab_config_path,
        args.notes,
    )

    print(f"Snapshot: {lab_config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
