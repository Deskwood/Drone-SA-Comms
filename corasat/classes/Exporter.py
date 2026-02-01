"""Results export, optional notebook export, and shared logging utilities."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Dict, List, Optional
import uuid

RESULTS_FIELDS = [
    "timestamp",
    "model",
    "seed",
    "norm_score",
    "rounds",
    "num_drones",
    "total_actions",
    "broadcasts",
    "broadcast_rate",
    "wait_actions",
    "wait_rate",
    "broadcast_effectiveness",
    "broadcast_effectiveness_per_broadcast",
    "coverage",
    "correct_edges",
    "correct_edge_rate",
    "false_edges",
    "false_edge_rate",
    "total_gt_edges",
    "rendezvous_success",
    "mission_score",
    "runtime_s",
    "avg_turn_duration_s",
    "timeout_count",
    "timeout_rate",
    "logfile",
    "run_id",
    "commit_sha",
    "config_hash",
]


def _resolve_results_path() -> Path:
    """Resolve the location for results.csv."""
    try:
        base = Path(__file__).resolve().parent.parent
    except NameError:
        base = Path.cwd()
    candidate = base / "results.csv"
    if candidate.parent.exists():
        return candidate
    return Path.cwd() / "results.csv"


RESULTS_PATH = _resolve_results_path()


def _resolve_log_dir(log_dir: str) -> Path:
    """Resolve the log directory relative to the Corasat root when needed."""
    path = Path(log_dir)
    if path.is_absolute():
        return path
    try:
        base = Path(__file__).resolve().parent.parent
    except NameError:
        base = Path.cwd()
    return base / path


def _next_run_log_path(log_dir: Path, date_tag: str) -> Path:
    """Return a new log file path for the date using the next run number."""
    base_name = f"simulation_{date_tag}"
    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)\.log$")
    run_numbers: List[int] = []
    try:
        for existing in log_dir.glob(f"{base_name}_*.log"):
            match = pattern.match(existing.name)
            if match:
                run_numbers.append(int(match.group(1)))
    except Exception:
        run_numbers = []
    next_run = max(run_numbers, default=0) + 1
    candidate = log_dir / f"{base_name}_{next_run:02d}.log"
    while candidate.exists():
        next_run += 1
        candidate = log_dir / f"{base_name}_{next_run:02d}.log"
    return candidate


def _format_logfile_entry(logfile: Optional[Path]) -> Optional[str]:
    """Return logfile path relative to the Code directory when possible."""
    if logfile is None:
        return None
    try:
        log_path = Path(logfile)
        base = RESULTS_PATH.parent
        rel_root = base.parent if base.parent.exists() else base
        return os.path.relpath(str(log_path), start=str(rel_root))
    except Exception:
        return str(logfile)


def _safe_commit_sha() -> Optional[str]:
    """Return the current git commit SHA, or None if unavailable."""
    candidates: List[Path] = []
    try:
        candidates.append(Path.cwd())
    except Exception:
        pass
    try:
        exporter_root = Path(__file__).resolve().parent
        candidates.extend([exporter_root, exporter_root.parent, exporter_root.parent.parent])
    except Exception:
        pass

    seen = set()
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(resolved),
                stderr=subprocess.DEVNULL,
            ).decode("utf-8").strip()
        except Exception:
            continue
    return None


def _config_hash_from_dict(cfg: Dict[str, Any]) -> Optional[str]:
    """Hash the config dictionary for run tracking."""
    try:
        cfg_txt = json.dumps(cfg, sort_keys=True)
        return hashlib.sha1(cfg_txt.encode("utf-8")).hexdigest()[:10]
    except Exception:
        return None


def _compute_coverage_ratio(sim: Any) -> Optional[float]:
    """Compute coverage ratio as visited unique tiles / total tiles."""
    try:
        total = sim.grid_size[0] * sim.grid_size[1]
        if not total:
            return None
        visited = set()
        for drone in getattr(sim, "drones", []):
            for pos in getattr(drone, "mission_report", []):
                if isinstance(pos, (list, tuple)) and len(pos) == 2:
                    visited.add(tuple(pos))
        return round(len(visited) / total, 4)
    except Exception:
        return None


def _build_run_id(seed: Optional[Any]) -> str:
    """Create a unique run identifier including timestamp and seed."""
    base = datetime.now().strftime("%Y%m%d-%H%M%S")
    seed_part = str(seed) if seed is not None else "noseed"
    return f"{base}-{seed_part}-{uuid.uuid4().hex[:6]}"


def _append_results_rows(rows: List[Dict[str, Any]]) -> None:
    """Append rows to results.csv, creating the file if needed."""
    if not rows:
        return
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=RESULTS_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def persist_run_results(run_exports: List[Dict[str, Any]]) -> None:
    """Persist per-seed run metrics to results.csv."""
    if not run_exports:
        return
    commit_sha = _safe_commit_sha()
    logfile = getattr(LOGGER, "log_path", None)
    logfile_entry = _format_logfile_entry(logfile)
    rows: List[Dict[str, Any]] = []
    for entry in run_exports:
        sim = entry.get("sim")
        config = entry.get("config")
        seed = entry.get("seed")
        runtime_s = entry.get("runtime_s")
        omit_scores = bool(entry.get("omit_scores"))
        timestamp = entry.get("timestamp") or datetime.now().isoformat()
        coverage = _compute_coverage_ratio(sim) if sim else None
        num_drones = None
        if sim is not None:
            num_drones = getattr(sim, "num_drones", None)
        if num_drones is None and isinstance(config, dict):
            num_drones = config.get("simulation", {}).get("num_drones")
        rounds_value = getattr(sim, "round", None) if sim else None
        max_rounds = getattr(sim, "max_rounds", None) if sim else None
        total_actions = getattr(sim, "total_actions", None) if sim else None
        if total_actions is None and isinstance(num_drones, (int, float)) and isinstance(max_rounds, (int, float)):
            total_actions = int(num_drones) * int(max_rounds)
        wait_actions = getattr(sim, "wait_actions", None) if sim else None
        broadcasts = getattr(sim, "broadcast_count", None) if sim else None
        wait_rate = None
        broadcast_rate = None
        timeout_rate = None
        if isinstance(total_actions, (int, float)) and total_actions > 0:
            if isinstance(wait_actions, (int, float)):
                wait_rate = round(wait_actions / total_actions, 5)
            timeout_count = getattr(sim, "timeout_count", None) if sim else None
            if isinstance(timeout_count, (int, float)):
                timeout_rate = round(timeout_count / total_actions, 5)
        if isinstance(num_drones, (int, float)) and num_drones > 0:
            if isinstance(broadcasts, (int, float)):
                broadcast_rate = round(broadcasts / num_drones, 5)
        avg_turn_duration_s = None
        runtime_value = runtime_s if isinstance(runtime_s, (int, float)) else None
        if runtime_value is None:
            runtime_value = getattr(sim, "runtime_s", None) if sim else None
        if isinstance(runtime_value, (int, float)) and isinstance(total_actions, (int, float)) and total_actions > 0:
            avg_turn_duration_s = round(runtime_value / total_actions, 5)
        score_value = None
        if not omit_scores and sim is not None:
            raw_score = getattr(sim, "score", None)
            if isinstance(raw_score, (int, float)):
                score_value = raw_score
        norm_score_value = None
        if score_value is not None and sim is not None:
            gt_edges = getattr(sim, "gt_edges", None)
            if gt_edges:
                norm_score_value = round(score_value / len(gt_edges), 5)
        broadcast_effectiveness = getattr(sim, "broadcast_effectiveness", None) if sim else None
        broadcast_effectiveness_per_broadcast = None
        if (
            broadcast_effectiveness is not None
            and isinstance(broadcasts, (int, float))
            and broadcasts > 0
        ):
            broadcast_effectiveness_per_broadcast = round(broadcast_effectiveness / broadcasts, 5)
        correct_edges = getattr(sim, "correct_edge_counter", None) if sim else None
        false_edges = getattr(sim, "false_edge_counter", None) if sim else None
        total_gt_edges = len(getattr(sim, "gt_edges", []) or []) if sim else 0
        correct_edge_rate = None
        false_edge_rate = None
        if total_gt_edges > 0:
            if isinstance(correct_edges, (int, float)):
                correct_edge_rate = round(correct_edges / total_gt_edges, 5)
            if isinstance(false_edges, (int, float)):
                false_edge_rate = round(false_edges / total_gt_edges, 5)
        row = {
            "run_id": _build_run_id(seed),
            "timestamp": timestamp,
            "commit_sha": commit_sha,
            "config_hash": _config_hash_from_dict(config) if config else None,
            "model": getattr(sim, "model", None) if sim else None,
            "seed": seed,
            "rounds": rounds_value,
            "num_drones": num_drones,
            "total_actions": total_actions,
            "coverage": coverage,
            "correct_edges": correct_edges,
            "correct_edge_rate": correct_edge_rate,
            "false_edges": false_edges,
            "false_edge_rate": false_edge_rate,
            "total_gt_edges": total_gt_edges,
            "rendezvous_success": getattr(sim, "rendezvous_success", None) if sim else None,
            "broadcasts": broadcasts,
            "broadcast_rate": broadcast_rate,
            "wait_actions": wait_actions,
            "wait_rate": wait_rate,
            "broadcast_effectiveness": broadcast_effectiveness,
            "broadcast_effectiveness_per_broadcast": broadcast_effectiveness_per_broadcast,
            "mission_score": score_value,
            "norm_score": norm_score_value,
            "runtime_s": round(runtime_s, 2) if isinstance(runtime_s, (int, float)) else None,
            "avg_turn_duration_s": avg_turn_duration_s,
            "timeout_count": getattr(sim, "timeout_count", None) if sim else None,
            "timeout_rate": timeout_rate,
            "logfile": logfile_entry,
        }
        rows.append(row)
    _append_results_rows(rows)
    LOGGER.log(f"results.csv updated with {len(rows)} run(s).")


class TimestampedLogger:
    """Log to file and stdout with timestamps and inter-log durations."""

    def __init__(self, log_dir: str = "logs", log_file: str = "simulation.log"):
        date_tag = datetime.now().strftime("%Y-%m-%d")
        log_dir_path = _resolve_log_dir(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        self.log_path = _next_run_log_path(log_dir_path, date_tag)

        root = logging.getLogger()
        for handler in list(root.handlers):
            try:
                handler.close()
            except Exception:
                pass
            root.removeHandler(handler)

        file_handler = logging.FileHandler(self.log_path, mode="a", encoding="utf-8", delay=False)
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter(fmt="%(levelname)s:%(name)s:%(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        root.setLevel(logging.INFO)
        root.addHandler(file_handler)
        root.addHandler(console_handler)

        logging.getLogger("httpx").setLevel(logging.INFO)

        self.start_time = time.time()
        self.last_time = self.start_time
        self.log("Logger initialized.")

    def _now(self) -> str:
        """Return a timestamp string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _duration(self) -> str:
        """Return elapsed time since last log and update the timer."""
        current_time = time.time()
        delta = current_time - self.last_time
        self.last_time = current_time
        return f"{delta:.3f}s"

    def log(self, message: str) -> None:
        """Log a message with timestamp and delta."""
        logging.info(f"[{self._now()}] (+{self._duration()}) {message}")

    def warn(self, message: str) -> None:
        """Log a warning message with timestamp and delta."""
        logging.warning(f"[{self._now()}] (+{self._duration()}) {message}")


LOGGER = TimestampedLogger()


def _resolve_config_path(config_path: str) -> Optional[Path]:
    """Resolve config.json location relative to CWD or repository."""
    path = Path(config_path)
    if path.is_absolute():
        return path if path.exists() else None
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    try:
        repo_path = Path(__file__).resolve().parent.parent / path
    except NameError:
        return None
    return repo_path if repo_path.exists() else None


def _load_config_for_export(config_path: str) -> Dict[str, Any]:
    """Load config.json for notebook export decisions."""
    resolved = _resolve_config_path(config_path)
    if resolved is None:
        return {}
    try:
        with open(resolved, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except Exception:
        return {}


def _running_in_notebook() -> bool:
    """Return True when running inside a Jupyter kernel."""
    try:
        from IPython.core.getipython import get_ipython
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _find_notebook_path() -> Optional[str]:
    """Locate the active notebook path or fall back to latest ipynb."""
    try:
        import ipynbname
        return str(ipynbname.path())
    except Exception:
        pass
    try:
        candidates = [nb_path for nb_path in os.listdir(".") if nb_path.endswith(".ipynb")]
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]
    except Exception:
        pass
    return None


def _export_notebook_to_py(nb_path: str, out_py: str = "run_simulation.py") -> None:
    """Export a notebook file to a Python script."""
    try:
        import nbformat
        from nbconvert.exporters import PythonExporter
    except Exception:
        LOGGER.log("Notebook export skipped (nbformat/nbconvert not available).")
        return
    try:
        nb = nbformat.read(nb_path, as_version=4)
        body, _ = PythonExporter().from_notebook_node(nb)
        with open(out_py, "w", encoding="utf-8") as file_handle:
            file_handle.write(body)
        LOGGER.log(f"Exported notebook '{nb_path}' -> '{out_py}'")
    except Exception as exc:
        LOGGER.log(f"ERROR: Notebook export failed: {exc}")


def _maybe_export_notebook(config_path: str = "config.json") -> None:
    """Run notebook export if config and environment allow it."""
    cfg = _load_config_for_export(config_path)
    if not cfg:
        return
    if not cfg.get("simulation", {}).get("create_py_export", True):
        return
    if not _running_in_notebook():
        return
    nb_path = _find_notebook_path()
    if nb_path:
        _export_notebook_to_py(nb_path, out_py="run_simulation.py")


try:
    _maybe_export_notebook()
except Exception:
    pass
