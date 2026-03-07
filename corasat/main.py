"""Entry point for Corasat campaign execution.

This module supports two campaign-controlled modes from one config file:
1) smoke run mode: run the configured campaign with only the first seeds, and
2) campaign run mode: run the full configured campaign.

Mode activation is controlled by ``campaign.smoke_run`` and ``campaign.campaign_run``.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

from classes.Simulation import Simulation

import classes.Core as core

CORASAT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = CORASAT_ROOT.parent

RUNTIME_ROOT_KEYS = (
    "board",
    "rules_path",
    "simulation",
    "gui",
    "logging",
    "decision_support",
    "prompt_requests",
)

PROFILE_SPECS: Dict[str, Dict[str, str]] = {
    "rules": {"subdir": "rules", "suffix": "_rules.txt", "format": "text"},
    "prompt": {"subdir": "prompt_requests", "suffix": "_prompt_requests.json", "format": "json"},
    "model": {"subdir": "model", "suffix": "_model.json", "format": "json"},
    "fine_tuning": {"subdir": "fine_tuning", "suffix": "_fine_tuning.json", "format": "json"},
    "action_policy": {"subdir": "action_policy", "suffix": "_action_policy.json", "format": "json"},
    "communication": {"subdir": "communication", "suffix": "_communication.json", "format": "json"},
    "decision_support": {
        "subdir": "decision_support",
        "suffix": "_decision_support.json",
        "format": "json",
    },
}

ACTION_POLICY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "A0": {
        "use_language_model": False,
        "use_decision_support": False,
        "runtime_model_override": "policy_random",
    },
    "A1": {
        "use_language_model": False,
        "use_decision_support": True,
        "runtime_model_override": "policy_decision_support",
    },
    "A2": {
        "use_language_model": True,
        "use_decision_support": False,
        "runtime_model_override": "",
    },
    "A3": {
        "use_language_model": True,
        "use_decision_support": True,
        "runtime_model_override": "",
    },
}

LAB_RESULTS_FIELDS = [
    "timestamp",
    "lab_id",
    "label",
    "seed_count_planned",
    "seed_count_recorded",
    "seed_count_completed",
    "seed_count_non_ok",
    "seed_count_failed",
    "seed_count_aborted",
    "seed_range",
    "mean_norm_score",
    "std_norm_score",
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_lm_tokens",
    "total_lm_inference_time_s",
    "total_lm_parse_failures",
    "total_lm_request_failures",
    "total_lm_request_timeouts",
    "total_runtime_s",
    "mean_runtime_s",
    "runtime_config",
    "lab_log",
    "lab_simulation_log",
    "notes",
]

ERROR_SUMMARY_FIELDS = [
    "timestamp",
    "scope",
    "lab_id",
    "label",
    "seed",
    "status",
    "reason",
    "runtime_config",
    "seed_report_path",
    "logfile",
    "lm_trace_log",
]

REPRO_COMPARE_FIELDS = [
    "mission_score",
    "norm_score",
    "correct_edges",
    "false_edges",
    "rendezvous_success",
    "prompt_tokens_total",
    "completion_tokens_total",
    "lm_total_tokens",
]

PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z0-9_]+)(?::([^}]+))?\}")

LM_CONVERSATION_OVERVIEW_FIELDS = [
    "timestamp",
    "campaign",
    "lab_id",
    "label",
    "seed",
    "status",
    "seed_outcome",
    "conversation_log",
    "simulation_log",
    "runtime_config",
    "rounds",
    "mission_score",
    "norm_score",
    "prompt_tokens_total",
    "completion_tokens_total",
    "lm_total_tokens",
    "lm_inference_time_s",
    "lm_parse_failures",
    "lm_request_failures",
    "lm_request_timeouts",
    "lm_last_error",
    "abort_reason",
    "error",
]


def _init_logger():
    """Return the shared TimestampedLogger if it is available."""
    try:
        from classes.Exporter import LOGGER as exporter_logger
    except Exception:
        return None
    return exporter_logger


_LOGGER = _init_logger()


def _log(message: str) -> None:
    """Log a message using the shared logger, falling back to print."""
    if _LOGGER is not None:
        try:
            _LOGGER.log(message)
            return
        except Exception:
            pass
    print(message)


def _set_shared_logger_timestamp_mode(include_timestamps: bool) -> None:
    if _LOGGER is None:
        return
    try:
        setter = getattr(_LOGGER, "set_include_timestamps", None)
        if callable(setter):
            setter(bool(include_timestamps))
    except Exception:
        pass


def _switch_shared_log_file(log_dir: Path, log_file: str, include_timestamps: bool) -> Optional[Path]:
    if _LOGGER is None:
        return None
    _set_shared_logger_timestamp_mode(include_timestamps)
    try:
        start_new_log = getattr(_LOGGER, "start_new_log", None)
        if callable(start_new_log):
            new_path = start_new_log(log_dir=str(log_dir), log_file=log_file)
            return Path(str(new_path))
    except Exception:
        return None
    return None


def _current_logfile_path() -> str:
    if _LOGGER is None:
        return ""
    try:
        path = getattr(_LOGGER, "log_path", None)
        if path:
            return str(path)
    except Exception:
        return ""
    return ""


def _reload_config(config_path: str = core.CONFIG_PATH) -> Dict[str, Any]:
    """Reload the shared Core CONFIG from disk."""
    _log(f"Load Config: {config_path}")
    return core.reload_config(config_path)


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _upsert_csv_row(
    path: Path,
    *,
    fieldnames: Sequence[str],
    row: Dict[str, Any],
    key_fields: Sequence[str],
) -> None:
    existing = _read_csv_rows(path)
    key = tuple(str(row.get(field, "")) for field in key_fields)
    filtered: List[Dict[str, Any]] = []
    for item in existing:
        item_key = tuple(str(item.get(field, "")) for field in key_fields)
        if item_key != key:
            filtered.append(item)
    filtered.append({field: row.get(field, "") for field in fieldnames})
    _write_csv(path, fieldnames, filtered)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _append_text_log(path: Path, message: str, include_timestamps: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if include_timestamps:
        prefix = datetime.now().astimezone().isoformat()
        line = f"[{prefix}] {message}"
    else:
        line = message
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _sanitize_filename_token(value: str, max_len: int = 48) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "-", str(value or "").strip())
    token = token.strip("-")
    if not token:
        token = "na"
    return token[:max(8, int(max_len))]


def _lab_config_tag(ids: Dict[str, str]) -> str:
    ordered_keys = [
        "rules_id",
        "prompt_id",
        "drone_support_id",
        "model_id",
        "fine_tuning_id",
        "action_policy_id",
        "communication_id",
    ]
    parts: List[str] = []
    for key in ordered_keys:
        value = str(ids.get(key) or "").strip().upper()
        if value:
            parts.append(value)
    return "-".join(parts) if parts else "CFG"


def _build_lab_log_stem(lab_id: str, ids: Dict[str, str], idea: str) -> str:
    lab_token = _sanitize_filename_token(lab_id, max_len=16)
    config_token = _sanitize_filename_token(_lab_config_tag(ids), max_len=48)
    idea_token = _sanitize_filename_token(idea, max_len=40)
    return f"{lab_token}__{config_token}__{idea_token}"


def _seed_outcome_tag(run_entry: Dict[str, Any]) -> str:
    status = str(run_entry.get("status") or "ok").strip().lower()
    if status != "ok":
        return "FAIL"
    sim = run_entry.get("sim")
    if sim is None:
        return "FAIL"
    parse_failures = int(getattr(sim, "lm_parse_failures", 0) or 0)
    request_failures = int(getattr(sim, "lm_request_failures", 0) or 0)
    request_timeouts = int(getattr(sim, "lm_request_timeouts", 0) or 0)
    if parse_failures > 0 or request_failures > 0 or request_timeouts > 0:
        return "FAIL"
    return "PASS"


def _rename_with_outcome_tag(path: Path, outcome_tag: str) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    if stem.endswith("__PASS") or stem.endswith("__FAIL"):
        stem = stem.rsplit("__", 1)[0]
    target = path.with_name(f"{stem}__{outcome_tag}{path.suffix}")
    if target.exists() and target != path:
        target.unlink()
    if target != path:
        try:
            path.replace(target)
        except PermissionError:
            try:
                shutil.copy2(path, target)
            except Exception:
                return path
    return target


def _finalize_seed_artifacts(
    run_entry: Dict[str, Any],
    *,
    lm_trace_tmp_path: Optional[Path],
) -> Dict[str, Any]:
    outcome_tag = _seed_outcome_tag(run_entry)
    run_entry["seed_outcome"] = outcome_tag

    logfile_raw = str(run_entry.get("logfile") or "").strip()
    if logfile_raw:
        logfile_path = Path(logfile_raw)
        if logfile_path.exists():
            renamed_log = _rename_with_outcome_tag(logfile_path, outcome_tag)
            run_entry["logfile"] = str(renamed_log)

    if lm_trace_tmp_path is None:
        run_entry["lm_trace_log"] = ""
        return run_entry

    final_name = lm_trace_tmp_path.name
    suffix = ".inprogress.jsonl"
    if final_name.endswith(suffix):
        final_name = final_name[: -len(suffix)] + f"__{outcome_tag}.jsonl"
    else:
        stem = lm_trace_tmp_path.stem
        final_name = f"{stem}__{outcome_tag}{lm_trace_tmp_path.suffix}"
    final_path = lm_trace_tmp_path.with_name(final_name)
    if final_path.exists() and final_path != lm_trace_tmp_path:
        final_path.unlink()
    if lm_trace_tmp_path.exists() and final_path != lm_trace_tmp_path:
        lm_trace_tmp_path.replace(final_path)
    elif lm_trace_tmp_path.exists():
        final_path = lm_trace_tmp_path

    run_entry["lm_trace_log"] = str(final_path)
    return run_entry


def _resolve_config_path(config_path: str) -> Path:
    candidate = Path(config_path)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    return (CORASAT_ROOT / candidate).resolve()


def _resolve_path(base_dir: Path, raw_value: str, fallback: str) -> Path:
    value = (raw_value or "").strip()
    if not value:
        value = fallback
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _remove_tree(path: Path) -> None:
    """Delete a directory tree, retrying once after clearing read-only flags."""
    if not path.exists():
        return

    def _onerror(func, target_path, _exc_info):
        try:
            os.chmod(target_path, stat.S_IWRITE)
            func(target_path)
        except Exception:
            raise

    shutil.rmtree(path, onerror=_onerror)


def _deep_merge(base: object, override: object) -> object:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = {key: copy.deepcopy(value) for key, value in base.items()}
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


def _runtime_config_base(master_config: Dict[str, Any]) -> Dict[str, Any]:
    runtime: Dict[str, Any] = {}
    for key in RUNTIME_ROOT_KEYS:
        if key in master_config:
            runtime[key] = copy.deepcopy(master_config[key])
    if runtime:
        return runtime
    return copy.deepcopy(master_config)


def _strip_profile_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in payload.items()
        if key not in {"id", "name", "description", "sha256_10"}
    }


def _profile_path(config_dir: Path, kind: str, profile_id: str) -> Path:
    spec = PROFILE_SPECS[kind]
    return config_dir / "profiles" / spec["subdir"] / f"{profile_id}{spec['suffix']}"


def _load_profile_json(config_dir: Path, kind: str, profile_id: str) -> Dict[str, Any]:
    path = _profile_path(config_dir, kind, profile_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing {kind} profile: {path}")
    payload = _read_json(path)
    return payload


def _path_for_runtime(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(CORASAT_ROOT.resolve())
        return rel.as_posix()
    except Exception:
        return str(path.resolve())


def _select_runtime_model(sim_cfg: Dict[str, Any], model_name: str) -> None:
    model_name = str(model_name or "").strip()
    if not model_name:
        return
    models = sim_cfg.get("models", [])
    if not isinstance(models, list):
        models = []
    normalized = [str(item) for item in models]
    if model_name not in normalized:
        normalized.append(model_name)
    sim_cfg["models"] = normalized
    sim_cfg["model_index"] = normalized.index(model_name)


def _resolve_seed_spec(seed_spec: Any) -> List[int]:
    """Resolve a seed specification into an explicit, ordered integer list."""
    if seed_spec is None:
        return []

    if isinstance(seed_spec, list):
        seeds: List[int] = []
        for value in seed_spec:
            try:
                seeds.append(int(value))
            except Exception:
                continue
        return seeds

    if isinstance(seed_spec, (int, float)):
        return [int(seed_spec)]

    if isinstance(seed_spec, str):
        return _parse_seed_spec_text(seed_spec)

    if isinstance(seed_spec, dict):
        explicit_values = (
            seed_spec.get("seed_values")
            if "seed_values" in seed_spec
            else seed_spec.get("seeds")
            if "seeds" in seed_spec
            else seed_spec.get("values")
        )
        if isinstance(explicit_values, list):
            seeds = []
            seen = set()
            for value in explicit_values:
                try:
                    seed = int(value)
                except Exception:
                    continue
                if seed in seen:
                    continue
                seen.add(seed)
                seeds.append(seed)
            if seeds:
                return seeds

        first = seed_spec.get("first_seed") if "first_seed" in seed_spec else seed_spec.get("first")
        last = seed_spec.get("last_seed") if "last_seed" in seed_spec else seed_spec.get("last")
        if first is None or last is None:
            return []
        try:
            start = int(first)
            end = int(last)
        except Exception:
            return []

        step = seed_spec.get("step", 1)
        try:
            step_i = int(step)
        except Exception:
            step_i = 1
        if step_i == 0:
            step_i = 1

        if start <= end and step_i < 0:
            step_i = abs(step_i)
        if start > end and step_i > 0:
            step_i = -step_i

        inclusive_end = end + (1 if step_i > 0 else -1)
        return list(range(start, inclusive_end, step_i))

    return []


def _parse_seed_spec_text(spec: str) -> List[int]:
    """Parse a compact seed spec like ``1-10,15,20-30``."""
    seeds: List[int] = []
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            if len(parts) != 2:
                continue
            try:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
            except Exception:
                continue
            step = 1 if start <= end else -1
            seeds.extend(list(range(start, end + step, step)))
            continue
        try:
            seeds.append(int(token))
        except Exception:
            continue

    ordered_unique: List[int] = []
    seen = set()
    for seed in seeds:
        if seed in seen:
            continue
        ordered_unique.append(seed)
        seen.add(seed)
    return ordered_unique


def _seed_object_from_values(seeds: Sequence[int]) -> Dict[str, Any]:
    if not seeds:
        return {"first_seed": 0, "last_seed": 0, "seed_values": []}
    ordered = list(seeds)
    payload: Dict[str, Any] = {
        "first_seed": int(ordered[0]),
        "last_seed": int(ordered[-1]),
    }
    if len(ordered) == 1:
        return payload
    step = ordered[1] - ordered[0]
    arithmetic = all((ordered[idx] - ordered[idx - 1]) == step for idx in range(2, len(ordered)))
    if arithmetic:
        if step not in (1, -1):
            payload["step"] = int(step)
        return payload
    payload["seed_values"] = [int(seed) for seed in ordered]
    return payload


def _seed_range_text(seeds: Sequence[int]) -> str:
    if not seeds:
        return ""
    ordered = sorted(int(seed) for seed in seeds)
    return f"{ordered[0]}-{ordered[-1]}"


def _set_global_seed(seed: Optional[int]) -> None:
    """Best-effort seeding helper that does not block runs during refactor."""
    if seed is None:
        return
    try:
        core.set_global_seed(seed)
    except Exception:
        return


def _safe_shutdown(sim: Simulation) -> None:
    """Attempt to shut down the simulation without propagating errors."""
    try:
        sim.shutdown()
    except Exception:
        pass


def _create_simulation(game_index: int, total_games: int, seed: Optional[int]):
    """Construct a Simulation instance, returning None on failure."""
    try:
        return Simulation(game_index=game_index, total_games=total_games, seed=seed)
    except Exception as exc:
        _log(f"Simulation init failed: {exc}")
        return None


def _persist_results(run_entry: Dict[str, Any]) -> None:
    """Persist run results using the Exporter module if available."""
    if not run_entry:
        return
    try:
        from classes.Exporter import persist_run_results
    except Exception:
        return
    try:
        persist_run_results([run_entry])
    except Exception as exc:
        _log(f"Failed to update results.csv: {exc}")


def _norm_score_from_sim(sim: Any) -> Optional[float]:
    if sim is None:
        return None
    gt_edges = getattr(sim, "gt_edges", None)
    if not gt_edges:
        return None
    score = getattr(sim, "score", None)
    if score is None:
        return None
    try:
        return float(score) / max(1, len(gt_edges))
    except Exception:
        return None


def _run_entry_to_seed_report(
    run_entry: Dict[str, Any],
    *,
    campaign_name: str,
    lab_id: str,
    runtime_config_path: Path,
) -> Dict[str, Any]:
    sim = run_entry.get("sim")
    seed = run_entry.get("seed")
    norm_score = _norm_score_from_sim(sim)
    status = str(run_entry.get("status") or "ok")
    report = {
        "campaign": campaign_name,
        "lab_id": lab_id,
        "seed": seed,
        "timestamp": run_entry.get("timestamp"),
        "status": status,
        "error": run_entry.get("error") or "",
        "abort_reason": run_entry.get("abort_reason") or "",
        "runtime_config": str(runtime_config_path),
        "logfile": run_entry.get("logfile") or "",
        "model": getattr(sim, "model", None) if sim else None,
        "runtime_s": run_entry.get("runtime_s"),
        "rounds": getattr(sim, "round", None) if sim else None,
        "mission_score": getattr(sim, "score", None) if sim else None,
        "norm_score": norm_score,
        "correct_edges": getattr(sim, "correct_edge_counter", None) if sim else None,
        "false_edges": getattr(sim, "false_edge_counter", None) if sim else None,
        "total_gt_edges": len(getattr(sim, "gt_edges", []) or []) if sim else None,
        "broadcasts": getattr(sim, "broadcast_count", None) if sim else None,
        "wait_actions": getattr(sim, "wait_actions", None) if sim else None,
        "total_actions": getattr(sim, "total_actions", None) if sim else None,
        "rendezvous_success": getattr(sim, "rendezvous_success", None) if sim else None,
        "timeout_count": getattr(sim, "timeout_count", None) if sim else None,
        "prompt_tokens_total": getattr(sim, "prompt_tokens_total", 0) if sim else 0,
        "completion_tokens_total": getattr(sim, "completion_tokens_total", 0) if sim else 0,
        "lm_total_tokens": getattr(sim, "lm_total_tokens", 0) if sim else 0,
        "lm_inference_time_s": getattr(sim, "lm_inference_time_s", 0.0) if sim else 0.0,
        "lm_parse_failures": getattr(sim, "lm_parse_failures", 0) if sim else 0,
        "lm_request_failures": getattr(sim, "lm_request_failures", 0) if sim else 0,
        "lm_request_timeouts": getattr(sim, "lm_request_timeouts", 0) if sim else 0,
        "lm_last_error": getattr(sim, "lm_last_error", "") if sim else "",
        "lm_error_events": list(getattr(sim, "lm_error_events", []) or []) if sim else [],
        "wait_reason_counts": dict(getattr(sim, "wait_reason_counts", {}) or {}) if sim else {},
        "wait_event_count": len(getattr(sim, "wait_events", []) or []) if sim else 0,
        "lm_trace_log": run_entry.get("lm_trace_log") or "",
        "seed_outcome": run_entry.get("seed_outcome") or "",
    }
    return report


def _lm_overview_row_from_seed(
    *,
    campaign_name: str,
    lab_id: str,
    label: str,
    runtime_config_path: Path,
    seed_payload: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "timestamp": seed_payload.get("timestamp", ""),
        "campaign": campaign_name,
        "lab_id": lab_id,
        "label": label,
        "seed": seed_payload.get("seed", ""),
        "status": seed_payload.get("status", ""),
        "seed_outcome": seed_payload.get("seed_outcome", ""),
        "conversation_log": seed_payload.get("lm_trace_log", ""),
        "simulation_log": seed_payload.get("logfile", ""),
        "runtime_config": str(runtime_config_path),
        "rounds": seed_payload.get("rounds", ""),
        "mission_score": seed_payload.get("mission_score", ""),
        "norm_score": seed_payload.get("norm_score", ""),
        "prompt_tokens_total": seed_payload.get("prompt_tokens_total", 0),
        "completion_tokens_total": seed_payload.get("completion_tokens_total", 0),
        "lm_total_tokens": seed_payload.get("lm_total_tokens", 0),
        "lm_inference_time_s": seed_payload.get("lm_inference_time_s", 0.0),
        "lm_parse_failures": seed_payload.get("lm_parse_failures", 0),
        "lm_request_failures": seed_payload.get("lm_request_failures", 0),
        "lm_request_timeouts": seed_payload.get("lm_request_timeouts", 0),
        "lm_last_error": seed_payload.get("lm_last_error", ""),
        "abort_reason": seed_payload.get("abort_reason", ""),
        "error": seed_payload.get("error", ""),
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def _stddev(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def _summarize_lab_seed_reports(seed_reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ok_reports = [item for item in seed_reports if str(item.get("status") or "ok") == "ok"]
    non_ok_reports = [item for item in seed_reports if str(item.get("status") or "ok") != "ok"]

    norm_scores = [float(item["norm_score"]) for item in ok_reports if isinstance(item.get("norm_score"), (int, float))]
    runtimes = [float(item["runtime_s"]) for item in ok_reports if isinstance(item.get("runtime_s"), (int, float))]

    prompt_tokens = sum(int(item.get("prompt_tokens_total") or 0) for item in ok_reports)
    completion_tokens = sum(int(item.get("completion_tokens_total") or 0) for item in ok_reports)
    lm_tokens = sum(int(item.get("lm_total_tokens") or 0) for item in ok_reports)
    lm_time = sum(float(item.get("lm_inference_time_s") or 0.0) for item in ok_reports)
    lm_parse_failures = sum(int(item.get("lm_parse_failures") or 0) for item in seed_reports)
    lm_request_failures = sum(int(item.get("lm_request_failures") or 0) for item in seed_reports)
    lm_request_timeouts = sum(int(item.get("lm_request_timeouts") or 0) for item in seed_reports)

    failed_statuses = {"init_error", "run_error"}
    aborted_statuses = {"watchdog_abort", "user_abort"}
    seed_count_failed = sum(1 for item in seed_reports if str(item.get("status") or "ok") in failed_statuses)
    seed_count_aborted = sum(1 for item in seed_reports if str(item.get("status") or "ok") in aborted_statuses)

    return {
        "seed_count_recorded": len(seed_reports),
        "seed_count_completed": len(ok_reports),
        "seed_count_non_ok": len(non_ok_reports),
        "seed_count_failed": seed_count_failed,
        "seed_count_aborted": seed_count_aborted,
        "mean_norm_score": round(_mean(norm_scores), 6),
        "std_norm_score": round(_stddev(norm_scores), 6),
        "total_runtime_s": round(sum(runtimes), 5),
        "mean_runtime_s": round(_mean(runtimes), 5),
        "total_prompt_tokens": int(prompt_tokens),
        "total_completion_tokens": int(completion_tokens),
        "total_lm_tokens": int(lm_tokens),
        "total_lm_inference_time_s": round(lm_time, 6),
        "total_lm_parse_failures": int(lm_parse_failures),
        "total_lm_request_failures": int(lm_request_failures),
        "total_lm_request_timeouts": int(lm_request_timeouts),
    }


def _summarize_error_entries(error_entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_status: Dict[str, int] = {}
    by_lab: Dict[str, int] = {}
    for entry in error_entries:
        status = str(entry.get("status") or "unknown").strip() or "unknown"
        lab_id = str(entry.get("lab_id") or "unknown").strip() or "unknown"
        by_status[status] = by_status.get(status, 0) + 1
        by_lab[lab_id] = by_lab.get(lab_id, 0) + 1

    return {
        "entry_count": len(error_entries),
        "by_status": by_status,
        "by_lab": by_lab,
    }


def run_seed(
    seed: Optional[int],
    game_index: int,
    total_games: int,
    *,
    config_path: str,
    seed_log_dir: Optional[Path] = None,
    seed_log_file: str = "",
    lm_trace_dir: Optional[Path] = None,
    include_timestamps: bool = True,
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Run a single seed and return (run_entry, abort_requested)."""
    resolved_log_name = seed_log_file or f"seed_{seed}.log"
    if seed_log_dir is not None:
        seed_log_dir.mkdir(parents=True, exist_ok=True)
        _switch_shared_log_file(seed_log_dir, resolved_log_name, include_timestamps)

    if context:
        os.environ["CORASAT_TRACE_CAMPAIGN"] = str(context.get("campaign", ""))
        os.environ["CORASAT_TRACE_LAB_ID"] = str(context.get("lab_id", ""))
    else:
        os.environ["CORASAT_TRACE_CAMPAIGN"] = ""
        os.environ["CORASAT_TRACE_LAB_ID"] = ""
    os.environ["CORASAT_TRACE_SEED"] = "" if seed is None else str(seed)

    lm_trace_tmp_path: Optional[Path] = None
    if lm_trace_dir is not None:
        lm_trace_dir.mkdir(parents=True, exist_ok=True)
        trace_stem = Path(resolved_log_name).stem
        lm_trace_tmp_path = lm_trace_dir / f"{trace_stem}.lmtrace.inprogress.jsonl"
        if lm_trace_tmp_path.exists():
            lm_trace_tmp_path.unlink()
        lm_trace_tmp_path.touch()
        os.environ["CORASAT_LM_TRACE_FILE"] = str(lm_trace_tmp_path)
    else:
        os.environ.pop("CORASAT_LM_TRACE_FILE", None)

    def _finalize(run_entry: Dict[str, Any], abort_requested: bool) -> Tuple[Optional[Dict[str, Any]], bool]:
        fallback_log_dir_raw = os.environ.get("CORASAT_LOG_DIR", "").strip()
        fallback_log_dir = Path(fallback_log_dir_raw) if fallback_log_dir_raw else (seed_log_dir or CORASAT_ROOT / "logs")
        _switch_shared_log_file(
            fallback_log_dir,
            "_campaign_runtime.log",
            include_timestamps,
        )
        finalized = _finalize_seed_artifacts(
            run_entry,
            lm_trace_tmp_path=lm_trace_tmp_path,
        )
        _persist_results(finalized)
        os.environ.pop("CORASAT_LM_TRACE_FILE", None)
        return finalized, abort_requested

    _log(f"==== Running seed {seed} (game {game_index}/{total_games}) ====")
    _set_global_seed(seed)
    config = _reload_config(config_path)

    sim = _create_simulation(game_index=game_index, total_games=total_games, seed=seed)
    if sim is None:
        run_entry: Dict[str, Any] = {
            "sim": None,
            "config": config,
            "seed": seed,
            "runtime_s": 0.0,
            "timestamp": datetime.now().isoformat(),
            "status": "init_error",
            "error": "Simulation init failed.",
            "logfile": _current_logfile_path(),
        }
        if context:
            run_entry["context"] = dict(context)
        return _finalize(run_entry, False)

    _log("Launching simulation.")
    run_started = time.time()
    run_success = False
    run_error: Optional[str] = None
    try:
        sim.run_simulation()
        run_success = True
    except KeyboardInterrupt:
        _log("Interrupted by user (Ctrl+C).")
        os.environ.pop("CORASAT_LM_TRACE_FILE", None)
        raise
    except Exception as exc:
        run_error = str(exc)
        _log(f"Simulation error: {exc}")
        _log(traceback.format_exc())
    finally:
        _safe_shutdown(sim)

    if not run_success:
        runtime_s = time.time() - run_started
        sim_runtime = getattr(sim, "runtime_s", None)
        if isinstance(sim_runtime, (int, float)) and sim_runtime > 0:
            runtime_s = sim_runtime
        run_entry = {
            "sim": sim,
            "config": config,
            "seed": seed,
            "runtime_s": runtime_s,
            "timestamp": datetime.now().isoformat(),
            "status": "run_error",
            "error": run_error or "Simulation run failed.",
            "omit_scores": True,
            "logfile": _current_logfile_path(),
        }
        if context:
            run_entry["context"] = dict(context)
        return _finalize(run_entry, False)

    if getattr(sim, "_watchdog_triggered", False):
        reason = getattr(sim, "_abort_reason", "") or "watchdog timeout"
        _log(f"Run aborted by watchdog: {reason}")
        runtime_s = time.time() - run_started
        sim_runtime = getattr(sim, "runtime_s", None)
        if isinstance(sim_runtime, (int, float)) and sim_runtime > 0:
            runtime_s = sim_runtime
        run_entry: Dict[str, Any] = {
            "sim": sim,
            "config": config,
            "seed": seed,
            "runtime_s": runtime_s,
            "timestamp": datetime.now().isoformat(),
            "status": "watchdog_abort",
            "abort_reason": reason,
            "omit_scores": True,
            "logfile": _current_logfile_path(),
        }
        if context:
            run_entry["context"] = dict(context)
        return _finalize(run_entry, False)

    if getattr(sim, "_abort_requested", False):
        reason = getattr(sim, "_abort_reason", "") or "GUI closed"
        _log(f"Run aborted: {reason} - stopping remaining seeds.")
        runtime_s = time.time() - run_started
        sim_runtime = getattr(sim, "runtime_s", None)
        if isinstance(sim_runtime, (int, float)) and sim_runtime > 0:
            runtime_s = sim_runtime
        run_entry = {
            "sim": sim,
            "config": config,
            "seed": seed,
            "runtime_s": runtime_s,
            "timestamp": datetime.now().isoformat(),
            "status": "user_abort",
            "abort_reason": reason,
            "omit_scores": True,
            "logfile": _current_logfile_path(),
        }
        if context:
            run_entry["context"] = dict(context)
        return _finalize(run_entry, True)

    runtime_s = time.time() - run_started
    sim_runtime = getattr(sim, "runtime_s", None)
    if isinstance(sim_runtime, (int, float)) and sim_runtime > 0:
        runtime_s = sim_runtime
    run_entry = {
        "sim": sim,
        "config": config,
        "seed": seed,
        "runtime_s": runtime_s,
        "timestamp": datetime.now().isoformat(),
        "status": "ok",
        "logfile": _current_logfile_path(),
    }
    if context:
        run_entry["context"] = dict(context)
    return _finalize(run_entry, False)


def _build_lab_runtime_config(
    *,
    master_config: Dict[str, Any],
    config_dir: Path,
    lab_entry: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    runtime_cfg = _runtime_config_base(master_config)
    sim_cfg = runtime_cfg.setdefault("simulation", {})
    ds_cfg = runtime_cfg.setdefault("decision_support", {})

    lab_id = str(lab_entry.get("id") or lab_entry.get("lab_id") or "").strip()
    rules_id = str(lab_entry.get("rules_id") or "").strip().upper()
    prompt_id = str(lab_entry.get("prompt_id") or "").strip().upper()
    ds_id = str(lab_entry.get("drone_support_id") or lab_entry.get("decision_support_id") or "").strip().upper()
    model_id = str(lab_entry.get("model_id") or "").strip().upper()
    fine_tuning_id = str(lab_entry.get("fine_tuning_id") or "").strip().upper()
    action_policy_id = str(lab_entry.get("action_policy_id") or "").strip().upper()
    communication_id = str(lab_entry.get("communication_id") or "").strip().upper()

    if rules_id:
        rules_path = _profile_path(config_dir, "rules", rules_id)
        if not rules_path.exists():
            raise FileNotFoundError(f"Missing rules profile for {lab_id}: {rules_path}")
        runtime_cfg["rules_path"] = _path_for_runtime(rules_path)

    if prompt_id:
        prompt_payload = _load_profile_json(config_dir, "prompt", prompt_id)
        runtime_cfg["prompt_requests"] = _strip_profile_metadata(prompt_payload)

    if ds_id:
        ds_payload = _load_profile_json(config_dir, "decision_support", ds_id)
        runtime_cfg["decision_support"] = _deep_merge(ds_cfg, _strip_profile_metadata(ds_payload))
        ds_cfg = runtime_cfg.setdefault("decision_support", {})

    if model_id:
        model_payload = _load_profile_json(config_dir, "model", model_id)
        selected_model = str(model_payload.get("selected_model") or model_payload.get("runtime_model") or "").strip()
        if selected_model:
            _select_runtime_model(sim_cfg, selected_model)

    if fine_tuning_id:
        ft_payload = _load_profile_json(config_dir, "fine_tuning", fine_tuning_id)
        sim_cfg["fine_tuning_enabled"] = bool(ft_payload.get("is_adapted", False))
        override_model = str(ft_payload.get("runtime_model_override") or "").strip()
        if override_model:
            _select_runtime_model(sim_cfg, override_model)

    action_defaults = dict(ACTION_POLICY_DEFAULTS.get(action_policy_id, {}))
    action_payload: Dict[str, Any] = {}
    if action_policy_id:
        try:
            action_payload = _load_profile_json(config_dir, "action_policy", action_policy_id)
        except FileNotFoundError:
            action_payload = {}

    use_lm = action_payload.get("use_language_model", action_defaults.get("use_language_model"))
    use_ds = action_payload.get("use_decision_support", action_defaults.get("use_decision_support"))
    action_model_override = action_payload.get("runtime_model_override", action_defaults.get("runtime_model_override"))

    if isinstance(use_lm, bool):
        sim_cfg["use_language_model"] = use_lm
    if isinstance(use_ds, bool):
        ds_cfg["enabled"] = use_ds
        if not use_ds:
            out_cfg = ds_cfg.setdefault("output", {})
            out_cfg["include_scoring"] = False
            out_cfg["include_summary"] = False
    action_override_model = str(action_model_override or "").strip()
    if action_override_model:
        _select_runtime_model(sim_cfg, action_override_model)

    if communication_id:
        comm_payload = _load_profile_json(config_dir, "communication", communication_id)
        comm_cfg = _strip_profile_metadata(comm_payload)
        mode = str(comm_cfg.get("mode") or "").strip()
        if mode:
            sim_cfg["communication_mode"] = mode
        sim_cfg["communication_id"] = communication_id
        if isinstance(sim_cfg.get("communication"), dict):
            merged_comm = _deep_merge(sim_cfg.get("communication", {}), comm_cfg)
            if isinstance(merged_comm, dict):
                sim_cfg["communication"] = merged_comm
        else:
            sim_cfg["communication"] = comm_cfg

    lab_override = lab_entry.get("runtime_override")
    if not isinstance(lab_override, dict):
        lab_override = lab_entry.get("config_override")
    if isinstance(lab_override, dict):
        merged = _deep_merge(runtime_cfg, lab_override)
        if isinstance(merged, dict):
            runtime_cfg = merged
            sim_cfg = runtime_cfg.setdefault("simulation", {})

    sim_cfg["lab_id"] = lab_id
    sim_cfg["rules_id"] = rules_id
    sim_cfg["prompt_id"] = prompt_id
    sim_cfg["drone_support_id"] = ds_id
    sim_cfg["model_id"] = model_id
    sim_cfg["fine_tuning_id"] = fine_tuning_id
    sim_cfg["action_policy_id"] = action_policy_id
    sim_cfg["communication_id"] = communication_id

    resolved = {
        "lab_id": lab_id,
        "rules_id": rules_id,
        "prompt_id": prompt_id,
        "drone_support_id": ds_id,
        "model_id": model_id,
        "fine_tuning_id": fine_tuning_id,
        "action_policy_id": action_policy_id,
        "communication_id": communication_id,
    }
    return runtime_cfg, resolved


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _build_optuna_args(optuna_cfg: Dict[str, Any]) -> List[str]:
    if isinstance(optuna_cfg.get("args"), list):
        return [str(token) for token in optuna_cfg.get("args", [])]

    args: List[str] = []
    mapping = {
        "trials": "--trials",
        "seeds": "--seeds",
        "study_name": "--study-name",
        "storage": "--storage",
        "sampler_seed": "--sampler-seed",
        "max_rounds": "--max-rounds",
        "seed_weights": "--seed-weights",
        "enqueue_config": "--enqueue-config",
        "output_dir": "--output-dir",
    }
    for key, flag in mapping.items():
        value = optuna_cfg.get(key, None)
        if value in (None, "", []):
            continue
        args.extend([flag, str(value)])
    if bool(optuna_cfg.get("no_prune", False)):
        args.append("--no-prune")
    if bool(optuna_cfg.get("no_apply_best", False)):
        args.append("--no-apply-best")
    if bool(optuna_cfg.get("tune_without_lm", False)):
        args.append("--tune-without-lm")
    if bool(optuna_cfg.get("verbose", False)):
        args.append("--verbose")
    return args


def _apply_smoke_optuna_overrides(
    optuna_cfg: Dict[str, Any],
    *,
    campaign_cfg: Dict[str, Any],
    lab_seeds: Sequence[int],
) -> Dict[str, Any]:
    """Return smoke-mode Optuna config with shortened defaults/overrides."""
    effective = copy.deepcopy(optuna_cfg)
    smoke_cfg = campaign_cfg.get("smoke_optuna", {})
    if not isinstance(smoke_cfg, dict):
        smoke_cfg = {}

    default_trials = smoke_cfg.get("trials", 3)
    try:
        trial_cap = max(1, int(default_trials))
    except Exception:
        trial_cap = 3

    current_trials = effective.get("trials")
    if current_trials is None:
        effective["trials"] = trial_cap
    else:
        try:
            effective["trials"] = max(1, min(int(current_trials), trial_cap))
        except Exception:
            effective["trials"] = trial_cap

    if smoke_cfg.get("seeds") not in (None, ""):
        effective["seeds"] = smoke_cfg.get("seeds")
    elif lab_seeds:
        effective["seeds"] = ",".join(str(int(seed)) for seed in list(lab_seeds)[:1])

    smoke_max_rounds = smoke_cfg.get("max_rounds")
    if smoke_max_rounds is not None:
        try:
            effective["max_rounds"] = max(1, int(smoke_max_rounds))
        except Exception:
            pass
    elif effective.get("max_rounds") is not None:
        try:
            effective["max_rounds"] = max(1, min(int(effective["max_rounds"]), 8))
        except Exception:
            pass

    if "no_prune" in smoke_cfg:
        effective["no_prune"] = bool(smoke_cfg.get("no_prune"))
    else:
        effective["no_prune"] = True

    extra_override = smoke_cfg.get("override")
    if isinstance(extra_override, dict):
        for key, value in extra_override.items():
            effective[key] = copy.deepcopy(value)

    return effective


def _select_lora_cfg_for_mode(
    lora_cfg: Dict[str, Any],
    *,
    campaign_cfg: Dict[str, Any],
    run_mode: str,
) -> Dict[str, Any]:
    if run_mode != "smoke":
        return lora_cfg
    smoke_cfg = campaign_cfg.get("smoke_lora", {})
    if not isinstance(smoke_cfg, dict):
        smoke_cfg = {}
    use_smoke_commands = bool(smoke_cfg.get("use_smoke_commands", True))
    if not use_smoke_commands:
        return lora_cfg

    smoke_commands = lora_cfg.get("smoke_commands")
    if not isinstance(smoke_commands, list):
        return lora_cfg

    effective = copy.deepcopy(lora_cfg)
    effective["commands"] = copy.deepcopy(smoke_commands)
    return effective


def _run_subprocess(argv: Sequence[str], *, cwd: Path, env: Dict[str, str], note: str = "") -> None:
    display = " ".join(argv)
    if note:
        _log(f"[run] {note}: {display}")
    else:
        _log(f"[run] {display}")
    subprocess.run(list(argv), cwd=str(cwd), env=env, check=True)


def _expand_command_token(token: str, context: Dict[str, str], env: Dict[str, str]) -> str:
    def _replace(match: re.Match[str]) -> str:
        kind = (match.group(1) or "").strip()
        arg = (match.group(2) or "").strip()
        if kind == "lab_logfile":
            if not arg:
                raise ValueError("Placeholder {lab_logfile:<key>} requires a key.")
            value = context.get(arg, "").strip()
            if not value:
                raise ValueError(
                    f"LoRA placeholder requested lab logfile '{arg}', but no logfile is registered yet."
                )
            return value
        if kind == "env":
            if not arg:
                raise ValueError("Placeholder {env:<VAR>} requires an environment variable name.")
            return str(env.get(arg, ""))
        if kind == "corasat_root":
            return str(CORASAT_ROOT)
        if kind == "code_root":
            return str(CODE_ROOT)
        raise ValueError(f"Unsupported placeholder {{{kind}{(':' + arg) if arg else ''}}}.")

    return PLACEHOLDER_RE.sub(_replace, token)


def _run_lora_commands(lora_cfg: Dict[str, Any], env: Dict[str, str], context: Dict[str, str]) -> None:
    commands = lora_cfg.get("commands", [])
    if not isinstance(commands, list):
        raise ValueError("lora.commands must be a list.")
    for index, command in enumerate(commands, start=1):
        if isinstance(command, str):
            argv = shlex.split(command)
        elif isinstance(command, list):
            argv = [str(token) for token in command]
        else:
            raise ValueError(f"Unsupported LoRA command at index {index}: {command!r}")
        if not argv:
            continue
        argv = [_expand_command_token(token, context=context, env=env) for token in argv]
        _run_subprocess(argv, cwd=CORASAT_ROOT, env=env, note=f"LoRA command {index}")


def _export_campaign_to_mt(
    *,
    master_config: Dict[str, Any],
    campaign_cfg: Dict[str, Any],
    config_path: Path,
    output_dir: Path,
    results_path: Path,
    lab_results_path: Path,
    manifest_path: Path,
) -> None:
    mt_cfg = campaign_cfg.get("mt_export", {})
    if not isinstance(mt_cfg, dict) or not bool(mt_cfg.get("enabled", False)):
        return
    try:
        from campaign_mt_export import export_campaign_artifacts
    except Exception as exc:
        _log(f"MT export unavailable: {exc}")
        return

    try:
        outputs = export_campaign_artifacts(
            corasat_root=CORASAT_ROOT,
            config_path=config_path,
            master_config=master_config,
            campaign_cfg=campaign_cfg,
            output_dir=output_dir,
            results_path=results_path,
            lab_results_path=lab_results_path,
            manifest_path=manifest_path,
            logger=_log,
        )
        if outputs:
            _log("MT export completed:")
            for key in sorted(outputs.keys()):
                _log(f"- {key}: {outputs[key]}")
    except Exception as exc:
        _log(f"MT export failed: {exc}")
        _log(traceback.format_exc())


def _concat_seed_logs(seed_reports: Sequence[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as target:
        for report in seed_reports:
            seed = report.get("seed")
            logfile = Path(str(report.get("logfile") or "").strip())
            if not logfile.exists():
                continue
            target.write(f"===== Seed {seed} :: {logfile.name} =====\n")
            target.write(logfile.read_text(encoding="utf-8", errors="ignore"))
            target.write("\n\n")


def _resolve_lab_id(lab_cfg: Dict[str, Any], index: int) -> str:
    """Resolve a stable lab id from config, with deterministic fallback."""
    resolved = str(lab_cfg.get("id") or lab_cfg.get("lab_id") or f"L{index - 1}").strip()
    if not resolved:
        return f"L{index - 1}"
    return resolved


def _validate_lab_profiles(labs_cfg: Sequence[Dict[str, Any]], config_dir: Path) -> None:
    """Fail fast when enabled labs reference profile files that do not exist."""
    profile_key_to_kind = {
        "rules_id": "rules",
        "prompt_id": "prompt",
        "model_id": "model",
        "fine_tuning_id": "fine_tuning",
        "action_policy_id": "action_policy",
        "communication_id": "communication",
        "drone_support_id": "decision_support",
        "decision_support_id": "decision_support",
    }

    for index, lab_cfg_raw in enumerate(labs_cfg, start=1):
        if not isinstance(lab_cfg_raw, dict):
            continue
        if not bool(lab_cfg_raw.get("enabled", True)):
            continue

        lab_id = _resolve_lab_id(lab_cfg_raw, index)
        for key, kind in profile_key_to_kind.items():
            raw_profile = lab_cfg_raw.get(key)
            if raw_profile in (None, ""):
                continue
            profile_id = str(raw_profile).strip().upper()
            if not profile_id:
                continue
            profile_path = _profile_path(config_dir, kind, profile_id)
            if not profile_path.exists():
                raise FileNotFoundError(
                    f"{lab_id}: missing profile '{profile_id}' for {key}: {profile_path}"
                )


def _lab_evaluate_runtime(lab_cfg: Dict[str, Any]) -> bool:
    return bool(lab_cfg.get("evaluate_runtime", True))


def _validate_lab_parent_diffs(labs_cfg: Sequence[Dict[str, Any]]) -> None:
    """Require one-variable parent-child runtime comparisons outside reference/prep labs."""
    enabled_labs: List[Dict[str, Any]] = []
    for index, lab_cfg_raw in enumerate(labs_cfg, start=1):
        if not isinstance(lab_cfg_raw, dict):
            continue
        if not bool(lab_cfg_raw.get("enabled", True)):
            continue
        lab_cfg = dict(lab_cfg_raw)
        lab_cfg["_resolved_lab_id"] = _resolve_lab_id(lab_cfg_raw, index)
        enabled_labs.append(lab_cfg)

    labs_by_id = {str(lab["_resolved_lab_id"]): lab for lab in enabled_labs}
    profile_fields = (
        "rules_id",
        "prompt_id",
        "drone_support_id",
        "model_id",
        "fine_tuning_id",
        "action_policy_id",
        "communication_id",
    )

    for lab_cfg in enabled_labs:
        lab_id = str(lab_cfg.get("_resolved_lab_id") or "").strip()
        parent_id = str(lab_cfg.get("parent_lab") or "").strip()
        if not lab_id or not parent_id:
            continue
        if str(lab_cfg.get("reference_lab") or "").strip():
            continue
        if not _lab_evaluate_runtime(lab_cfg):
            continue

        parent_cfg = labs_by_id.get(parent_id)
        if parent_cfg is None:
            raise ValueError(f"{lab_id}: parent_lab '{parent_id}' must reference an enabled lab defined earlier.")

        diff_items: List[str] = []
        short_names = {
            "rules_id": "R",
            "prompt_id": "P",
            "drone_support_id": "DS",
            "model_id": "M",
            "fine_tuning_id": "FT",
            "action_policy_id": "A",
            "communication_id": "C",
        }
        for field in profile_fields:
            left = str(lab_cfg.get(field) or "").strip().upper()
            right = str(parent_cfg.get(field) or "").strip().upper()
            if left != right:
                diff_items.append(short_names[field])

        child_optuna = lab_cfg.get("optuna", {})
        parent_optuna = parent_cfg.get("optuna", {})
        if not isinstance(child_optuna, dict):
            child_optuna = {}
        if not isinstance(parent_optuna, dict):
            parent_optuna = {}
        if json.dumps(child_optuna, sort_keys=True) != json.dumps(parent_optuna, sort_keys=True):
            diff_items.append("optuna")

        if len(diff_items) != 1:
            changed = ", ".join(diff_items) if diff_items else "none"
            raise ValueError(
                f"{lab_id}: runtime-evaluated non-reference labs must differ from parent_lab by exactly one "
                f"profile item or one Optuna stage. Found {changed} relative to {parent_id}."
            )


def _reproducibility_cfg(campaign_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = campaign_cfg.get("reproducibility", {})
    if not isinstance(cfg, dict):
        return {}
    return cfg


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coerce_positive_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        number = int(value)
    except Exception:
        return None
    if number <= 0:
        return None
    return number


def _resolve_sim_model_name(sim_cfg: Dict[str, Any]) -> str:
    model_name = ""
    models = sim_cfg.get("models") if isinstance(sim_cfg, dict) and "models" in sim_cfg else None
    index_raw = sim_cfg.get("model_index") if isinstance(sim_cfg, dict) and "model_index" in sim_cfg else None
    if isinstance(models, list) and index_raw is not None:
        try:
            idx = int(index_raw)
        except Exception:
            idx = None
        if idx is not None and 0 <= idx < len(models):
            model_name = str(models[idx] or "").strip()
    if not model_name and isinstance(sim_cfg, dict):
        if "selected_model" in sim_cfg:
            model_name = str(sim_cfg.get("selected_model") or "").strip()
        elif "runtime_model" in sim_cfg:
            model_name = str(sim_cfg.get("runtime_model") or "").strip()
    return model_name


def _apply_watchdog_timeout_overrides(
    *,
    runtime_cfg: Dict[str, Any],
    lab_cfg: Dict[str, Any],
    campaign_cfg: Dict[str, Any],
    run_mode: str,
) -> None:
    """Apply mode/lab/model watchdog overrides to runtime simulation config."""
    sim_cfg = runtime_cfg.setdefault("simulation", {})
    if not isinstance(sim_cfg, dict):
        return

    candidates: List[int] = []
    if "watchdog_timeout_s" in sim_cfg:
        current_timeout = _coerce_positive_int(sim_cfg.get("watchdog_timeout_s"))
        if current_timeout is not None:
            candidates.append(current_timeout)

    mode_key = "smoke_watchdog_timeout_s" if run_mode == "smoke" else "campaign_watchdog_timeout_s"
    mode_timeout = _coerce_positive_int(campaign_cfg.get(mode_key))
    if mode_timeout is not None:
        candidates.append(mode_timeout)

    lab_id = str(lab_cfg.get("id") or lab_cfg.get("lab_id") or "").strip()
    lab_timeout_map = campaign_cfg.get("watchdog_timeout_by_lab", {})
    if isinstance(lab_timeout_map, dict) and lab_id:
        lab_timeout = _coerce_positive_int(lab_timeout_map.get(lab_id))
        if lab_timeout is not None:
            candidates.append(lab_timeout)

    model_timeout_map = campaign_cfg.get("watchdog_timeout_by_model", {})
    if isinstance(model_timeout_map, dict):
        model_name = _resolve_sim_model_name(sim_cfg)
        if model_name:
            model_timeout = _coerce_positive_int(model_timeout_map.get(model_name))
            if model_timeout is not None:
                candidates.append(model_timeout)

    if not candidates:
        return

    # Use the most permissive timeout among active sources to avoid spurious watchdog aborts.
    sim_cfg["watchdog_timeout_s"] = max(candidates)


def _lab_is_stochastic_by_id(lab_id: str, lab_cfg_by_id: Dict[str, Dict[str, Any]], cache: Dict[str, bool]) -> bool:
    if lab_id in cache:
        return cache[lab_id]
    cfg = lab_cfg_by_id.get(lab_id)
    if not isinstance(cfg, dict):
        cache[lab_id] = False
        return False

    fine_tuning_id = str(cfg.get("fine_tuning_id") or "").strip().upper()
    stochastic = bool(cfg.get("optuna", {}).get("enabled", False)) or bool(cfg.get("lora", {}).get("enabled", False))
    if fine_tuning_id and fine_tuning_id != "FT0":
        stochastic = True

    reference_lab_id = str(cfg.get("reference_lab") or "").strip()
    if not stochastic and reference_lab_id and reference_lab_id != lab_id:
        stochastic = _lab_is_stochastic_by_id(reference_lab_id, lab_cfg_by_id, cache)

    cache[lab_id] = stochastic
    return stochastic


def _compare_results_for_reproducibility(
    *,
    current_rows: Sequence[Dict[str, str]],
    reference_rows: Sequence[Dict[str, str]],
    labs_cfg: Sequence[Dict[str, Any]],
    repro_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    fields_cfg = repro_cfg.get("fields")
    if isinstance(fields_cfg, list) and fields_cfg:
        fields = [str(item) for item in fields_cfg if str(item or "").strip()]
    else:
        fields = list(REPRO_COMPARE_FIELDS)
    if not fields:
        fields = list(REPRO_COMPARE_FIELDS)

    det_tol = _coerce_float(repro_cfg.get("numeric_tolerance"))
    if det_tol is None or det_tol < 0:
        det_tol = 0.0
    stoch_tol = _coerce_float(repro_cfg.get("stochastic_numeric_tolerance"))
    if stoch_tol is None or stoch_tol < 0:
        stoch_tol = det_tol

    allow_missing_keys = bool(repro_cfg.get("allow_missing_keys", False))
    max_stochastic_mismatch_rate = _coerce_float(repro_cfg.get("max_stochastic_mismatch_rate"))
    if max_stochastic_mismatch_rate is None:
        max_stochastic_mismatch_rate = 0.0
    max_stochastic_mismatch_rate = max(0.0, min(1.0, float(max_stochastic_mismatch_rate)))

    def _key(row: Dict[str, str]) -> Tuple[str, str]:
        return (str(row.get("lab_id") or "").strip(), str(row.get("seed") or "").strip())

    current_by_key = {_key(row): row for row in current_rows if _key(row) != ("", "")}
    reference_by_key = {_key(row): row for row in reference_rows if _key(row) != ("", "")}

    shared_keys = sorted(set(current_by_key.keys()) & set(reference_by_key.keys()))
    missing_in_current = sorted(set(reference_by_key.keys()) - set(current_by_key.keys()))
    missing_in_reference = sorted(set(current_by_key.keys()) - set(reference_by_key.keys()))

    enabled_labs = [lab for lab in labs_cfg if isinstance(lab, dict) and bool(lab.get("enabled", True))]
    lab_cfg_by_id: Dict[str, Dict[str, Any]] = {
        str(lab.get("id") or lab.get("lab_id") or "").strip(): lab for lab in enabled_labs
    }
    stochastic_cache: Dict[str, bool] = {}

    field_stats: Dict[str, Dict[str, Any]] = {}
    for field in fields:
        field_stats[field] = {
            "checked": 0,
            "matched": 0,
            "mismatched": 0,
            "deterministic_mismatched": 0,
            "stochastic_mismatched": 0,
        }

    mismatches: List[Dict[str, Any]] = []
    deterministic_checks = 0
    deterministic_mismatches = 0
    stochastic_checks = 0
    stochastic_mismatches = 0

    for lab_id, seed in shared_keys:
        current_row = current_by_key[(lab_id, seed)]
        reference_row = reference_by_key[(lab_id, seed)]
        is_stochastic = _lab_is_stochastic_by_id(lab_id, lab_cfg_by_id, stochastic_cache)
        mode = "stochastic" if is_stochastic else "deterministic"
        tolerance = stoch_tol if is_stochastic else det_tol

        for field in fields:
            current_value = current_row.get(field)
            reference_value = reference_row.get(field)
            field_stats[field]["checked"] += 1

            current_num = _coerce_float(current_value)
            reference_num = _coerce_float(reference_value)
            matched = False
            abs_delta: Optional[float] = None

            if current_num is not None and reference_num is not None:
                abs_delta = abs(current_num - reference_num)
                matched = abs_delta <= tolerance
            else:
                matched = str(current_value or "") == str(reference_value or "")

            if is_stochastic:
                stochastic_checks += 1
            else:
                deterministic_checks += 1

            if matched:
                field_stats[field]["matched"] += 1
                continue

            field_stats[field]["mismatched"] += 1
            if is_stochastic:
                field_stats[field]["stochastic_mismatched"] += 1
                stochastic_mismatches += 1
            else:
                field_stats[field]["deterministic_mismatched"] += 1
                deterministic_mismatches += 1

            mismatches.append(
                {
                    "lab_id": lab_id,
                    "seed": seed,
                    "mode": mode,
                    "field": field,
                    "current": current_value,
                    "reference": reference_value,
                    "abs_delta": "" if abs_delta is None else round(abs_delta, 12),
                    "tolerance": tolerance,
                }
            )

    missing_key_violation = (not allow_missing_keys) and (bool(missing_in_current) or bool(missing_in_reference))
    stochastic_mismatch_rate = (
        (float(stochastic_mismatches) / float(stochastic_checks)) if stochastic_checks > 0 else 0.0
    )

    pass_status = True
    reasons: List[str] = []
    if missing_key_violation:
        pass_status = False
        reasons.append("missing_keys")
    if deterministic_mismatches > 0:
        pass_status = False
        reasons.append("deterministic_mismatch")
    if stochastic_mismatch_rate > max_stochastic_mismatch_rate:
        pass_status = False
        reasons.append("stochastic_mismatch_rate_exceeded")

    return {
        "pass": pass_status,
        "failure_reasons": reasons,
        "fields": fields,
        "tolerances": {
            "deterministic_numeric_tolerance": det_tol,
            "stochastic_numeric_tolerance": stoch_tol,
            "allow_missing_keys": allow_missing_keys,
            "max_stochastic_mismatch_rate": max_stochastic_mismatch_rate,
        },
        "shared_key_count": len(shared_keys),
        "missing_in_current_count": len(missing_in_current),
        "missing_in_reference_count": len(missing_in_reference),
        "missing_in_current": [{"lab_id": item[0], "seed": item[1]} for item in missing_in_current],
        "missing_in_reference": [{"lab_id": item[0], "seed": item[1]} for item in missing_in_reference],
        "deterministic_checks": deterministic_checks,
        "deterministic_mismatches": deterministic_mismatches,
        "stochastic_checks": stochastic_checks,
        "stochastic_mismatches": stochastic_mismatches,
        "stochastic_mismatch_rate": round(stochastic_mismatch_rate, 6),
        "field_stats": field_stats,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def _run_reproducibility_check(
    *,
    campaign_cfg: Dict[str, Any],
    output_dir: Path,
    results_path: Path,
    labs_cfg: Sequence[Dict[str, Any]],
    include_timestamps: bool,
    campaign_log_path: Path,
) -> Optional[Dict[str, Any]]:
    repro_cfg = _reproducibility_cfg(campaign_cfg)
    if not bool(repro_cfg.get("enabled", False)):
        return None

    reference_token = str(repro_cfg.get("reference_results_csv") or "").strip()
    if not reference_token:
        _append_text_log(
            campaign_log_path,
            "Reproducibility check enabled, but reference_results_csv is empty; skipping.",
            include_timestamps,
        )
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "reference_results_csv_missing",
        }

    reference_path = _resolve_path(output_dir, reference_token, "")
    if not reference_path.exists() or not reference_path.is_file():
        message = f"Reproducibility reference not found: {reference_path}"
        if bool(repro_cfg.get("fail_on_violation", False)):
            raise FileNotFoundError(message)
        _append_text_log(campaign_log_path, message, include_timestamps)
        return {
            "enabled": True,
            "status": "failed",
            "reason": "reference_results_csv_not_found",
            "reference_results_csv": str(reference_path),
        }

    current_rows = _read_csv_rows(results_path)
    reference_rows = _read_csv_rows(reference_path)
    comparison = _compare_results_for_reproducibility(
        current_rows=current_rows,
        reference_rows=reference_rows,
        labs_cfg=labs_cfg,
        repro_cfg=repro_cfg,
    )
    comparison["enabled"] = True
    comparison["status"] = "passed" if comparison.get("pass") else "failed"
    comparison["created_at"] = datetime.now().astimezone().isoformat()
    comparison["current_results_csv"] = str(results_path)
    comparison["reference_results_csv"] = str(reference_path)

    report_name = str(repro_cfg.get("report_json") or "reproducibility_report.json").strip()
    mismatch_name = str(repro_cfg.get("mismatches_csv") or "reproducibility_mismatches.csv").strip()
    report_path = _resolve_path(output_dir, report_name, "reproducibility_report.json")
    mismatch_path = _resolve_path(output_dir, mismatch_name, "reproducibility_mismatches.csv")

    _write_json(report_path, comparison)
    mismatch_rows = comparison.get("mismatches", [])
    if isinstance(mismatch_rows, list) and mismatch_rows:
        _write_csv(
            mismatch_path,
            ["lab_id", "seed", "mode", "field", "current", "reference", "abs_delta", "tolerance"],
            mismatch_rows,
        )
    else:
        _write_csv(
            mismatch_path,
            ["lab_id", "seed", "mode", "field", "current", "reference", "abs_delta", "tolerance"],
            [],
        )

    _append_text_log(
        campaign_log_path,
        "Reproducibility check: "
        f"status={comparison['status']}, shared={comparison.get('shared_key_count', 0)}, "
        f"det_mismatch={comparison.get('deterministic_mismatches', 0)}, "
        f"stoch_mismatch={comparison.get('stochastic_mismatches', 0)}.",
        include_timestamps,
    )
    _append_text_log(campaign_log_path, f"Reproducibility report: {report_path}", include_timestamps)
    _append_text_log(campaign_log_path, f"Reproducibility mismatches: {mismatch_path}", include_timestamps)

    if bool(repro_cfg.get("fail_on_violation", False)) and not bool(comparison.get("pass", False)):
        raise RuntimeError(
            "Reproducibility check failed "
            f"(deterministic_mismatches={comparison.get('deterministic_mismatches', 0)}, "
            f"stochastic_mismatch_rate={comparison.get('stochastic_mismatch_rate', 0.0)})."
        )

    return {
        "enabled": True,
        "status": comparison.get("status"),
        "pass": bool(comparison.get("pass", False)),
        "report_json": str(report_path),
        "mismatches_csv": str(mismatch_path),
        "shared_key_count": comparison.get("shared_key_count", 0),
        "deterministic_mismatches": comparison.get("deterministic_mismatches", 0),
        "stochastic_mismatches": comparison.get("stochastic_mismatches", 0),
        "stochastic_mismatch_rate": comparison.get("stochastic_mismatch_rate", 0.0),
        "reference_results_csv": str(reference_path),
    }


def _run_campaign(
    master_config: Dict[str, Any],
    config_path: Path,
    *,
    run_mode: str = "campaign",
    seed_limit: Optional[int] = None,
    skip_optuna: bool = False,
    skip_lora: bool = False,
    output_dir_override: Optional[Path] = None,
    export_mt: bool = True,
) -> int:
    campaign_cfg = master_config.get("campaign", {})
    if not isinstance(campaign_cfg, dict):
        raise ValueError("campaign must be a JSON object when provided.")
    if seed_limit is not None and int(seed_limit) < 1:
        raise ValueError("seed_limit must be >= 1 when provided.")

    config_dir = config_path.parent
    campaign_name_base = str(campaign_cfg.get("name") or campaign_cfg.get("campaign_name") or "campaign").strip()
    if not campaign_name_base:
        campaign_name_base = "campaign"
    campaign_name = campaign_name_base if run_mode == "campaign" else f"{campaign_name_base}_{run_mode}"

    if output_dir_override is not None:
        output_dir = output_dir_override.resolve()
    else:
        output_dir = _resolve_path(
            config_dir,
            str(campaign_cfg.get("output_dir") or ""),
            f"campaign_runs/{campaign_name_base}",
        )
    fresh_output = bool(campaign_cfg.get("fresh_output", True))
    stop_on_error = bool(campaign_cfg.get("stop_on_error", True))
    stop_on_watchdog_abort = bool(campaign_cfg.get("stop_on_watchdog_abort", False))

    global_logging_cfg = master_config.get("logging", {}) if isinstance(master_config.get("logging"), dict) else {}
    campaign_logging_cfg = campaign_cfg.get("logging", {}) if isinstance(campaign_cfg.get("logging"), dict) else {}
    include_timestamps = bool(
        campaign_logging_cfg.get(
            "include_timestamps",
            global_logging_cfg.get("include_timestamps", True),
        )
    )
    _set_shared_logger_timestamp_mode(include_timestamps)
    repro_cfg = _reproducibility_cfg(campaign_cfg)
    runtime_logs_root = (output_dir / "runtime_logs").resolve()

    if fresh_output and output_dir.exists():
        _remove_tree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_logs_root.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.csv"
    lab_results_path = output_dir / "lab_results.csv"
    campaign_log_path = output_dir / "campaign.log"
    runtime_dir = output_dir / "runtime_configs"
    labs_root = output_dir / "labs"
    lm_error_summary_csv = runtime_logs_root / "lm_error_summary.csv"
    _write_csv(lm_error_summary_csv, LM_CONVERSATION_OVERVIEW_FIELDS, [])

    os.environ["CORASAT_RESULTS_CSV"] = str(results_path)
    os.environ["CORASAT_LOG_DIR"] = str(runtime_logs_root / "shared_logs")
    os.environ["CORASAT_LOG_TIMESTAMPS"] = _bool_text(include_timestamps)
    os.environ["CORASAT_DETERMINISTIC_RUN_ID"] = _bool_text(bool(repro_cfg.get("deterministic_run_id", False)))
    repro_strict = bool(repro_cfg.get("set_deterministic_env", False))
    os.environ["CORASAT_REPRO_STRICT"] = _bool_text(repro_strict)
    if repro_strict:
        os.environ["PYTHONHASHSEED"] = str(repro_cfg.get("pythonhashseed", 0))
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(repro_cfg.get("cublas_workspace_config", ":4096:8"))
        os.environ["TOKENIZERS_PARALLELISM"] = str(repro_cfg.get("tokenizers_parallelism", "false"))
        os.environ["CORASAT_TORCH_DETERMINISTIC"] = "true"
    else:
        os.environ.pop("CORASAT_TORCH_DETERMINISTIC", None)

    labs_cfg = campaign_cfg.get("labs", [])
    if not isinstance(labs_cfg, list) or not labs_cfg:
        raise ValueError("campaign.labs must be a non-empty list.")

    # Preflight checks: duplicate ids and forward/missing references.
    seen_enabled_lab_ids: Dict[str, int] = {}
    for index, lab_cfg_raw in enumerate(labs_cfg, start=1):
        if not isinstance(lab_cfg_raw, dict):
            continue
        if not bool(lab_cfg_raw.get("enabled", True)):
            continue
        lab_id = _resolve_lab_id(lab_cfg_raw, index)
        previous = seen_enabled_lab_ids.get(lab_id)
        if previous is not None:
            raise ValueError(
                f"Duplicate enabled lab id '{lab_id}' in campaign.labs "
                f"(positions {previous} and {index})."
            )
        seen_enabled_lab_ids[lab_id] = index

    completed_ids: set[str] = set()
    for index, lab_cfg_raw in enumerate(labs_cfg, start=1):
        if not isinstance(lab_cfg_raw, dict):
            continue
        if not bool(lab_cfg_raw.get("enabled", True)):
            continue
        lab_id = _resolve_lab_id(lab_cfg_raw, index)
        reference_lab_id = str(lab_cfg_raw.get("reference_lab") or "").strip()
        if reference_lab_id and reference_lab_id not in completed_ids:
            raise ValueError(
                f"{lab_id}: reference_lab '{reference_lab_id}' must reference an enabled lab "
                "defined earlier in campaign.labs."
            )
        completed_ids.add(lab_id)

    _validate_lab_profiles(labs_cfg, config_dir)
    _validate_lab_parent_diffs(labs_cfg)

    global_seed_spec = campaign_cfg.get("seed_list")
    if global_seed_spec is None:
        global_seed_spec = master_config.get("simulation", {}).get("seed_list")
    if not _resolve_seed_spec(global_seed_spec):
        raise ValueError("campaign seed_list resolved to an empty set.")

    global_max_rounds = int(
        campaign_cfg.get("max_rounds")
        if campaign_cfg.get("max_rounds") is not None
        else master_config.get("simulation", {}).get("max_rounds", 32)
    )
    global_use_gui = bool(
        campaign_cfg.get("use_gui")
        if campaign_cfg.get("use_gui") is not None
        else master_config.get("simulation", {}).get("use_gui", False)
    )

    manifest: Dict[str, Any] = {
        "campaign_name": campaign_name,
        "run_mode": run_mode,
        "config_path": str(config_path),
        "started_at": datetime.now().astimezone().isoformat(),
        "output_dir": str(output_dir),
        "runtime_logs_root": str(runtime_logs_root),
        "results_csv": str(results_path),
        "lab_results_csv": str(lab_results_path),
        "lm_error_summary_csv": str(lm_error_summary_csv),
        "include_timestamps": include_timestamps,
        "seed_limit": int(seed_limit) if seed_limit is not None else None,
        "skip_optuna": bool(skip_optuna),
        "skip_lora": bool(skip_lora),
        "export_mt": bool(export_mt),
        "labs": [],
    }

    _append_text_log(
        campaign_log_path,
        f"Campaign '{campaign_name}' started (mode={run_mode}, seed_limit={seed_limit}, "
        f"skip_optuna={skip_optuna}, skip_lora={skip_lora}).",
        include_timestamps,
    )
    _append_text_log(campaign_log_path, f"Output dir: {output_dir}", include_timestamps)
    _append_text_log(campaign_log_path, f"Runtime logs dir: {runtime_logs_root}", include_timestamps)
    _append_text_log(
        campaign_log_path,
        f"LM error summary CSV: {lm_error_summary_csv}",
        include_timestamps,
    )

    lab_rows: List[Dict[str, Any]] = []
    lab_rows_by_id: Dict[str, Dict[str, Any]] = {}
    lab_reports_by_id: Dict[str, Dict[str, Any]] = {}
    completed_lab_logfiles: Dict[str, str] = {}
    error_entries: List[Dict[str, Any]] = []
    error_summary_json_path = output_dir / "error_summary.json"
    error_summary_csv_path = output_dir / "error_summary.csv"

    def _persist_error_summary() -> Dict[str, Any]:
        payload = {
            "campaign_name": campaign_name,
            "run_mode": run_mode,
            "generated_at": datetime.now().astimezone().isoformat(),
            "summary": _summarize_error_entries(error_entries),
            "entries": error_entries,
        }
        _write_json(error_summary_json_path, payload)
        _write_csv(error_summary_csv_path, ERROR_SUMMARY_FIELDS, error_entries)
        return payload

    for index, lab_cfg_raw in enumerate(labs_cfg, start=1):
        if not isinstance(lab_cfg_raw, dict):
            continue
        if not bool(lab_cfg_raw.get("enabled", True)):
            continue

        lab_cfg = copy.deepcopy(lab_cfg_raw)
        lab_id = _resolve_lab_id(lab_cfg, index)
        lab_label = str(lab_cfg.get("label") or lab_id).strip()
        lab_notes = str(lab_cfg.get("notes") or "").strip()
        lab_fine_tuning_id = str(lab_cfg.get("fine_tuning_id") or "").strip().upper()
        lab_config_ids = {
            "rules_id": str(lab_cfg.get("rules_id") or "").strip().upper(),
            "prompt_id": str(lab_cfg.get("prompt_id") or "").strip().upper(),
            "drone_support_id": str(
                lab_cfg.get("drone_support_id") or lab_cfg.get("decision_support_id") or ""
            ).strip().upper(),
            "model_id": str(lab_cfg.get("model_id") or "").strip().upper(),
            "fine_tuning_id": str(lab_cfg.get("fine_tuning_id") or "").strip().upper(),
            "action_policy_id": str(lab_cfg.get("action_policy_id") or "").strip().upper(),
            "communication_id": str(lab_cfg.get("communication_id") or "").strip().upper(),
        }

        if skip_lora and lab_fine_tuning_id == "FT1":
            _append_text_log(
                campaign_log_path,
                f"[{lab_id}] skipped (skip_lora=true and fine_tuning_id={lab_fine_tuning_id}).",
                include_timestamps,
            )
            continue

        lab_dir = labs_root / lab_id
        seeds_dir = lab_dir / "seed_reports"
        lab_runtime_logs_dir = runtime_logs_root / lab_id
        seed_logs_dir = lab_runtime_logs_dir / "seed_logs"
        lm_trace_dir = lab_runtime_logs_dir / "lm_conversations"
        lab_log_stem = _build_lab_log_stem(lab_id, lab_config_ids, lab_notes or lab_label)
        lab_log_path = lab_runtime_logs_dir / f"{lab_log_stem}.lab.log"
        lab_report_path = lab_dir / "lab_report.json"
        lab_simulation_log_path = lab_runtime_logs_dir / f"{lab_log_stem}.sim.log"
        reference_lab_id = str(lab_cfg.get("reference_lab") or "").strip()

        _append_text_log(
            campaign_log_path,
            f"[{lab_id}] start | log={lab_log_path.name}",
            include_timestamps,
        )
        _append_text_log(lab_log_path, f"Lab {lab_id} started.", include_timestamps)

        if reference_lab_id:
            source_report = lab_reports_by_id.get(reference_lab_id)
            source_row = lab_rows_by_id.get(reference_lab_id)
            if source_report is None or source_row is None:
                raise ValueError(f"{lab_id}: reference_lab '{reference_lab_id}' not found among completed labs.")

            runtime_cfg, resolved_ids = _build_lab_runtime_config(
                master_config=master_config,
                config_dir=config_dir,
                lab_entry=lab_cfg,
            )
            runtime_config_path = runtime_dir / f"{lab_id}_runtime.json"
            _write_json(runtime_config_path, runtime_cfg)

            source_metrics = source_report.get("metrics", {})
            source_seed_reports = source_report.get("seed_reports", [])
            source_lab_sim_log = str(source_report.get("lab_simulation_log") or "")

            _append_text_log(
                lab_log_path,
                f"Referenced results from {reference_lab_id}. No simulation executed for this lab.",
                include_timestamps,
            )

            lab_report = {
                "campaign": campaign_name,
                "lab_id": lab_id,
                "label": lab_label,
                "notes": lab_notes,
                "reference_lab": reference_lab_id,
                "resolved_ids": resolved_ids,
                "seed_count_planned": source_report.get("seed_count_planned"),
                "seed_range": source_report.get("seed_range"),
                "seed_spec": source_report.get("seed_spec"),
                "runtime_config": str(runtime_config_path),
                "lab_log": str(lab_log_path),
                "lab_simulation_log": source_lab_sim_log,
                "lm_error_summary_csv": str(lm_error_summary_csv),
                "seed_reports": list(source_seed_reports) if isinstance(source_seed_reports, list) else [],
                "metrics": source_metrics if isinstance(source_metrics, dict) else {},
            }
            _write_json(lab_report_path, lab_report)

            lab_row = {
                "timestamp": datetime.now().astimezone().isoformat(),
                "lab_id": lab_id,
                "label": lab_label,
                "seed_count_planned": source_row.get("seed_count_planned"),
                "seed_count_recorded": source_row.get("seed_count_recorded"),
                "seed_count_completed": source_row.get("seed_count_completed"),
                "seed_count_non_ok": source_row.get("seed_count_non_ok"),
                "seed_count_failed": source_row.get("seed_count_failed"),
                "seed_count_aborted": source_row.get("seed_count_aborted"),
                "seed_range": source_row.get("seed_range"),
                "mean_norm_score": source_row.get("mean_norm_score"),
                "std_norm_score": source_row.get("std_norm_score"),
                "total_prompt_tokens": source_row.get("total_prompt_tokens"),
                "total_completion_tokens": source_row.get("total_completion_tokens"),
                "total_lm_tokens": source_row.get("total_lm_tokens"),
                "total_lm_inference_time_s": source_row.get("total_lm_inference_time_s"),
                "total_lm_parse_failures": source_row.get("total_lm_parse_failures"),
                "total_lm_request_failures": source_row.get("total_lm_request_failures"),
                "total_lm_request_timeouts": source_row.get("total_lm_request_timeouts"),
                "total_runtime_s": source_row.get("total_runtime_s"),
                "mean_runtime_s": source_row.get("mean_runtime_s"),
                "runtime_config": str(runtime_config_path),
                "lab_log": str(lab_log_path),
                "lab_simulation_log": source_lab_sim_log,
                "notes": f"{lab_notes} (referenced from {reference_lab_id})".strip(),
            }
            lab_rows.append(lab_row)
            lab_rows_by_id[lab_id] = dict(lab_row)
            manifest["labs"].append(lab_report)
            lab_reports_by_id[lab_id] = dict(lab_report)

            completed_lab_logfiles[lab_id] = source_lab_sim_log
            completed_lab_logfiles[lab_label] = source_lab_sim_log

            _append_text_log(
                campaign_log_path,
                f"[{lab_id}] referenced from {reference_lab_id}.",
                include_timestamps,
            )
            _append_text_log(lab_log_path, "Lab completed (reference).", include_timestamps)
            continue

        try:
            runtime_cfg, resolved_ids = _build_lab_runtime_config(
                master_config=master_config,
                config_dir=config_dir,
                lab_entry=lab_cfg,
            )

            lab_seed_spec = (
                lab_cfg.get("seed_list")
                if lab_cfg.get("seed_list") is not None
                else lab_cfg.get("seeds")
                if lab_cfg.get("seeds") is not None
                else global_seed_spec
            )
            lab_seeds = _resolve_seed_spec(lab_seed_spec)
            if seed_limit is not None:
                lab_seeds = lab_seeds[: int(seed_limit)]
            if not lab_seeds:
                raise ValueError(f"{lab_id}: seed_list resolved to empty list.")

            runtime_cfg.setdefault("simulation", {})["seed_list"] = _seed_object_from_values(lab_seeds)
            runtime_cfg["simulation"]["max_rounds"] = int(lab_cfg.get("max_rounds", global_max_rounds))
            runtime_cfg["simulation"]["use_gui"] = bool(lab_cfg.get("use_gui", global_use_gui))
            _apply_watchdog_timeout_overrides(
                runtime_cfg=runtime_cfg,
                lab_cfg=lab_cfg,
                campaign_cfg=campaign_cfg,
                run_mode=run_mode,
            )

            runtime_cfg["campaign_context"] = {
                "campaign_name": campaign_name,
                "lab_id": lab_id,
                "label": lab_label,
            }

            runtime_config_path = runtime_dir / f"{lab_id}_runtime.json"
            _write_json(runtime_config_path, runtime_cfg)

            env_base = dict(os.environ)

            optuna_cfg = lab_cfg.get("optuna", {})
            if isinstance(optuna_cfg, dict) and bool(optuna_cfg.get("enabled", False)):
                if skip_optuna:
                    _append_text_log(
                        lab_log_path,
                        f"Optuna skipped for {lab_id} in mode '{run_mode}'.",
                        include_timestamps,
                    )
                else:
                    effective_optuna_cfg = (
                        _apply_smoke_optuna_overrides(optuna_cfg, campaign_cfg=campaign_cfg, lab_seeds=lab_seeds)
                        if run_mode == "smoke"
                        else optuna_cfg
                    )
                    optuna_output_dir = lab_dir / "optuna"
                    optuna_output_dir.mkdir(parents=True, exist_ok=True)
                    optuna_env = dict(env_base)
                    optuna_env["CORASAT_RESULTS_CSV"] = str(optuna_output_dir / "results_optuna.csv")
                    optuna_env["CORASAT_LOG_DIR"] = str(optuna_output_dir / "logs")
                    optuna_env["CORASAT_LOG_TIMESTAMPS"] = _bool_text(include_timestamps)

                    optuna_args = _build_optuna_args(effective_optuna_cfg)
                    if "--output-dir" not in optuna_args:
                        optuna_args.extend(["--output-dir", str(optuna_output_dir)])

                    _run_subprocess(
                        [
                            sys.executable,
                            str(CORASAT_ROOT / "optuna_tuner.py"),
                            "--config",
                            str(runtime_config_path),
                        ]
                        + optuna_args,
                        cwd=CORASAT_ROOT,
                        env=optuna_env,
                        note=f"Optuna for {lab_id}",
                    )

            lora_cfg = lab_cfg.get("lora", {})
            if isinstance(lora_cfg, dict) and bool(lora_cfg.get("enabled", False)):
                if skip_lora:
                    _append_text_log(
                        lab_log_path,
                        f"LoRA fine-tuning skipped for {lab_id} in mode '{run_mode}'.",
                        include_timestamps,
                    )
                else:
                    effective_lora_cfg = _select_lora_cfg_for_mode(
                        lora_cfg,
                        campaign_cfg=campaign_cfg,
                        run_mode=run_mode,
                    )
                    if run_mode == "smoke" and effective_lora_cfg is lora_cfg:
                        _append_text_log(
                            lab_log_path,
                            "Smoke mode: no lora.smoke_commands configured; using default lora.commands.",
                            include_timestamps,
                        )
                    _run_lora_commands(effective_lora_cfg, env_base, context=completed_lab_logfiles)

            if not _lab_evaluate_runtime(lab_cfg):
                lab_simulation_log_path.parent.mkdir(parents=True, exist_ok=True)
                lab_simulation_log_path.write_text("", encoding="utf-8")
                stage_note = f"{lab_notes} (preparation stage; no simulation executed)".strip()
                lab_report = {
                    "campaign": campaign_name,
                    "lab_id": lab_id,
                    "label": lab_label,
                    "notes": stage_note,
                    "resolved_ids": resolved_ids,
                    "runtime_evaluated": False,
                    "seed_count_planned": 0,
                    "seed_range": "",
                    "seed_spec": "",
                    "runtime_config": str(runtime_config_path),
                    "lab_log": str(lab_log_path),
                    "lab_simulation_log": str(lab_simulation_log_path),
                    "lm_error_summary_csv": str(lm_error_summary_csv),
                    "seed_reports": [],
                    "metrics": {},
                }
                _write_json(lab_report_path, lab_report)

                lab_row = {
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "lab_id": lab_id,
                    "label": lab_label,
                    "seed_count_planned": 0,
                    "seed_count_recorded": 0,
                    "seed_count_completed": 0,
                    "seed_count_non_ok": 0,
                    "seed_count_failed": 0,
                    "seed_count_aborted": 0,
                    "seed_range": "",
                    "mean_norm_score": "",
                    "std_norm_score": "",
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_lm_tokens": 0,
                    "total_lm_inference_time_s": 0.0,
                    "total_lm_parse_failures": 0,
                    "total_lm_request_failures": 0,
                    "total_lm_request_timeouts": 0,
                    "total_runtime_s": 0.0,
                    "mean_runtime_s": "",
                    "runtime_config": str(runtime_config_path),
                    "lab_log": str(lab_log_path),
                    "lab_simulation_log": str(lab_simulation_log_path),
                    "notes": stage_note,
                }
                lab_rows.append(lab_row)
                lab_rows_by_id[lab_id] = dict(lab_row)
                manifest["labs"].append(lab_report)
                lab_reports_by_id[lab_id] = dict(lab_report)
                completed_lab_logfiles[lab_id] = str(lab_simulation_log_path)
                completed_lab_logfiles[lab_label] = str(lab_simulation_log_path)

                _append_text_log(
                    campaign_log_path,
                    f"[{lab_id}] completed as preparation-only stage.",
                    include_timestamps,
                )
                _append_text_log(lab_log_path, "Lab completed (preparation only).", include_timestamps)
                continue

            seed_reports: List[Dict[str, Any]] = []
            total_games = max(1, len(lab_seeds))

            for game_index, seed in enumerate(lab_seeds, start=1):
                seed_log_name = f"{lab_id}_seed_{int(seed):04d}.log"
                _append_text_log(lab_log_path, f"Seed {seed} started.", include_timestamps)
                seed_log_path = seed_logs_dir / seed_log_name
                seed_trace_stem = Path(seed_log_name).stem
                seed_trace_inprogress = lm_trace_dir / f"{seed_trace_stem}.lmtrace.inprogress.jsonl"
                _upsert_csv_row(
                    lm_error_summary_csv,
                    fieldnames=LM_CONVERSATION_OVERVIEW_FIELDS,
                    row={
                        "timestamp": datetime.now().astimezone().isoformat(),
                        "campaign": campaign_name,
                        "lab_id": lab_id,
                        "label": lab_label,
                        "seed": seed,
                        "status": "running",
                        "seed_outcome": "RUNNING",
                        "conversation_log": str(seed_trace_inprogress),
                        "simulation_log": str(seed_log_path),
                        "runtime_config": str(runtime_config_path),
                        "rounds": "",
                        "mission_score": "",
                        "norm_score": "",
                        "prompt_tokens_total": 0,
                        "completion_tokens_total": 0,
                        "lm_total_tokens": 0,
                        "lm_inference_time_s": 0.0,
                        "lm_parse_failures": 0,
                        "lm_request_failures": 0,
                        "lm_request_timeouts": 0,
                        "lm_last_error": "",
                        "abort_reason": "",
                        "error": "",
                    },
                    key_fields=("campaign", "lab_id", "seed"),
                )
                run_entry, abort_requested = run_seed(
                    seed,
                    game_index,
                    total_games,
                    config_path=str(runtime_config_path),
                    seed_log_dir=seed_logs_dir,
                    seed_log_file=seed_log_name,
                    lm_trace_dir=lm_trace_dir,
                    include_timestamps=include_timestamps,
                    context={"campaign": campaign_name, "lab_id": lab_id},
                )
                if run_entry:
                    seed_payload = _run_entry_to_seed_report(
                        run_entry,
                        campaign_name=campaign_name,
                        lab_id=lab_id,
                        runtime_config_path=runtime_config_path,
                    )
                    seed_report_path = seeds_dir / f"seed_{int(seed):04d}.json"
                    _write_json(seed_report_path, seed_payload)
                    seed_payload["seed_report_path"] = str(seed_report_path)
                    seed_reports.append(seed_payload)
                    overview_row = _lm_overview_row_from_seed(
                        campaign_name=campaign_name,
                        lab_id=lab_id,
                        label=lab_label,
                        runtime_config_path=runtime_config_path,
                        seed_payload=seed_payload,
                    )
                    _upsert_csv_row(
                        lm_error_summary_csv,
                        fieldnames=LM_CONVERSATION_OVERVIEW_FIELDS,
                        row=overview_row,
                        key_fields=("campaign", "lab_id", "seed"),
                    )

                    status = str(seed_payload.get("status") or "ok")
                    norm_score = seed_payload.get("norm_score")
                    parse_failures = int(seed_payload.get("lm_parse_failures") or 0)
                    request_failures = int(seed_payload.get("lm_request_failures") or 0)
                    timeout_failures = int(seed_payload.get("lm_request_timeouts") or 0)
                    if status == "ok" and isinstance(norm_score, (int, float)):
                        _append_text_log(
                            lab_log_path,
                            f"Seed {seed} completed. norm_score={float(norm_score):.5f} "
                            f"(parse_failures={parse_failures}, request_failures={request_failures}, request_timeouts={timeout_failures})",
                            include_timestamps,
                        )
                    elif status == "ok":
                        _append_text_log(
                            lab_log_path,
                            f"Seed {seed} completed. norm_score=n/a "
                            f"(parse_failures={parse_failures}, request_failures={request_failures}, request_timeouts={timeout_failures})",
                            include_timestamps,
                        )
                    else:
                        status_note = str(seed_payload.get("error") or seed_payload.get("abort_reason") or "").strip()
                        suffix = f" ({status_note})" if status_note else ""
                        error_entries.append(
                            {
                                "timestamp": datetime.now().astimezone().isoformat(),
                                "scope": "seed",
                                "lab_id": lab_id,
                                "label": lab_label,
                                "seed": seed_payload.get("seed"),
                                "status": status,
                                "reason": status_note,
                                "runtime_config": str(runtime_config_path),
                                "seed_report_path": str(seed_report_path),
                                "logfile": str(seed_payload.get("logfile") or ""),
                                "lm_trace_log": str(seed_payload.get("lm_trace_log") or ""),
                            }
                        )
                        _append_text_log(
                            lab_log_path,
                            f"Seed {seed} completed with status={status}{suffix}.",
                            include_timestamps,
                        )
                        if stop_on_error and status in {"init_error", "run_error"}:
                            raise RuntimeError(f"{lab_id}: seed {seed} failed with status={status}{suffix}")
                        if stop_on_watchdog_abort and status == "watchdog_abort":
                            raise RuntimeError(
                                f"{lab_id}: seed {seed} aborted by watchdog{suffix}. "
                                "Increase watchdog timeout or reduce workload."
                            )
                    _append_text_log(
                        campaign_log_path,
                        f"[{lab_id}] seed {game_index}/{total_games} (seed={seed}) status={status}"
                        + (
                            f", norm_score={float(norm_score):.5f}"
                            if status == "ok" and isinstance(norm_score, (int, float))
                            else ""
                        )
                        + (
                            f", parse_failures={parse_failures}, request_failures={request_failures}, "
                            f"request_timeouts={timeout_failures}"
                        ),
                        include_timestamps,
                    )

                if abort_requested:
                    _append_text_log(
                        lab_log_path,
                        f"Abort requested by simulation after seed {seed}; stopping remaining seeds.",
                        include_timestamps,
                    )
                    break

            _concat_seed_logs(seed_reports, lab_simulation_log_path)
            completed_lab_logfiles[lab_id] = str(lab_simulation_log_path)
            completed_lab_logfiles[lab_label] = str(lab_simulation_log_path)

            summary = _summarize_lab_seed_reports(seed_reports)
            lab_report = {
                "campaign": campaign_name,
                "lab_id": lab_id,
                "label": lab_label,
                "notes": lab_notes,
                "resolved_ids": resolved_ids,
                "seed_count_planned": len(lab_seeds),
                "seed_range": _seed_range_text(lab_seeds),
                "seed_spec": lab_seed_spec,
                "runtime_config": str(runtime_config_path),
                "lab_log": str(lab_log_path),
                "lab_simulation_log": str(lab_simulation_log_path),
                "lm_error_summary_csv": str(lm_error_summary_csv),
                "seed_reports": [item.get("seed_report_path", "") for item in seed_reports],
                "metrics": summary,
            }
            _write_json(lab_report_path, lab_report)

            lab_row = {
                "timestamp": datetime.now().astimezone().isoformat(),
                "lab_id": lab_id,
                "label": lab_label,
                "seed_count_planned": len(lab_seeds),
                "seed_count_recorded": summary["seed_count_recorded"],
                "seed_count_completed": summary["seed_count_completed"],
                "seed_count_non_ok": summary["seed_count_non_ok"],
                "seed_count_failed": summary["seed_count_failed"],
                "seed_count_aborted": summary["seed_count_aborted"],
                "seed_range": _seed_range_text(lab_seeds),
                "mean_norm_score": summary["mean_norm_score"],
                "std_norm_score": summary["std_norm_score"],
                "total_prompt_tokens": summary["total_prompt_tokens"],
                "total_completion_tokens": summary["total_completion_tokens"],
                "total_lm_tokens": summary["total_lm_tokens"],
                "total_lm_inference_time_s": summary["total_lm_inference_time_s"],
                "total_lm_parse_failures": summary["total_lm_parse_failures"],
                "total_lm_request_failures": summary["total_lm_request_failures"],
                "total_lm_request_timeouts": summary["total_lm_request_timeouts"],
                "total_runtime_s": summary["total_runtime_s"],
                "mean_runtime_s": summary["mean_runtime_s"],
                "runtime_config": str(runtime_config_path),
                "lab_log": str(lab_log_path),
                "lab_simulation_log": str(lab_simulation_log_path),
                "notes": lab_notes,
            }
            lab_rows.append(lab_row)
            lab_rows_by_id[lab_id] = dict(lab_row)
            manifest["labs"].append(lab_report)
            lab_reports_by_id[lab_id] = dict(lab_report)

            _append_text_log(
                campaign_log_path,
                f"[{lab_id}] completed with {summary['seed_count_completed']}/{len(lab_seeds)} seeds.",
                include_timestamps,
            )
            _append_text_log(lab_log_path, "Lab completed.", include_timestamps)
        except Exception as exc:
            error_message = f"[{lab_id}] failed: {exc}"
            error_entries.append(
                {
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "scope": "lab",
                    "lab_id": lab_id,
                    "label": lab_label,
                    "seed": "",
                    "status": "lab_exception",
                    "reason": str(exc),
                    "runtime_config": "",
                    "seed_report_path": "",
                    "logfile": str(lab_log_path),
                    "lm_trace_log": "",
                }
            )
            _persist_error_summary()
            _append_text_log(campaign_log_path, error_message, include_timestamps)
            _append_text_log(lab_log_path, error_message, include_timestamps)
            _append_text_log(lab_log_path, traceback.format_exc(), include_timestamps)
            if stop_on_error:
                raise

    _write_csv(lab_results_path, LAB_RESULTS_FIELDS, lab_rows)

    campaign_totals = {
        "labs_executed": len(lab_rows),
        "seed_count_recorded": sum(int(row.get("seed_count_recorded") or 0) for row in lab_rows),
        "seed_count_completed": sum(int(row.get("seed_count_completed") or 0) for row in lab_rows),
        "seed_count_non_ok": sum(int(row.get("seed_count_non_ok") or 0) for row in lab_rows),
        "seed_count_failed": sum(int(row.get("seed_count_failed") or 0) for row in lab_rows),
        "seed_count_aborted": sum(int(row.get("seed_count_aborted") or 0) for row in lab_rows),
        "total_prompt_tokens": sum(int(row.get("total_prompt_tokens") or 0) for row in lab_rows),
        "total_completion_tokens": sum(int(row.get("total_completion_tokens") or 0) for row in lab_rows),
        "total_lm_tokens": sum(int(row.get("total_lm_tokens") or 0) for row in lab_rows),
        "total_lm_inference_time_s": round(
            sum(float(row.get("total_lm_inference_time_s") or 0.0) for row in lab_rows),
            6,
        ),
        "total_lm_parse_failures": sum(int(row.get("total_lm_parse_failures") or 0) for row in lab_rows),
        "total_lm_request_failures": sum(int(row.get("total_lm_request_failures") or 0) for row in lab_rows),
        "total_lm_request_timeouts": sum(int(row.get("total_lm_request_timeouts") or 0) for row in lab_rows),
        "total_runtime_s": round(sum(float(row.get("total_runtime_s") or 0.0) for row in lab_rows), 6),
    }

    manifest["finished_at"] = datetime.now().astimezone().isoformat()
    manifest["totals"] = campaign_totals

    error_summary_payload = _persist_error_summary()
    manifest["error_summary"] = {
        "json_path": str(error_summary_json_path),
        "csv_path": str(error_summary_csv_path),
        "entry_count": error_summary_payload["summary"].get("entry_count", 0),
        "by_status": error_summary_payload["summary"].get("by_status", {}),
    }

    repro_result = _run_reproducibility_check(
        campaign_cfg=campaign_cfg,
        output_dir=output_dir,
        results_path=results_path,
        labs_cfg=labs_cfg,
        include_timestamps=include_timestamps,
        campaign_log_path=campaign_log_path,
    )
    if repro_result is not None:
        manifest["reproducibility"] = repro_result

    manifest_path = output_dir / "campaign_report.json"
    _write_json(manifest_path, manifest)

    if export_mt:
        _export_campaign_to_mt(
            master_config=master_config,
            campaign_cfg=campaign_cfg,
            config_path=config_path,
            output_dir=output_dir,
            results_path=results_path,
            lab_results_path=lab_results_path,
            manifest_path=manifest_path,
        )
    else:
        _append_text_log(campaign_log_path, "MT export skipped for this run mode.", include_timestamps)

    _append_text_log(campaign_log_path, f"Campaign completed (mode={run_mode}).", include_timestamps)
    _append_text_log(campaign_log_path, f"Manifest: {manifest_path}", include_timestamps)
    _append_text_log(
        campaign_log_path,
        f"Error summary: {error_summary_json_path} | entries={error_summary_payload['summary'].get('entry_count', 0)}",
        include_timestamps,
    )

    _log(f"Campaign completed (mode={run_mode}).")
    _log(f"- Output dir: {output_dir}")
    _log(f"- Results: {results_path}")
    _log(f"- Lab results: {lab_results_path}")
    _log(
        f"- Error summary: {error_summary_json_path} "
        f"(entries={error_summary_payload['summary'].get('entry_count', 0)})"
    )
    if repro_result is not None:
        _log(f"- Reproducibility: {repro_result.get('status')}")
    _log(f"- Campaign report: {manifest_path}")
    return 0


def _run_configured_modes(master_config: Dict[str, Any], config_path: Path) -> int:
    campaign_cfg = master_config.get("campaign")
    if not isinstance(campaign_cfg, dict):
        raise ValueError("config must define a 'campaign' object.")

    labs = campaign_cfg.get("labs")
    if not isinstance(labs, list) or not labs:
        raise ValueError("campaign.labs must be a non-empty list.")

    smoke_run = bool(campaign_cfg.get("smoke_run", False))
    campaign_run = bool(campaign_cfg.get("campaign_run", True))
    if not smoke_run and not campaign_run:
        raise ValueError("At least one mode must be enabled: campaign.smoke_run or campaign.campaign_run.")

    config_dir = config_path.parent
    campaign_name = str(campaign_cfg.get("name") or campaign_cfg.get("campaign_name") or "campaign").strip() or "campaign"
    base_output_dir = _resolve_path(
        config_dir,
        str(campaign_cfg.get("output_dir") or ""),
        f"campaign_runs/{campaign_name}",
    )

    if smoke_run:
        smoke_seed_limit_raw = (
            campaign_cfg.get("smoke_first_n_seeds")
            if campaign_cfg.get("smoke_first_n_seeds") is not None
            else campaign_cfg.get("smoke_seed_limit", 2)
        )
        try:
            smoke_seed_limit = int(smoke_seed_limit_raw)
        except Exception as exc:
            raise ValueError(
                "campaign.smoke_first_n_seeds (or smoke_seed_limit) must be an integer, "
                f"got: {smoke_seed_limit_raw!r}"
            ) from exc
        if smoke_seed_limit < 1:
            raise ValueError("campaign.smoke_first_n_seeds must be >= 1.")

        smoke_output_dir = _resolve_path(
            config_dir,
            str(campaign_cfg.get("smoke_output_dir") or ""),
            str(base_output_dir) + "_smoke",
        )
        smoke_skip_optuna = bool(campaign_cfg.get("smoke_skip_optuna", True))
        smoke_skip_lora = bool(campaign_cfg.get("smoke_skip_lora", True))
        smoke_export_mt = bool(campaign_cfg.get("smoke_mt_export", False))
        smoke_repro_cfg = (
            campaign_cfg.get("smoke_reproducibility", {})
            if isinstance(campaign_cfg.get("smoke_reproducibility"), dict)
            else {}
        )
        smoke_run_twice = bool(smoke_repro_cfg.get("enabled", False))

        if smoke_run_twice:
            smoke_run1_output = _resolve_path(
                config_dir,
                str(smoke_repro_cfg.get("run1_output_dir") or ""),
                str(smoke_output_dir) + "_run1",
            )
            smoke_run2_output = _resolve_path(
                config_dir,
                str(smoke_repro_cfg.get("run2_output_dir") or ""),
                str(smoke_output_dir) + "_run2",
            )
            smoke_report_path = _resolve_path(
                config_dir,
                str(smoke_repro_cfg.get("report_path") or ""),
                f"campaign_runs/{campaign_name}_smoke_report.json",
            )
            fail_on_deviation = bool(smoke_repro_cfg.get("fail_on_deviation", False))
            run2_fail_on_violation = bool(
                smoke_repro_cfg.get("fail_on_violation", fail_on_deviation)
            )

            run1_error = ""
            run2_error = ""
            run1_success = False
            run2_success = False

            _log(
                "Starting smoke run #1 "
                f"(seed_limit={smoke_seed_limit}, skip_optuna={smoke_skip_optuna}, skip_lora={smoke_skip_lora})."
            )
            run1_master = copy.deepcopy(master_config)
            run1_campaign = run1_master.get("campaign", {})
            if isinstance(run1_campaign, dict):
                repro = run1_campaign.get("reproducibility")
                if not isinstance(repro, dict):
                    repro = {}
                    run1_campaign["reproducibility"] = repro
                repro["enabled"] = False
                repro["reference_results_csv"] = ""

            try:
                _run_campaign(
                    run1_master,
                    config_path,
                    run_mode="smoke",
                    seed_limit=smoke_seed_limit,
                    skip_optuna=smoke_skip_optuna,
                    skip_lora=smoke_skip_lora,
                    output_dir_override=smoke_run1_output,
                    export_mt=smoke_export_mt,
                )
                run1_success = True
            except Exception as exc:
                run1_error = str(exc)

            if run1_success:
                _log(
                    "Starting smoke run #2 "
                    f"(seed_limit={smoke_seed_limit}, skip_optuna={smoke_skip_optuna}, skip_lora={smoke_skip_lora})."
                )
                run2_master = copy.deepcopy(master_config)
                run2_campaign = run2_master.get("campaign", {})
                if isinstance(run2_campaign, dict):
                    repro = run2_campaign.get("reproducibility")
                    if not isinstance(repro, dict):
                        repro = {}
                        run2_campaign["reproducibility"] = repro
                    repro["enabled"] = True
                    repro["reference_results_csv"] = str((smoke_run1_output / "results.csv").resolve())
                    repro["fail_on_violation"] = run2_fail_on_violation

                try:
                    _run_campaign(
                        run2_master,
                        config_path,
                        run_mode="smoke",
                        seed_limit=smoke_seed_limit,
                        skip_optuna=smoke_skip_optuna,
                        skip_lora=smoke_skip_lora,
                        output_dir_override=smoke_run2_output,
                        export_mt=smoke_export_mt,
                    )
                    run2_success = True
                except Exception as exc:
                    run2_error = str(exc)

            run1_summary = _build_smoke_run_summary(smoke_run1_output)
            run2_summary = _build_smoke_run_summary(smoke_run2_output)
            run2_repro = (
                run2_summary.get("reproducibility", {})
                if isinstance(run2_summary.get("reproducibility"), dict)
                else {}
            )

            det_mismatch = int(run2_repro.get("deterministic_mismatches") or 0)
            stoch_mismatch = int(run2_repro.get("stochastic_mismatches") or 0)
            mismatch_total = int(
                run2_repro.get("mismatch_count")
                or (det_mismatch + stoch_mismatch)
            )
            deviations_encountered = mismatch_total > 0 or not bool(run2_repro.get("pass", False))

            smoke_report = {
                "created_at": datetime.now().astimezone().isoformat(),
                "mode": "double_smoke",
                "seed_limit": smoke_seed_limit,
                "smoke_run_success": bool(run1_success and run2_success),
                "deviations_encountered": bool(deviations_encountered),
                "overall_success": bool(run1_success and run2_success and not deviations_encountered),
                "run_1": {
                    "success": run1_success,
                    "error": run1_error,
                    **run1_summary,
                },
                "run_2": {
                    "success": run2_success,
                    "error": run2_error,
                    **run2_summary,
                },
                "reproducibility": {
                    "status": run2_repro.get("status", "missing"),
                    "pass": bool(run2_repro.get("pass", False)),
                    "shared_key_count": int(run2_repro.get("shared_key_count") or 0),
                    "deterministic_mismatches": det_mismatch,
                    "stochastic_mismatches": stoch_mismatch,
                    "stochastic_mismatch_rate": float(run2_repro.get("stochastic_mismatch_rate") or 0.0),
                    "mismatch_count": mismatch_total,
                    "report_json": run2_repro.get("report_json", ""),
                    "mismatches_csv": run2_repro.get("mismatches_csv", ""),
                },
            }
            _write_json(smoke_report_path, smoke_report)
            _log(f"Smoke reproducibility report: {smoke_report_path}")

            if bool(campaign_cfg.get("stop_on_error", True)) and (not run1_success or not run2_success):
                raise RuntimeError(
                    "One or both smoke runs failed. "
                    f"See {smoke_report_path}"
                )
            if fail_on_deviation and deviations_encountered:
                raise RuntimeError(
                    "Smoke reproducibility report indicates deviations. "
                    f"See {smoke_report_path}"
                )
        else:
            _log(
                "Starting smoke run "
                f"(seed_limit={smoke_seed_limit}, skip_optuna={smoke_skip_optuna}, skip_lora={smoke_skip_lora})."
            )
            _run_campaign(
                master_config,
                config_path,
                run_mode="smoke",
                seed_limit=smoke_seed_limit,
                skip_optuna=smoke_skip_optuna,
                skip_lora=smoke_skip_lora,
                output_dir_override=smoke_output_dir,
                export_mt=smoke_export_mt,
            )

    if campaign_run:
        _log("Starting full campaign run.")
        _run_campaign(master_config, config_path, run_mode="campaign")

    return 0


def _has_campaign_labs(config: Dict[str, Any]) -> bool:
    campaign_cfg = config.get("campaign")
    if not isinstance(campaign_cfg, dict):
        return False
    labs = campaign_cfg.get("labs")
    return isinstance(labs, list) and len(labs) > 0


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = _read_json(path)
        if isinstance(payload, dict):
            return payload
        return {}
    except Exception:
        return {}


def _build_smoke_run_summary(output_dir: Path) -> Dict[str, Any]:
    manifest_path = output_dir / "campaign_report.json"
    manifest = _read_json_if_exists(manifest_path)
    totals = manifest.get("totals", {}) if isinstance(manifest.get("totals"), dict) else {}
    repro = manifest.get("reproducibility", {}) if isinstance(manifest.get("reproducibility"), dict) else {}
    error_summary = manifest.get("error_summary", {}) if isinstance(manifest.get("error_summary"), dict) else {}
    summary = {
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "results_csv": str(output_dir / "results.csv"),
        "lab_results_csv": str(output_dir / "lab_results.csv"),
        "error_summary_csv": str(output_dir / "error_summary.csv"),
        "status": "unknown",
        "totals": totals,
        "reproducibility": repro,
        "error_summary": error_summary,
    }
    if manifest:
        summary["status"] = "completed"
    return summary


def main() -> int:
    """CLI entry point for smoke/campaign execution."""
    parser = argparse.ArgumentParser(description="Run Corasat campaign(s) from config.json.")
    parser.add_argument(
        "--config",
        default=core.CONFIG_PATH,
        help="Campaign config path (default: config.json).",
    )
    args = parser.parse_args()

    config_path = _resolve_config_path(str(args.config))
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    master_config = _read_json(config_path)

    try:
        if not _has_campaign_labs(master_config):
            raise SystemExit("Single-runtime mode was removed. Provide config.campaign.labs in the config file.")
        return int(_run_configured_modes(master_config, config_path))
    except KeyboardInterrupt:
        _log("Interrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
