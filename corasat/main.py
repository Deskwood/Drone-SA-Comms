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
    "total_runtime_s",
    "mean_runtime_s",
    "runtime_config",
    "lab_log",
    "lab_simulation_log",
    "notes",
]

PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z0-9_]+)(?::([^}]+))?\}")


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


def _append_text_log(path: Path, message: str, include_timestamps: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if include_timestamps:
        prefix = datetime.now().astimezone().isoformat()
        line = f"[{prefix}] {message}"
    else:
        line = message
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


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
    }
    return report


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
    }


def run_seed(
    seed: Optional[int],
    game_index: int,
    total_games: int,
    *,
    config_path: str,
    seed_log_dir: Optional[Path] = None,
    seed_log_file: str = "",
    include_timestamps: bool = True,
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Run a single seed and return (run_entry, abort_requested)."""
    if seed_log_dir is not None:
        seed_log_dir.mkdir(parents=True, exist_ok=True)
        resolved_log_name = seed_log_file or f"seed_{seed}.log"
        _switch_shared_log_file(seed_log_dir, resolved_log_name, include_timestamps)

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
        _persist_results(run_entry)
        return run_entry, False

    _log("Launching simulation.")
    run_started = time.time()
    run_success = False
    run_error: Optional[str] = None
    try:
        sim.run_simulation()
        run_success = True
    except KeyboardInterrupt:
        _log("Interrupted by user (Ctrl+C).")
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
        _persist_results(run_entry)
        return run_entry, False

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
        _persist_results(run_entry)
        return run_entry, False

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
        _persist_results(run_entry)
        return run_entry, True

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
    _persist_results(run_entry)
    return run_entry, False


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

    global_logging_cfg = master_config.get("logging", {}) if isinstance(master_config.get("logging"), dict) else {}
    campaign_logging_cfg = campaign_cfg.get("logging", {}) if isinstance(campaign_cfg.get("logging"), dict) else {}
    include_timestamps = bool(
        campaign_logging_cfg.get(
            "include_timestamps",
            global_logging_cfg.get("include_timestamps", True),
        )
    )
    _set_shared_logger_timestamp_mode(include_timestamps)

    if fresh_output and output_dir.exists():
        _remove_tree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.csv"
    lab_results_path = output_dir / "lab_results.csv"
    campaign_log_path = output_dir / "campaign.log"
    runtime_dir = output_dir / "runtime_configs"
    labs_root = output_dir / "labs"

    os.environ["CORASAT_RESULTS_CSV"] = str(results_path)
    os.environ["CORASAT_LOG_DIR"] = str(output_dir / "logs")
    os.environ["CORASAT_LOG_TIMESTAMPS"] = _bool_text(include_timestamps)

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
        "results_csv": str(results_path),
        "lab_results_csv": str(lab_results_path),
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

    lab_rows: List[Dict[str, Any]] = []
    lab_rows_by_id: Dict[str, Dict[str, Any]] = {}
    lab_reports_by_id: Dict[str, Dict[str, Any]] = {}
    completed_lab_logfiles: Dict[str, str] = {}

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

        if skip_lora and lab_fine_tuning_id == "FT1":
            _append_text_log(
                campaign_log_path,
                f"[{lab_id}] skipped (skip_lora=true and fine_tuning_id={lab_fine_tuning_id}).",
                include_timestamps,
            )
            continue

        lab_dir = labs_root / lab_id
        seeds_dir = lab_dir / "seed_reports"
        seed_logs_dir = lab_dir / "seed_logs"
        lab_log_path = lab_dir / "lab.log"
        lab_report_path = lab_dir / "lab_report.json"
        lab_simulation_log_path = lab_dir / "lab_simulation.log"
        reference_lab_id = str(lab_cfg.get("reference_lab") or "").strip()

        _append_text_log(campaign_log_path, f"[{lab_id}] start", include_timestamps)
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

            seed_reports: List[Dict[str, Any]] = []
            total_games = max(1, len(lab_seeds))

            for game_index, seed in enumerate(lab_seeds, start=1):
                seed_log_name = f"{lab_id}_seed_{int(seed):04d}.log"
                _append_text_log(lab_log_path, f"Seed {seed} started.", include_timestamps)
                run_entry, abort_requested = run_seed(
                    seed,
                    game_index,
                    total_games,
                    config_path=str(runtime_config_path),
                    seed_log_dir=seed_logs_dir,
                    seed_log_file=seed_log_name,
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

                    status = str(seed_payload.get("status") or "ok")
                    norm_score = seed_payload.get("norm_score")
                    if status == "ok" and isinstance(norm_score, (int, float)):
                        _append_text_log(
                            lab_log_path,
                            f"Seed {seed} completed. norm_score={float(norm_score):.5f}",
                            include_timestamps,
                        )
                    elif status == "ok":
                        _append_text_log(
                            lab_log_path,
                            f"Seed {seed} completed. norm_score=n/a",
                            include_timestamps,
                        )
                    else:
                        status_note = str(seed_payload.get("error") or seed_payload.get("abort_reason") or "").strip()
                        suffix = f" ({status_note})" if status_note else ""
                        _append_text_log(
                            lab_log_path,
                            f"Seed {seed} completed with status={status}{suffix}.",
                            include_timestamps,
                        )
                        if stop_on_error and status in {"init_error", "run_error"}:
                            raise RuntimeError(f"{lab_id}: seed {seed} failed with status={status}{suffix}")

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
        "total_runtime_s": round(sum(float(row.get("total_runtime_s") or 0.0) for row in lab_rows), 6),
    }

    manifest["finished_at"] = datetime.now().astimezone().isoformat()
    manifest["totals"] = campaign_totals

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

    _log(f"Campaign completed (mode={run_mode}).")
    _log(f"- Output dir: {output_dir}")
    _log(f"- Results: {results_path}")
    _log(f"- Lab results: {lab_results_path}")
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
        smoke_seed_limit_raw = campaign_cfg.get("smoke_seed_limit", 2)
        try:
            smoke_seed_limit = int(smoke_seed_limit_raw)
        except Exception as exc:
            raise ValueError(f"campaign.smoke_seed_limit must be an integer, got: {smoke_seed_limit_raw!r}") from exc
        if smoke_seed_limit < 1:
            raise ValueError("campaign.smoke_seed_limit must be >= 1.")

        smoke_output_dir = _resolve_path(
            config_dir,
            str(campaign_cfg.get("smoke_output_dir") or ""),
            str(base_output_dir) + "_smoke",
        )
        smoke_skip_optuna = bool(campaign_cfg.get("smoke_skip_optuna", True))
        smoke_skip_lora = bool(campaign_cfg.get("smoke_skip_lora", True))
        smoke_export_mt = bool(campaign_cfg.get("smoke_mt_export", False))

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
