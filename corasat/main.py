"""Entry point for Corasat simulation and campaign runs.

This module supports two modes from one config file:
1) single runtime mode (legacy): run the seeds in one runtime config, and
2) campaign mode: run multiple labs in sequence using ID-based profile tuples.

In campaign mode, ``config.json`` is the single source of truth.
``run_campaign.py`` and ``lab_state.py`` are kept as compatibility wrappers.
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
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from classes.Simulation import Simulation

if TYPE_CHECKING:
    import pygame

import classes.Core as core

# Legacy shared constants (kept here until core modules are reorganized).
COLORS = ["white", "black"]
FIGURE_TYPES = ["king", "queen", "rook", "bishop", "knight", "pawn"]
DIRECTION_MAP: Dict[str, Tuple[int, int]] = {
    "north": (0, 1),
    "south": (0, -1),
    "east": (1, 0),
    "west": (-1, 0),
    "northeast": (1, 1),
    "northwest": (-1, 1),
    "southeast": (1, -1),
    "southwest": (-1, -1),
}
VALID_DIRECTIONS = set(DIRECTION_MAP.keys())
FIGURE_IMAGES: Dict[Tuple[str, str], "pygame.Surface"] = {}

CORASAT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = CORASAT_ROOT.parent

RUNTIME_ROOT_KEYS = (
    "board",
    "rules_path",
    "simulation",
    "gui",
    "decision_support",
    "prompt_requests",
)

PROFILE_SPECS: Dict[str, Dict[str, str]] = {
    "rules": {"subdir": "rules", "suffix": "_rules.txt", "format": "text"},
    "prompt": {"subdir": "prompt_requests", "suffix": "_prompt_requests.json", "format": "json"},
    "model": {"subdir": "model", "suffix": "_model.json", "format": "json"},
    "fine_tuning": {"subdir": "fine_tuning", "suffix": "_fine_tuning.json", "format": "json"},
    "action_policy": {"subdir": "action_policy", "suffix": "_action_policy.json", "format": "json"},
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
    "seed_count_completed",
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


def _seed_list_from_config(config: Dict[str, Any]) -> List[Optional[int]]:
    """Extract the simulation seed list and apply a fallback when missing."""
    simulation_cfg = config.get("simulation", {}) or {}
    seed_spec = simulation_cfg.get("seed_list", [])
    seed_list = _resolve_seed_spec(seed_spec)

    if not seed_list:
        _log("No seed list found in config; defaulting to [0].")
        return [0]
    if len(seed_list) > 10:
        _log(f"Seed list: {seed_list[:3]} .. {seed_list[-3:]} (total {len(seed_list)})")
    else:
        _log(f"Seed list: {seed_list}")
    return list(seed_list)


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
    if len(ordered) <= 2:
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
    report = {
        "campaign": campaign_name,
        "lab_id": lab_id,
        "seed": seed,
        "timestamp": run_entry.get("timestamp"),
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
    norm_scores = [float(item["norm_score"]) for item in seed_reports if isinstance(item.get("norm_score"), (int, float))]
    runtimes = [float(item["runtime_s"]) for item in seed_reports if isinstance(item.get("runtime_s"), (int, float))]

    prompt_tokens = sum(int(item.get("prompt_tokens_total") or 0) for item in seed_reports)
    completion_tokens = sum(int(item.get("completion_tokens_total") or 0) for item in seed_reports)
    lm_tokens = sum(int(item.get("lm_total_tokens") or 0) for item in seed_reports)
    lm_time = sum(float(item.get("lm_inference_time_s") or 0.0) for item in seed_reports)

    return {
        "seed_count_completed": len(seed_reports),
        "mean_norm_score": round(_mean(norm_scores), 6),
        "std_norm_score": round(_stddev(norm_scores), 6),
        "total_runtime_s": round(sum(runtimes), 5),
        "mean_runtime_s": round(_mean(runtimes), 5),
        "total_prompt_tokens": int(prompt_tokens),
        "total_completion_tokens": int(completion_tokens),
        "total_lm_tokens": int(lm_tokens),
        "total_lm_inference_time_s": round(lm_time, 6),
    }


def _compute_average_norm_score(
    run_exports: List[Dict[str, Any]],
    seeds: List[Optional[int]],
) -> Optional[float]:
    """Compute average normalized score for completed seed runs."""
    if not run_exports:
        return None
    scores: List[float] = []
    valid_seeds = {str(seed) for seed in seeds}
    for entry in run_exports:
        sim = entry.get("sim")
        seed = entry.get("seed")
        if sim is None or seed is None:
            continue
        if str(seed) not in valid_seeds:
            continue
        gt_edges = getattr(sim, "gt_edges", None)
        if not gt_edges:
            continue
        score = getattr(sim, "score", None)
        if score is None:
            continue
        scores.append(float(score) / max(1, len(gt_edges)))
    if not scores:
        return None
    return sum(scores) / len(scores)


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
        return None, False

    _log("Launching simulation.")
    run_started = time.time()
    run_success = False
    try:
        sim.run_simulation()
        run_success = True
    except KeyboardInterrupt:
        _log("Interrupted by user (Ctrl+C).")
        raise
    except Exception as exc:
        _log(f"Simulation error: {exc}")
        _log(traceback.format_exc())
    finally:
        _safe_shutdown(sim)

    if not run_success:
        return None, False

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
        return None, True

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
        "logfile": _current_logfile_path(),
    }
    if context:
        run_entry["context"] = dict(context)
    _persist_results(run_entry)
    return run_entry, False


def run_all_seeds(config_path: str = core.CONFIG_PATH) -> List[Dict[str, Any]]:
    """Run all configured seeds in sequence for a single runtime config."""
    run_exports: List[Dict[str, Any]] = []
    config = _reload_config(config_path)
    include_timestamps = bool(config.get("logging", {}).get("include_timestamps", True))
    _set_shared_logger_timestamp_mode(include_timestamps)

    seeds = _seed_list_from_config(config)
    total_games = max(1, len(seeds))

    for game_index, seed in enumerate(seeds, start=1):
        try:
            run_entry, abort_requested = run_seed(
                seed,
                game_index,
                total_games,
                config_path=config_path,
                include_timestamps=include_timestamps,
            )
        except KeyboardInterrupt:
            break
        if run_entry:
            run_exports.append(run_entry)
        if abort_requested:
            break
    avg_norm = _compute_average_norm_score(run_exports, seeds)
    if avg_norm is not None:
        _log(f"Average normalized score over {len(seeds)} seeds: {avg_norm:.5f}")
    else:
        _log("Average normalized score: n/a (no completed runs)")
    return run_exports


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

    resolved = {
        "lab_id": lab_id,
        "rules_id": rules_id,
        "prompt_id": prompt_id,
        "drone_support_id": ds_id,
        "model_id": model_id,
        "fine_tuning_id": fine_tuning_id,
        "action_policy_id": action_policy_id,
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


def _run_campaign(master_config: Dict[str, Any], config_path: Path) -> int:
    campaign_cfg = master_config.get("campaign", {})
    if not isinstance(campaign_cfg, dict):
        raise ValueError("campaign must be a JSON object when provided.")

    config_dir = config_path.parent
    campaign_name = str(campaign_cfg.get("name") or campaign_cfg.get("campaign_name") or "campaign").strip()
    if not campaign_name:
        campaign_name = "campaign"

    output_dir = _resolve_path(
        config_dir,
        str(campaign_cfg.get("output_dir") or ""),
        f"campaign_runs/{campaign_name}",
    )
    fresh_output = bool(campaign_cfg.get("fresh_output", True))
    stop_on_error = bool(campaign_cfg.get("stop_on_error", True))

    logging_cfg = campaign_cfg.get("logging", {}) if isinstance(campaign_cfg.get("logging"), dict) else {}
    include_timestamps = bool(logging_cfg.get("include_timestamps", True))
    _set_shared_logger_timestamp_mode(include_timestamps)

    if fresh_output and output_dir.exists():
        shutil.rmtree(output_dir)
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

    global_seed_spec = campaign_cfg.get("seed_list")
    if global_seed_spec is None:
        global_seed_spec = master_config.get("simulation", {}).get("seed_list")
    global_seeds = _resolve_seed_spec(global_seed_spec)
    if not global_seeds:
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
        "config_path": str(config_path),
        "started_at": datetime.now().astimezone().isoformat(),
        "output_dir": str(output_dir),
        "results_csv": str(results_path),
        "lab_results_csv": str(lab_results_path),
        "include_timestamps": include_timestamps,
        "labs": [],
    }

    _append_text_log(campaign_log_path, f"Campaign '{campaign_name}' started.", include_timestamps)
    _append_text_log(campaign_log_path, f"Output dir: {output_dir}", include_timestamps)

    lab_rows: List[Dict[str, Any]] = []
    completed_lab_logfiles: Dict[str, str] = {}

    for index, lab_cfg_raw in enumerate(labs_cfg, start=1):
        if not isinstance(lab_cfg_raw, dict):
            continue
        if not bool(lab_cfg_raw.get("enabled", True)):
            continue

        lab_cfg = copy.deepcopy(lab_cfg_raw)
        lab_id = str(lab_cfg.get("id") or lab_cfg.get("lab_id") or f"L{index - 1}").strip()
        if not lab_id:
            lab_id = f"L{index - 1}"
        lab_label = str(lab_cfg.get("label") or lab_cfg.get("title") or lab_id).strip()
        lab_notes = str(lab_cfg.get("notes") or "").strip()

        lab_dir = labs_root / lab_id
        seeds_dir = lab_dir / "seed_reports"
        seed_logs_dir = lab_dir / "seed_logs"
        lab_log_path = lab_dir / "lab.log"
        lab_report_path = lab_dir / "lab_report.json"
        lab_simulation_log_path = lab_dir / "lab_simulation.log"

        _append_text_log(campaign_log_path, f"[{lab_id}] start", include_timestamps)
        _append_text_log(lab_log_path, f"Lab {lab_id} started.", include_timestamps)

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
                optuna_output_dir = lab_dir / "optuna"
                optuna_output_dir.mkdir(parents=True, exist_ok=True)
                optuna_env = dict(env_base)
                optuna_env["CORASAT_RESULTS_CSV"] = str(optuna_output_dir / "results_optuna.csv")
                optuna_env["CORASAT_LOG_DIR"] = str(optuna_output_dir / "logs")
                optuna_env["CORASAT_LOG_TIMESTAMPS"] = _bool_text(include_timestamps)

                optuna_args = _build_optuna_args(optuna_cfg)
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
                _run_lora_commands(lora_cfg, env_base, context=completed_lab_logfiles)

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

                    norm_score = seed_payload.get("norm_score")
                    if isinstance(norm_score, (int, float)):
                        _append_text_log(
                            lab_log_path,
                            f"Seed {seed} completed. norm_score={float(norm_score):.5f}",
                            include_timestamps,
                        )
                    else:
                        _append_text_log(
                            lab_log_path,
                            f"Seed {seed} completed. norm_score=n/a",
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
                "seed_reports": [item.get("seed_report_path", "") for item in seed_reports],
                "metrics": summary,
            }
            _write_json(lab_report_path, lab_report)

            lab_row = {
                "timestamp": datetime.now().astimezone().isoformat(),
                "lab_id": lab_id,
                "label": lab_label,
                "seed_count_planned": len(lab_seeds),
                "seed_count_completed": summary["seed_count_completed"],
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
            manifest["labs"].append(lab_report)

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
        "seed_count_completed": sum(int(row.get("seed_count_completed") or 0) for row in lab_rows),
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

    _append_text_log(campaign_log_path, "Campaign completed.", include_timestamps)
    _append_text_log(campaign_log_path, f"Manifest: {manifest_path}", include_timestamps)

    _log("Campaign completed.")
    _log(f"- Output dir: {output_dir}")
    _log(f"- Results: {results_path}")
    _log(f"- Lab results: {lab_results_path}")
    _log(f"- Campaign report: {manifest_path}")
    return 0


def _should_run_campaign(config: Dict[str, Any]) -> bool:
    campaign_cfg = config.get("campaign")
    if not isinstance(campaign_cfg, dict):
        return False
    labs = campaign_cfg.get("labs")
    return isinstance(labs, list) and len(labs) > 0


def main() -> int:
    """CLI entry point for single-run and campaign execution."""
    parser = argparse.ArgumentParser(description="Run Corasat simulation(s) from config.json.")
    parser.add_argument(
        "--config",
        default=core.CONFIG_PATH,
        help="Runtime/campaign config path (default: config.json).",
    )
    args = parser.parse_args()

    config_path = _resolve_config_path(str(args.config))
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    master_config = _read_json(config_path)

    try:
        if _should_run_campaign(master_config):
            return int(_run_campaign(master_config, config_path))
        run_all_seeds(str(config_path))
        return 0
    except KeyboardInterrupt:
        _log("Interrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
