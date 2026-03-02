"""Optuna-based tuner for Corasat decision_support parameters.

Example:
  python optuna_tuner.py --trials 60 --seeds 1,2,3 --study-name corasat_tuning
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import optuna


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config.json"

PARAM_SPACE = {
    "decision_support.scoring.move.waypoint_progress_bonus": (0.0, 2.0, 0.05),
    "decision_support.scoring.move.waypoint_delay_penalty": (-6.0, -0.1, 0.05),
    "decision_support.scoring.move.unknown_tile_bonus": (0.0, 3.0, 0.05),
    "decision_support.scoring.move.no_figures_left_behind_bonus": (0.0, 3.0, 0.05),
    "decision_support.scoring.move.neighborhood_potential": (0.0, 0.5, 0.01),
    "decision_support.scoring.move.figure_hint_bonus": (0.0, 2.0, 0.05),
    "decision_support.scoring.move.possible_target_bonus": (0.0, 2.5, 0.05),
    "decision_support.scoring.move.neighborhood_weight_any_figure": (0.0, 2.0, 0.05),
    "decision_support.scoring.move.neighborhood_weight_possible_target": (0.0, 2.0, 0.05),
    "decision_support.scoring.move.border_bonus": (0.0, 2.0, 0.05),
    "decision_support.scoring.move.cross_track_penalty_per_step_squared": (-4.0, 0.0, 0.05),
    "decision_support.scoring.move.sector_compliance_bonus": (0.0, 2.0, 0.05),
    "decision_support.scoring.move.revisit_penalty": (-3.0, 0.0, 0.05),
    "decision_support.scoring.broadcast.base_broadcast_value": (-2.0, 0.0, 0.05),
    "decision_support.scoring.broadcast.first_turn_coordination_bonus": (0.0, 6.0, 0.05),
    "decision_support.scoring.broadcast.last_turn_coordination_bonus": (0.0, 12.0, 0.05),
    "decision_support.scoring.wait.base_wait_value": (-3.0, 0.0, 0.05),
}


def _format_json(value: object, indent: int = 2, level: int = 0) -> str:
    if isinstance(value, dict):
        if not value:
            return "{}"
        pad = " " * (indent * level)
        lines = ["{"]
        items = list(value.items())
        for idx, (key, val) in enumerate(items):
            rendered = _format_json(val, indent, level + 1)
            comma = "," if idx < len(items) - 1 else ""
            lines.append(" " * (indent * (level + 1)) + json.dumps(key) + ": " + rendered + comma)
        lines.append(pad + "}")
        return "\n".join(lines)
    if isinstance(value, list):
        return json.dumps(value)
    return json.dumps(value)


def _load_config(config_path: Path) -> Dict[str, object]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def _write_config(cfg: Dict[str, object], config_path: Path) -> None:
    config_path.write_text(_format_json(cfg) + "\n", encoding="utf-8")


def _set_path(cfg: Dict[str, object], dotted_key: str, value: float) -> None:
    ref = cfg
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        ref = ref[key]
    ref[keys[-1]] = value


def _get_path(cfg: Dict[str, object], dotted_key: str) -> object:
    ref: object = cfg
    for key in dotted_key.split("."):
        if not isinstance(ref, dict) or key not in ref:
            raise KeyError(dotted_key)
        ref = ref[key]
    return ref


def _params_from_config(cfg: Dict[str, object]) -> Dict[str, float]:
    params: Dict[str, float] = {}
    missing: List[str] = []
    for key in PARAM_SPACE:
        try:
            value = _get_path(cfg, key)
        except KeyError:
            missing.append(key)
            continue
        try:
            params[key] = float(value)
        except (TypeError, ValueError):
            missing.append(key)
    if missing:
        raise KeyError(f"Missing or invalid PARAM_SPACE keys in enqueue config: {', '.join(missing)}")
    return params


def _parse_seeds(seed_text: str) -> List[int]:
    seeds: List[int] = []
    seen: set[int] = set()
    for token in seed_text.split(","):
        token = token.strip()
        if not token:
            continue

        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            step = 1 if end >= start else -1
            for seed in range(start, end + step, step):
                if seed in seen:
                    continue
                seen.add(seed)
                seeds.append(seed)
            continue

        seed = int(token)
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)
    return seeds


def _parse_seed_weights(seed_text: str) -> Dict[int, float]:
    weights: Dict[int, float] = {}
    if not seed_text:
        return weights
    for token in seed_text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid seed weight '{token}'. Expected format seed:weight.")
        seed_str, weight_str = token.split(":", 1)
        seed = int(seed_str.strip())
        weight = float(weight_str.strip())
        weights[seed] = weight
    return weights


def _weighted_mean(seed_scores: Dict[int, float], seed_weights: Dict[int, float]) -> float:
    total_weight = 0.0
    weighted_sum = 0.0
    for seed, score in seed_scores.items():
        weight = seed_weights.get(seed, 1.0)
        weighted_sum += weight * score
        total_weight += weight
    return weighted_sum / total_weight if total_weight else 0.0


def _run_seed_via_main(config_path: Path, seed: int, game_index: int, total_games: int) -> Tuple[str, Optional[float]]:
    # Import lazily to avoid side effects when running argparse/help paths.
    from main import run_seed

    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        run_entry, _abort_requested = run_seed(
            seed=seed,
            game_index=game_index,
            total_games=total_games,
            config_path=str(config_path),
            include_timestamps=False,
            context={"mode": "optuna"},
        )
    finally:
        logging.disable(previous_disable)
    if not isinstance(run_entry, dict):
        return "run_error", None

    status = str(run_entry.get("status") or "run_error")
    if status != "ok":
        return status, None

    sim = run_entry.get("sim")
    if sim is None:
        return "run_error", None

    gt_edges = getattr(sim, "gt_edges", None)
    score = getattr(sim, "score", None)
    if not gt_edges or score is None:
        return "run_error", None

    try:
        norm_score = float(score) / max(1, len(gt_edges))
    except Exception:
        return "run_error", None
    return "ok", norm_score


def _suggest_params(trial: optuna.Trial) -> Dict[str, float]:
    params: Dict[str, float] = {}
    for key, (low, high, step) in PARAM_SPACE.items():
        params[key] = round(trial.suggest_float(key, low, high, step=step), 2)
    return params


def _evaluate_trial(
    trial: optuna.Trial,
    base_config: Dict[str, object],
    config_path: Path,
    seeds: List[int],
    seed_weights: Dict[int, float],
    max_rounds: Optional[int],
    log_writer: csv.DictWriter,
    weight_display: str,
) -> float:
    params = _suggest_params(trial)
    seed_scores: Dict[int, float] = {}

    for step_idx, seed in enumerate(seeds):
        cfg = copy.deepcopy(base_config)
        cfg.setdefault("simulation", {})["seed_list"] = [seed]
        cfg["simulation"]["use_gui"] = False
        if max_rounds is not None:
            cfg["simulation"]["max_rounds"] = max_rounds
        for key, value in params.items():
            _set_path(cfg, key, value)
        _write_config(cfg, config_path)

        status, score = _run_seed_via_main(
            config_path,
            seed=seed,
            game_index=step_idx + 1,
            total_games=len(seeds),
        )
        if status != "ok" or score is None:
            score = 0.0
        seed_scores[seed] = score

        interim_mean = _weighted_mean(seed_scores, seed_weights)
        trial.report(interim_mean, step=step_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_score = _weighted_mean(seed_scores, seed_weights)

    row = {
        "trial": trial.number,
        "value": f"{mean_score:.5f}",
        "seed_scores": "; ".join(f"{seed}:{score:.5f}" for seed, score in seed_scores.items()),
        "seed_weights": weight_display,
    }
    for key in PARAM_SPACE:
        row[key] = params[key]
    log_writer.writerow(row)

    return mean_score


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optuna tuner for Corasat decision_support parameters.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Runtime config path to tune (default: corasat/config.json).",
    )
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials.")
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Comma-separated seed list.")
    parser.add_argument("--study-name", type=str, default="corasat_tuning", help="Optuna study name.")
    parser.add_argument("--storage", type=str, default="", help="Optuna storage URL (leave empty for in-memory).")
    parser.add_argument("--sampler-seed", type=int, default=0, help="Random seed for the sampler.")
    parser.add_argument("--max-rounds", type=int, default=0, help="Override simulation max_rounds during tuning.")
    parser.add_argument(
        "--seed-weights",
        type=str,
        default="",
        help="Optional comma-separated seed:weight pairs, e.g. 5:3,8:3.",
    )
    parser.add_argument(
        "--enqueue-config",
        type=str,
        default="",
        help="Optional config.json path to enqueue as the first trial.",
    )
    parser.add_argument("--no-prune", action="store_true", help="Disable pruning.")
    parser.add_argument("--no-apply-best", action="store_true", help="Do not write best params back to config.json.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for optuna trial CSV logs (relative to corasat root by default).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable Optuna info logging.")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (BASE_DIR / config_path).resolve()
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    seeds = _parse_seeds(args.seeds)
    if not seeds:
        raise SystemExit("No seeds provided.")

    seed_weights = _parse_seed_weights(args.seed_weights)
    weight_display = "; ".join(f"{seed}:{seed_weights[seed]:.2f}" for seed in sorted(seed_weights))

    optuna.logging.set_verbosity(optuna.logging.INFO if args.verbose else optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1) if not args.no_prune else None

    storage = args.storage or None
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=bool(storage),
    )

    original_cfg = _load_config(config_path)
    max_rounds = args.max_rounds if args.max_rounds > 0 else None

    if args.enqueue_config:
        enqueue_path = Path(args.enqueue_config).resolve()
        enqueue_cfg = json.loads(enqueue_path.read_text(encoding="utf-8"))
        enqueue_params = _params_from_config(enqueue_cfg)
        study.enqueue_trial(enqueue_params)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (BASE_DIR / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = output_dir / f"optuna_runs_{timestamp}.csv"
    fieldnames = ["trial", "value", "seed_scores", "seed_weights"] + list(PARAM_SPACE.keys())

    best_params: Dict[str, float] = {}
    best_value: Optional[float] = None

    try:
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            def _objective(trial: optuna.Trial) -> float:
                value = _evaluate_trial(
                    trial,
                    original_cfg,
                    config_path,
                    seeds,
                    seed_weights,
                    max_rounds,
                    writer,
                    weight_display,
                )
                nonlocal best_params, best_value
                if best_value is None or value > best_value:
                    best_value = value
                    best_params = {key: trial.params[key] for key in PARAM_SPACE.keys()}
                return value

            study.optimize(_objective, n_trials=args.trials)
    finally:
        cfg = _load_config(config_path)
        cfg.setdefault("simulation", {})["seed_list"] = original_cfg.get("simulation", {}).get("seed_list", [])
        cfg["simulation"]["use_gui"] = original_cfg.get("simulation", {}).get("use_gui", True)
        if not args.no_apply_best and best_params:
            for key, value in best_params.items():
                _set_path(cfg, key, float(value))
        _write_config(cfg, config_path)

    best_display = "n/a" if best_value is None else f"{best_value:.5f}"
    print(f"Best objective: {best_display}")
    print(f"Log saved to: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
