"""Entry point for Corasat simulation runs.

Loads configuration from config.json, iterates over the seed list, and runs the
simulation loop. This module intentionally avoids CLI parameters for now; the
config file is the single source of runtime settings while refactoring.
"""
from __future__ import annotations

from datetime import datetime
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
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

RUN_EXPORTS: List[Dict[str, Any]] = []


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


def _reload_config(config_path: str = core.CONFIG_PATH) -> Dict[str, Any]:
    """Reload the shared Core CONFIG from disk."""
    _log(f"Load Config: {config_path}")
    return core.reload_config(config_path)


def _seed_list_from_config(config: Dict[str, Any]) -> List[Optional[int]]:
    """Extract the simulation seed list and apply a fallback when missing."""
    seed_list = config.get("simulation", {}).get("seed_list", [])
    if not seed_list:
        _log("No seed list found in config; defaulting to [0].")
        return [0]
    if len(seed_list) > 10:
        _log(f"Seed list: {seed_list[:3]} .. {seed_list[-3:]} (total {len(seed_list)})")
    else:
        _log(f"Seed list: {seed_list}")
    return list(seed_list)


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


def _create_simulation(game_index: int, total_games: int):
    """Construct a Simulation instance, returning None on failure."""
    try:
        return Simulation(game_index=game_index, total_games=total_games)
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


def run_seed(seed: Optional[int], game_index: int, total_games: int) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Run a single seed and return (run_entry, abort_requested)."""
    _log(f"==== Running seed {seed} (game {game_index}/{total_games}) ====")
    _set_global_seed(seed)
    config = _reload_config(core.CONFIG_PATH)

    sim = _create_simulation(game_index=game_index, total_games=total_games)
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

    if getattr(sim, "_abort_requested", False):
        _log("GUI closed by user - stopping remaining seeds.")
        return None, True

    runtime_s = time.time() - run_started
    run_entry = {
        "sim": sim,
        "config": config,
        "seed": seed,
        "runtime_s": runtime_s,
        "timestamp": datetime.now().isoformat(),
    }
    RUN_EXPORTS.append(run_entry)
    _persist_results(run_entry)
    return run_entry, False


def run_all_seeds(config_path: str = core.CONFIG_PATH) -> List[Dict[str, Any]]:
    """Run all configured seeds in sequence."""
    config = _reload_config(config_path)
    seeds = _seed_list_from_config(config)
    total_games = max(1, len(seeds))

    for game_index, seed in enumerate(seeds, start=1):
        try:
            _, abort_requested = run_seed(seed, game_index, total_games)
        except KeyboardInterrupt:
            break
        if abort_requested:
            break
    avg_norm = _compute_average_norm_score(RUN_EXPORTS, seeds)
    if avg_norm is not None:
        _log(f"Average normalized score over {len(seeds)} seeds: {avg_norm:.5f}")
    else:
        _log("Average normalized score: n/a (no completed runs)")
    return RUN_EXPORTS


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


def main() -> None:
    """CLI entry point for running the configured simulation seeds."""
    run_all_seeds(core.CONFIG_PATH)


if __name__ == "__main__":
    main()
