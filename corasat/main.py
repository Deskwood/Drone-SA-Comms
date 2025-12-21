# Main, Imports,General Constants, Config Loading
# =========================

# Imports
from __future__ import annotations
import os, json, time, random, logging, colorsys, nbformat, pygame, csv, hashlib, uuid, subprocess, math, torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
from nbconvert.exporters import PythonExporter
from ollama import chat as ollama_chat

from classes.Exporter import LOGGER
from classes.Simulation import Simulation

# General Constants
COLORS = ["white", "black"]
FIGURE_TYPES = ["king", "queen", "rook", "bishop", "knight", "pawn"]
DIRECTION_MAP: Dict[str, Tuple[int, int]] = {
    "north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0),
    "northeast": (1, 1), "northwest": (-1, 1), "southeast": (1, -1), "southwest": (-1, -1)
}
VALID_DIRECTIONS = set(DIRECTION_MAP.keys())
FIGURE_IMAGES: Dict[Tuple[str,str], pygame.Surface] = {}


# Configuration Loading
def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from a JSON file."""
    LOGGER.log(f"Load Config: {config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing config file: {config_path}")

    return cfg
CONFIG = load_config("config.json")


# Main
if __name__ == "__main__":
    seed_list = CONFIG.get("simulation", {}).get("seed_list", [])
    if not seed_list:
        seed_list = [0]
        LOGGER.log("No seed list found in config; defaulting to [0].")
    elif len(seed_list) > 10:
        LOGGER.log(f"Seed list: {seed_list[:3]} .. {seed_list[-3:]} (total {len(seed_list)})")
    else:
        LOGGER.log(f"Seed list: {seed_list}")
    seeds = list(seed_list or [None])
    total_games = max(1, len(seeds))
    for game_index, seed in enumerate(seeds, start=1):
        LOGGER.log(f"==== Running seed {seed} (game {game_index}/{total_games}) ====")
        try:
            set_global_seed(seed)
        except Exception:
            pass
        CONFIG = load_config("config.json")
        try:
            LOGGER.log("Launching simulation.")
            run_started = time.time()
            SIM = Simulation(game_index=game_index, total_games=total_games)
            SIM.run_simulation()
            runtime_s = time.time() - run_started
            finished_at = datetime.now().isoformat()
            if getattr(SIM, "_abort_requested", False):
                LOGGER.log("GUI closed by user - stopping remaining seeds.")
                break
            run_entry = {
                "sim": SIM,
                "config": CONFIG,
                "seed": seed,
                "runtime_s": runtime_s,
                "timestamp": finished_at,
            }
            RUN_EXPORTS.append(run_entry)
            try:
                persist_run_results([run_entry])
            except Exception as exc:
                LOGGER.log(f"Failed to update results.csv: {exc}")
        except KeyboardInterrupt:
            LOGGER.log("Interrupted by user (Ctrl+C).")
            try:
                SIM.shutdown()
            except Exception:
                pass
            break
        except Exception as exc:
            LOGGER.log(f"Simulation error: {exc}")
        finally:
            try:
                SIM.shutdown()
            except Exception:
                pass