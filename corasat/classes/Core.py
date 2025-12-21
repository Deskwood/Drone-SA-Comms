"""Core domain types and helpers for the Corasat simulation."""
from __future__ import annotations

import colorsys
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # Optional dependency during refactor.
    np = None

try:
    import torch
except Exception:  # Optional dependency during refactor.
    torch = None

try:
    import pygame
except Exception:  # Optional dependency during refactor.
    pygame = None

try:
    from classes.Exporter import LOGGER
except Exception:
    LOGGER = None

CONFIG_PATH = "config.json"


def _log(message: str) -> None:
    """Log via shared logger when available, otherwise print."""
    if LOGGER is not None:
        try:
            LOGGER.log(message)
            return
        except Exception:
            pass
    print(message)


def load_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    try:
        with open(config_path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except FileNotFoundError:
        _log(f"Missing config file: {config_path}")
        return {}
    except Exception as exc:
        _log(f"Failed to load config: {exc}")
        return {}


def reload_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Reload the module-level CONFIG mapping."""
    global CONFIG
    CONFIG = load_config(config_path)
    return CONFIG


CONFIG: Dict[str, Any] = load_config(CONFIG_PATH)

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


def direction_from_vector(vector: Tuple[int, int]) -> str:
    """Map a (dx, dy) vector to a named compass direction."""
    for direction, vec in DIRECTION_MAP.items():
        if vec == vector:
            return direction
    return str(vector)


def hsv_to_rgb255(h_deg: float, s: float, v: float) -> Tuple[int, int, int]:
    """Convert HSV (degrees, 0..1, 0..1) into RGB 0..255 integers."""
    r, g, b = colorsys.hsv_to_rgb(h_deg / 360.0, max(0, min(1, s)), max(0, min(1, v)))
    return (int(r * 255), int(g * 255), int(b * 255))


def set_global_seed(seed: Optional[int]) -> None:
    """Seed Python, NumPy, and Torch RNGs when available."""
    if seed is None:
        _log("No seed set.")
        return
    random.seed(seed)
    if np is not None:
        np.random.seed(seed % (2**32 - 1))
    if torch is not None:
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
    _log(f"Global seed set to {seed}.")


def chebyshev_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Return the Chebyshev distance between two points."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def cartesian_to_chess(pos: Tuple[int, int]) -> str:
    """Convert (x, y) into chess coordinate string (e.g., (0,0) -> 'a1')."""
    x, y = pos
    return f"{chr(ord('a') + x)}{y + 1}"


def chess_to_cartesian(value: str) -> Tuple[int, int]:
    """Convert chess coordinate string (e.g., 'e4') into (x, y)."""
    value = value.strip().lower()
    if len(value) < 2:
        raise ValueError(f"Invalid chess coordinate: {value}")
    return ord(value[0]) - ord("a"), int(value[1:]) - 1


def _edge_point_to_chess(point: Any) -> str:
    if isinstance(point, Waypoint):
        return point.to_chess()
    if isinstance(point, (tuple, list)) and len(point) == 2:
        return cartesian_to_chess((int(point[0]), int(point[1])))
    return str(point)


def format_edge(
    source_type: str,
    source_color: str,
    target_color: str,
    edge: Tuple[Any, Any],
) -> str:
    """Format a chess edge in notation (e.g., Qe4xg6)."""
    src, dst = edge
    piece_symbol = {
        "king": "K",
        "queen": "Q",
        "rook": "R",
        "bishop": "B",
        "knight": "N",
        "pawn": "",
    }.get(source_type, "?")
    capture_symbol = "x" if source_color != target_color else "-"
    return f"{piece_symbol}{_edge_point_to_chess(src)}{capture_symbol}{_edge_point_to_chess(dst)}"


def on_board(x: int, y: int, width: Optional[int] = None, height: Optional[int] = None) -> bool:
    """Return True when (x, y) is inside the board boundaries."""
    if width is None or height is None:
        board_cfg = CONFIG.get("board", {})
        width = int(board_cfg.get("width", 8))
        height = int(board_cfg.get("height", 8))
    return 0 <= x < width and 0 <= y < height


def load_figure_images() -> Dict[Tuple[str, str], Any]:
    """Load figure images from the configured directory."""
    if pygame is None:
        _log("pygame not available; skipping figure image load.")
        return {}
    images: Dict[Tuple[str, str], Any] = {}
    base_path = CONFIG.get("gui", {}).get("figure_image_dir", "figures")

    def try_load(path: str):
        return pygame.image.load(path) if os.path.exists(path) else None

    for color in COLORS:
        for figure_type in FIGURE_TYPES:
            candidates = [
                f"{color}{figure_type}.png",
                f"{color.capitalize()}{figure_type}.png",
                f"{color}{figure_type.capitalize()}.png",
                f"{color.capitalize()}{figure_type.capitalize()}.png",
            ]
            img = None
            for name in candidates:
                path = os.path.join(base_path, name)
                img = try_load(path)
                if img:
                    break
            if img:
                images[(color, figure_type)] = img
            elif LOGGER is not None:
                LOGGER.log(f"ERROR: Image not found for {color} {figure_type} in {base_path}")
    return images


class _Tile:
    """Board tile with figure occupancy, drone list, and targeting counts."""

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.targeted_by: Dict[str, int] = {"white": 0, "black": 0}
        self.figure: Optional[_Figure] = None
        self.drones: List[Any] = []

    def set_figure(self, figure: Optional["_Figure"]) -> None:
        """Place or remove a figure on this tile."""
        self.figure = figure

    def add_drone(self, drone: Any) -> None:
        """Add a drone to this tile if not already present."""
        if drone not in self.drones:
            self.drones.append(drone)

    def remove_drone(self, drone: Any) -> None:
        """Remove a drone from this tile if present."""
        if drone in self.drones:
            self.drones.remove(drone)

    def reset_targeted_by_amounts(self) -> None:
        """Clear cached attack/defense counts."""
        self.targeted_by = {"white": 0, "black": 0}

    def add_targeted_by_amount(self, color: str, amount: int = 1) -> None:
        """Increment the targeter count for the given color."""
        self.targeted_by[color] += amount


class _Figure:
    """Chess figure with cached attack and defense metadata."""

    def __init__(self, position: Tuple[int, int], color: str, figure_type: str):
        self.position = position
        self.color = color
        self.figure_type = figure_type
        self.defended_by = 0
        self.attacked_by = 0
        self.target_positions: List[Tuple[int, int]] = []

    def calculate_figure_targets(self, board: List[List["_Tile"]]) -> None:
        """Populate target_positions using chess movement rules."""
        self.target_positions = []
        width = int(CONFIG.get("board", {}).get("width", 8))
        height = int(CONFIG.get("board", {}).get("height", 8))

        def _on_board(nx: int, ny: int) -> bool:
            return 0 <= nx < width and 0 <= ny < height

        if self.figure_type in ("queen", "rook", "bishop"):
            if self.figure_type == "rook":
                directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            elif self.figure_type == "bishop":
                directions = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
            else:
                directions = [
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                    (1, 1),
                    (-1, -1),
                    (1, -1),
                    (-1, 1),
                ]
            for dx, dy in directions:
                x, y = self.position
                while True:
                    x += dx
                    y += dy
                    if not _on_board(x, y):
                        break
                    self.target_positions.append((x, y))
                    if board[x][y].figure is not None:
                        break

        elif self.figure_type == "knight":
            for dx, dy in [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]:
                x = self.position[0] + dx
                y = self.position[1] + dy
                if _on_board(x, y):
                    self.target_positions.append((x, y))

        elif self.figure_type == "king":
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                x = self.position[0] + dx
                y = self.position[1] + dy
                if _on_board(x, y):
                    self.target_positions.append((x, y))

        elif self.figure_type == "pawn":
            diagonals = [(1, 1), (-1, 1)] if self.color == "white" else [(1, -1), (-1, -1)]
            for dx, dy in diagonals:
                x = self.position[0] + dx
                y = self.position[1] + dy
                if _on_board(x, y):
                    self.target_positions.append((x, y))


class _Local_Tile:
    """Local knowledge of a board tile (type/color certainty and targeters)."""

    def __init__(self, true_figure: Optional["_Figure"]):
        self.true_figure = true_figure
        self.figure_type = "unknown"
        self.figure_color = "unknown"
        self.confirmed_targeter_count = 0

    def identify_true_figure_type_and_color(self) -> None:
        """Record exact type and color if a figure is present."""
        if self.true_figure is not None:
            self.figure_type = self.true_figure.figure_type
            self.figure_color = self.true_figure.color
        else:
            self.figure_type = "n/a"
            self.figure_color = "n/a"

    def identify_true_figure_color(self) -> None:
        """Record color only when a figure is present."""
        if self.true_figure is not None:
            if self.figure_type == "unknown":
                self.figure_type = "any figure"
            self.figure_color = self.true_figure.color
        else:
            self.figure_type = "n/a"
            self.figure_color = "n/a"

    def clear_targeter_count(self) -> None:
        """Reset the confirmed targeter count."""
        self.confirmed_targeter_count = 0

    def increase_targeter_count(self) -> None:
        """Increment the confirmed targeter count."""
        self.confirmed_targeter_count += 1


class Waypoint:
    """Board coordinate with optional turn and wait constraints."""

    def __init__(self, coordinate: Any, turn: Optional[int] = None, wait: Optional[int] = None):
        self.x: Optional[int] = None
        self.y: Optional[int] = None
        self.turn = turn
        self.wait = wait

        if isinstance(coordinate, str):
            s = coordinate.strip().lower()
            if len(s) < 2:
                raise ValueError(f"Invalid chess coordinate: {coordinate}")
            col, row = s[0], s[1:]
            self.x = ord(col) - ord("a")
            self.y = int(row) - 1
        elif isinstance(coordinate, (tuple, list)):
            if len(coordinate) != 2:
                raise ValueError("Cartesian coordinate must be a 2-tuple/list")
            self.x = int(coordinate[0])
            self.y = int(coordinate[1])
        else:
            raise ValueError("Coordinate must be chess notation or an (x, y) tuple")

    def to_chess(self) -> str:
        """Return chess notation (e.g., 'e4')."""
        if self.x is None or self.y is None:
            raise ValueError("Waypoint coordinates are not set")
        col = chr(ord("a") + self.x)
        row = str(self.y + 1)
        return f"{col}{row}"


class Sector:
    """Axis-aligned rectangular sector defined by two waypoints."""

    def __init__(self, upper_left: Optional[Waypoint] = None, lower_right: Optional[Waypoint] = None):
        if upper_left is None:
            upper_left = Waypoint((0, int(CONFIG.get("board", {}).get("height", 8)) - 1))
        if lower_right is None:
            lower_right = Waypoint((int(CONFIG.get("board", {}).get("width", 8)) - 1, 0))
        self.upper_left = upper_left
        self.lower_right = lower_right

    def equals(self, other: "Sector") -> bool:
        """Return True when two sectors have the same bounds."""
        return (
            self.upper_left.x == other.upper_left.x
            and self.upper_left.y == other.upper_left.y
            and self.lower_right.x == other.lower_right.x
            and self.lower_right.y == other.lower_right.y
        )

    def change(self, upper_left: Optional[Waypoint] = None, lower_right: Optional[Waypoint] = None) -> None:
        """Update sector boundaries."""
        if upper_left is not None:
            self.upper_left = upper_left
        if lower_right is not None:
            self.lower_right = lower_right
