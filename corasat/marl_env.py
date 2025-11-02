"""
Multi-agent reinforcement learning environment for the CORASAT drone scenario.

The goal of the drones matches the original LLM-driven simulation in
``run_simulation.ipynb``: explore the chess board, gather intelligence about
pieces, and report attacking edges.  This module re-implements the domain
mechanics in a framework-friendly form so that multiple learning agents can be
trained without relying on language models.

Key features:
* Action space identical to the LLM simulation (wait, broadcast, move in eight
  compass directions).
* Observations expose the same information the language agents received,
  translated into numerical tensors suitable for RL algorithms.
* Environment returns per-agent rewards based on newly discovered edges and
  penalises inefficient behaviour such as illegal moves or excessive waiting.
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = THIS_DIR / "config.json"


def load_config(config_path: Optional[Path | str] = None) -> dict:
    """Load the simulation configuration."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Domain constants and helpers (mirrors run_simulation.ipynb)
# ---------------------------------------------------------------------------

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

# Fixed ordering used for action and observation encoding.
ORDERED_DIRECTIONS: Tuple[str, ...] = (
    "north",
    "south",
    "east",
    "west",
    "northeast",
    "northwest",
    "southeast",
    "southwest",
)

RECENT_POSITION_HISTORY = 4
ACTION_LOOKUP: Tuple[Tuple[str, Optional[str]], ...] = (
    ("wait", None),
    ("broadcast", None),
    ("move", "north"),
    ("move", "south"),
    ("move", "east"),
    ("move", "west"),
    ("move", "northeast"),
    ("move", "northwest"),
    ("move", "southeast"),
    ("move", "southwest"),
)

ACTION_NAMES = tuple(
    "wait" if kind == "wait" else (kind if direction is None else f"{kind}:{direction}")
    for kind, direction in ACTION_LOOKUP
)


COLOR_ENCODING = {
    "unknown": 0,
    "n/a": 1,
    "white": 2,
    "black": 3,
}
COLOR_SCALE = max(COLOR_ENCODING.values()) or 1


TYPE_ENCODING = {
    "unknown": 0,
    "n/a": 1,
    "any figure": 2,
    "king": 3,
    "queen": 4,
    "rook": 5,
    "bishop": 6,
    "knight": 7,
    "pawn": 8,
    "a possible target": 9,
}
TYPE_SCALE = max(TYPE_ENCODING.values()) or 1


def cartesian_to_chess(pos: Tuple[int, int]) -> str:
    x, y = pos
    return f"{chr(ord('a') + x)}{y + 1}"


def chess_to_cartesian(value: str) -> Tuple[int, int]:
    value = value.strip().lower()
    if len(value) < 2:
        raise ValueError(f"Invalid chess coordinate: {value}")
    return ord(value[0]) - ord("a"), int(value[1:]) - 1


def format_edge(
    source_type: str,
    source_color: str,
    target_color: str,
    edge: Tuple[Tuple[int, int], Tuple[int, int]],
) -> str:
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
    return f"{piece_symbol}{cartesian_to_chess(src)}{capture_symbol}{cartesian_to_chess(dst)}"


def on_board(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


# ---------------------------------------------------------------------------
# Board primitives
# ---------------------------------------------------------------------------


@dataclass
class Tile:
    x: int
    y: int
    targeted_by: Dict[str, int] = field(default_factory=lambda: {"white": 0, "black": 0})
    figure: Optional["Figure"] = None
    drones: List["DroneState"] = field(default_factory=list)

    def add_drone(self, drone: "DroneState") -> None:
        if drone not in self.drones:
            self.drones.append(drone)

    def remove_drone(self, drone: "DroneState") -> None:
        if drone in self.drones:
            self.drones.remove(drone)

    def reset_targeted_by(self) -> None:
        self.targeted_by = {"white": 0, "black": 0}

    def set_figure(self, figure: Optional["Figure"]) -> None:
        self.figure = figure


@dataclass
class Figure:
    position: Tuple[int, int]
    color: str
    figure_type: str
    defended_by: int = 0
    attacked_by: int = 0
    target_positions: List[Tuple[int, int]] = field(default_factory=list)

    def calculate_targets(self, board: List[List[Tile]], width: int, height: int) -> None:
        # Walk directionally appropriate rays to determine squares threatened by this figure.
        self.target_positions = []

        def _on_board(nx: int, ny: int) -> bool:
            return 0 <= nx < width and 0 <= ny < height

        if self.figure_type in ("queen", "rook", "bishop"):
            if self.figure_type == "rook":
                directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            elif self.figure_type == "bishop":
                directions = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
            else:  # queen
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
            for dx, dy in [
                (2, 1),
                (2, -1),
                (-2, 1),
                (-2, -1),
                (1, 2),
                (1, -2),
                (-1, 2),
                (-1, -2),
            ]:
                nx = self.position[0] + dx
                ny = self.position[1] + dy
                if _on_board(nx, ny):
                    self.target_positions.append((nx, ny))

        elif self.figure_type == "king":
            for dx, dy in [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (-1, -1),
                (1, -1),
                (-1, 1),
            ]:
                nx = self.position[0] + dx
                ny = self.position[1] + dy
                if _on_board(nx, ny):
                    self.target_positions.append((nx, ny))

        elif self.figure_type == "pawn":
            diagonals = [(1, 1), (-1, 1)] if self.color == "white" else [(1, -1), (-1, -1)]
            for dx, dy in diagonals:
                nx = self.position[0] + dx
                ny = self.position[1] + dy
                if _on_board(nx, ny):
                    self.target_positions.append((nx, ny))


# ---------------------------------------------------------------------------
# Drone state
# ---------------------------------------------------------------------------


class DroneState:
    """Holds the local knowledge and behaviour of a single drone."""

    def __init__(self, env: "CorasatMultiAgentEnv", drone_id: int, position: Tuple[int, int]):
        self.env = env
        self.id = drone_id
        self.position = position
        self.memory: List[str] = []
        self.local_board: Dict[str, Dict[str, str]] = {}
        self.identified_edges: List[str] = []
        self.visited_tiles: set[Tuple[int, int]] = set()
        self.position_history: Deque[Tuple[int, int]] = deque(maxlen=RECENT_POSITION_HISTORY)
        self.position_history.append(position)
        self.reset_local_board()

    # ------------------------------------------------------------------ utils
    def reset_local_board(self) -> None:
        self.local_board = {}
        for x in range(self.env.width):
            for y in range(self.env.height):
                key = cartesian_to_chess((x, y))
                self.local_board[key] = {"color": "unknown", "type": "unknown"}
        self.identified_edges = []
        self.visited_tiles = {self.position}
        self.memory = [f"VISITED:{self.position[0]},{self.position[1]}"]
        self.position_history.clear()
        self.position_history.append(self.position)

    def legal_moves(self) -> List[str]:
        # Filter out moves that would leave the board — used for both RL and GUI logic.
        moves: List[str] = []
        for direction, (dx, dy) in DIRECTION_MAP.items():
            nx, ny = self.position[0] + dx, self.position[1] + dy
            if on_board(nx, ny, self.env.width, self.env.height):
                moves.append(direction)
        return moves

    def record_position(self, position: Tuple[int, int]) -> None:
        # Append the location the drone just visited to its rolling history buffer.
        self.position_history.append(position)

    # -------------------------------------------------------------- observations
    def encode_observation(self) -> np.ndarray:
        """Return the fixed-length observation vector exposed to the shared RL policy."""
        env = self.env
        round_norm = env.round / max(1, env.max_rounds)
        score_norm = env.score / max(1, len(env.gt_edges))
        visited_ratio = len(self.visited_tiles) / max(1, env.total_tiles)
        drone_score = env.drone_stats[self.id]
        correct_norm = drone_score["correct"] / max(1, len(env.gt_edges))
        identified_ratio = len(self.identified_edges) / max(1, len(env.gt_edges))

        x_norm = self.position[0] / max(1, env.width - 1)
        y_norm = self.position[1] / max(1, env.height - 1)
        legal_mask = [
            1.0 if direction in self.legal_moves() else 0.0 for direction in ORDERED_DIRECTIONS
        ]
        tile = env.board[self.position[0]][self.position[1]]
        broadcast_available = 1.0 if len(tile.drones) > 1 else 0.0

        total_drones = max(1, env.num_drones)
        relative_index = (self.id - 1) / max(1, total_drones - 1)

        history_coords: List[float] = []
        norm_width = max(1, env.width - 1)
        norm_height = max(1, env.height - 1)
        for hx, hy in self.position_history:
            history_coords.append(hx / norm_width)
            history_coords.append(hy / norm_height)
        while len(history_coords) < RECENT_POSITION_HISTORY * 2:
            history_coords.append(0.0)

        movement_context: List[float] = []
        for direction in ORDERED_DIRECTIONS:
            dx, dy = DIRECTION_MAP[direction]
            tx, ty = self.position[0] + dx, self.position[1] + dy
            if not on_board(tx, ty, env.width, env.height):
                movement_context.extend([1.0, 0.0, 0.0, 0.0])  # illegal/off-board
                continue
            key = cartesian_to_chess((tx, ty))
            info = self.local_board[key]
            tile_type = info["type"]
            if tile_type == "unknown":
                movement_context.extend([0.0, 1.0, 0.0, 0.0])  # unknown / unexplored
            elif tile_type in FIGURE_TYPES or tile_type == "any figure":
                movement_context.extend([0.0, 0.0, 1.0, 0.0])  # known figure
            elif tile_type == "a possible target":
                movement_context.extend([0.0, 1.0, 0.0, 0.0])  # treat as unknown target
            else:
                # considered empty / safe (n/a)
                movement_context.extend([0.0, 0.0, 0.0, 1.0])

        board_features: List[float] = []
        for y in range(env.height):
            for x in range(env.width):
                key = cartesian_to_chess((x, y))
                info = self.local_board[key]
                occupancy = info["type"]
                if occupancy in FIGURE_TYPES or occupancy == "any figure":
                    occ_value = 1.0
                elif occupancy == "n/a":
                    occ_value = -1.0
                elif occupancy == "a possible target":
                    occ_value = 0.5
                else:
                    occ_value = 0.0
                board_features.append(COLOR_ENCODING.get(info["color"], 0) / COLOR_SCALE)
                board_features.append(TYPE_ENCODING.get(info["type"], 0) / TYPE_SCALE)
                board_features.append(occ_value)

        header = [
            round_norm,
            score_norm,
            visited_ratio,
            correct_norm,
            identified_ratio,
            x_norm,
            y_norm,
            broadcast_available,
            relative_index,
        ]
        observation = np.array(
            header + history_coords + legal_mask + movement_context + board_features,
            dtype=np.float32,
        )
        return observation

    # ----------------------------------------------------------------- gameplay
    def update_board_report(self) -> Dict[str, object]:
        """Refresh observable tiles and return discovery metrics."""
        env = self.env
        sx, sy = self.position
        tile = env.board[sx][sy]
        key = cartesian_to_chess((sx, sy))
        resolved_tiles = 0
        new_possible_targets = 0

        def _count_resolution(previous: Dict[str, str], new: Dict[str, str]) -> int:
            prev_unknown = previous["type"] == "unknown" or previous["color"] == "unknown"
            newly_known = new["type"] not in ("unknown", "a possible target") or new["color"] not in (
                "unknown",
            )
            # Treat "any figure" / "n/a" as resolved knowledge
            newly_known = newly_known or new["type"] in ("any figure", "n/a")
            return 1 if prev_unknown and newly_known else 0

        def _register_update(board_key: str, new_info: Dict[str, str]) -> None:
            nonlocal resolved_tiles, new_possible_targets
            previous = dict(self.local_board[board_key])
            resolved_tiles += _count_resolution(previous, new_info)
            if previous["type"] != "a possible target" and new_info["type"] == "a possible target":
                new_possible_targets += 1
            self.local_board[board_key] = new_info

        if tile.figure:
            _register_update(
                key,
                {
                    "color": tile.figure.color,
                    "type": tile.figure.figure_type,
                },
            )
        else:
            _register_update(key, {"color": "n/a", "type": "n/a"})

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = sx + dx, sy + dy
                if not on_board(nx, ny, env.width, env.height):
                    continue
                neighbor_tile = env.board[nx][ny]
                neighbor_key = cartesian_to_chess((nx, ny))
                if neighbor_tile.figure:
                    if self.local_board[neighbor_key]["type"] == "unknown":
                        _register_update(
                            neighbor_key,
                            {
                                "color": neighbor_tile.figure.color,
                                "type": "any figure",
                            },
                        )
                else:
                    _register_update(neighbor_key, {"color": "n/a", "type": "n/a"})

        previous_edges = set(self.identified_edges)
        self._identify_edges()
        new_edges = [edge for edge in self.identified_edges if edge not in previous_edges]

        visited_token = f"VISITED:{sx},{sy}"
        if visited_token not in self.memory:
            self.memory.append(visited_token)
        self.visited_tiles.add((sx, sy))
        return {
            "edges": new_edges,
            "resolved_tiles": resolved_tiles,
            "new_possible_targets": new_possible_targets,
        }

    def _identify_edges(self) -> None:
        env = self.env
        width, height = env.width, env.height
        for bx in range(width):
            for by in range(height):
                board_key = cartesian_to_chess((bx, by))
                figure_type = self.local_board[board_key]["type"]
                figure_color = self.local_board[board_key]["color"]
                if figure_type not in FIGURE_TYPES:
                    continue

                if figure_type in ("queen", "rook", "bishop"):
                    if figure_type == "rook":
                        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    elif figure_type == "bishop":
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
                    slider = True
                else:
                    slider = False
                    if figure_type == "knight":
                        directions = [
                            (2, 1),
                            (2, -1),
                            (-2, 1),
                            (-2, -1),
                            (1, 2),
                            (1, -2),
                            (-1, 2),
                            (-1, -2),
                        ]
                    elif figure_type == "king":
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
                    elif figure_type == "pawn":
                        directions = (
                            [(1, 1), (-1, 1)]
                            if figure_color == "white"
                            else [(1, -1), (-1, -1)]
                        )
                    else:
                        directions = []

                for dx, dy in directions:
                    tx, ty = bx, by
                    while True:
                        tx += dx
                        ty += dy
                        if not on_board(tx, ty, width, height):
                            break
                        target_key = cartesian_to_chess((tx, ty))
                        target_info = self.local_board[target_key]
                        if target_info["type"] == "unknown":
                            self.local_board[target_key]["type"] = "a possible target"
                        if target_info["type"] == "a possible target":
                            break
                        if target_info["type"] in FIGURE_TYPES or target_info["type"] == "any figure":
                            edge = format_edge(
                                figure_type,
                                figure_color,
                                target_info["color"],
                                ((bx, by), (tx, ty)),
                            )
                            if edge not in self.identified_edges and (bx, by) != (tx, ty):
                                self.identified_edges.append(edge)
                            break
                        if not slider:
                            break

    def provide_intelligence(self, target: "DroneState") -> Dict[str, object]:
        """Share knowledge with another drone and report discovery metrics."""
        resolved_tiles = 0
        new_possible_targets = 0

        def _count_resolution(previous: Dict[str, str], new: Dict[str, str]) -> int:
            prev_unknown = previous["type"] == "unknown" or previous["color"] == "unknown"
            newly_known = new["type"] not in ("unknown", "a possible target") or new["color"] not in (
                "unknown",
            )
            newly_known = newly_known or new["type"] in ("any figure", "n/a")
            return 1 if prev_unknown and newly_known else 0

        for key, info in self.local_board.items():
            target_record = target.local_board.get(key, {"color": "unknown", "type": "unknown"})
            previous = dict(target_record)
            if target_record["color"] == "unknown" and info["color"] != "unknown":
                target.local_board[key]["color"] = info["color"]
            if target_record["type"] == "unknown" and info["type"] != "unknown":
                target.local_board[key]["type"] = info["type"]
            resolved_tiles += _count_resolution(previous, target.local_board[key])
            if previous["type"] != "a possible target" and target.local_board[key]["type"] == "a possible target":
                new_possible_targets += 1
        previous_edges = set(target.identified_edges)
        target._identify_edges()
        new_edges = [edge for edge in target.identified_edges if edge not in previous_edges]
        return {
            "edges": new_edges,
            "resolved_tiles": resolved_tiles,
            "new_possible_targets": new_possible_targets,
        }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class CorasatMultiAgentEnv:
    """Lightweight multi-agent environment for MARL experiments."""

    def __init__(
        self,
        config_path: Optional[Path | str] = None,
        reward_config: Optional[Dict[str, float]] = None,
    ):
        self.config = load_config(config_path)
        self.reward_config = {
            "step_penalty": -0.01,
            "illegal_move_penalty": -0.2,
            "wait_penalty": -0.05,
            "broadcast_penalty": -0.05,
            "broadcast_assist_share": 0.5,
            "new_tile_bonus": 0.05,
            "info_discovery_bonus": 0.02,
            "possible_target_bonus": 0.03,
            "intel_share_bonus": 0.02,
        }
        if reward_config:
            self.reward_config.update(reward_config)

        self.width = self.config["board"]["width"]
        self.height = self.config["board"]["height"]
        self.total_tiles = self.width * self.height

        sim_cfg = self.config["simulation"]
        self.num_drones = int(sim_cfg["num_drones"])
        self.max_rounds = int(sim_cfg["max_rounds"])
        self.random_seed = sim_cfg.get("random_seed")
        self.randomize_figures = bool(
            sim_cfg.get("randomize_figures", False) or self.random_seed is not None
        )

        self.board: List[List[Tile]] = []
        self.figures: List[Figure] = []
        self.gt_edges: List[str] = []
        self.agent_ids: Tuple[int, ...] = tuple(range(1, self.num_drones + 1))

        self.round: int = 1
        self.score: float = 0.0
        self.reported_edges: List[str] = []
        self.drone_stats: Dict[int, Dict[str, int]] = {
            drone_id: {"correct": 0, "false": 0} for drone_id in self.agent_ids
        }

        self.drones: Dict[int, DroneState] = {}
        self.start_position: Tuple[int, int] = (0, 0)
        self.observation_size: Optional[int] = None

        # Internal RNG used for figure placement. Seeded at reset().
        self._rng = random.Random()

    # ------------------------------------------------------------------ helpers
    def _initial_seed(self, seed: Optional[int]) -> None:
        if seed is not None:
            self._rng.seed(seed)
        elif self.random_seed is not None:
            self._rng.seed(self.random_seed)
        else:
            self._rng.seed()

    def _clone_figures_config(self) -> dict:
        return copy.deepcopy(self.config.get("figures", {}))

    def _randomize_figures(self, figures_cfg: dict) -> dict:
        all_tiles = [(x, y) for x in range(self.width) for y in range(self.height)]
        self._rng.shuffle(all_tiles)
        cursor = 0
        requested: List[Tuple[str, str, int]] = []
        for color in COLORS:
            for figure_type in FIGURE_TYPES:
                entries = figures_cfg.get(color, {}).get(figure_type, [])
                if entries:
                    count = len(entries)
                else:
                    if figure_type in ("king", "queen"):
                        count = 1
                    elif figure_type in ("rook", "bishop", "knight"):
                        count = 2
                    else:
                        count = 3
                requested.append((color, figure_type, count))

        randomized: Dict[str, Dict[str, List[List[int]]]] = {
            color: {ftype: [] for ftype in FIGURE_TYPES} for color in COLORS
        }
        for color, figure_type, count in requested:
            picks = all_tiles[cursor : cursor + count]
            cursor += count
            randomized[color][figure_type] = [[px, py] for px, py in picks]
        return randomized

    # ---------------------------------------------------------------- environment
    def reset(self, seed: Optional[int] = None) -> Dict[int, np.ndarray]:
        """Reset the environment and return initial observations."""
        self._initial_seed(seed)
        self.round = 1
        self.score = 0.0
        self.reported_edges = []
        for stats in self.drone_stats.values():
            stats["correct"] = 0
            stats["false"] = 0

        figures_cfg = self._clone_figures_config()
        if self.randomize_figures:
            figures_cfg = self._randomize_figures(figures_cfg)

        self._build_board(figures_cfg)
        self._create_drones()

        observations = {
            drone_id: drone.encode_observation() for drone_id, drone in self.drones.items()
        }
        if observations:
            self.observation_size = len(next(iter(observations.values())))
        return observations

    def _build_board(self, figures_cfg: dict) -> None:
        self.board = [[Tile(x, y) for y in range(self.height)] for x in range(self.width)]
        self.figures = []

        for color in COLORS:
            for figure_type in FIGURE_TYPES:
                for position in figures_cfg.get(color, {}).get(figure_type, []):
                    figure = Figure(tuple(position), color, figure_type)
                    self.figures.append(figure)
                    self.board[figure.position[0]][figure.position[1]].set_figure(figure)

        for column in self.board:
            for tile in column:
                tile.reset_targeted_by()

        for figure in self.figures:
            figure.calculate_targets(self.board, self.width, self.height)

        for figure in self.figures:
            for tx, ty in figure.target_positions:
                target_tile = self.board[tx][ty]
                target_tile.targeted_by[figure.color] += 1

        for figure in self.figures:
            tile = self.board[figure.position[0]][figure.position[1]]
            if figure.color == "white":
                figure.defended_by = tile.targeted_by["white"]
                figure.attacked_by = tile.targeted_by["black"]
            else:
                figure.defended_by = tile.targeted_by["black"]
                figure.attacked_by = tile.targeted_by["white"]

        self.gt_edges = []
        for figure in self.figures:
            for tx, ty in figure.target_positions:
                target_figure = self.board[tx][ty].figure
                if target_figure is not None:
                    edge = format_edge(
                        figure.figure_type,
                        figure.color,
                        target_figure.color,
                        (figure.position, target_figure.position),
                    )
                    self.gt_edges.append(edge)

        white_king = next(
            (fig for fig in self.figures if fig.color == "white" and fig.figure_type == "king"),
            None,
        )
        self.start_position = white_king.position if white_king else (0, 0)

    def _create_drones(self) -> None:
        self.drones = {}
        for column in self.board:
            for tile in column:
                tile.drones.clear()

        for drone_id in self.agent_ids:
            drone = DroneState(self, drone_id, self.start_position)
            self.drones[drone_id] = drone
            self.board[self.start_position[0]][self.start_position[1]].add_drone(drone)
            report = drone.update_board_report()
            self._integrate_edges(report.get("edges", []), drone_id)

    # ------------------------------------------------------------------- stepping
    def step(
        self, actions: Dict[int, int]
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict[int, bool], Dict[int, dict]]:
        """Apply actions for all drones and advance the simulation by one round."""
        rewards: Dict[int, float] = {agent_id: self.reward_config["step_penalty"] for agent_id in self.agent_ids}
        infos: Dict[int, dict] = {agent_id: {} for agent_id in self.agent_ids}

        for drone_id in self.agent_ids:
            action_index = int(actions.get(drone_id, 0))
            rewards[drone_id] += self._apply_action(drone_id, action_index, infos)

        self.round += 1
        done = self.round > self.max_rounds or len(self.reported_edges) >= len(self.gt_edges)
        terminated = {agent_id: done for agent_id in self.agent_ids}
        truncated = {agent_id: False for agent_id in self.agent_ids}

        observations = {
            drone_id: drone.encode_observation() for drone_id, drone in self.drones.items()
        }
        for drone_id in self.agent_ids:
            infos[drone_id].update(
                {
                    "round": self.round,
                    "score": self.score,
                    "correct_edges": self.drone_stats[drone_id]["correct"],
                    "false_edges": self.drone_stats[drone_id]["false"],
                    "reported_edges": len(self.reported_edges),
                }
            )
        return observations, rewards, terminated, truncated, infos

    def _apply_action(self, drone_id: int, action_index: int, infos: Dict[int, dict]) -> float:
        """Translate the discrete action into environment side effects and reward shaping."""
        drone = self.drones[drone_id]
        if action_index < 0 or action_index >= len(ACTION_LOOKUP):
            return self.reward_config["illegal_move_penalty"]

        action, payload = ACTION_LOOKUP[action_index]
        total_reward = 0.0

        if action == "wait":
            total_reward += self.reward_config["wait_penalty"]

        elif action == "move":
            if payload not in drone.legal_moves():
                total_reward += self.reward_config["illegal_move_penalty"]
            else:
                total_reward += self._move_drone(drone, payload)

        elif action == "broadcast":
            total_reward += self._broadcast(drone)

        else:
            total_reward += self.reward_config["illegal_move_penalty"]

        infos[drone_id]["action"] = ACTION_NAMES[action_index]
        return total_reward

    def _move_drone(self, drone: DroneState, direction: str) -> float:
        dx, dy = DIRECTION_MAP[direction]
        sx, sy = drone.position
        nx, ny = sx + dx, sy + dy
        if not on_board(nx, ny, self.width, self.height):
            return self.reward_config["illegal_move_penalty"]

        old_tile = self.board[sx][sy]
        new_tile = self.board[nx][ny]
        old_tile.remove_drone(drone)
        new_tile.add_drone(drone)
        drone.position = (nx, ny)
        drone.record_position((nx, ny))

        reward = 0.0
        if (nx, ny) not in drone.visited_tiles:
            reward += self.reward_config["new_tile_bonus"]

        report = drone.update_board_report()
        reward += self.reward_config["info_discovery_bonus"] * float(
            report.get("resolved_tiles", 0)
        )
        reward += self.reward_config["possible_target_bonus"] * float(
            report.get("new_possible_targets", 0)
        )
        reward += self._integrate_edges(report.get("edges", []), drone.id)
        return reward

    def _broadcast(self, drone: DroneState) -> float:
        tile = self.board[drone.position[0]][drone.position[1]]
        others = [candidate for candidate in tile.drones if candidate.id != drone.id]
        if not others:
            return self.reward_config["broadcast_penalty"]

        total_reward = self.reward_config["broadcast_penalty"]
        positive_gain = 0.0
        info_bonus = 0.0
        for target in others:
            report = drone.provide_intelligence(target)
            reward = self._integrate_edges(report.get("edges", []), target.id)
            positive_gain += max(0.0, reward)
            info_bonus += self.reward_config["intel_share_bonus"] * float(
                report.get("resolved_tiles", 0)
            )
        assist = positive_gain * self.reward_config["broadcast_assist_share"]
        return total_reward + assist + info_bonus

    def _integrate_edges(self, edges: Iterable[str], owner_id: int) -> float:
        """Fold reported edges into mission statistics and compute the associated reward."""
        reward = 0.0
        for edge in edges:
            if edge in self.reported_edges:
                continue
            self.reported_edges.append(edge)
            if edge in self.gt_edges:
                reward += 1.0
                self.score += 1.0
                self.drone_stats[owner_id]["correct"] += 1
            else:
                reward -= 1.0
                self.score -= 1.0
                self.drone_stats[owner_id]["false"] += 1
        return reward

    # ----------------------------------------------------------------- utilities
    def get_score_summary(self) -> dict:
        """Return current scoreboard metrics."""
        return {
            "score": self.score,
            "reported_edges": len(self.reported_edges),
            "correct_edges": sum(stats["correct"] for stats in self.drone_stats.values()),
            "false_edges": sum(stats["false"] for stats in self.drone_stats.values()),
            "gt_edges": len(self.gt_edges),
        }


__all__ = [
    "CorasatMultiAgentEnv",
    "DroneState",
    "ACTION_LOOKUP",
    "ACTION_NAMES",
    "ORDERED_DIRECTIONS",
    "load_config",
]
