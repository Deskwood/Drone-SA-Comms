"""Support components for drone behavior and decision-making."""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from classes.Core import (
    CONFIG,
    FIGURE_TYPES,
    DIRECTION_MAP,
    cartesian_to_chess,
    chebyshev_distance,
    chess_to_cartesian,
    direction_from_vector,
    format_edge,
    on_board,
)
from classes.Exporter import LOGGER
if TYPE_CHECKING:
    from classes.Drone import _Drone


def _board_center_cartesian() -> Tuple[int, int]:
    width = max(1, int(CONFIG.get("board", {}).get("width", 8)))
    height = max(1, int(CONFIG.get("board", {}).get("height", 8)))
    center_x = min(width - 1, width // 2)
    center_y = min(height - 1, height // 2)
    return (center_x, center_y)


class _Drone_Knowledge:
    """Local board knowledge, memory, and intel sharing helpers."""

    def __init__(self, drone: _Drone):
        self.drone = drone
        self.drone.memory = ""
        self.drone.rx_buffer = ""
        self.drone.local_board = self._make_empty_local_board()
        self.drone.identified_edges = []
        self.drone.info_exchange_rounds = {}

    def _make_empty_local_board(self) -> Dict[str, Dict[str, str]]:
        """Initialize local board knowledge with unknown placeholders."""
        local_board: Dict[str, Dict[str, str]] = {}
        width = int(CONFIG.get("board", {}).get("width", 8))
        height = int(CONFIG.get("board", {}).get("height", 8))
        for bx in range(width):
            for by in range(height):
                local_board[cartesian_to_chess((bx, by))] = {"color": "unknown", "type": "unknown"}
        return local_board

    def visible_neighbor_figures(self) -> str:
        """Return visibility summary of neighboring figures by direction."""
        neighbors = []
        sx, sy = self.drone.position
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = sx + dx, sy + dy
                if not on_board(nx, ny):
                    continue
                tile = self.drone.sim.board[nx][ny]
                if tile.figure:
                    color = tile.figure.color or "unknown"
                    tile_key = cartesian_to_chess((nx, ny))
                    info = self.drone.local_board.get(tile_key, {})
                    figure_type = info.get("type")
                    if figure_type in FIGURE_TYPES:
                        desc = f"{color} {figure_type}"
                    else:
                        desc = f"{color} unknown"
                    neighbors.append(f"{direction_from_vector((dx, dy))}: {desc}")
        return ", ".join(neighbors)

    def collected_figure_information_text(self) -> str:
        """Summarize known figure intel from the local board."""
        entries = []
        width = int(CONFIG.get("board", {}).get("width", 8))
        height = int(CONFIG.get("board", {}).get("height", 8))
        for bx in range(width):
            for by in range(height):
                info = self.drone.local_board[cartesian_to_chess((bx, by))]
                if info["type"] in FIGURE_TYPES:
                    entries.append(f"{cartesian_to_chess((bx, by))}: {info['color']} {info['type']}")
                elif info["type"] == "a possible target":
                    entries.append(f"{cartesian_to_chess((bx, by))}: {info['type']}")
        return ", ".join(entries)

    def identify_edges(self) -> None:
        """Identify edges based on local board knowledge and report to simulation."""
        drone = self.drone
        width = int(CONFIG.get("board", {}).get("width", 8))
        height = int(CONFIG.get("board", {}).get("height", 8))

        for bx in range(width):
            for by in range(height):
                board_chess = cartesian_to_chess((bx, by))
                figure_type = drone.local_board[board_chess]["type"]
                figure_color = drone.local_board[board_chess]["color"]
                if figure_type not in FIGURE_TYPES:
                    continue

                if figure_type in ("queen", "rook", "bishop"):
                    is_slider = True
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
                else:
                    is_slider = False
                    if figure_type == "knight":
                        directions = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
                    elif figure_type == "king":
                        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
                    else:
                        directions = [(1, 1), (-1, 1)] if figure_color == "white" else [(1, -1), (-1, -1)]

                for dx, dy in directions:
                    tx, ty = bx, by
                    while True:
                        tx += dx
                        ty += dy
                        if not on_board(tx, ty, width, height):
                            break
                        target_chess = cartesian_to_chess((tx, ty))
                        target_info = drone.local_board[target_chess]
                        if target_info["type"] == "unknown":
                            target_info["type"] = "a possible target"
                        if target_info["type"] == "a possible target":
                            break
                        if target_info["type"] in FIGURE_TYPES or target_info["type"] == "any figure":
                            edge = format_edge(figure_type, figure_color, target_info["color"], ((bx, by), (tx, ty)))
                            if edge not in drone.identified_edges and (bx, by) != (tx, ty):
                                drone.identified_edges.append(edge)
                                if edge not in drone.sim.reported_edges:
                                    if edge in drone.sim.gt_edges:
                                        correct_marker = "- CORRECT: "
                                    else:
                                        correct_marker = "- FALSE: "
                                    edge_identified_message = f"{correct_marker}{edge}"
                                    LOGGER.log(edge_identified_message)
                                    if drone.sim.gui:
                                        drone.sim.gui.post_info(edge_identified_message)
                            break
                        if not is_slider:
                            break

        drone.sim.report_edges(drone.identified_edges)

    def update_board_report(self) -> None:
        """Update local board knowledge based on current observations."""
        drone = self.drone
        sx, sy = drone.position
        tile = drone.sim.board[sx][sy]
        if tile.figure:
            drone.local_board[cartesian_to_chess((sx, sy))] = {
                "color": tile.figure.color,
                "type": tile.figure.figure_type,
            }
        else:
            drone.local_board[cartesian_to_chess((sx, sy))] = {"color": "n/a", "type": "n/a"}

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = sx + dx, sy + dy
                if not on_board(nx, ny):
                    continue
                neighbor_tile = drone.sim.board[nx][ny]
                key = cartesian_to_chess((nx, ny))
                if neighbor_tile.figure:
                    if drone.local_board[key]["type"] == "unknown":
                        drone.local_board[key] = {"color": neighbor_tile.figure.color, "type": "any figure"}
                else:
                    drone.local_board[key] = {"color": "n/a", "type": "n/a"}

        self.identify_edges()

    def provide_intelligence_to(self, target_drone: "_Drone") -> None:
        """Share local board knowledge and edges with another drone."""
        drone = self.drone
        for pos, info in drone.local_board.items():
            tgt_info = target_drone.local_board.get(pos, {"color": "unknown", "type": "unknown"})
            if tgt_info["color"] == "unknown" and info["color"] != "unknown":
                target_drone.local_board[pos]["color"] = info["color"]
            if tgt_info["type"] == "unknown" and info["type"] != "unknown":
                target_drone.local_board[pos]["type"] = info["type"]

        for edge in drone.identified_edges:
            if edge not in target_drone.identified_edges:
                target_drone.identified_edges.append(edge)

        current_round = drone.sim.round
        drone.info_exchange_rounds[target_drone.id] = current_round
        target_drone.info_exchange_rounds[drone.id] = current_round
        target_drone.update_board_report()


class _Drone_Mission_Support:
    """Mission planning, sector assignment, and navigation helpers."""

    def __init__(self, drone: "_Drone"):
        self.drone = drone
        drone.assigned_sector = self._default_sector_assignment()
        drone.rendezvous_directive = None
        drone.mission_plan = []
        drone.current_leg_index = 0
        drone.mission_report = [tuple(drone.position)]
        try:
            self.build_initial_mission_plan()
        except Exception:
            drone.mission_plan = []

    def consume_rx_buffer(self) -> str:
        """Return and clear broadcast RX messages after mission support reads them."""
        lock = getattr(self.drone.sim, "state_lock", None)
        if lock:
            with lock:
                buf = self.drone.rx_buffer
                self.drone.rx_buffer = ""
                return buf
        buf = self.drone.rx_buffer
        self.drone.rx_buffer = ""
        return buf

    def _default_sector_assignment(self) -> Dict[str, str]:
        """Assign a default sector split among all drones."""
        return self._sector_assignment_for_drone(self.drone.id)

    def _sector_assignment_for_drone(self, drone_id: int) -> Dict[str, str]:
        width = int(CONFIG.get("board", {}).get("width", 8))
        height = int(CONFIG.get("board", {}).get("height", 8))
        num_drones = max(1, int(CONFIG.get("simulation", {}).get("num_drones", 1)))
        block_width = max(1, math.ceil(width / num_drones))
        idx = max(0, int(drone_id) - 1)
        start_col = min(idx * block_width, width - 1)
        end_col = min(width - 1, start_col + block_width - 1)
        if start_col > end_col:
            start_col = end_col
        return {
            "upper_left": cartesian_to_chess((start_col, height - 1)),
            "lower_right": cartesian_to_chess((end_col, 0)),
        }

    def _sector_to_key(self, sector: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if not isinstance(sector, dict):
            return None, None, None
        return (
            str(sector.get("upper_left")).lower() if sector.get("upper_left") else None,
            str(sector.get("lower_right")).lower() if sector.get("lower_right") else None,
            str(sector.get("label")).lower() if sector.get("label") else None,
        )

    def _sector_changed(self, old: Optional[Dict[str, Any]], new: Optional[Dict[str, Any]]) -> bool:
        return self._sector_to_key(old) != self._sector_to_key(new)

    def _to_chess(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if hasattr(value, "to_chess"):
            try:
                return value.to_chess()
            except Exception:
                return None
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return cartesian_to_chess((int(value[0]), int(value[1])))
        if isinstance(value, str):
            return value.strip().lower()
        return None

    def _normalize_sector_assignment(self, sector: Any) -> Dict[str, Any]:
        if sector is None:
            return self._default_sector_assignment()
        if isinstance(sector, dict):
            upper = self._to_chess(sector.get("upper_left"))
            lower = self._to_chess(sector.get("lower_right"))
            label = sector.get("label")
            normalized: Dict[str, Any] = {}
            if upper and lower:
                normalized["upper_left"] = upper
                normalized["lower_right"] = lower
            if label:
                normalized["label"] = label
            if normalized:
                return normalized
            return self._default_sector_assignment()
        if isinstance(sector, str):
            return {"label": sector}
        return self._default_sector_assignment()

    def _build_sector_perimeter_waypoints(
        self,
        bounds: Tuple[int, int, int, int],
        start_pos: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        min_x, max_x, min_y, max_y = bounds
        corners = [
            (min_x, max_y),  # upper-left
            (max_x, max_y),  # upper-right
            (max_x, min_y),  # lower-right
            (min_x, min_y),  # lower-left
        ]

        unique_corners: List[Tuple[int, int]] = []
        for corner in corners:
            if corner not in unique_corners:
                unique_corners.append(corner)

        if not unique_corners:
            return [start_pos]

        start_idx = min(
            range(len(unique_corners)),
            key=lambda idx: (chebyshev_distance(start_pos, unique_corners[idx]), idx),
        )
        start_corner = unique_corners[start_idx]
        if len(unique_corners) == 1:
            perimeter = [start_corner]
        else:
            dist_current = chebyshev_distance(start_pos, start_corner)
            cw_next = unique_corners[(start_idx + 1) % len(unique_corners)]
            ccw_next = unique_corners[(start_idx - 1) % len(unique_corners)]
            dist_cw = chebyshev_distance(start_pos, cw_next)
            dist_ccw = chebyshev_distance(start_pos, ccw_next)

            if dist_cw > dist_current and dist_ccw <= dist_current:
                clockwise = True
            elif dist_ccw > dist_current and dist_cw <= dist_current:
                clockwise = False
            else:
                clockwise = True

            perimeter = []
            if clockwise:
                for offset in range(len(unique_corners)):
                    perimeter.append(unique_corners[(start_idx + offset) % len(unique_corners)])
            else:
                for offset in range(len(unique_corners)):
                    perimeter.append(unique_corners[(start_idx - offset) % len(unique_corners)])

        if len(perimeter) > 1 and perimeter[-1] != perimeter[0]:
            perimeter = perimeter + [perimeter[0]]

        waypoints = [start_pos] + perimeter
        cleaned: List[Tuple[int, int]] = []
        for wp in waypoints:
            if not cleaned or cleaned[-1] != wp:
                cleaned.append(wp)
        return cleaned

    def sector_bounds(self, sector: Optional[Dict[str, Any]] = None) -> Optional[Tuple[int, int, int, int]]:
        sector = sector or self.drone.assigned_sector
        if not isinstance(sector, dict):
            return None
        upper_left = sector.get("upper_left")
        lower_right = sector.get("lower_right")
        if not upper_left or not lower_right:
            return None
        try:
            ul_x, ul_y = chess_to_cartesian(str(upper_left))
            lr_x, lr_y = chess_to_cartesian(str(lower_right))
        except Exception:
            return None
        min_x = min(ul_x, lr_x)
        max_x = max(ul_x, lr_x)
        min_y = min(lr_y, ul_y)
        max_y = max(lr_y, ul_y)
        return (min_x, max_x, min_y, max_y)

    def sector_summary(self) -> str:
        sector = self.drone.assigned_sector
        if isinstance(sector, dict) and sector.get("upper_left") and sector.get("lower_right"):
            return f"{sector.get('upper_left')} -> {sector.get('lower_right')}"
        if isinstance(sector, dict) and sector.get("label"):
            return str(sector.get("label"))
        return "unassigned"

    def set_sector_assignment(self, sector: Any) -> None:
        normalized = self._normalize_sector_assignment(sector)
        changed = self._sector_changed(self.drone.assigned_sector, normalized)
        self.drone.assigned_sector = normalized
        if changed:
            self.build_mission_plan_for_sector(normalized)
            LOGGER.log(f"Drone {self.drone.id} adopted sector {self.sector_summary()}")
        self.drone.current_leg_index = 0 if changed else min(
            self.drone.current_leg_index, max(0, len(self.drone.mission_plan) - 1)
        )
        self.sanitize_leg_index()

    def build_initial_mission_plan(self) -> List[dict]:
        return self.build_mission_plan_for_sector(self.drone.assigned_sector)

    def build_mission_plan_for_sector(self, sector: Optional[Dict[str, Any]]) -> List[dict]:
        bounds = self.sector_bounds(sector)
        if not bounds:
            sector = self._default_sector_assignment()
            bounds = self.sector_bounds(sector)
            self.drone.assigned_sector = sector
        if not bounds:
            self.drone.mission_plan = []
            return []
        min_x, max_x, min_y, max_y = bounds

        plan: List[dict] = []
        leg_id = 1
        current_turn = max(1, int(getattr(self.drone.sim, "round", 1) or 1))
        start_pos = tuple(self.drone.position)

        max_rounds = max(1, CONFIG.get("simulation", {}).get("max_rounds", 1))
        rv_turn_limit = max(1, max_rounds - 1)
        rv_turn = rv_turn_limit
        rv_cart = None
        if self.drone.rendezvous_directive:
            rv_cart = self.drone.rendezvous_directive.get("target_cartesian")
            if rv_cart is None and self.drone.rendezvous_directive.get("target"):
                rv_cart = list(chess_to_cartesian(self.drone.rendezvous_directive["target"]))
            requested_turn = self.drone.rendezvous_directive.get("turn") or rv_turn
            rv_turn = max(1, min(requested_turn, rv_turn_limit))
        if rv_cart is None:
            rv_cart = _board_center_cartesian()

        waypoints = self._build_sector_perimeter_waypoints(bounds, start_pos)
        if tuple(rv_cart) != waypoints[-1]:
            waypoints.append(tuple(rv_cart))
        legs = list(zip(waypoints[:-1], waypoints[1:]))
        distances = [chebyshev_distance(a, b) for a, b in legs]
        min_total_steps = sum(distances)
        remaining_turns = max(0, rv_turn - current_turn + 1)
        available_turns = max(0, remaining_turns - 1)
        slack_turns = max(0, available_turns - min_total_steps)

        slack_by_leg = [0] * len(distances)
        if min_total_steps > 0 and slack_turns > 0:
            raw_slack = [slack_turns * (dist / min_total_steps) for dist in distances]
            slack_by_leg = [int(math.floor(value)) for value in raw_slack]
            remainder = slack_turns - sum(slack_by_leg)
            if remainder:
                fractions = [raw - base for raw, base in zip(raw_slack, slack_by_leg)]
                order = sorted(
                    range(len(fractions)),
                    key=lambda idx: (-fractions[idx], -distances[idx], idx),
                )
                for idx in order[:remainder]:
                    slack_by_leg[idx] += 1

        cumulative_dist = 0
        cumulative_slack = 0
        last_arrival = current_turn
        for idx, ((wp_start, wp_end), dist) in enumerate(zip(legs, distances)):
            cumulative_dist += dist
            cumulative_slack += slack_by_leg[idx] if slack_by_leg else 0
            arrival_turn = current_turn + cumulative_dist + cumulative_slack
            arrival_turn = max(arrival_turn, last_arrival)
            last_arrival = arrival_turn

            duration_turns = 1 if wp_end == tuple(rv_cart) else 0
            dx = wp_end[0] - wp_start[0]
            dy = wp_end[1] - wp_start[1]
            if dx == 0 and dy != 0:
                orientation = "vertical"
            elif dy == 0 and dx != 0:
                orientation = "horizontal"
            else:
                orientation = "diagonal"
            plan.append(
                {
                    "leg_id": leg_id,
                    "turn": arrival_turn,
                    "leg_start": [wp_start[0], wp_start[1]],
                    "leg_end": [wp_end[0], wp_end[1]],
                    "duration_turns": duration_turns,
                    "orientation": orientation,
                }
            )
            leg_id += 1

        self.drone.current_leg_index = 0
        self.drone.mission_plan = plan
        self.sanitize_leg_index()
        self.advance_leg_progress()
        return plan

    def sanitize_leg_index(self) -> None:
        if not self.drone.mission_plan:
            self.drone.current_leg_index = 0
            return
        self.drone.current_leg_index = max(
            0, min(self.drone.current_leg_index, len(self.drone.mission_plan) - 1)
        )

    def current_mission_leg(self) -> Optional[dict]:
        if not self.drone.mission_plan:
            return None
        self.sanitize_leg_index()
        return self.drone.mission_plan[self.drone.current_leg_index]

    def advance_leg_progress(self) -> None:
        while True:
            leg = self.current_mission_leg()
            if not leg:
                return
            arrival_turn = max(1, int(leg.get("turn", 0) or 1))
            duration = max(0, int(leg.get("duration_turns", 0) or 0))
            current_round = max(1, int(getattr(self.drone.sim, "round", 1) or 1))
            leg_end = leg.get("leg_end")
            tolerance = int(leg.get("tolerance_steps", 0) or 0)
            at_leg_end = False
            if isinstance(leg_end, (list, tuple)) and len(leg_end) == 2:
                try:
                    end_pos = (int(leg_end[0]), int(leg_end[1]))
                    at_leg_end = chebyshev_distance(self.drone.position, end_pos) <= tolerance
                except (TypeError, ValueError):
                    at_leg_end = False
            leg_expired = current_round > arrival_turn + duration
            if leg_expired and not at_leg_end:
                if self.drone.current_leg_index < len(self.drone.mission_plan) - 1:
                    self.drone.current_leg_index += 1
                    continue
                self.drone.current_leg_index = len(self.drone.mission_plan) - 1
                return
            if at_leg_end and (duration == 0 or current_round >= arrival_turn + duration):
                if self.drone.current_leg_index < len(self.drone.mission_plan) - 1:
                    self.drone.current_leg_index += 1
                    continue
                self.drone.current_leg_index = len(self.drone.mission_plan) - 1
            return

    def leg_segment_bounds(self, leg: dict) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        try:
            start_vec = leg.get("leg_start")
            end_vec = leg.get("leg_end")
            if start_vec is None or end_vec is None:
                return None
            start = (int(start_vec[0]), int(start_vec[1]))
            end = (int(end_vec[0]), int(end_vec[1]))
            return start, end
        except (TypeError, ValueError):
            return None

    def distance_to_leg(self, pos: Tuple[int, int], leg: Optional[dict]) -> Optional[int]:
        if leg is None:
            return None
        bounds = self.leg_segment_bounds(leg)
        if not bounds:
            return None
        (sx, sy), (ex, ey) = bounds
        px, py = pos
        vx, vy = ex - sx, ey - sy
        wx, wy = px - sx, py - sy
        seg_len_sq = vx * vx + vy * vy
        if seg_len_sq == 0:
            return int(round(math.hypot(wx, wy)))
        t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len_sq))
        proj_x = sx + t * vx
        proj_y = sy + t * vy
        dist = math.hypot(px - proj_x, py - proj_y)
        return int(round(dist))

    def distance_to_leg_end(self, pos: Tuple[int, int], leg: Optional[dict]) -> Optional[int]:
        if leg is None:
            return None
        target_vec = leg.get("leg_end")
        if not target_vec:
            return None
        tx, ty = target_vec
        return int(round(math.hypot(pos[0] - tx, pos[1] - ty)))

    def next_mission_waypoint(self) -> Optional[dict]:
        return self.current_mission_leg()

    def legal_movement_steps(self) -> List[dict]:
        sx, sy = self.drone.position
        reachable_tiles = []
        for name, (dx, dy) in DIRECTION_MAP.items():
            nx, ny = sx + dx, sy + dy
            if on_board(nx, ny):
                reachable_tiles.append({"direction": name, "new_position": (nx, ny)})
        return reachable_tiles

    def apply_plan_directive(self, payload: Dict[str, object]) -> None:
        if not isinstance(payload, dict):
            return
        plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else None
        if plan is None:
            return

        assignments = plan.get("assignments")
        if isinstance(assignments, list):
            for assignment in assignments:
                if not isinstance(assignment, dict):
                    continue
                drone_ref = assignment.get("drone")
                if drone_ref is None:
                    continue
                if str(drone_ref).strip() == str(self.drone.id):
                    sector = assignment.get("sector")
                    if sector is not None:
                        self.set_sector_assignment(sector)
                    break

        rendezvous = plan.get("rendezvous")
        if isinstance(rendezvous, dict):
            tile = rendezvous.get("tile")
            turn = rendezvous.get("turn")
            if tile:
                try:
                    cart = chess_to_cartesian(str(tile))
                    if not on_board(*cart):
                        LOGGER.log(f"Drone {self.drone.id} received out-of-bounds rendezvous '{tile}'")
                        return
                    max_rounds = max(1, CONFIG.get("simulation", {}).get("max_rounds", 1))
                    rv_turn_limit = max(1, max_rounds - 1)
                    requested_turn = int(turn) if turn is not None else rv_turn_limit
                    rv_turn = max(1, min(requested_turn, rv_turn_limit))
                    if rv_turn != requested_turn:
                        LOGGER.log(
                            f"Drone {self.drone.id} clamped rendezvous turn {requested_turn} -> {rv_turn}"
                        )
                    waypoint = {
                        "turn": rv_turn,
                        "target": cartesian_to_chess(cart),
                        "target_cartesian": [cart[0], cart[1]],
                        "distance_steps": chebyshev_distance(self.drone.position, cart),
                        "tolerance_steps": 0,
                        "notes": "Rendezvous directive.",
                    }
                    self.drone.rendezvous_directive = waypoint

                    try:
                        self.build_mission_plan_for_sector(self.drone.assigned_sector)
                    except Exception as exc:
                        LOGGER.log(f"Drone {self.drone.id} failed to rebuild mission plan with rendezvous: {exc}")
                    LOGGER.log(f"Drone {self.drone.id} adopted rendezvous {waypoint['target']} by turn {rv_turn}")
                except Exception as exc:
                    LOGGER.log(f"Drone {self.drone.id} failed to apply rendezvous directive: {exc}")
        self.sanitize_leg_index()


class _Drone_Decision_Support:
    """Decision scoring, summaries, and prompt assembly."""

    def __init__(self, drone: "_Drone"):
        self.drone = drone

    def snapshot(self) -> Dict[str, object]:
        """Compute decision support scores and helper metadata."""
        return self._compute_decision_support()

    def format_lines(self, snapshot: Dict[str, object]) -> Tuple[List[str], List[str]]:
        """Format decision support output for UI display."""
        drone = self.drone
        lines: List[str] = []
        ledger_lines: List[str] = []

        max_rounds = CONFIG["simulation"]["max_rounds"]
        lines.append(f"Round: {drone.sim.round}/{max_rounds}")
        lines.append(f"Position: {cartesian_to_chess(drone.position)}")

        same_tile_drones = [
            f"Drone {other.id}"
            for other in drone.sim.board[drone.position[0]][drone.position[1]].drones
            if other.id != drone.id
        ]
        visible_drones = ", ".join(same_tile_drones) if same_tile_drones else "None"
        fig_here = "None"
        tile_here = drone.sim.board[drone.position[0]][drone.position[1]]
        if tile_here.figure:
            fig_here = tile_here.figure.figure_type
        lines.append(f"Visible: drones={visible_drones}; figure={fig_here}")

        legal_entries: List[str] = []
        for step in drone._legal_movement_steps():
            new_pos = step["new_position"]
            tile = cartesian_to_chess(new_pos)
            figure_label = None
            tile_state = drone.sim.board[new_pos[0]][new_pos[1]]
            if tile_state.figure:
                color = tile_state.figure.color or "unknown"
                info = drone.local_board.get(tile, {})
                figure_type = info.get("type")
                if figure_type in FIGURE_TYPES:
                    figure_label = f"{color} {figure_type}"
                else:
                    figure_label = f"{color} unknown"
            if figure_label:
                legal_entries.append(f"{step['direction']} to {tile} ({figure_label})")
            else:
                legal_entries.append(f"{step['direction']} to {tile}")
        legal_moves = ", ".join(legal_entries) or "none"
        lines.append(f"Legal moves: {legal_moves}")

        memory_text = drone.memory.strip()
        memory_text = memory_text.replace("\n", " | ") if memory_text else "None"
        lines.append(f"Memory (previous rounds): {memory_text}")

        if getattr(drone, "rendezvous_directive", None):
            rv = drone.rendezvous_directive
            lines.append(f"Rendezvous: {rv['target']} on turn {rv['turn']}")

        if drone.id == 1 and getattr(drone.sim, "round", None) == 1 and getattr(drone.sim, "turn", None) == 1:
            total_drones = CONFIG["simulation"].get("num_drones", len(getattr(drone.sim, "drones", [])))
            lines.append(
                "Special directive: Drone 1 must broadcast a coverage plan on the opening turn "
                f"(JSON plan->assignments for all {total_drones} drones) before other actions; "
                "overrides decision support ranking."
            )

        waypoint = snapshot.get("next_waypoint")
        if waypoint:
            if waypoint.get("leg_start") and waypoint.get("leg_end"):
                leg_id = waypoint.get("leg_id", "?")
                start_vec = waypoint.get("leg_start")
                end_vec = waypoint.get("leg_end")
                start = cartesian_to_chess(tuple(start_vec)) if start_vec else "?"
                target = cartesian_to_chess(tuple(end_vec)) if end_vec else "?"
                lines.append(f"Plan focus: leg {leg_id} {start} -> {target} (turn {waypoint.get('turn', '?')})")
            else:
                target_vec = waypoint.get("leg_end")
                target = cartesian_to_chess(tuple(target_vec)) if target_vec else "?"
                lines.append(f"Plan focus: {target} by turn {waypoint.get('turn', '?')}")
        timing = snapshot.get("waypoint_timing")
        if timing:
            dist = timing.get("distance")
            turns_left = timing.get("turns_remaining")
            slack = timing.get("slack")
            if dist is not None and turns_left is not None:
                lines.append(
                    f"Waypoint timing: {dist} steps away, {turns_left} turns left (slack {slack}, incl. current turn)."
                )
                if slack is not None:
                    if slack < 0:
                        lines.append(
                            "Timing rule: behind schedule -> prioritize reducing Chebyshev distance to the waypoint."
                        )
                    else:
                        lines.append("Timing rule: on schedule -> no timing bias unless distance exceeds turns left.")

        if getattr(drone, "assigned_sector", None):
            lines.append(f"Sector directive: {drone.sector_summary()}")

        suggestion = snapshot.get("coordination_suggestion")
        if suggestion:
            lines.append("Suggested coordination broadcast JSON:")
            plan_json = json.dumps(suggestion.get("plan", {}), indent=2)
            for line in plan_json.splitlines():
                lines.append(f"  {line}")

        score_entries = snapshot.get("scores", [])
        sorted_scores = sorted(
            score_entries,
            key=lambda entry: entry.get("score", float("-inf")),
            reverse=True,
        )
        rank_lookup = {id(entry): idx for idx, entry in enumerate(sorted_scores)}

        def _qualitative_label(rank: int, score: float) -> str:
            if rank == 0:
                return "Best choice"
            if score >= 5:
                return "Excellent choice"
            if score >= 2:
                return "Good choice"
            if score >= 0:
                return "Okay choice"
            if score > -2:
                return "Risky choice"
            return "Bad choice"

        def _format_score(value: float) -> str:
            return f"{value:.2f}"

        excluded_component_keys = {"unknown_tile_bonus", "possible_target"}

        move_component_keys = [
            "waypoint_progress",
            "waypoint_regression",
            "cross_track_penalty",
            "sector_compliance_bonus",
            "no_figures_left_behind",
            "figure_hint",
            "revisit_penalty",
            "neighborhood_potential",
        ]

        def _build_score_table(
            entries: List[Dict[str, object]],
            title: str,
            component_order: Optional[List[str]] = None,
        ) -> List[str]:
            if not entries:
                return []
            seen_keys = {
                key for entry in entries for key in (entry.get("components") or {}).keys()
            } - excluded_component_keys
            component_keys = sorted(seen_keys)
            if component_order:
                component_order = [key for key in component_order if key not in excluded_component_keys]
                extras = sorted(key for key in seen_keys if key not in component_order)
                component_keys = list(component_order) + extras
            headers = ["Action", "Choice", "Score"] + component_keys + ["Notes"]
            rows: List[List[str]] = []
            max_note_len = 120
            for entry in entries:
                components = entry.get("components") or {}
                action = (entry.get("action") or "-").strip()
                label = (entry.get("label") or "").strip()
                if action == "move" and label:
                    action_text = f"{action} {label}"
                elif action in {"broadcast", "wait"}:
                    action_text = action
                elif label and label != "-":
                    action_text = f"{action} {label}"
                else:
                    action_text = action
                rank = rank_lookup.get(id(entry), 0)
                qualitative = _qualitative_label(rank, entry.get("score", 0.0))
                score_text = _format_score(entry.get("score", 0.0))
                comp_values = [f"{components.get(key, 0.0):+.2f}" for key in component_keys]
                notes = "; ".join(entry.get("notes") or [])
                if not notes:
                    notes = "n/a"
                elif len(notes) > max_note_len:
                    notes = notes[:max_note_len - 3].rstrip() + "..."
                rows.append([action_text, qualitative, score_text] + comp_values + [notes])

            widths = []
            for col_idx, header in enumerate(headers):
                width = len(header)
                for row in rows:
                    width = max(width, len(row[col_idx]))
                widths.append(width)

            aligns = ["left", "left", "right"] + ["right"] * len(component_keys) + ["left"]

            def _pad(text: str, width: int, align: str) -> str:
                return text.rjust(width) if align == "right" else text.ljust(width)

            header_line = " | ".join(
                _pad(headers[idx], widths[idx], aligns[idx]) for idx in range(len(headers))
            )
            divider_line = "-+-".join("-" * widths[idx] for idx in range(len(headers)))
            table_lines = []
            if title:
                table_lines.append(title)
            table_lines.append(header_line)
            table_lines.append(divider_line)
            for row in rows:
                table_lines.append(
                    " | ".join(
                        _pad(row[idx], widths[idx], aligns[idx]) for idx in range(len(headers))
                    )
                )
            return table_lines

        if sorted_scores:
            move_entries = [entry for entry in sorted_scores if entry.get("action") == "move"]
            other_entries = [entry for entry in sorted_scores if entry.get("action") != "move"]
            lines.extend(_build_score_table(move_entries, "Movement scoring:", move_component_keys))
            if other_entries:
                lines.append("Other actions scoring:")
                for entry in other_entries:
                    action = (entry.get("action") or "-").strip()
                    label = (entry.get("label") or "").strip()
                    if label and action not in {"broadcast", "wait"}:
                        action_text = f"{action} {label}"
                    else:
                        action_text = action
                    score_text = _format_score(entry.get("score", 0.0))
                    components = entry.get("components") or {}
                    comp_text = ", ".join(
                        f"{key}={value:+.2f}" for key, value in components.items()
                    ) or "n/a"
                    notes = "; ".join(entry.get("notes") or []) or "n/a"
                    lines.append(f"{action_text}: score {score_text}; components: {comp_text}; notes: {notes}")

            best_entry = sorted_scores[0]
            best_action = (best_entry.get("action") or "-").strip()
            best_label = (best_entry.get("label") or "").strip()
            if best_action == "move" and best_label:
                best_action_text = f"{best_action} {best_label}"
            elif best_label and best_action not in {"broadcast", "wait"}:
                best_action_text = f"{best_action} {best_label}"
            else:
                best_action_text = best_action
            best_score_text = _format_score(best_entry.get("score", 0.0))
            summary_notes = "; ".join(best_entry.get("notes") or [])
            if not summary_notes:
                reason_text = "highest score"
            else:
                max_reason_len = 140
                reason_text = summary_notes
                if len(reason_text) > max_reason_len:
                    reason_text = reason_text[: max_reason_len - 3].rstrip() + "..."
            if reason_text.endswith("."):
                summary_line = (
                    f"Decision Support Summary: best choice {best_action_text} "
                    f"(score {best_score_text}) because {reason_text}"
                )
            else:
                summary_line = (
                    f"Decision Support Summary: best choice {best_action_text} "
                    f"(score {best_score_text}) because {reason_text}."
                )
            lines.append(summary_line)

        for ledger in snapshot.get("intel_ledger", []):
            last_round = ledger.get("last_round")
            age = ledger.get("age", 0)
            if last_round is None:
                ledger_lines.append(f"Drone {ledger['drone']}: never shared (age {age} rounds)")
            else:
                ledger_lines.append(f"Drone {ledger['drone']}: last shared round {last_round} (age {age})")

        return lines, ledger_lines

    def build_situation(self, snapshot: Dict[str, object]) -> str:
        """Assemble the situation text that fuels the language model prompt."""
        drone = self.drone
        collected_figure_information = drone.knowledge.collected_figure_information_text()

        lines: List[str] = []
        lines.append(f"Collected figure information: {collected_figure_information or 'None'}")
        rx_buffer = drone.mission_support.consume_rx_buffer().strip()
        if rx_buffer:
            rx_buffer = rx_buffer.replace("\n", " | ")
            if len(rx_buffer) > 300:
                rx_buffer = rx_buffer[:300].rstrip() + "...(truncated)"
        lines.append(f"Broadcast Rx Buffer: {rx_buffer or 'None'}")

        ds_lines, ledger_lines = self.format_lines(snapshot)
        if ds_lines:
            lines.append("Decision Support:")
            lines.extend([f"  {entry}" for entry in ds_lines])
        if ledger_lines:
            lines.append("Intel Share Ledger:")
            lines.extend([f"  {entry}" for entry in ledger_lines])

        situation_description = "\n".join(lines)
        LOGGER.log(f"Drone {drone.id} Situation:\n{situation_description}")
        return situation_description

    def build_prompt(self, rules: str) -> Dict[str, object]:
        """Create the prompt payload for the language model."""
        snapshot = self.snapshot()
        rules = ""
        situation = self.build_situation(snapshot)

        prompt_requests = CONFIG.get("prompt_requests", {})
        cues = "\n".join(
            [
                prompt_requests.get("rationale", ""),
                prompt_requests.get("action", ""),
                prompt_requests.get("action_move", ""),
                prompt_requests.get("action_broadcast", ""),
                prompt_requests.get("memory_update", ""),
            ]
        ).strip()
        user_content = situation if not cues else situation + "\n\n" + cues

        messages = [
            {"role": "system", "content": rules},
            {"role": "user", "content": user_content},
        ]
        prompt_char_len = len(user_content) + len(rules)
        return {
            "messages": messages,
            "prompt_char_len": prompt_char_len,
            "snapshot": snapshot,
            "user_content": user_content,
        }

    def _compute_neighborhood_potential(
        self,
        pos: Tuple[int, int],
        border_bonus: float,
        any_figure_weight: float,
        possible_target_weight: float,
        unknown_weight: float,
    ) -> Tuple[float, Dict[str, float]]:
        drone = self.drone
        weights = {
            "any figure": any_figure_weight,
            "a possible target": possible_target_weight,
            "unknown": unknown_weight,
            "border": border_bonus,
        }
        total = 0.0
        contributions: Dict[str, float] = {}
        directions = [
            (0, 1, "n"),
            (1, 1, "ne"),
            (1, 0, "e"),
            (1, -1, "se"),
            (0, -1, "s"),
            (-1, -1, "sw"),
            (-1, 0, "w"),
            (-1, 1, "nw"),
        ]
        for dx, dy, label in directions:
            nx, ny = pos[0] + dx, pos[1] + dy
            if not on_board(nx, ny):
                neighbor_type = "border"
            else:
                key = cartesian_to_chess((nx, ny))
                info = drone.local_board.get(key, {"type": "unknown"})
                neighbor_type = info.get("type")
            weight = weights.get(neighbor_type, 0.0)
            contributions[label] = weight
            total += weight
        return total, contributions

    def _distance_to_sector(self, point: Tuple[int, int], bounds: Tuple[int, int, int, int]) -> Optional[int]:
        if point is None or bounds is None:
            return None
        px, py = point
        min_x, max_x, min_y, max_y = bounds
        dx = 0
        if px < min_x:
            dx = min_x - px
        elif px > max_x:
            dx = px - max_x
        dy = 0
        if py < min_y:
            dy = min_y - py
        elif py > max_y:
            dy = py - max_y
        return max(dx, dy)

    def _build_coordination_suggestion(self) -> Dict[str, Any]:
        drone = self.drone
        sim = drone.sim
        board_w = int(CONFIG.get("board", {}).get("width", 8))
        board_h = int(CONFIG.get("board", {}).get("height", 8))
        num_drones = max(1, sim.num_drones)
        assignments: List[Dict[str, Any]] = []
        for idx in range(num_drones):
            sector = drone.mission_support._sector_assignment_for_drone(idx + 1)
            upper_left = sector.get("upper_left")
            lower_right = sector.get("lower_right")
            assignments.append(
                {
                    "drone": idx + 1,
                    "sector": {"upper_left": upper_left, "lower_right": lower_right},
                }
            )
        rendezvous_turn = max(1, CONFIG.get("simulation", {}).get("max_rounds", 1) - 1)
        rendezvous_tile = cartesian_to_chess(_board_center_cartesian())
        return {
            "plan": {
                "assignments": assignments,
                "rendezvous": {"tile": rendezvous_tile, "turn": rendezvous_turn},
            }
        }

    def _compute_decision_support(self) -> Dict[str, object]:
        drone = self.drone
        ds_cfg = CONFIG.get("decision_support", {})
        scoring_cfg = ds_cfg.get("scoring", {})

        move_cfg = scoring_cfg.get("move", {})
        broadcast_cfg = scoring_cfg.get("broadcast", {})
        wait_cfg = scoring_cfg.get("wait", {})

        waypoint_progress_bonus = move_cfg.get("waypoint_progress_bonus", 1.0)
        waypoint_delay_penalty = move_cfg.get("waypoint_delay_penalty", -1.0)
        unknown_tile_bonus = move_cfg.get("unknown_tile_bonus", 1.0)
        no_figures_left_behind_bonus = move_cfg.get("no_figures_left_behind_bonus", 1.0)
        possible_target_bonus = move_cfg.get("possible_target_bonus", 1.2)
        figure_hint_bonus = move_cfg.get("figure_hint_bonus", 0.6)
        neighborhood_potential_factor = move_cfg.get("neighborhood_potential", 0.2)
        neighborhood_weight_any_figure = move_cfg.get("neighborhood_weight_any_figure", 3.0)
        neighborhood_weight_possible_target = move_cfg.get("neighborhood_weight_possible_target", 1.0)
        border_bonus = move_cfg.get("border_bonus", 0.0)
        revisit_penalty = move_cfg.get("revisit_penalty", -1.0)
        cross_track_penalty_per_step_squared = move_cfg.get("cross_track_penalty_per_step_squared", 0.6)
        sector_compliance_bonus = move_cfg.get("sector_compliance_bonus", 0.8)

        move_params = {
            "waypoint_progress_bonus": waypoint_progress_bonus,
            "waypoint_delay_penalty": waypoint_delay_penalty,
            "unknown_tile_bonus": unknown_tile_bonus,
            "no_figures_left_behind_bonus": no_figures_left_behind_bonus,
            "possible_target_bonus": possible_target_bonus,
            "figure_hint_bonus": figure_hint_bonus,
            "neighborhood_potential": neighborhood_potential_factor,
            "neighborhood_weight_any_figure": neighborhood_weight_any_figure,
            "neighborhood_weight_possible_target": neighborhood_weight_possible_target,
            "border_bonus": border_bonus,
            "revisit_penalty": revisit_penalty,
            "cross_track_penalty_per_step_squared": cross_track_penalty_per_step_squared,
            "sector_compliance_bonus": sector_compliance_bonus,
        }

        broadcast_base_value = broadcast_cfg.get("base_broadcast_value", -0.5)
        coordination_broadcast_bonus = broadcast_cfg.get("first_turn_coordination_bonus", 2.5)
        last_turn_coordination_bonus = broadcast_cfg.get("last_turn_coordination_bonus", 0.0)
        is_first_coordination_turn = (
            drone.id == 1 and getattr(drone.sim, "round", None) == 1 and getattr(drone.sim, "turn", None) == 1
        )

        wait_base_value = wait_cfg.get("base_wait_value", -1.0)

        scores: List[Dict[str, object]] = []
        max_rounds = max(1, CONFIG.get("simulation", {}).get("max_rounds", 1))
        current_round = drone.sim.round
        is_last_round = current_round == max_rounds
        try:
            drone._advance_leg_progress()
        except Exception:
            pass
        next_wp = drone._next_mission_waypoint()
        target_pos = tuple(next_wp["leg_end"]) if next_wp else None
        current_leg = next_wp if next_wp and next_wp.get("leg_start") else None
        current_leg_distance = drone._distance_to_leg(drone.position, current_leg) if current_leg else None
        final_leg_active = False
        mission_plan = getattr(drone, "mission_plan", None)
        if isinstance(mission_plan, list) and mission_plan:
            try:
                final_leg_active = int(drone.current_leg_index) >= len(mission_plan) - 1
            except (TypeError, ValueError):
                final_leg_active = False
        timing_turns_remaining: Optional[int] = None
        timing_distance: Optional[int] = None
        timing_slack: Optional[int] = None
        if target_pos and next_wp and next_wp.get("turn") is not None:
            try:
                timing_turns_remaining = max(0, int(next_wp["turn"]) - current_round + 1)
                timing_distance = chebyshev_distance(drone.position, target_pos)
                timing_slack = timing_turns_remaining - timing_distance
            except (TypeError, ValueError):
                timing_turns_remaining = None
                timing_distance = None
                timing_slack = None
        sector_bounds = drone._sector_bounds() if hasattr(drone, "_sector_bounds") else None
        current_sector_distance = self._distance_to_sector(drone.position, sector_bounds) if sector_bounds else None
        nearest_sector_unknown: Optional[Dict[str, object]] = None
        if sector_bounds:
            min_x, max_x, min_y, max_y = sector_bounds
            best_unknown_dist: Optional[int] = None
            best_unknown_pos: Optional[Tuple[int, int]] = None
            for sx in range(min_x, max_x + 1):
                for sy in range(min_y, max_y + 1):
                    tile_key = cartesian_to_chess((sx, sy))
                    tile_report = drone.local_board.get(tile_key, {})
                    if tile_report.get("type") == "unknown":
                        pos = (sx, sy)
                        dist = chebyshev_distance(drone.position, pos)
                        if best_unknown_dist is None or dist < best_unknown_dist:
                            best_unknown_dist = dist
                            best_unknown_pos = pos
            if best_unknown_pos is not None and best_unknown_dist is not None:
                nearest_sector_unknown = {"tile": cartesian_to_chess(best_unknown_pos), "distance": best_unknown_dist}

        visited_tiles: set = set()
        for pos in getattr(drone, "mission_report", []) or []:
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                try:
                    visited_tiles.add((int(pos[0]), int(pos[1])))
                except (TypeError, ValueError):
                    continue

        unidentified_neighbors: List[Tuple[int, int]] = []
        if target_pos:
            sx, sy = drone.position
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = sx + dx, sy + dy
                    if not on_board(nx, ny):
                        continue
                    tile_key = cartesian_to_chess((nx, ny))
                    tile_info = drone.local_board.get(tile_key, {"type": "unknown"})
                    if tile_info.get("type") == "any figure":
                        unidentified_neighbors.append((nx, ny))

        def _adjacent_to_unidentified(pos: Tuple[int, int]) -> bool:
            for ux, uy in unidentified_neighbors:
                if max(abs(pos[0] - ux), abs(pos[1] - uy)) <= 1:
                    return True
            return False

        for step in drone._legal_movement_steps():
            direction = step["direction"]
            new_pos = tuple(step["new_position"])
            tile_key = cartesian_to_chess(new_pos)
            tile_info = drone.local_board.get(tile_key, {"color": "unknown", "type": "unknown"})

            score = 0.0
            score_components: Dict[str, float] = {}
            notes: List[str] = []

            def _add_component(name: str, value: float) -> None:
                if not value:
                    return
                score_components[name] = score_components.get(name, 0.0) + value

            # Cross-track penalty relative to current mission leg
            if current_leg and not final_leg_active:
                new_leg_distance = drone._distance_to_leg(new_pos, current_leg)
                if current_leg_distance is not None and new_leg_distance is not None:
                    delta_leg_distance = new_leg_distance - current_leg_distance
                    cross_track_penalty = cross_track_penalty_per_step_squared * new_leg_distance**2
                    score += cross_track_penalty
                    _add_component("cross_track_penalty", cross_track_penalty)
                    if new_leg_distance != 0:
                        notes.append(f"not on current leg (dist {new_leg_distance})")
                    if delta_leg_distance < 0:
                        notes.append("approaching current leg")
                    elif delta_leg_distance > 0:
                        notes.append("diverting from current leg")

            # Waypoint timing evaluation
            if target_pos:
                current_dist = chebyshev_distance(drone.position, target_pos)
                new_dist = chebyshev_distance(new_pos, target_pos)
                turns_remaining = timing_turns_remaining
                if turns_remaining is None and next_wp and next_wp.get("turn") is not None:
                    try:
                        turns_remaining = max(0, int(next_wp["turn"]) - current_round + 1)
                    except (TypeError, ValueError):
                        turns_remaining = None
                turns_remaining_after_move = (
                    max(0, turns_remaining - 1) if turns_remaining is not None else None
                )
                slack = (
                    turns_remaining_after_move - new_dist if turns_remaining_after_move is not None else None
                )

                if new_dist < current_dist:
                    score += waypoint_progress_bonus
                    _add_component("waypoint_progress", waypoint_progress_bonus)
                    notes.append("closer to waypoint")
                elif new_dist > current_dist:
                    progress_penalty = -abs(waypoint_progress_bonus)
                    score += progress_penalty
                    _add_component("waypoint_progress", progress_penalty)
                    notes.append("farther from waypoint")

                if slack is not None and slack < 0:
                    regression_penalty = waypoint_delay_penalty * abs(slack)
                    score += regression_penalty
                    _add_component("waypoint_regression", regression_penalty)
                    notes.append("late for waypoint")

                if (
                    no_figures_left_behind_bonus
                    and unidentified_neighbors
                    and slack is not None
                    and slack > 0
                    and new_dist == current_dist
                    and _adjacent_to_unidentified(new_pos)
                ):
                    score += no_figures_left_behind_bonus
                    _add_component("no_figures_left_behind", no_figures_left_behind_bonus)
                    notes.append("holding progress to inspect nearby unknown figure")

                if next_wp.get("duration_turns", 0) > 0 and slack is not None:
                    buffer_turns = max(0, slack)
                    suffix = "turn" if buffer_turns == 1 else "turns"
                    notes.append(f"{buffer_turns} {suffix} left to reach rendezvous on time")
                    if buffer_turns == 0 and slack < 0:
                        notes.append("rendezvous already overdue - move immediately")

            # Tile type bonuses
            if tile_info["type"] == "unknown":
                score += unknown_tile_bonus
                _add_component("unknown_tile_bonus", unknown_tile_bonus)
                notes.append("unidentified tile")
            elif tile_info["type"] == "a possible target":
                score += possible_target_bonus
                _add_component("possible_target", possible_target_bonus)
                notes.append("possible blocker")
            elif tile_info["type"] == "any figure":
                score += figure_hint_bonus
                _add_component("figure_hint", figure_hint_bonus)
                notes.append("figure nearby")

            # Revisit penalty
            if new_pos in visited_tiles:
                score += revisit_penalty
                _add_component("revisit_penalty", revisit_penalty)
                notes.append("revisiting tile")

            # Neighborhood potential
            neighborhood_potential, neighborhood_breakdown = self._compute_neighborhood_potential(
                new_pos,
                border_bonus,
                neighborhood_weight_any_figure,
                neighborhood_weight_possible_target,
                unknown_tile_bonus,
            )
            if neighborhood_potential and neighborhood_potential_factor:
                neighbor_bonus = neighborhood_potential * neighborhood_potential_factor
                score += neighbor_bonus
                _add_component("neighborhood_potential", neighbor_bonus)
                if neighborhood_breakdown:
                    bonus_parts = {
                        label: value * neighborhood_potential_factor
                        for label, value in neighborhood_breakdown.items()
                    }
                    bonus_detail = ", ".join(f"{label}:{value:.2f}" for label, value in bonus_parts.items())
                    notes.append(f"neighbor bonuses {bonus_detail} (total {neighbor_bonus:.2f})")

            # Sector compliance bonus
            if sector_bounds and not final_leg_active:
                new_sector_distance = self._distance_to_sector(new_pos, sector_bounds)
                if new_sector_distance is not None:
                    if sector_compliance_bonus and (
                        new_sector_distance == 0
                        or (
                            current_sector_distance is not None
                            and new_sector_distance < current_sector_distance
                        )
                    ):
                        score += sector_compliance_bonus
                        _add_component("sector_compliance_bonus", sector_compliance_bonus)
                        if new_sector_distance == 0:
                            notes.append("within assigned sector")
                        else:
                            notes.append("moving toward assigned sector")


            scores.append(
                {
                    "action": "move",
                    "label": direction,
                    "score": round(score, 2),
                    "components": dict(score_components),
                    "notes": notes,
                }
            )

        tile = drone.sim.board[drone.position[0]][drone.position[1]]
        recipients = [d for d in tile.drones if d.id != drone.id]
        broadcast_components: Dict[str, float] = {}
        broadcast_notes: List[str] = []
        broadcast_score = broadcast_base_value
        if recipients:
            ages: List[int] = []
            for target in recipients:
                last_round = drone.info_exchange_rounds.get(target.id)
                age = current_round - last_round if last_round is not None else current_round
                ages.append(max(age, 0))
            avg_age = sum(ages) / len(ages) if ages else 0.0
            broadcast_score = 0.0
            broadcast_notes.append(f"{len(recipients)} co-located drones")
            if avg_age > 0:
                broadcast_notes.append("recipients have stale intel")
        else:
            broadcast_components["broadcast_base"] = broadcast_base_value
            broadcast_notes.append("no co-located drones")

        if is_first_coordination_turn and coordination_broadcast_bonus:
            broadcast_score += coordination_broadcast_bonus
            broadcast_components["coordination_bonus"] = coordination_broadcast_bonus
            broadcast_notes.append("first-turn coverage assignment priority")

        if recipients and is_last_round and last_turn_coordination_bonus:
            broadcast_score += last_turn_coordination_bonus
            broadcast_components["last_turn_coordination_bonus"] = last_turn_coordination_bonus
            broadcast_notes.append("last-round coordination priority")

        scores.append(
            {
                "action": "broadcast",
                "label": "share",
                "score": round(broadcast_score, 2),
                "components": {k: v for k, v in broadcast_components.items()},
                "notes": broadcast_notes,
            }
        )

        wait_score = wait_base_value
        wait_notes: List[str] = ["no progress"]
        wait_components: Dict[str, float] = {"wait_base": wait_base_value}
        scores.append(
            {
                "action": "wait",
                "label": "hold",
                "score": round(wait_score, 2),
                "components": {k: v for k, v in wait_components.items()},
                "notes": wait_notes,
            }
        )

        scores.sort(key=lambda entry: entry["score"], reverse=True)

        intel_ledger: List[Dict[str, object]] = []
        for other_id in sorted(drone.info_exchange_rounds.keys()):
            last_round = drone.info_exchange_rounds[other_id]
            age = current_round - last_round if last_round is not None else current_round
            intel_ledger.append({"drone": other_id, "last_round": last_round, "age": max(age, 0)})

        coordination_suggestion: Optional[Dict[str, Any]] = None
        if is_first_coordination_turn:
            try:
                coordination_suggestion = self._build_coordination_suggestion()
            except Exception:
                coordination_suggestion = None

        return {
            "scores": scores,
            "move_params": move_params,
            "next_waypoint": next_wp,
            "waypoint_timing": {
                "distance": timing_distance,
                "turns_remaining": timing_turns_remaining,
                "slack": timing_slack,
            }
            if timing_turns_remaining is not None and timing_distance is not None
            else None,
            "intel_ledger": intel_ledger,
            "coordination_suggestion": coordination_suggestion,
            "sector_unknown_target": nearest_sector_unknown,
        }


class _Drone_Language_Model:
    """Language model interactions for drone decisions."""

    def __init__(self, drone: "_Drone", model: str):
        self.drone = drone
        self.model = model

    def generate(self, messages: List[dict], temperature: float, prompt_char_len: Optional[int] = None) -> List[dict]:
        """Return a list of messages including the model response."""
        if self.model == "manual":
            try:
                import pyperclip

                pyperclip.copy(messages[-1]["content"])
            except Exception:
                pass
            content = input("Paste model result: ")
            messages.append({"role": "assistant", "content": content})
            if prompt_char_len is not None:
                approx_tokens = max(1, math.ceil(prompt_char_len / 4))
                print(f"Context length: ~{approx_tokens} tokens ({prompt_char_len} chars)")
            return messages

        try:
            from ollama import chat as ollama_chat
        except Exception as exc:
            LOGGER.log(f"Ollama unavailable: {exc}")
            raise

        request_started = time.perf_counter()
        response = ollama_chat(
            model=self.model,
            messages=messages,
            stream=False,
            format="json",
            options={"temperature": float(temperature)},
        )
        elapsed = time.perf_counter() - request_started
        content = response["message"]["content"]
        messages.append({"role": "assistant", "content": content})
        prompt_tokens = response.get("prompt_eval_count")
        completion_tokens = response.get("eval_count")
        eval_duration = response.get("eval_duration")
        duration_seconds = None
        if isinstance(eval_duration, (int, float)) and eval_duration > 0:
            duration_seconds = eval_duration / 1_000_000_000
        elif elapsed > 0:
            duration_seconds = elapsed

        if prompt_tokens is not None:
            ctx_msg = f"Context length: {prompt_tokens} tokens"
            if prompt_char_len is not None:
                ctx_msg += f" ({prompt_char_len} chars)"
            print(ctx_msg)
        elif prompt_char_len is not None:
            approx_tokens = max(1, math.ceil(prompt_char_len / 4))
            print(f"Context length: ~{approx_tokens} tokens ({prompt_char_len} chars)")

        if (
            isinstance(completion_tokens, (int, float))
            and completion_tokens > 0
            and duration_seconds
            and duration_seconds > 0
        ):
            tokens_per_second = completion_tokens / duration_seconds
            print(f"Tokens per second: {tokens_per_second:.2f} tok/s ({completion_tokens} completion tokens)")
        return messages


class _Drone_Aftermath:
    """Execute LM decisions, update state, and return turn outcomes."""

    def __init__(self, drone: "_Drone"):
        self.drone = drone

    def _normalize_token(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            if not value:
                return ""
            return str(value[0])
        return str(value)

    def _normalize_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return "\n".join(str(item) for item in value)
        if isinstance(value, dict):
            try:
                return json.dumps(value)
            except Exception:
                return str(value)
        return str(value)

    @contextmanager
    def _state_guard(self):
        lock = getattr(self.drone.sim, "state_lock", None)
        if lock:
            with lock:
                yield
            return
        yield

    def execute_turn(self, messages: List[dict]) -> Dict[str, object]:
        """Parse LM output, execute the action, and return a summary for Simulation."""
        result, parse_error = self._parse_messages(messages)
        if result is None:
            result = {}

        errors: List[str] = []
        if parse_error:
            errors.append(parse_error)

        is_first_coordination_turn = (
            self.drone.id == 1 and getattr(self.drone.sim, "round", None) == 1 and getattr(self.drone.sim, "turn", None) == 1
        )
        if is_first_coordination_turn:
            raw_message = result.get("message")
            _, payload = self._parse_broadcast_message(raw_message)
            valid_plan = False
            if isinstance(payload, dict):
                plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else None
                if isinstance(plan, dict):
                    assignments = plan.get("assignments")
                    rendezvous = plan.get("rendezvous")
                    valid_plan = isinstance(assignments, list) and isinstance(rendezvous, dict) and bool(assignments)
            action_txt = str(result.get("action") or "").strip().lower()
            if action_txt != "broadcast" or not valid_plan:
                coordination_payload = None
                try:
                    coordination_payload = self.drone.decision_support._build_coordination_suggestion()
                except Exception:
                    coordination_payload = None
                if isinstance(coordination_payload, dict) and coordination_payload.get("plan"):
                    result["action"] = "broadcast"
                    result["direction"] = None
                    result["message"] = coordination_payload
                    if not result.get("rationale"):
                        result["rationale"] = "Auto broadcast coverage plan."

        current_round = max(1, int(getattr(self.drone.sim, "round", 1) or 1))
        max_rounds = max(1, int(CONFIG.get("simulation", {}).get("max_rounds", 1)))
        rv_directive = getattr(self.drone, "rendezvous_directive", None)
        if isinstance(rv_directive, dict):
            rv_turn = rv_directive.get("turn")
            rv_cart = rv_directive.get("target_cartesian")
            if rv_cart is None and rv_directive.get("target"):
                try:
                    rv_cart = list(chess_to_cartesian(rv_directive["target"]))
                except Exception:
                    rv_cart = None
            try:
                rv_turn = int(rv_turn) if rv_turn is not None else None
            except (TypeError, ValueError):
                rv_turn = None
            if rv_turn and rv_cart and tuple(rv_cart) == tuple(self.drone.position) and current_round >= rv_turn:
                result["direction"] = None
                if current_round >= max_rounds:
                    result["action"] = "broadcast"
                    if not result.get("message"):
                        result["message"] = "Final rendezvous broadcast."
                    if not result.get("rationale"):
                        result["rationale"] = "Holding rendezvous for final coordination."
                else:
                    result["action"] = "wait"
                    if not result.get("rationale"):
                        result["rationale"] = "Holding rendezvous for final coordination."

        action = self._normalize_token(result.get("action") or "wait").strip().lower()
        rationale = self._normalize_text(result.get("rationale")).strip()
        moved = False
        direction = None
        broadcast_message = ""
        broadcast_payload = None

        if action == "move":
            direction = self._normalize_token(result.get("direction")).strip().lower()
            moved, move_error = self._execute_move(direction)
            if move_error:
                errors.append(move_error)
        elif action == "broadcast":
            broadcast_message, broadcast_payload, broadcast_error = self._execute_broadcast(result.get("message"))
            if broadcast_error:
                errors.append(broadcast_error)
        else:
            action = "wait"

        self._update_memory(result)

        decision_snapshot = None
        try:
            decision_snapshot = self.drone.decision_support.snapshot()
        except Exception:
            decision_snapshot = None

        return {
            "result": result,
            "action": action,
            "direction": direction,
            "broadcast_message": broadcast_message,
            "broadcast_payload": broadcast_payload,
            "rationale": rationale,
            "position": self.drone.position,
            "moved": moved,
            "errors": errors,
            "decision_support": decision_snapshot,
        }

    def execute_move(self, direction: str) -> Tuple[bool, Optional[str]]:
        """Execute a movement request outside the normal LM pipeline."""
        return self._execute_move(direction)

    def _parse_messages(self, messages: Optional[List[dict]]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not messages:
            return None, "ERROR: Empty model response; defaulting to wait."
        try:
            return json.loads(messages[-1]["content"]), None
        except Exception as exc:
            LOGGER.log(f"Drone {self.drone.id} response parse failed: {exc}")
            return None, "ERROR: Invalid JSON response; defaulting to wait."

    def _execute_move(self, direction: str) -> Tuple[bool, Optional[str]]:
        drone = self.drone
        legal_directions = [step["direction"] for step in drone._legal_movement_steps()]
        if direction not in legal_directions:
            return False, f"Invalid direction: '{direction}', allowed={legal_directions} -> Aborted movement"

        dx, dy = DIRECTION_MAP.get(direction, (0, 0))
        nx, ny = drone.position[0] + dx, drone.position[1] + dy
        if not on_board(nx, ny):
            error = f"Invalid direction: '{direction}', allowed={legal_directions} -> Aborted movement"
            LOGGER.log(f"ERROR: Drone {drone.id} attempted to move to {(nx, ny)}.")
            return False, error

        with self._state_guard():
            drone.sim.board[drone.position[0]][drone.position[1]].remove_drone(drone)
            drone.position = (nx, ny)
            drone.sim.board[nx][ny].add_drone(drone)
            drone.mission_report.append(drone.position)
            drone.knowledge.update_board_report()
            drone.mission_support.advance_leg_progress()
            LOGGER.log(f"Drone {drone.id} moved to {cartesian_to_chess(drone.position)}.")
        return True, None

    def _parse_broadcast_message(self, raw_message) -> Tuple[str, Optional[object]]:
        if isinstance(raw_message, str):
            msg = raw_message.strip()
            if not msg:
                return "", None
            try:
                return msg, json.loads(msg)
            except Exception:
                return msg, None
        if isinstance(raw_message, (dict, list)):
            payload = raw_message
            try:
                msg = json.dumps(raw_message)
            except Exception:
                msg = str(raw_message)
            return msg, payload
        return "", None

    def _execute_broadcast(self, raw_message) -> Tuple[str, Optional[object], Optional[str]]:
        drone = self.drone
        msg, payload = self._parse_broadcast_message(raw_message)
        new_edges = 0
        if not msg:
            with self._state_guard():
                try:
                    drone.sim.broadcast_count += 1
                except Exception:
                    pass
            LOGGER.log(f"Broadcast by Drone {drone.id} added {new_edges} new edge(s).")
            return "", payload, "ERROR: Empty broadcast."

        with self._state_guard():
            pre_edges = set(getattr(drone.sim, "reported_edges", []) or [])
            try:
                drone.sim.broadcast_count += 1
            except Exception:
                pass

            if payload:
                drone.mission_support.apply_plan_directive(payload)

            tile = drone.sim.board[drone.position[0]][drone.position[1]]
            for target_drone in tile.drones:
                if target_drone.id != drone.id:
                    target_drone.rx_buffer += f"Drone {drone.id} broadcasted: {msg}\n"
                if payload:
                    target_drone.mission_support.apply_plan_directive(payload)
                drone.knowledge.provide_intelligence_to(target_drone)
            post_edges = set(getattr(drone.sim, "reported_edges", []) or [])
            new_edges = len(post_edges - pre_edges)
        LOGGER.log(f"Broadcast by Drone {drone.id} added {new_edges} new edge(s).")
        return msg, payload, None

    def _sanitize_memory(self, mem_txt: str, prior_mem: str) -> str:
        allowed_prefixes = ("PLAN:", "SECTOR:", "RV:", "FIG:", "POSSIBLE:", "NOTE:", "VISITED:")
        max_chars = 800
        max_line_len = 200
        blocked_tokens = ("legal movements", "decision support", "broadcast rx buffer")

        def _lines(text: str) -> List[str]:
            return [line.strip() for line in text.splitlines() if line.strip()]

        prior_lines = _lines(prior_mem)
        new_lines = _lines(mem_txt)

        selected: List[str] = []
        for line in new_lines:
            if line.lower().startswith("memory:"):
                line = line.split(":", 1)[1].strip()
            if line.startswith(allowed_prefixes):
                selected.append(line[:max_line_len])
                continue
            lowered = line.lower()
            if any(token in lowered for token in blocked_tokens):
                continue
            if any(token in lowered for token in ("sector", "rendezvous", "plan")) and len(line) <= 120:
                selected.append(line[:max_line_len])

        def _ensure_prefix(prefix: str) -> None:
            if any(line.startswith(prefix) for line in selected):
                return
            for line in prior_lines:
                if line.startswith(prefix):
                    selected.append(line[:max_line_len])
                    return

        for prefix in ("PLAN:", "SECTOR:", "RV:"):
            _ensure_prefix(prefix)

        for line in prior_lines:
            if line.startswith("VISITED:") and line not in selected:
                selected.append(line[:max_line_len])

        if not selected:
            for line in new_lines:
                lowered = line.lower()
                if any(token in lowered for token in blocked_tokens):
                    continue
                selected.append(line[:max_line_len])
                if len(selected) >= 3:
                    break

        deduped: List[str] = []
        for line in selected:
            if line not in deduped:
                deduped.append(line)

        joined = "\n".join(deduped)
        if len(joined) > max_chars:
            trimmed: List[str] = []
            total = 0
            for line in deduped:
                extra = len(line) + (1 if trimmed else 0)
                if total + extra > max_chars:
                    continue
                trimmed.append(line)
                total += extra
            joined = "\n".join(trimmed) if trimmed else joined[:max_chars]
        return joined

    def _update_memory(self, result: Dict[str, Any]) -> None:
        mem_txt = self._normalize_text(result.get("memory")).strip()
        if mem_txt:
            self.drone.memory = self._sanitize_memory(mem_txt, self.drone.memory)
        self.drone.memory = self._sanitize_memory(self.drone.memory, self.drone.memory)
        vx, vy = self.drone.position
        token = f"VISITED: {cartesian_to_chess((vx, vy))}"
        if token not in self.drone.memory:
            self.drone.memory += ("" if self.drone.memory.endswith("\n") else "\n") + token
