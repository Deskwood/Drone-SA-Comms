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
                    neighbors.append(f"{direction_from_vector((dx, dy))}: {tile.figure.color}")
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
            perimeter.append(perimeter[0])

        waypoints = [start_pos] + perimeter + [start_pos]
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
        current_turn = 1
        start_pos = tuple(self.drone.position)

        max_rounds = max(1, CONFIG.get("simulation", {}).get("max_rounds", 1))
        rv_turn = max_rounds - 1
        rv_cart = None
        if self.drone.rendezvous_directive:
            rv_cart = self.drone.rendezvous_directive.get("target_cartesian")
            if rv_cart is None and self.drone.rendezvous_directive.get("target"):
                rv_cart = list(chess_to_cartesian(self.drone.rendezvous_directive["target"]))
            rv_turn = max(1, self.drone.rendezvous_directive.get("turn") or rv_turn)
        if rv_cart is None:
            rv_cart = start_pos

        waypoints = self._build_sector_perimeter_waypoints(bounds, start_pos)
        if tuple(rv_cart) != waypoints[-1]:
            waypoints.append(tuple(rv_cart))
        legs = list(zip(waypoints[:-1], waypoints[1:]))
        distances = [chebyshev_distance(a, b) for a, b in legs]
        total_dist = sum(distances)
        available_turns = max(0, rv_turn - current_turn)
        slack = max(0, available_turns - total_dist)

        cumulative = 0
        last_arrival = current_turn
        for idx, ((wp_start, wp_end), dist) in enumerate(zip(legs, distances)):
            cumulative += dist
            if total_dist > 0 and available_turns >= total_dist:
                extra = int(round(slack * (cumulative / total_dist)))
                arrival_turn = current_turn + cumulative + extra
            else:
                arrival_turn = current_turn + cumulative
            if idx == len(legs) - 1 and available_turns >= total_dist:
                arrival_turn = rv_turn
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
            if current_round >= arrival_turn + duration:
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
                    requested_turn = int(turn) if turn is not None else max_rounds - 1
                    rv_turn = max(1, min(requested_turn, max_rounds))
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
            formatted = f"{value:.2f}"
            if "." in formatted:
                formatted = formatted.rstrip("0").rstrip(".")
            return formatted

        for idx, entry in enumerate(sorted_scores):
            components = entry.get("components", {})
            if components:
                sorted_components = sorted(components.items(), key=lambda item: (-abs(item[1]), item[0]))
                component_text = ", ".join(
                    f"{name} ({value:+.2f})" for name, value in sorted_components[:3]
                )
            else:
                component_text = "n/a"
            notes = entry.get("notes") or []
            note_text = " | notes: " + "; ".join(notes) if notes else ""
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
            qualitative = _qualitative_label(idx, entry.get("score", 0.0))
            score_text = _format_score(entry.get("score", 0.0))
            lines.append(f"{action_text} -> {qualitative} ({score_text}) | factors: {component_text}{note_text}")

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
        same_tile_drones = [
            f"Drone {other.id}"
            for other in drone.sim.board[drone.position[0]][drone.position[1]].drones
            if other.id != drone.id
        ]

        fig_here = "None"
        if drone.sim.board[drone.position[0]][drone.position[1]].figure:
            fig_here = drone.sim.board[drone.position[0]][drone.position[1]].figure.figure_type

        neighbor_figures = drone.knowledge.visible_neighbor_figures()
        legal_steps = drone._legal_movement_steps()
        legal_movements = ", ".join(
            [f"{lms['direction']} to {cartesian_to_chess(lms['new_position'])}" for lms in legal_steps]
        )
        allowed_directions = ", ".join([step["direction"] for step in legal_steps]) or "none"
        collected_figure_information = drone.knowledge.collected_figure_information_text()

        lines: List[str] = []
        lines.append(f"Current round number: {drone.sim.round} of {CONFIG['simulation']['max_rounds']} rounds.")
        lines.append(f"Current position: {cartesian_to_chess(drone.position)}")
        lines.append("AllowedActions: wait, move, broadcast")
        lines.append(f"AllowedDirections: {allowed_directions}")
        lines.append(f"Legal movements: {legal_movements}")
        lines.append(f"Visible drones at position: {', '.join(same_tile_drones) if same_tile_drones else 'None'}")
        lines.append(f"Visible figure at position: {fig_here}")
        lines.append(f"Visible neighbor figures: {neighbor_figures or 'None'}")
        lines.append(f"Memory: {drone.memory}")
        lines.append(f"Collected figure information: {collected_figure_information}")
        if getattr(drone, "assigned_sector", None):
            lines.append(f"Assigned coverage sector: {drone.sector_summary()}")
        if getattr(drone, "rendezvous_directive", None):
            rv = drone.rendezvous_directive
            lines.append(f"Rendezvous directive: {rv['target']} on turn {rv['turn']}")
        rx_buffer = drone.mission_support.consume_rx_buffer().strip()
        if rx_buffer:
            rx_buffer = rx_buffer.replace("\n", " | ")
            if len(rx_buffer) > 300:
                rx_buffer = rx_buffer[:300].rstrip() + "...(truncated)"
        lines.append(f"Broadcast Rx Buffer: {rx_buffer or 'None'}")
        if drone.id == 1 and getattr(drone.sim, "round", None) == 1 and getattr(drone.sim, "turn", None) == 1:
            total_drones = CONFIG["simulation"].get("num_drones", len(getattr(drone.sim, "drones", [])))
            lines.append(
                "Special directive (MANDATORY): As Drone 1 on the opening turn, broadcast a coverage plan assigning "
                "every drone their sector before taking other actions."
            )
            lines.append(
                f"Ensure the broadcast is valid JSON with plan->assignments entries for all {total_drones} drones "
                "and describe each sector clearly. This overrides decision support ranking."
            )

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

    def _count_unknown_neighbors(self, pos: Tuple[int, int]) -> int:
        drone = self.drone
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = pos[0] + dx, pos[1] + dy
                if not on_board(nx, ny):
                    continue
                key = cartesian_to_chess((nx, ny))
                info = drone.local_board.get(key, {"color": "unknown", "type": "unknown"})
                if info["type"] in {"unknown", "a possible target"} or info["color"] == "unknown":
                    count += 1
        return count

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
        rendezvous_tile = cartesian_to_chess(sim.drone_base)
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
        board_w = int(CONFIG.get("board", {}).get("width", 8))
        board_h = int(CONFIG.get("board", {}).get("height", 8))

        move_cfg = scoring_cfg.get("move", {})
        broadcast_cfg = scoring_cfg.get("broadcast", {})
        wait_cfg = scoring_cfg.get("wait", {})

        waypoint_progress_reward = move_cfg.get("waypoint_progress_reward_per_step", 1.0)
        waypoint_regression_penalty = move_cfg.get("waypoint_regression_penalty_per_step", -1.0)
        deadline_slack_bonus = move_cfg.get("deadline_slack_bonus", 0.2)
        deadline_slack_penalty = move_cfg.get("deadline_slack_penalty", -1.0)
        tolerance_bonus = move_cfg.get("tolerance_bonus", 0.5)
        unknown_tile_bonus = move_cfg.get("unknown_tile_bonus", 1.0)
        possible_target_bonus = move_cfg.get("possible_target_bonus", 1.2)
        figure_hint_bonus = move_cfg.get("figure_hint_bonus", 0.6)
        known_figure_penalty = move_cfg.get("known_figure_penalty", -0.2)
        known_empty_penalty = move_cfg.get("known_empty_penalty", -0.6)
        unknown_color_bonus = move_cfg.get("unknown_color_bonus", 0.5)
        unknown_neighbor_bonus = move_cfg.get("unknown_neighbor_bonus_per_tile", 0.2)
        novel_tile_bonus = move_cfg.get("novel_tile_bonus", 0.6)
        revisit_penalty = move_cfg.get("revisit_penalty", -1.0)
        leg_alignment_reward = move_cfg.get("leg_alignment_reward", 0.6)
        leg_alignment_penalty = move_cfg.get("leg_alignment_penalty", -0.6)
        leg_travel_reward = move_cfg.get("leg_travel_reward", 0.9)
        leg_travel_penalty = move_cfg.get("leg_travel_penalty", -0.3)
        leg_sideways_reward = move_cfg.get("leg_sideways_reward", 1.2)
        leg_sideways_probe_bonus = move_cfg.get("leg_sideways_inspection_bonus", leg_sideways_reward)
        leg_along_penalty = move_cfg.get("leg_along_penalty", -0.7)
        leg_start_progress_reward = move_cfg.get("leg_start_progress_reward", 1.2)
        leg_start_regression_penalty = move_cfg.get("leg_start_regression_penalty", -1.0)
        sector_alignment_reward = move_cfg.get("sector_alignment_reward", 0.8)
        sector_inside_bonus = move_cfg.get("sector_inside_bonus", 0.5)
        late_penalty_multiplier = move_cfg.get("late_penalty_multiplier", 2.0)
        border_edge_bonus = move_cfg.get("board_edge_bias_bonus", 0.2)
        border_edge_range = int(move_cfg.get("board_edge_bias_range", 1))
        sector_unknown_probe_bonus = move_cfg.get("sector_unknown_probe_bonus", 0.0)
        sector_unknown_probe_min_slack = float(move_cfg.get("sector_unknown_probe_min_slack", 1))

        broadcast_base_penalty = broadcast_cfg.get("base_penalty", -0.5)
        broadcast_recipient_factor = broadcast_cfg.get("recipient_factor", 0.8)
        broadcast_staleness_factor = broadcast_cfg.get("staleness_factor", 0.4)
        coordination_broadcast_bonus = broadcast_cfg.get("first_turn_coordination_bonus", 2.5)
        is_first_coordination_turn = (
            drone.id == 1 and getattr(drone.sim, "round", None) == 1 and getattr(drone.sim, "turn", None) == 1
        )

        wait_default_score = wait_cfg.get("default_score", -1.0)
        wait_idle_component = wait_cfg.get("idle_penalty_component", -1.0)
        wait_holding_score = wait_cfg.get("holding_position_score", -0.2)
        wait_holding_component = wait_cfg.get("holding_pattern_component", 0.3)

        scores: List[Dict[str, object]] = []
        current_round = drone.sim.round
        next_wp = drone._next_mission_waypoint()
        target_pos = tuple(next_wp["leg_end"]) if next_wp else None
        current_leg = next_wp if next_wp and next_wp.get("leg_start") else None
        current_leg_distance = drone._distance_to_leg(drone.position, current_leg) if current_leg else None
        current_leg_end_distance = drone._distance_to_leg_end(drone.position, current_leg) if current_leg else None
        current_leg_start: Optional[Tuple[int, int]] = None
        current_leg_start_distance: Optional[int] = None
        if current_leg:
            start_vec = current_leg.get("leg_start")
            if isinstance(start_vec, (list, tuple)) and len(start_vec) == 2:
                try:
                    current_leg_start = (int(start_vec[0]), int(start_vec[1]))
                    current_leg_start_distance = chebyshev_distance(drone.position, current_leg_start)
                except (TypeError, ValueError):
                    current_leg_start = None
                    current_leg_start_distance = None

        sector_bounds = drone._sector_bounds() if hasattr(drone, "_sector_bounds") else None
        current_sector_distance = self._distance_to_sector(drone.position, sector_bounds) if sector_bounds else None
        sector_unknown_tiles: List[Tuple[int, int]] = []
        nearest_sector_unknown: Optional[Dict[str, object]] = None
        closest_probe_distance: Optional[int] = None
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
                        sector_unknown_tiles.append(pos)
                        dist = chebyshev_distance(drone.position, pos)
                        if best_unknown_dist is None or dist < best_unknown_dist:
                            best_unknown_dist = dist
                            best_unknown_pos = pos
            if best_unknown_pos is not None and best_unknown_dist is not None:
                nearest_sector_unknown = {"tile": cartesian_to_chess(best_unknown_pos), "distance": best_unknown_dist}
                closest_probe_distance = best_unknown_dist

        visited_tiles: set = set()
        for pos in getattr(drone, "mission_report", []) or []:
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                try:
                    visited_tiles.add((int(pos[0]), int(pos[1])))
                except (TypeError, ValueError):
                    continue

        for step in drone._legal_movement_steps():
            direction = step["direction"]
            new_pos = tuple(step["new_position"])
            tile_key = cartesian_to_chess(new_pos)
            tile_info = drone.local_board.get(tile_key, {"color": "unknown", "type": "unknown"})

            score = 0.0
            score_components: Dict[str, float] = {}
            notes: List[str] = []
            new_start_distance: Optional[int] = None

            if current_leg:
                new_leg_distance = drone._distance_to_leg(new_pos, current_leg)
                if current_leg_start is not None:
                    new_start_distance = chebyshev_distance(new_pos, current_leg_start)
                    if current_leg_start_distance is not None and current_leg_start_distance > 0:
                        delta_start = current_leg_start_distance - new_start_distance
                        score_components["leg_start_focus"] = round(delta_start, 2)
                        if delta_start > 0:
                            score += leg_start_progress_reward
                            notes.append("closing distance to leg start")
                        elif delta_start < 0:
                            score += leg_start_regression_penalty
                            notes.append("drifting from leg start")
                if current_leg_distance is not None and new_leg_distance is not None:
                    delta_leg = current_leg_distance - new_leg_distance
                    score_components["leg_alignment"] = round(delta_leg, 2)
                    if delta_leg > 0:
                        score += leg_alignment_reward
                        notes.append("aligning with coverage leg")
                    elif delta_leg < 0:
                        score += leg_alignment_penalty
                        notes.append("drifting from coverage leg")
                    if current_leg_distance and current_leg_distance > 0 and new_leg_distance == 0:
                        notes.append("entered coverage leg corridor")
                    orientation = (current_leg.get("orientation") or "").lower()
                    dx = new_pos[0] - drone.position[0]
                    dy = new_pos[1] - drone.position[1]
                    if current_leg_distance and current_leg_distance > 0:
                        if orientation == "vertical":
                            if abs(dx) == 1 and abs(dy) == 0 and new_leg_distance < current_leg_distance:
                                score += leg_sideways_reward
                                notes.append("sidestep toward vertical leg")
                            if abs(dx) == 0 and abs(dy) == 1 and delta_leg <= 0:
                                score += leg_along_penalty
                                notes.append("slide along vertical leg before aligning")
                        elif orientation == "horizontal":
                            if abs(dy) == 1 and abs(dx) == 0 and new_leg_distance < current_leg_distance:
                                score += leg_sideways_reward
                                notes.append("sidestep toward horizontal leg")
                            if abs(dy) == 0 and abs(dx) == 1 and delta_leg <= 0:
                                score += leg_along_penalty
                                notes.append("slide along horizontal leg before aligning")
                if current_leg_distance == 0 and new_leg_distance == 1:
                    score += leg_sideways_probe_bonus
                    score_components["leg_sideways_probe"] = round(leg_sideways_probe_bonus, 2)
                    notes.append("sideways probe off leg")
                if new_leg_distance == 0 and current_leg_end_distance is not None:
                    if new_start_distance is not None and new_start_distance > 0:
                        if current_leg_start_distance is not None and current_leg_start_distance > 0:
                            score += leg_start_regression_penalty
                            notes.append("skipping leg start")
                    else:
                        new_end_distance = drone._distance_to_leg_end(new_pos, current_leg)
                        if new_end_distance is not None:
                            delta_end = current_leg_end_distance - new_end_distance
                            if delta_end > 0:
                                score += leg_travel_reward
                                score_components["leg_travel"] = round(delta_end, 2)
                                notes.append("progress along current leg")
                            elif delta_end < 0:
                                score += leg_travel_penalty
                                notes.append("retreat along current leg")

            if target_pos:
                current_dist = chebyshev_distance(drone.position, target_pos)
                new_dist = chebyshev_distance(new_pos, target_pos)
                turns_remaining = max(0, next_wp["turn"] - (current_round + 1))
                slack = turns_remaining - new_dist

                delta = current_dist - new_dist
                score_components["plan_progress"] = delta
                if delta > 0:
                    score += waypoint_progress_reward
                    notes.append("closer to waypoint")
                elif delta < 0:
                    regression_penalty = waypoint_regression_penalty
                    if slack < 0:
                        regression_penalty += late_penalty_multiplier * abs(slack)
                    score += regression_penalty
                    notes.append("farther from waypoint")

                score_components["deadline_margin"] = float(slack)
                if slack >= 0:
                    score += deadline_slack_bonus
                    notes.append("on schedule")
                else:
                    slack_penalty = deadline_slack_penalty * max(1.0, late_penalty_multiplier)
                    score += slack_penalty
                    notes.append("missing deadline")

                if next_wp.get("duration_turns", 0) > 0:
                    buffer_turns = max(0, slack)
                    score_components["rendezvous_buffer"] = float(buffer_turns)
                    suffix = "turn" if buffer_turns == 1 else "turns"
                    notes.append(f"{buffer_turns} {suffix} left to reach rendezvous on time")
                    if buffer_turns == 0:
                        if slack < 0:
                            notes.append("rendezvous already overdue - move immediately")
                        else:
                            notes.append("critical rendezvous move - no buffer remaining")

                if new_dist <= 0:
                    score += tolerance_bonus
                    score_components["within_tolerance"] = 1.0

                if sector_unknown_tiles and sector_unknown_probe_bonus and slack >= sector_unknown_probe_min_slack:
                    if closest_probe_distance is not None and slack >= closest_probe_distance:
                        score += sector_unknown_probe_bonus
                        score_components["sector_unknown_probe"] = round(sector_unknown_probe_bonus, 2)
                        notes.append(
                            f"sector probe feasible (nearest unknown {closest_probe_distance} steps away without delaying leg)"
                        )

            if tile_info["type"] == "unknown":
                score += unknown_tile_bonus
                score_components["discover_type"] = unknown_tile_bonus
                notes.append("unidentified tile")
            elif tile_info["type"] == "a possible target":
                score += possible_target_bonus
                score_components["possible_target"] = possible_target_bonus
                notes.append("possible blocker")
            elif tile_info["type"] == "any figure":
                score += figure_hint_bonus
                score_components["figure_hint"] = figure_hint_bonus
                notes.append("figure nearby")
            elif tile_info["type"] in FIGURE_TYPES and known_figure_penalty:
                score += known_figure_penalty
                score_components["known_figure"] = known_figure_penalty
                notes.append("known figure tile")
            elif tile_info["type"] == "n/a" and known_empty_penalty:
                score += known_empty_penalty
                score_components["known_empty"] = known_empty_penalty
                notes.append("known empty tile")

            if tile_info["color"] == "unknown":
                score += unknown_color_bonus
                score_components["discover_color"] = unknown_color_bonus

            if new_pos in visited_tiles:
                score += revisit_penalty
                score_components["revisit_penalty"] = revisit_penalty
                notes.append("revisiting tile")
            elif novel_tile_bonus:
                score += novel_tile_bonus
                score_components["new_tile"] = novel_tile_bonus
                notes.append("new tile")

            unknown_neighbors = self._count_unknown_neighbors(new_pos)
            if unknown_neighbors:
                score += unknown_neighbors * unknown_neighbor_bonus
                score_components["unknown_neighbors"] = float(unknown_neighbors)
                notes.append(f"{unknown_neighbors} unknown neighbors")

            if sector_bounds:
                new_sector_distance = self._distance_to_sector(new_pos, sector_bounds)
                if new_sector_distance is not None:
                    if sector_alignment_reward and current_sector_distance is not None and new_sector_distance < current_sector_distance:
                        score += sector_alignment_reward
                        score_components["sector_alignment"] = round(current_sector_distance - new_sector_distance, 2)
                        notes.append("moving toward assigned sector")
                    if sector_inside_bonus and new_sector_distance == 0:
                        score += sector_inside_bonus
                        score_components["sector_inside"] = round(sector_inside_bonus, 2)
                        notes.append("within assigned sector")

            if border_edge_bonus and border_edge_range >= 0:
                border_distance = min(
                    new_pos[0],
                    board_w - 1 - new_pos[0],
                    new_pos[1],
                    board_h - 1 - new_pos[1],
                )
                if border_distance <= border_edge_range:
                    bias_strength = max(1, border_edge_range - border_distance + 1)
                    bonus = border_edge_bonus * bias_strength
                    score += bonus
                    score_components["border_bias"] = round(bonus, 2)
                    notes.append("nudged toward board edge")

            scores.append(
                {
                    "action": "move",
                    "label": direction,
                    "score": round(score, 2),
                    "components": {k: round(v, 2) for k, v in score_components.items() if abs(v) >= 0.01},
                    "notes": notes,
                }
            )

        tile = drone.sim.board[drone.position[0]][drone.position[1]]
        recipients = [d for d in tile.drones if d.id != drone.id]
        broadcast_components: Dict[str, float] = {"recipients": float(len(recipients))}
        broadcast_notes: List[str] = []
        broadcast_score = broadcast_base_penalty
        if recipients:
            ages: List[int] = []
            for target in recipients:
                last_round = drone.info_exchange_rounds.get(target.id)
                age = current_round - last_round if last_round is not None else current_round
                ages.append(max(age, 0))
            avg_age = sum(ages) / len(ages) if ages else 0.0
            broadcast_components["avg_staleness"] = round(avg_age, 2)
            broadcast_components["max_staleness"] = float(max(ages) if ages else 0)
            broadcast_score = len(recipients) * broadcast_recipient_factor + avg_age * broadcast_staleness_factor
            if avg_age > 0:
                broadcast_notes.append("recipients have stale intel")
        else:
            broadcast_notes.append("no co-located drones")

        if is_first_coordination_turn and coordination_broadcast_bonus:
            broadcast_score += coordination_broadcast_bonus
            broadcast_components["coordination_bonus"] = float(coordination_broadcast_bonus)
            broadcast_notes.append("first-turn coverage assignment priority")

        scores.append(
            {
                "action": "broadcast",
                "label": "share",
                "score": round(broadcast_score, 2),
                "components": {k: v for k, v in broadcast_components.items()},
                "notes": broadcast_notes,
            }
        )

        wait_score = wait_default_score
        wait_notes: List[str] = ["no progress"]
        wait_components: Dict[str, float] = {"idle_penalty": wait_idle_component}
        if target_pos:
            dist_to_target = chebyshev_distance(drone.position, target_pos)
            if dist_to_target == 0 and next_wp and current_round < next_wp["turn"]:
                wait_score = wait_holding_score
                wait_notes = ["holding position at waypoint"]
                wait_components["holding_pattern"] = wait_holding_component
        scores.append(
            {
                "action": "wait",
                "label": "hold",
                "score": round(wait_score, 2),
                "components": {k: round(v, 2) for k, v in wait_components.items()},
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
            "next_waypoint": next_wp,
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

        action = (result.get("action") or "wait").strip().lower()
        rationale = (result.get("rationale") or "").strip()
        moved = False
        direction = None
        broadcast_message = ""
        broadcast_payload = None

        if action == "move":
            direction = (result.get("direction") or "").strip().lower()
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
        if not msg:
            with self._state_guard():
                try:
                    drone.sim.broadcast_count += 1
                except Exception:
                    pass
            return "", payload, "ERROR: Empty broadcast."

        with self._state_guard():
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
        mem_txt = (result.get("memory") or "").strip()
        if mem_txt:
            self.drone.memory = self._sanitize_memory(mem_txt, self.drone.memory)
        self.drone.memory = self._sanitize_memory(self.drone.memory, self.drone.memory)
        vx, vy = self.drone.position
        token = f"VISITED:{vx},{vy}"
        if token not in self.drone.memory:
            self.drone.memory += ("" if self.drone.memory.endswith("\n") else "\n") + token
