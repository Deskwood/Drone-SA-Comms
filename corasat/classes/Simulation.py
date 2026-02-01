"""Simulation orchestrator for Corasat games."""
from __future__ import annotations

import json
import os
import pprint
import random
import threading
import time
import pygame
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional

from classes.Core import (
    CONFIG,
    COLORS,
    FIGURE_TYPES,
    _Figure,
    _Tile,
    cartesian_to_chess,
    chess_to_cartesian,
    format_edge,
    hsv_to_rgb255,
    resolve_data_path,
)
from classes.Drone import _Drone
from classes.Exporter import LOGGER
from classes.GUI import _SimulationGUI


class Simulation:
    """Run a full Corasat game, coordinating drones, board state, and GUI."""

    def __init__(self, game_index: int = 1, total_games: int = 1, seed: Optional[int] = None):
        if not CONFIG["simulation"].get("use_gui", True):
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        self.turn = 1
        self.round = 1
        self.seed = seed
        self.rules = self._load_rules()

        self.grid_size = (CONFIG["board"]["width"], CONFIG["board"]["height"])
        self.max_rounds = CONFIG["simulation"]["max_rounds"]
        self.num_drones = CONFIG["simulation"]["num_drones"]
        self.models = CONFIG["simulation"]["models"]
        self.model_index = CONFIG["simulation"]["model_index"]
        self.model = self.models[self.model_index]
        LOGGER.log(f"Using model: {self.model}")

        self.total_games = self._normalize_game_count(total_games)
        self.game_index = self._normalize_game_index(game_index, self.total_games)
        self.gui = self._create_gui()

        self.plans: Dict[int, List[str]] = {}
        self._last_logged_plans: Dict[int, str] = {}
        self.state_lock = threading.Lock()

        self.board = [[_Tile(x, y) for y in range(self.grid_size[1])] for x in range(self.grid_size[0])]
        self.figures: List[_Figure] = []
        self.drones: List[_Drone] = []

        self.gt_edges: List[str] = []
        self.reported_edges: List[str] = []
        self.correct_edge_counter = 0
        self.false_edge_counter = 0
        self.score = 0.0
        self.broadcast_count = 0
        self.wait_actions = 0
        self.total_actions = 0
        self.wait_rate: Optional[float] = None
        self.broadcast_rate: Optional[float] = None
        self.total_visits = 0
        self.unique_visits = 0
        self.redundant_visits = 0
        self._broadcast_correct_sum = 0
        self.broadcast_effectiveness: Optional[float] = None
        self.broadcast_effectiveness_per_broadcast: Optional[float] = None
        self.runtime_s: Optional[float] = None
        self.avg_turn_duration_s: Optional[float] = None
        self.timeout_count = 0
        self.timeout_reason: Optional[str] = None
        self._metrics_finalized = False
        self._run_start_time: Optional[float] = None

        self._create_figures()
        self.drone_base = self._resolve_drone_base()
        self._create_drones()

        self.executor = ThreadPoolExecutor(max_workers=1)
        self._current_future: Optional[Future] = None
        self._thinking = False
        self._abort_requested = False
        self._abort_reason = ""
        self._watchdog_triggered = False
        self.rendezvous_success: Optional[bool] = None
        self._rendezvous_checked = False
        self._rendezvous_turn = max(1, self.max_rounds - 1)
        self._rendezvous_tile = self._board_center_cartesian()

    def _board_center_cartesian(self) -> tuple:
        width = max(1, int(CONFIG.get("board", {}).get("width", 8)))
        height = max(1, int(CONFIG.get("board", {}).get("height", 8)))
        center_x = min(width - 1, width // 2)
        center_y = min(height - 1, height // 2)
        return (center_x, center_y)

    def _rendezvous_target(self) -> tuple:
        for drone in self.drones:
            rv = getattr(drone, "rendezvous_directive", None)
            if isinstance(rv, dict):
                cart = rv.get("target_cartesian")
                if isinstance(cart, (list, tuple)) and len(cart) == 2:
                    try:
                        return (int(cart[0]), int(cart[1]))
                    except (TypeError, ValueError):
                        pass
                tile = rv.get("target")
                if isinstance(tile, str):
                    try:
                        return chess_to_cartesian(tile)
                    except Exception:
                        pass
        return self._rendezvous_tile

    def _check_rendezvous(self) -> None:
        target = self._rendezvous_target()
        positions = []
        success = True
        for drone in self.drones:
            positions.append(cartesian_to_chess(drone.position))
            if tuple(drone.position) != tuple(target):
                success = False
        self.rendezvous_success = success
        target_label = cartesian_to_chess(target)
        LOGGER.log(
            f"Rendezvous check (turn {self._rendezvous_turn}, target {target_label}): "
            f"{'success' if success else 'failed'}; positions={positions}"
        )

    def _load_rules(self) -> str:
        """Load and personalize rules for this simulation instance."""
        rules_path = CONFIG.get("rules_path", "rules.txt")
        resolved = resolve_data_path(str(rules_path))
        with open(resolved, "r", encoding="utf-8") as file_handle:
            rules = file_handle.read()
        return rules.replace("NUMBER_OF_DRONES", str(CONFIG["simulation"]["num_drones"]))

    def _normalize_game_count(self, total_games: int) -> int:
        if not isinstance(total_games, int) or total_games <= 0:
            return 1
        return total_games

    def _normalize_game_index(self, game_index: int, total_games: int) -> int:
        if not isinstance(game_index, int) or game_index <= 0:
            game_index = 1
        return min(game_index, total_games)

    def _create_gui(self):
        """Instantiate GUI when enabled and pygame is available."""
        if not CONFIG["simulation"].get("use_gui", True):
            return None
        if pygame is None:
            LOGGER.log("pygame not available; GUI disabled.")
            return None
        return _SimulationGUI(self)

    def _resolve_drone_base(self) -> tuple:
        """Select the starting tile for drones (white king if available)."""
        if self.figures:
            white_king = next(
                (figure for figure in self.figures if figure.color == "white" and figure.figure_type == "king"),
                self.figures[0],
            )
            return white_king.position
        return (0, 0)

    def _progress_string(self, round_num: Optional[int] = None, turn_num: Optional[int] = None) -> str:
        round_value = round_num if round_num is not None else self.round
        turn_value = turn_num if turn_num is not None else self.turn
        round_value = round_value if round_value else 1
        turn_value = turn_value if turn_value else 1
        return f"{self.game_index}.{round_value}.{turn_value}/{self.total_games}.{self.max_rounds}.{self.num_drones}"

    def progress_caption(self, round_num: Optional[int] = None, turn_num: Optional[int] = None) -> str:
        return f"Simulation - Round {self._progress_string(round_num, turn_num)}"

    # Figures
    def _create_figures(self) -> None:
        """Populate the board with figures and precompute ground truth edges."""
        LOGGER.log("Creating figures based on configuration.")
        self.figures = []
        figures_cfg = CONFIG.get("figures", {})
        sim_cfg = CONFIG.get("simulation", {})
        width, height = self.grid_size
        randomize = bool(sim_cfg.get("randomize_figures", False))
        rng = random
        if randomize:
            try:
                all_tiles = [(x, y) for x in range(width) for y in range(height)]
                rng.shuffle(all_tiles)
                cursor = 0
                req = []
                for color in COLORS:
                    for ftype in FIGURE_TYPES:
                        lst = figures_cfg.get(color, {}).get(ftype, [])
                        cnt = len(lst) if lst else (
                            1 if ftype in ("king", "queen") else (2 if ftype in ("rook", "bishop", "knight") else 3)
                        )
                        req.append((color, ftype, cnt))
                out = {c: {t: [] for t in FIGURE_TYPES} for c in COLORS}
                for color, ftype, cnt in req:
                    picks = all_tiles[cursor:cursor + cnt]
                    cursor += cnt
                    out[color][ftype] = [list(p) for p in picks]
                figures_cfg = out
                LOGGER.log("Figure positions RANDOMIZED.")
            except Exception as exc:
                LOGGER.log(f"Randomization failed ({exc}); using configured positions.")
        for color in COLORS:
            for figure_type in FIGURE_TYPES:
                for position in figures_cfg.get(color, {}).get(figure_type, []):
                    self.figures.append(_Figure(tuple(position), color, figure_type))
        for figure in self.figures:
            self.board[figure.position[0]][figure.position[1]].set_figure(figure)

        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                self.board[x][y].reset_targeted_by_amounts()
        for figure in self.figures:
            figure.calculate_figure_targets(self.board)
        for figure in self.figures:
            for tx, ty in figure.target_positions:
                self.board[tx][ty].add_targeted_by_amount(figure.color, 1)

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
            for x_tgt, y_tgt in figure.target_positions:
                target_figure = self.board[x_tgt][y_tgt].figure
                if target_figure is not None:
                    edge = format_edge(
                        figure.figure_type,
                        figure.color,
                        target_figure.color,
                        (figure.position, target_figure.position),
                    )
                    self.gt_edges.append(edge)
        LOGGER.log(f"GT Edges computed: {len(self.gt_edges)}")

    # Drones
    def _create_drones(self) -> None:
        """Create drones and place them on the base tile."""
        LOGGER.log(f"Creating {self.num_drones} drones.")
        for drone_index in range(self.num_drones):
            new_drone = _Drone(
                sim=self,
                id=drone_index + 1,
                position=self.drone_base,
                model=self.model,
                rules=self.rules,
            )
            hue_deg = (drone_index / max(1, self.num_drones)) * 360.0
            new_drone.render_color = hsv_to_rgb255(hue_deg, 0.85, 0.95)
            self.drones.append(new_drone)
        base_tile = self.board[self.drone_base[0]][self.drone_base[1]]
        for new_drone in self.drones:
            base_tile.add_drone(new_drone)

    # Edges/score
    def report_edges(self, edges: List[str]) -> None:
        """Merge new edges and update global score counters."""
        for edge in edges:
            if edge not in self.reported_edges:
                self.reported_edges.append(edge)
        correct_edge_counter = 0
        false_edge_counter = 0
        for edge in self.reported_edges:
            if edge in self.gt_edges:
                correct_edge_counter += 1
            else:
                false_edge_counter += 1
        self.correct_edge_counter = correct_edge_counter
        self.false_edge_counter = false_edge_counter
        self.score = correct_edge_counter - false_edge_counter

    def _log_final_summary(self) -> None:
        self._finalize_metrics()
        LOGGER.log("#" * 60)
        LOGGER.log("FINAL EDGE SUMMARY")
        LOGGER.log(f"Reported edges: {len(self.reported_edges)}")
        LOGGER.log(f"  - Correct:    {self.correct_edge_counter}")
        LOGGER.log(f"  - False:      {self.false_edge_counter}")
        LOGGER.log(
            f"Score:          {self.score} / {len(self.gt_edges)} = "
            f"{self.score / max(1, len(self.gt_edges)):.2%}"
        )
        if self.rendezvous_success is not None:
            LOGGER.log(f"Rendezvous success: {self.rendezvous_success}")
        if self.broadcast_effectiveness is not None:
            LOGGER.log(
                "Broadcast effectiveness (sum correct edges per broadcast / gt edges): "
                f"{self.broadcast_effectiveness:.4f}"
            )
        if self.broadcast_effectiveness_per_broadcast is not None:
            LOGGER.log(
                "Broadcast effectiveness per broadcast (avg correct edges / gt edges): "
                f"{self.broadcast_effectiveness_per_broadcast:.4f}"
            )
        if self.avg_turn_duration_s is not None:
            LOGGER.log(f"Average turn duration (s): {self.avg_turn_duration_s:.5f}")
        LOGGER.log(f"Timeouts: {self.timeout_count}")
        if self.broadcast_rate is not None:
            LOGGER.log(f"Broadcast rate (per drone/seed): {self.broadcast_rate:.4f}")
        if self.wait_rate is not None:
            LOGGER.log(f"Wait rate: {self.wait_rate:.4f}")
        LOGGER.log(
            f"Wait actions: {self.wait_actions} / {self.total_actions} | "
            f"Visits: total={self.total_visits}, unique={self.unique_visits}, redundant={self.redundant_visits}"
        )

        LOGGER.log("\nEdge summary:")
        correct_edges = []
        false_edges = []
        for edge in self.reported_edges:
            if edge in self.gt_edges:
                correct_edges.append(edge)
            else:
                false_edges.append(edge)
        for edge in correct_edges:
            LOGGER.log(f"  CORRECT: {edge}")
        for edge in false_edges:
            LOGGER.log(f"  FALSE:   {edge}")

        if self.gui:
            try:
                self.gui.draw_field()
            except Exception:
                pass
        self.post_info("=== FINAL EDGE SUMMARY ===")
        self.post_info(
            f"Reported: {len(self.reported_edges)}  Correct: {self.correct_edge_counter}  "
            f"False: {self.false_edge_counter}  Score: {self.score} / {len(self.gt_edges)}"
        )

    # GUI/log helper
    def post_info(self, msg: str) -> None:
        if self.gui:
            self.gui.post_info(msg)

    def _compute_visit_metrics(self) -> tuple:
        total_visits = 0
        visited = set()
        for drone in self.drones:
            start = getattr(drone, "start_position", None)
            if isinstance(start, (list, tuple)) and len(start) == 2:
                total_visits += 1
                visited.add(tuple(start))
            for pos in getattr(drone, "mission_report", []) or []:
                if isinstance(pos, (list, tuple)) and len(pos) == 2:
                    total_visits += 1
                    visited.add(tuple(pos))
        return total_visits, len(visited)

    def _finalize_metrics(self) -> None:
        if self._metrics_finalized:
            return
        self._metrics_finalized = True
        expected_actions = max(1, int(self.max_rounds)) * max(1, int(self.num_drones))
        if self.total_actions < expected_actions:
            missing = expected_actions - self.total_actions
            self.wait_actions += missing
            self.total_actions = expected_actions
        total_visits, unique_visits = self._compute_visit_metrics()
        self.total_visits = total_visits
        self.unique_visits = unique_visits
        self.redundant_visits = max(0, total_visits - unique_visits)
        total_gt = len(self.gt_edges)
        if total_gt > 0:
            self.broadcast_effectiveness = round(self._broadcast_correct_sum / total_gt, 5)
        else:
            self.broadcast_effectiveness = None
        if self.total_actions > 0:
            self.wait_rate = round(self.wait_actions / self.total_actions, 5)
        if self.num_drones > 0:
            self.broadcast_rate = round(self.broadcast_count / self.num_drones, 5)
        if self.broadcast_effectiveness is not None and self.broadcast_count > 0:
            self.broadcast_effectiveness_per_broadcast = round(
                self.broadcast_effectiveness / self.broadcast_count,
                5,
            )
        if self.runtime_s is not None and self.total_actions > 0:
            self.avg_turn_duration_s = round(self.runtime_s / self.total_actions, 5)

    def _log_round_mission_plans(self, round_number: int) -> None:
        """Log mission plan changes at the start of each round."""
        try:
            updates = []
            for drone in self.drones:
                plan = getattr(drone, "mission_plan", [])
                snapshot = {"sector": getattr(drone, "assigned_sector", None), "plan": plan}
                try:
                    signature = json.dumps(snapshot, sort_keys=True, default=lambda o: getattr(o, "__dict__", str(o)))
                except Exception:
                    signature = json.dumps(str(snapshot))
                previous = self._last_logged_plans.get(drone.id)
                if signature != previous:
                    updates.append((drone, plan, signature))
            if not updates:
                return

            LOGGER.log(f"Mission plan update for round {round_number}:")
            for drone, plan, signature in updates:
                self._last_logged_plans[drone.id] = signature
                LOGGER.log(f"  Drone {drone.id} mission plan:")
                try:
                    LOGGER.log(f"    Sector: {drone.sector_summary()}")
                except Exception:
                    pass
                for entry in plan:
                    if isinstance(entry, dict) and entry.get("leg_start") and entry.get("leg_end"):
                        start_vec = entry.get("leg_start")
                        end_vec = entry.get("leg_end")
                        start = cartesian_to_chess(tuple(start_vec)) if start_vec else "?"
                        end = cartesian_to_chess(tuple(end_vec)) if end_vec else "?"
                        leg_label = entry.get("leg_id", "?")
                        turn = entry.get("turn", "?")
                        duration = entry.get("duration_turns", 0)
                        LOGGER.log(f"    Leg {leg_label}: {start} -> {end} (turn {turn}, duration {duration})")

                def _format_plan_entry(item: object) -> object:
                    if not isinstance(item, dict):
                        return item
                    formatted = dict(item)
                    for key in ("leg_start", "leg_end"):
                        value = formatted.get(key)
                        if isinstance(value, (list, tuple)) and len(value) == 2:
                            try:
                                formatted[key] = cartesian_to_chess((int(value[0]), int(value[1])))
                            except (TypeError, ValueError):
                                pass
                    return formatted

                plan_log = [_format_plan_entry(entry) for entry in plan]
                for line in pprint.pformat(plan_log, indent=2, width=120).splitlines():
                    LOGGER.log(f"    {line}")
        except Exception as exc:
            LOGGER.log(f"Error logging mission plans: {exc}")

    # Turn orchestration
    def _try_finish_drone_turn(self, drone: _Drone) -> bool:
        if self._current_future is None or not self._current_future.done():
            return False

        try:
            outcome = self._current_future.result()
            if not isinstance(outcome, dict):
                raise ValueError("Drone turn outcome is not a dict.")
            result = outcome.get("result")
            if isinstance(result, dict):
                LOGGER.log(f"Drone {drone.id} response:\n{pprint.pformat(result, indent=4, width=200)}")
            else:
                LOGGER.log(f"Drone {drone.id} response: <no parseable result>")

            self.post_info(f"Drone {drone.id}:")
            rationale = outcome.get("rationale", "")
            self.post_info(f"Rationale: {rationale}")

            for error in outcome.get("errors", []) or []:
                self.post_info(error)

            action = outcome.get("action", "wait")
            self.total_actions += 1
            if action == "wait":
                self.wait_actions += 1
            elif action == "broadcast":
                self._broadcast_correct_sum += self.correct_edge_counter
            if action == "move":
                direction = outcome.get("direction")
                if outcome.get("moved"):
                    self.post_info(f"Move {direction} to {cartesian_to_chess(drone.position)}")
                else:
                    self.post_info("Wait")
            elif action == "broadcast":
                msg = outcome.get("broadcast_message") or ""
                if not msg:
                    self.post_info("ERROR: Empty broadcast.")
                else:
                    self.post_info("Broadcast")
                self.post_info(msg or "<no message>")
            else:
                self.post_info("Wait")

            self.post_info("\n")
        except Exception as exc:
            LOGGER.log(f"Error finishing Drone {drone.id}'s turn: {exc}")

        self._thinking = False
        self._current_future = None
        return True

    # Main loop
    def run_simulation(self) -> None:
        max_rounds = CONFIG["simulation"].get("max_rounds", 10)
        use_gui = bool(CONFIG["simulation"].get("use_gui", True)) and self.gui is not None
        watchdog_s = CONFIG.get("simulation", {}).get("watchdog_timeout_s", 900)
        if not isinstance(watchdog_s, (int, float)) or watchdog_s <= 0:
            watchdog_s = None
        start_time = time.time()
        self._run_start_time = start_time

        running = True
        clock = None
        if use_gui and pygame:
            clock = pygame.time.Clock()
            pygame.display.set_caption(self.progress_caption(round_num=1, turn_num=1))
            self.gui.draw_field()

        current_round = 1
        drone_index = 0
        pending = False

        try:
            while running:
                if use_gui and pygame:
                    try:
                        events = pygame.event.get()
                    except Exception as exc:
                        LOGGER.log(f"GUI event handling error: {exc}")
                        events = []
                    for event in events:
                        if event.type == pygame.QUIT:
                            running = False
                            self._abort_requested = True
                            self._abort_reason = "GUI closed"
                        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                            running = False
                            self._abort_requested = True
                            self._abort_reason = "GUI closed (ESC)"

                if watchdog_s is not None and (time.time() - start_time) > watchdog_s:
                    self._watchdog_triggered = True
                    minutes = int(round(watchdog_s / 60))
                    self._abort_reason = f"watchdog timeout after {minutes} minutes"
                    self.timeout_count = 1
                    self.timeout_reason = self._abort_reason
                    LOGGER.log(f"Watchdog abort: {self._abort_reason}")
                    running = False
                    break

                current_drone = self.drones[drone_index]
                if not pending:
                    if current_round > max_rounds:
                        break
                    self.round = current_round
                    self.turn = drone_index + 1
                    caption = self.progress_caption(current_round, self.turn)
                    LOGGER.log("#" * 50)
                    LOGGER.log(caption)
                    if drone_index == 0:
                        self._log_round_mission_plans(current_round)
                    if use_gui and pygame:
                        pygame.display.set_caption(caption)
                    self._thinking = True
                    self._current_future = self.executor.submit(current_drone.take_turn)
                    pending = True

                if pending:
                    if self._try_finish_drone_turn(current_drone):
                        drone_index += 1
                        if drone_index >= self.num_drones:
                            drone_index = 0
                            current_round += 1
                            completed_round = current_round - 1
                            if (
                                not self._rendezvous_checked
                                and completed_round == self._rendezvous_turn
                            ):
                                self._rendezvous_checked = True
                                self._check_rendezvous()
                        pending = False

                if use_gui and pygame:
                    self.gui.draw_field()
                    clock.tick(60)
        except KeyboardInterrupt:
            LOGGER.log("KeyboardInterrupt received; shutting down gracefully.")
            running = False
        finally:
            try:
                if self._run_start_time is not None:
                    self.runtime_s = time.time() - self._run_start_time
                self._log_final_summary()
            except Exception:
                pass
            self.shutdown()

    def shutdown(self) -> None:
        if getattr(self, "_shutdown_complete", False):
            return
        self._shutdown_complete = True
        try:
            if getattr(self, "_current_future", None) and not self._current_future.done():
                self._current_future.cancel()
        except Exception:
            pass
        try:
            if getattr(self, "executor", None):
                self.executor.shutdown(wait=False, cancel_futures=True)
                self.executor = None
        except Exception:
            pass
        try:
            if self.gui:
                surface = None
                if pygame and pygame.display.get_init():
                    try:
                        surface = pygame.display.get_surface()
                    except Exception:
                        surface = None
                if surface is not None:
                    self.gui.draw_field()
                    self.gui.save_screenshot()
        except Exception as exc:
            LOGGER.log(f"Error during final GUI draw: {exc}")
        if pygame:
            pygame.display.quit()
            pygame.quit()
        LOGGER.log("Clean shutdown complete.")
