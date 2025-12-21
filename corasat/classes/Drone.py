"""Drone agent orchestrating per-turn behavior via support components."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from classes.Core import CONFIG
from classes.Drone_Support import (
    _Drone_Aftermath,
    _Drone_Decision_Support,
    _Drone_Knowledge,
    _Drone_Language_Model,
    _Drone_Mission_Support,
)

class _Drone:
    """Drone agent handling sector assignment, intel, movement, and model I/O."""

    def __init__(self, sim: object, id: int, position: Tuple[int, int], model: str, rules: str):
        self.sim = sim
        self.id = id
        self.position = tuple(position)
        self.model = model

        rules = rules.replace("DRONE_ID", str(id))
        rules = rules.replace("NUMBER_OF_DRONES", str(CONFIG["simulation"]["num_drones"]))
        rules = rules.replace("NUMBER_OF_ROUNDS", str(CONFIG["simulation"]["max_rounds"]))
        self.rules = rules

        self.knowledge = _Drone_Knowledge(self)
        self.mission_support = _Drone_Mission_Support(self)
        self.decision_support = _Drone_Decision_Support(self)
        self.language_model = _Drone_Language_Model(self, model)
        self.aftermath = _Drone_Aftermath(self)

    # --- Mission plan wrappers ---
    def _apply_rendezvous_to_plan(self, plan: List[dict]) -> List[dict]:
        return plan

    def _sanitize_leg_index(self) -> None:
        self.mission_support.sanitize_leg_index()

    def _current_mission_leg(self) -> Optional[dict]:
        return self.mission_support.current_mission_leg()

    def _advance_leg_progress(self) -> None:
        self.mission_support.advance_leg_progress()

    def _leg_segment_bounds(self, leg: dict) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        return self.mission_support.leg_segment_bounds(leg)

    def _distance_to_leg(self, pos: Tuple[int, int], leg: Optional[dict]) -> Optional[int]:
        return self.mission_support.distance_to_leg(pos, leg)

    def _distance_to_leg_end(self, pos: Tuple[int, int], leg: Optional[dict]) -> Optional[int]:
        return self.mission_support.distance_to_leg_end(pos, leg)

    def _build_initial_mission_plan(self) -> List[dict]:
        return self.mission_support.build_initial_mission_plan()

    def _build_mission_plan_for_sector(self, sector: Optional[Dict[str, object]]) -> List[dict]:
        return self.mission_support.build_mission_plan_for_sector(sector)

    def _next_mission_waypoint(self) -> Optional[dict]:
        return self.mission_support.next_mission_waypoint()

    def _set_sector_assignment(self, sector: object) -> None:
        self.mission_support.set_sector_assignment(sector)

    def _sector_bounds(self, sector: Optional[Dict[str, object]] = None) -> Optional[Tuple[int, int, int, int]]:
        return self.mission_support.sector_bounds(sector)

    def sector_summary(self) -> str:
        return self.mission_support.sector_summary()

    def _apply_plan_directive(self, payload: Dict[str, object]) -> None:
        self.mission_support.apply_plan_directive(payload)

    # --- Decision support wrappers ---
    def _decision_support_snapshot(self) -> Dict[str, object]:
        return self.decision_support.snapshot()

    def _format_decision_support_lines(self, snapshot: Dict[str, object]) -> Tuple[List[str], List[str]]:
        return self.decision_support.format_lines(snapshot)

    # --- Knowledge wrappers ---
    def _identify_edges(self) -> None:
        self.knowledge.identify_edges()

    def update_board_report(self) -> None:
        self.knowledge.update_board_report()

    def _provide_intelligence_to(self, target_drone: "_Drone") -> None:
        self.knowledge.provide_intelligence_to(target_drone)

    # --- Movement wrappers ---
    def _legal_movement_steps(self) -> List[dict]:
        return self.mission_support.legal_movement_steps()

    def _move(self, direction: str) -> bool:
        moved, _ = self.aftermath.execute_move(direction)
        return moved

    # --- Situation & model I/O ---
    def generate_full_model_response(self) -> List[dict]:
        temperature = CONFIG["simulation"].get("temperature", 0.7)
        prompt_data = self.decision_support.build_prompt(self.rules)
        prompt_char_len = prompt_data["prompt_char_len"]
        print(f"Context length (chars): {prompt_char_len}")
        return self.language_model.generate(prompt_data["messages"], temperature, prompt_char_len)

    def take_turn(self) -> Dict[str, object]:
        """Run the full turn pipeline and return an outcome for Simulation."""
        messages = self.generate_full_model_response()
        return self.aftermath.execute_turn(messages)
