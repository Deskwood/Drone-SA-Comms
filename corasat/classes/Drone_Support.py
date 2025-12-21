# Drone Helper Classes
# =========================

class _Drone_Knowledge:
    """Encapsulates knowledge and board state for drones."""

    def __init__(self, drone: _Drone):
        self.drone = drone
        self.local_board = self._make_empty_local_board()

    def _make_empty_local_board(self) -> Dict[str, _Local_Tile]:
        """Initialize local board knowledge with unknown placeholders."""
        local_board = {}
        for bx in range(CONFIG["board"]["width"]):
            for by in range(CONFIG["board"]["height"]):
                local_board[Waypoint((bx, by)).to_chess()] = _Local_Tile(self.drone.sim.board, Waypoint((bx, by)))
        return local_board

class _Drone_Mission_Support:
    """Encapsulates mission management for drones."""

    def __init__(self, drone: _Drone):
        self.drone = drone
        self.assigned_sector = self._assign_sector(
            upper_left=Waypoint((0, CONFIG["board"]["height"] - 1)),
            lower_right=Waypoint((CONFIG["board"]["width"] - 1, 0))
        )

    def _assign_sector(self, upper_left: Waypoint, lower_right: Waypoint) -> Dict[str, Any]:
        return {
            "upper_left": upper_left,
            "lower_right": lower_right
        }
    
    def _sector_changed(self, old: Tuple[Waypoint, Waypoint], new: Tuple[Waypoint, Waypoint]) -> bool:
        if old is None and new is None:
            return True
        return any(old.get(key) != new.get(key) for key in ("upper_left", "lower_right"))
    
    def _sector_to_string(self, sector: Optional[Dict[str, Any]]) -> str:
        return f"{sector.get("upper_left")} -> {sector.get("lower_right")}"
    
    def _set_sector_assignment(self, sector: Tuple[str, str]) -> None:
        normalized = self._normalize_sector_assignment(sector)
        changed = self._sector_changed(getattr(self, "assigned_sector", None), normalized)
        self.assigned_sector = normalized
        if changed:
            self.mission_plan = self._build_mission_plan_for_sector(normalized)
            LOGGER.log(f"Drone {self.id} adopted sector {self.sector_summary()}")
        self.current_leg_index = 0 if changed else min(self.current_leg_index, max(0, len(self.mission_plan) - 1))
        self._sanitize_leg_index()

    def format_lines(self, snapshot: Dict[str, object]) -> Tuple[List[str], List[str]]:
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
                lines.append(f"Plan focus: leg {leg_id} {start} -> {target} (turn {waypoint.get('turn','?')})")
            else:
                target_vec = waypoint.get("leg_end")
                target = cartesian_to_chess(tuple(target_vec)) if target_vec else "?"
                lines.append(f"Plan focus: {target} by turn {waypoint.get('turn','?')}")

        if getattr(drone, 'assigned_sector', None):
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
            key=lambda entry: entry.get("score", float('-inf')),
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
                sorted_components = sorted(
                    components.items(),
                    key=lambda item: (-abs(item[1]), item[0])
                )
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
            lines.append(
                f"{action_text} -> {qualitative} ({score_text}) | factors: {component_text}{note_text}"
            )

        for ledger in snapshot.get("intel_ledger", []):
            last_round = ledger.get("last_round")
            age = ledger.get("age", 0)
            if last_round is None:
                ledger_lines.append(
                    f"Drone {ledger['drone']}: never shared (age {age} rounds)"
                )
            else:
                ledger_lines.append(
                    f"Drone {ledger['drone']}: last shared round {last_round} (age {age})"
                )

        return lines, ledger_lines

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
                if info["type"] == "unknown" or info["color"] == "unknown" or info["type"] == "a possible target":
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
        board_w = CONFIG["board"]["width"]
        board_h = CONFIG["board"]["height"]
        num_drones = max(1, sim.num_drones)
        block_width = max(1, math.ceil(board_w / num_drones))
        assignments: List[Dict[str, Any]] = []
        for idx in range(num_drones):
            start_col = min(idx * block_width, board_w - 1)
            end_col = min(board_w - 1, start_col + block_width - 1)
            if start_col > end_col:
                start_col = end_col
            upper_left = cartesian_to_chess((start_col, board_h - 1))
            lower_right = cartesian_to_chess((end_col, 0))
            assignments.append({
                "drone": idx + 1,
                "sector": {
                    "upper_left": upper_left,
                    "lower_right": lower_right,
                }
            })
        rendezvous_turn = max(1, CONFIG["simulation"].get("max_rounds", 1) - 1)
        rendezvous_tile = cartesian_to_chess(sim.drone_base)
        return {
            "plan": {
                "assignments": assignments,
                "rendezvous": {
                    "tile": rendezvous_tile,
                    "turn": rendezvous_turn,
                }
            }
        }

    def _compute_decision_support(self) -> Dict[str, object]:
        drone = self.drone
        ds_cfg = CONFIG.get("decision_support", {})
        scoring_cfg = ds_cfg.get("scoring", {})
        board_w = CONFIG["board"]["width"]
        board_h = CONFIG["board"]["height"]

        move_cfg = scoring_cfg.get("move", {})
        broadcast_cfg = scoring_cfg.get("broadcast", {})
        wait_cfg = scoring_cfg.get("wait", {})

        # Move scoring parameters
        waypoint_progress_reward = move_cfg.get("waypoint_progress_reward_per_step", 1.0)  # per step closer
        waypoint_regression_penalty = move_cfg.get("waypoint_regression_penalty_per_step", -1.0) # per step further
        deadline_slack_bonus = move_cfg.get("deadline_slack_bonus", 0.2) # per turn slack
        deadline_slack_penalty = move_cfg.get("deadline_slack_penalty", -1.0) # per turn overdue
        tolerance_bonus = move_cfg.get("tolerance_bonus", 0.5)
        unknown_tile_bonus = move_cfg.get("unknown_tile_bonus", 1.0)
        possible_target_bonus = move_cfg.get("possible_target_bonus", 1.2)
        figure_hint_bonus = move_cfg.get("figure_hint_bonus", 0.6)
        unknown_color_bonus = move_cfg.get("unknown_color_bonus", 0.5)
        unknown_neighbor_bonus = move_cfg.get("unknown_neighbor_bonus_per_tile", 0.2)
        leg_alignment_reward = move_cfg.get("leg_alignment_reward", 0.6)
        leg_alignment_penalty = move_cfg.get("leg_alignment_penalty", -0.6)
        leg_travel_reward = move_cfg.get("leg_travel_reward", 0.9)
        leg_travel_penalty = move_cfg.get("leg_travel_penalty", -0.3)
        leg_sideways_reward = move_cfg.get("leg_sideways_reward", 1.2)
        leg_sideways_probe_bonus = move_cfg.get("leg_sideways_inspection_bonus", leg_sideways_reward)
        leg_along_penalty = move_cfg.get("leg_along_penalty", -0.7)
        leg_start_progress_reward = move_cfg.get("leg_start_progress_reward", 1.2)
        leg_start_regression_penalty = move_cfg.get("leg_start_regression_penalty", -1.0)
        leg_neighbor_scan_bonus = move_cfg.get("leg_neighbor_scan_bonus", 0.4)
        sector_alignment_reward = move_cfg.get("sector_alignment_reward", 0.8)
        sector_inside_bonus = move_cfg.get("sector_inside_bonus", 0.5)
        late_penalty_multiplier = move_cfg.get("late_penalty_multiplier", 2.0)
        border_edge_bonus = move_cfg.get("board_edge_bias_bonus", 0.2)
        border_edge_range = int(move_cfg.get("board_edge_bias_range", 1))
        sector_unknown_probe_bonus = move_cfg.get("sector_unknown_probe_bonus", 0.0)
        sector_unknown_probe_min_slack = float(move_cfg.get("sector_unknown_probe_min_slack", 1))

        # Broadcast scoring parameters
        broadcast_base_penalty = broadcast_cfg.get("base_penalty", -0.5)
        broadcast_recipient_factor = broadcast_cfg.get("recipient_factor", 0.8)
        broadcast_staleness_factor = broadcast_cfg.get("staleness_factor", 0.4)
        coordination_broadcast_bonus = broadcast_cfg.get("first_turn_coordination_bonus", 2.5)
        is_first_coordination_turn = (
            drone.id == 1
            and getattr(drone.sim, "round", None) == 1
            and getattr(drone.sim, "turn", None) == 1
        )

        # Wait scoring parameters
        wait_default_score = wait_cfg.get("default_score", -1.0)
        wait_idle_component = wait_cfg.get("idle_penalty_component", -1.0)
        wait_holding_score = wait_cfg.get("holding_position_score", -0.2)
        wait_holding_component = wait_cfg.get("holding_pattern_component", 0.3)
        wait_over_activity_penalty = wait_cfg.get("over_activity_penalty", 0.0)

        scores: List[Dict[str, object]] = []
        best_move_score = float('-inf')
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
                nearest_sector_unknown = {
                    "tile": cartesian_to_chess(best_unknown_pos),
                    "distance": best_unknown_dist,
                }
                closest_probe_distance = best_unknown_dist

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
                    if current_leg_start_distance is not None and current_leg_start_distance > 0 and new_start_distance is not None:
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
                    if abs(new_pos[0] - drone.position[0]) <= 1 and abs(new_pos[1] - drone.position[1]) <= 1:
                        score += leg_sideways_probe_bonus
                        score_components["leg_sideways_probe"] = round(leg_sideways_probe_bonus, 2)
                        notes.append("sideways probe off leg")
                if new_leg_distance == 0 and current_leg_end_distance is not None:
                    if new_start_distance is not None and new_start_distance > 0 and current_leg_start_distance is not None and current_leg_start_distance > 0:
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
                # Evaluate progress toward next waypoint
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

                # Evaluate rendezvous buffer
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

                # Evaluate tolerance achievement
                tolerance = 0
                if new_dist <= tolerance:
                    score += tolerance_bonus
                    score_components["within_tolerance"] = 1.0

                if (
                    sector_unknown_tiles
                    and sector_unknown_probe_bonus
                    and slack >= sector_unknown_probe_min_slack
                ):
                    sector_probe_bonus = sector_unknown_probe_bonus
                    if closest_probe_distance is not None and slack >= closest_probe_distance:
                        score += sector_unknown_probe_bonus
                        score_components["sector_unknown_probe"] = round(sector_unknown_probe_bonus, 2)
                        notes.append(
                            f"sector probe feasible (nearest unknown {closest_probe_distance} steps away without delaying leg)"
                        )

            # Evaluate tile information gain
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

            # Evaluate color information gain
            if tile_info["color"] == "unknown":
                score += unknown_color_bonus
                score_components["discover_color"] = unknown_color_bonus

            # Evaluate unknown neighbors
            unknown_neighbors = self._count_unknown_neighbors(new_pos)
            if unknown_neighbors:
                score += unknown_neighbors * unknown_neighbor_bonus
                score_components["unknown_neighbors"] = float(unknown_neighbors)
                notes.append(f"{unknown_neighbors} unknown neighbors")

            if sector_bounds:
                new_sector_distance = self._distance_to_sector(new_pos, sector_bounds)
                if new_sector_distance is not None:
                    if (sector_alignment_reward and current_sector_distance is not None
                            and new_sector_distance < current_sector_distance):
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

            best_move_score = max(best_move_score, score)

            scores.append({
                "action": "move",
                "label": direction,
                "score": round(score, 2),
                "components": {k: round(v, 2) for k, v in score_components.items() if abs(v) >= 0.01},
                "notes": notes,
            })

        # Broadcast scoring
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
                if age < 0:
                    age = 0
                ages.append(age)
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

        scores.append({
            "action": "broadcast",
            "label": "share",
            "score": round(broadcast_score, 2),
            "components": {k: v for k, v in broadcast_components.items()},
            "notes": broadcast_notes,
        })

        # Wait scoring
        wait_score = wait_default_score
        wait_notes: List[str] = ["no progress"]
        wait_components: Dict[str, float] = {"idle_penalty": wait_idle_component}
        if target_pos:
            dist_to_target = chebyshev_distance(drone.position, target_pos)
            if dist_to_target == 0 and next_wp and current_round < next_wp["turn"]:
                wait_score = wait_holding_score
                wait_notes = ["holding position at waypoint"]
                wait_components["holding_pattern"] = wait_holding_component
        scores.append({
            "action": "wait",
            "label": "hold",
            "score": round(wait_score, 2),
            "components": {k: round(v, 2) for k, v in wait_components.items()},
            "notes": wait_notes,
        })

        scores.sort(key=lambda entry: entry["score"], reverse=True)

        intel_ledger: List[Dict[str, object]] = []
        for other_id in sorted(drone.info_exchange_rounds.keys()):
            last_round = drone.info_exchange_rounds[other_id]
            age = current_round - last_round if last_round is not None else current_round
            if age < 0:
                age = 0
            intel_ledger.append({
                "drone": other_id,
                "last_round": last_round,
                "age": age,
            })

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

class _Drone_LM:
    """Encapsulates language model interactions for drones."""

    def __init__(self, drone: object, model: str):
        self.drone = drone
        self.model = model

class _Drone_Decision_Support:
    """Encapsulates decision scoring and summaries for drones."""

    def __init__(self, drone: object):
        self.drone = drone

class _Drone_Operator:
    """Applies LM outputs to drone actions."""

    def __init__(self, drone: object):
        self.drone = drone