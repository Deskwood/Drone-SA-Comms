# Decision Support Parameters (current implementation)

Source of truth: `Code/corasat/config.json` and `Code/corasat/classes/Drone_Support.py` (`_compute_decision_support`).

Notes:
- Move scores start at 0.0 and add each component; the final score is rounded to 2 decimals in the log.
- All movement components use the drone's `local_board` knowledge (not ground truth).
- Chebyshev distance is used for all distance calculations.

## Derived terms used below

| Term | Definition |
| --- | --- |
| `current_dist` | Chebyshev distance from current position to `target_pos` (waypoint leg_end). |
| `new_dist` | Chebyshev distance from candidate move position to `target_pos`. |
| `turns_remaining` | `max(0, next_wp["turn"] - current_round)`. |
| `slack` | `turns_remaining - new_dist` (computed per candidate move). |
| `current_leg_distance` | Distance from current position to the leg corridor. |
| `new_leg_distance` | Distance from candidate position to the leg corridor. |
| `current_leg_start_distance` | Distance from current position to leg_start. |
| `current_leg_end_distance` | Distance from current position to leg_end. |
| `new_end_distance` | Distance from candidate position to leg_end. |
| `visited_tiles` | Set of positions in `drone.mission_report`. |
| `unknown_neighbors` | Count of adjacent tiles where local_board type is `unknown` or `a possible target`, or color is `unknown`. |
| `border_distance` | `min(x, w-1-x, y, h-1-y)` for candidate position. |

## Movement scoring components (log column keys)

| Component key | Config key(s) | Current value(s) | Adds to score when... |
| --- | --- | --- | --- |
| `waypoint_progress` | `decision_support.scoring.move.waypoint_progress_reward_per_step` | `1.1390557352137365` | `slack < 0` **after the move** and `new_dist < current_dist`. Adds the reward once (not multiplied by delta). |
| `waypoint_regression` | `decision_support.scoring.move.waypoint_regression_penalty_per_step`, `decision_support.scoring.move.late_penalty_multiplier` | `-0.316352144799844`, `2.0` | `slack < 0` and `new_dist > current_dist`. Adds `waypoint_regression_penalty_per_step + late_penalty_multiplier * abs(slack)`. |
| `deadline_penalty` | `decision_support.scoring.move.deadline_slack_penalty`, `decision_support.scoring.move.late_penalty_multiplier` | `-1.2848331439337206`, `2.0` | `slack < 0`. Adds `deadline_slack_penalty * max(1.0, late_penalty_multiplier)`. |
| `tolerance_bonus` | `decision_support.scoring.move.tolerance_bonus` | `0.21237937241286878` | `new_dist <= 0` (candidate move reaches the target). |
| `leg_start_progress` | `decision_support.scoring.move.leg_start_progress_reward` | `1.1733817270595754` | `current_leg_start_distance > 0` and the move reduces distance to `leg_start`. |
| `leg_start_regression` | `decision_support.scoring.move.leg_start_regression_penalty` | `-0.2581200428311292` | Move increases distance to `leg_start`, or the move enters the leg corridor (`new_leg_distance == 0`) while still not at `leg_start` ("skipping leg start"). |
| `leg_alignment` | `decision_support.scoring.move.leg_alignment_reward`, `decision_support.scoring.move.leg_alignment_penalty` | `0.6`, `-0.6` | Compares `current_leg_distance - new_leg_distance`. Positive delta adds reward, negative delta adds penalty. |
| `leg_travel` | `decision_support.scoring.move.leg_travel_reward`, `decision_support.scoring.move.leg_travel_penalty` | `0.3682416522100012`, `-0.299138984543186` | When on the leg corridor (`new_leg_distance == 0`) and not skipping leg start: compares `current_leg_end_distance - new_end_distance` to reward progress or penalize retreat. |
| `leg_sideways_reward` | `decision_support.scoring.move.leg_sideways_reward` | `0.5973113343180227` | If off the leg corridor and a perpendicular move (relative to leg orientation) reduces distance to the leg corridor. |
| `leg_sideways_probe` | `decision_support.scoring.move.leg_sideways_inspection_bonus` | `0.7519634110819631` | If currently on the leg corridor and the move steps off it (`current_leg_distance == 0` and `new_leg_distance == 1`). |
| `leg_along_penalty` | `decision_support.scoring.move.leg_along_penalty` | `-0.5662462252883823` | If off the leg corridor and a move along the leg axis does **not** reduce leg distance (`delta_leg <= 0`). |
| `sector_alignment` | `decision_support.scoring.move.sector_alignment_reward` | `0.7850850533649799` | If sector bounds exist and the move reduces distance to the sector bounds. |
| `sector_inside` | `decision_support.scoring.move.sector_inside_bonus` | `0.959147417154021` | If sector bounds exist and the candidate move is inside them (`new_sector_distance == 0`). |
| `sector_unknown_probe` | `decision_support.scoring.move.sector_unknown_probe_bonus`, `decision_support.scoring.move.sector_unknown_probe_min_slack` | `0.6`, `1.1676590478688627` | Sector bounds exist, there are unknown tiles in the sector, `slack >= min_slack`, and `slack >= closest_probe_distance`. |
| `discover_type` | `decision_support.scoring.move.unknown_tile_bonus` | `1.0` | Destination tile has local_board type `unknown`. |
| `possible_target` | `decision_support.scoring.move.possible_target_bonus` | `1.2` | Destination tile has local_board type `a possible target`. |
| `figure_hint` | `decision_support.scoring.move.figure_hint_bonus` | `3.0` | Destination tile has local_board type `any figure`. |
| `known_figure` | `decision_support.scoring.move.known_figure_penalty` | `-0.4` | Destination tile has a known figure type and penalty is non-zero. |
| `known_empty` | `decision_support.scoring.move.known_empty_penalty` | `-0.8` | Destination tile has local_board type `n/a` and penalty is non-zero. |
| `discover_color` | `decision_support.scoring.move.unknown_color_bonus` | `0.5` | Destination tile has local_board color `unknown`. |
| `revisit_penalty` | `decision_support.scoring.move.revisit_penalty` | `-1.2` | Destination tile is in `visited_tiles`. |
| `new_tile` | `decision_support.scoring.move.novel_tile_bonus` | `0.9` | Destination tile is **not** in `visited_tiles` and `novel_tile_bonus` is non-zero. |
| `unknown_neighbors` | `decision_support.scoring.move.unknown_neighbor_bonus_per_tile` | `0.45` | Adds `unknown_neighbor_bonus_per_tile * unknown_neighbors` when `unknown_neighbors > 0`. |
| `border_bias` | `decision_support.scoring.move.board_edge_bias_bonus`, `decision_support.scoring.move.board_edge_bias_range` | `0.05`, `1` | If `border_distance <= board_edge_bias_range`, adds `board_edge_bias_bonus * max(1, board_edge_bias_range - border_distance + 1)`. |

## Broadcast scoring components (log column keys)

| Component key | Config key(s) | Current value(s) | Adds to score when... |
| --- | --- | --- | --- |
| `broadcast_base` | `decision_support.scoring.broadcast.base_penalty` | `-0.5` | No co-located drones; score starts at this base penalty. |
| `broadcast_recipients` | `decision_support.scoring.broadcast.recipient_factor` | `0.8` | Co-located drones exist; adds `count(recipients) * recipient_factor`. |
| `broadcast_staleness` | `decision_support.scoring.broadcast.staleness_factor` | `0.4` | Co-located drones exist; adds `avg_age * staleness_factor` (age since last exchange). |
| `coordination_bonus` | `decision_support.scoring.broadcast.first_turn_coordination_bonus` | `2.5` | Drone 1 on round 1, turn 1; adds once. |

## Wait scoring components (log column keys)

| Component key | Config key(s) | Current value(s) | Adds to score when... |
| --- | --- | --- | --- |
| `wait_idle` | `decision_support.scoring.wait.default_score` | `-1.4` | Default wait score when not holding position at a waypoint. |
| `wait_holding` | `decision_support.scoring.wait.holding_position_score` | `-0.2` | If at `target_pos` **and** `current_round < next_wp["turn"]`, replaces `wait_idle`. |

## Non-score directives shown in Decision Support

| Directive | Trigger condition | Effect |
| --- | --- | --- |
| `Special directive` | Drone 1, round 1, turn 1 | Instructs to broadcast the coverage plan before any other action; overrides decision support ranking in the prompt. |
| `Plan focus`, `Waypoint timing`, `Sector directive` | Mission plan + sector assignment | Informational; the scoring components above derive from these fields. |

## Config keys currently unused in scoring

| Config key | Current value | Status |
| --- | --- | --- |
| `decision_support.scoring.move.deadline_slack_bonus` | `-0.16352244315206657` | Read but not used in scoring. |
| `decision_support.scoring.broadcast.min_staleness_rounds` | `2` | Not referenced in code. |
| `decision_support.scoring.broadcast.fresh_penalty` | `-2.0` | Not referenced in code. |
| `decision_support.scoring.wait.idle_penalty_component` | `-1.0` | Read but not used in scoring. |
| `decision_support.scoring.wait.holding_pattern_component` | `0.3` | Read but not used in scoring. |
