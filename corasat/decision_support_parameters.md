# Decision Support Parameters (current implementation)

Source of truth: `Code/corasat/config.json` and `Code/corasat/classes/Drone_Support.py` (`_compute_decision_support`).

Notes:
- Move scores start at 0.0 and add each component; the final score is rounded to 2 decimals in the log.
- All movement components use the drone's `local_board` knowledge (not ground truth).
- Chebyshev distance is used for all distance calculations.
- Neighborhood potential notes include per-direction bonuses (n, ne, e, se, s, sw, w, nw) scaled by `neighborhood_potential`, plus the total.
- Parameter values live in `Code/corasat/config.json` under `decision_support.scoring`.

## Derived terms used below

| Term | Definition |
| --- | --- |
| `current_dist` | Chebyshev distance from current position to `target_pos` (waypoint leg_end). |
| `new_dist` | Chebyshev distance from candidate move position to `target_pos`. |
| `turns_remaining` | `max(0, next_wp["turn"] - current_round)`. |
| `slack` | `turns_remaining - new_dist` (computed per candidate move). |
| `current_leg_distance` | Distance from current position to the leg corridor. |
| `new_leg_distance` | Distance from candidate position to the leg corridor. |
| `visited_tiles` | Set of positions in `drone.mission_report`. |
| `neighborhood_potential_sum` | Sum of neighbor weights around the candidate tile: `any figure` = `neighborhood_weight_any_figure`, `a possible target` = `neighborhood_weight_possible_target`, `unknown` = `unknown_tile_bonus`, off-board neighbor = `border_bonus`, all other types = 0. |

## Movement scoring components (log column keys)

| Component key | Config key(s) | Adds to score when... |
| --- | --- | --- |
| `waypoint_progress` | `decision_support.scoring.move.waypoint_progress_bonus` | `new_dist <= current_dist` (candidate move is not farther from the waypoint). Adds the reward once (not multiplied by delta). |
| `waypoint_regression` | `decision_support.scoring.move.waypoint_delay_penalty` | `turns_remaining` available and `slack < 0` (i.e., `new_dist > turns_remaining`). Adds `waypoint_delay_penalty * abs(slack)`. |
| `leg_approach_bonus` | `decision_support.scoring.move.leg_alignment_bonus_per_step` | Compares `current_leg_distance - new_leg_distance`. Positive delta adds `leg_alignment_bonus_per_step * delta_leg` (highest when the new tile is on the leg). |
| `sector_compliance_bonus` | `decision_support.scoring.move.sector_compliance_bonus` | If sector bounds exist and the candidate move is inside them (`new_sector_distance == 0`) or reduces distance to the sector. |
| `unknown_tile_bonus` | `decision_support.scoring.move.unknown_tile_bonus` | Destination tile has local_board type `unknown`. |
| `possible_target` | `decision_support.scoring.move.possible_target_bonus` | Destination tile has local_board type `a possible target`. |
| `figure_hint` | `decision_support.scoring.move.figure_hint_bonus` | Destination tile has local_board type `any figure`. |
| `revisit_penalty` | `decision_support.scoring.move.revisit_penalty` | Destination tile is in `visited_tiles`. |
| `neighborhood_potential` | `decision_support.scoring.move.neighborhood_potential`, `decision_support.scoring.move.border_bonus`, `decision_support.scoring.move.neighborhood_weight_any_figure`, `decision_support.scoring.move.neighborhood_weight_possible_target`, `decision_support.scoring.move.unknown_tile_bonus` | Adds `neighborhood_potential * neighborhood_potential_sum` when `neighborhood_potential_sum > 0`. Log notes show per-direction bonuses (n, ne, e, se, s, sw, w, nw). |

## Move parameter mapping (config -> log column keys)

| Config key | Log column(s) |
| --- | --- |
| `decision_support.scoring.move.waypoint_progress_bonus` | `waypoint_progress` |
| `decision_support.scoring.move.waypoint_delay_penalty` | `waypoint_regression` |
| `decision_support.scoring.move.leg_alignment_bonus_per_step` | `leg_approach_bonus` |
| `decision_support.scoring.move.sector_compliance_bonus` | `sector_compliance_bonus` |
| `decision_support.scoring.move.unknown_tile_bonus` | `unknown_tile_bonus`, `neighborhood_potential` |
| `decision_support.scoring.move.possible_target_bonus` | `possible_target` |
| `decision_support.scoring.move.figure_hint_bonus` | `figure_hint` |
| `decision_support.scoring.move.revisit_penalty` | `revisit_penalty` |
| `decision_support.scoring.move.neighborhood_potential` | `neighborhood_potential` |
| `decision_support.scoring.move.neighborhood_weight_any_figure` | `neighborhood_potential` |
| `decision_support.scoring.move.neighborhood_weight_possible_target` | `neighborhood_potential` |
| `decision_support.scoring.move.border_bonus` | `neighborhood_potential` |

## Broadcast scoring components (log column keys)

| Component key | Config key(s) | Adds to score when... |
| --- | --- | --- |
| `broadcast_base` | `decision_support.scoring.broadcast.base_broadcast_value` | No co-located drones; score starts at this base value. |
| `coordination_bonus` | `decision_support.scoring.broadcast.first_turn_coordination_bonus` | Drone 1 on round 1, turn 1; adds once. |

## Wait scoring components (log column keys)

| Component key | Config key(s) | Adds to score when... |
| --- | --- | --- |
| `wait_base` | `decision_support.scoring.wait.base_wait_value` | Default wait score when not holding position at a waypoint. |
| `wait_planned` | `decision_support.scoring.wait.planned_waiting_bonus` | If at `target_pos` **and** `current_round < next_wp["turn"]`, replaces `wait_base`. |

## Non-score directives shown in Decision Support

| Directive | Trigger condition | Effect |
| --- | --- | --- |
| `Special directive` | Drone 1, round 1, turn 1 | Instructs to broadcast the coverage plan before any other action; overrides decision support ranking in the prompt. |
| `Plan focus`, `Waypoint timing`, `Sector directive` | Mission plan + sector assignment | Informational; the scoring components above derive from these fields. |

## Config keys currently unused in scoring

None documented here. If you add unused keys, list them without values and reference the config key.
