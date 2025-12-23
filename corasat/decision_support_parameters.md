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
| `visited_tiles` | Set of positions in `drone.mission_report`. |
| `neighborhood_potential_sum` | Sum of neighbor weights around the candidate tile: `any figure` = 3, `a possible target` = 1, off-board neighbor = `border_bonus`, all other types = 0. |

## Movement scoring components (log column keys)

| Component key | Config key(s) | Current value(s) | Adds to score when... |
| --- | --- | --- | --- |
| `waypoint_progress` | `decision_support.scoring.move.waypoint_progress_reward_per_step` | `1.1390557352137365` | `new_dist <= current_dist` (candidate move is not farther from the waypoint). Adds the reward once (not multiplied by delta). |
| `waypoint_regression` | `decision_support.scoring.move.waypoint_regression_penalty_per_step`, `decision_support.scoring.move.late_penalty_multiplier` | `-0.316352144799844`, `2.0` | `turns_remaining` available and `slack < 0` (i.e., `new_dist > turns_remaining`). Adds `waypoint_regression_penalty_per_step + late_penalty_multiplier * abs(slack)`. |
| `leg_alignment` | `decision_support.scoring.move.leg_alignment_reward`, `decision_support.scoring.move.leg_alignment_penalty` | `0.6`, `-0.6` | Compares `current_leg_distance - new_leg_distance`. Positive delta adds reward, negative delta adds penalty. |
| `sector_alignment` | `decision_support.scoring.move.sector_alignment_reward` | `0.7850850533649799` | If sector bounds exist and the move reduces distance to the sector bounds. |
| `unknown_tile_bonus` | `decision_support.scoring.move.unknown_tile_bonus` | `1.0` | Destination tile has local_board type `unknown`. |
| `possible_target` | `decision_support.scoring.move.possible_target_bonus` | `1.2` | Destination tile has local_board type `a possible target`. |
| `figure_hint` | `decision_support.scoring.move.figure_hint_bonus` | `3.0` | Destination tile has local_board type `any figure`. |
| `revisit_penalty` | `decision_support.scoring.move.revisit_penalty` | `-1.2` | Destination tile is in `visited_tiles`. |
| `neighborhood_potential` | `decision_support.scoring.move.neighborhood_potential`, `decision_support.scoring.move.border_bonus` | `0.45`, `0.05` | Adds `neighborhood_potential * neighborhood_potential_sum` when `neighborhood_potential_sum > 0`. |

## Move parameter values (config -> log column keys)

| Config key | Current value | Log column(s) |
| --- | --- | --- |
| `decision_support.scoring.move.waypoint_progress_reward_per_step` | `1.1390557352137365` | `waypoint_progress` |
| `decision_support.scoring.move.waypoint_regression_penalty_per_step` | `-0.316352144799844` | `waypoint_regression` |
| `decision_support.scoring.move.late_penalty_multiplier` | `2.0` | `waypoint_regression` |
| `decision_support.scoring.move.leg_alignment_reward` | `0.6` | `leg_alignment` |
| `decision_support.scoring.move.leg_alignment_penalty` | `-0.6` | `leg_alignment` |
| `decision_support.scoring.move.sector_alignment_reward` | `0.7850850533649799` | `sector_alignment` |
| `decision_support.scoring.move.unknown_tile_bonus` | `1.0` | `unknown_tile_bonus` |
| `decision_support.scoring.move.possible_target_bonus` | `1.2` | `possible_target` |
| `decision_support.scoring.move.figure_hint_bonus` | `3.0` | `figure_hint` |
| `decision_support.scoring.move.revisit_penalty` | `-1.2` | `revisit_penalty` |
| `decision_support.scoring.move.neighborhood_potential` | `0.45` | `neighborhood_potential` |
| `decision_support.scoring.move.border_bonus` | `0.05` | `neighborhood_potential` |

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
| `decision_support.scoring.broadcast.min_staleness_rounds` | `2` | Not referenced in code. |
| `decision_support.scoring.broadcast.fresh_penalty` | `-2.0` | Not referenced in code. |
| `decision_support.scoring.wait.idle_penalty_component` | `-1.0` | Read but not used in scoring. |
| `decision_support.scoring.wait.holding_pattern_component` | `0.3` | Read but not used in scoring. |
