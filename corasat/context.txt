You are assisting with my Master Thesis research environment called Collaborative Radio Silent Agent Tuning (Corasat).

CONTEXT SUMMARY
- Corasat consists of G games.
- Each game has R rounds.
- Each round has T turns (one per drone, in sequential order: Drone 1, Drone 2, …).

TURN PROMPT CONTENT
Each turn, the active drone receives the following:
1. RULES: loaded from rules.txt. These may be rewritten by the OPTIMIZER model between games, but CONSTRAINTS must remain logically unchanged.
2. SITUATION: environment information for this turn. Includes:
- drone ID
- round number r and total rounds R
- total number of drones
- which drones are co-located
- which figure is co-located (color + type)
- compass directions of neighboring figures (color only)
- the drone’s current location
  (Note: chessboard coordinates may be converted to Cartesian, e.g., A2 → (0,1).)
3. MEMORY: text stored by this drone from its last turn.
4. RX_BUFFER: broadcasts received from co-located drones in the previous round. This buffer is cleared once read.

DRONE OUTPUT REQUIREMENTS
The drone must output:
- Action: one of {wait, move, broadcast}.
- Specifier:
-- wait → none
-- move → compass direction
-- broadcast → message text
- Rationale: reasoning for debugging.
- New MEMORY: arbitrary content (task list, strategy, notes, etc.) carried into its next turn.

CONSTRAINTS
- Drones always remain on valid chessboard tiles.
- Movement follows king moves (8 directions).
- Drones cannot move across the border. If a move is invalid, the action defaults to wait.
- Drones may only communicate by broadcast, and only with drones on the same tile in that turn.
- All drones co-start on the same tile in the first round.
- Each drone performs exactly one action per turn.
- Chess figures define nodes; attacked/defended relations define edges. This ground truth is known only to the simulation.
- No early termination. All R rounds are always executed.

PERFORMANCE METRICS
- Recall = |Identified edges ∩ Ground Truth| ÷ |Ground Truth|
- Precision is also tracked.
- Example composite score = (True Positives – False Positives) ÷ |Ground Truth|

OPTIMIZER ROLE
- After all G games are completed, logs and config.json are provided to the OPTIMIZER model.
- The OPTIMIZER improves rules.txt, may adjust simulation code, and provides hints to maximize recall.

YOUR ROLE IN THIS CHAT
You are not the simulation.
You help design, refine, and tune the RULES, OPTIMIZER strategies, memory structures, broadcast protocols, and the general framework.
Your focus is efficiency, clarity, compliance with constraints, and strategies for improving recall and precision.