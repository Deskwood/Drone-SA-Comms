# Drone
# =========================
class _Drone:
    """Drone agent handling sector assignment, intel, movement, and model I/O."""
    # --- Setup & state ---
    def __init__(self, sim: object, id: int, position: Tuple[int, int], model: str, rules: str, ):
        self.sim = sim

        rules = rules \
            .replace("DRONE_ID", str(id)) \
            .replace("NUMBER_OF_DRONES", str(CONFIG["simulation"]["num_drones"])) \
            .replace("NUMBER_OF_ROUNDS", str(CONFIG["simulation"]["max_rounds"]))
        self.knowledge = _Drone_Knowledge(self, id, position, rules)
        self.decision_support = _Drone_Decision_Support(self)
        self.language_model = _Drone_LM(self, model)
        self.operator = _Drone_Operator(self)

    # --- Sector assignment ---


    def _apply_rendezvous_to_plan(self, plan: List[dict]) -> List[dict]:
        return plan

    def _sanitize_leg_index(self) -> None:
        if not self.mission_plan:
            self.current_leg_index = 0
            return
        self.current_leg_index = max(0, min(self.current_leg_index, len(self.mission_plan) - 1))

    def _current_mission_leg(self) -> Optional[dict]:
        if not self.mission_plan:
            return None
        self._sanitize_leg_index()
        return self.mission_plan[self.current_leg_index]

    def _advance_leg_progress(self) -> None:
        while True:
            leg = self._current_mission_leg()
            if not leg:
                return
            arrival_turn = max(1, int(leg.get("turn", 0) or 1))
            duration = max(0, int(leg.get("duration_turns", 0) or 0))
            current_round = max(1, int(getattr(self.sim, "round", 1) or 1))
            if current_round >= arrival_turn + duration:
                if self.current_leg_index < len(self.mission_plan) - 1:
                    self.current_leg_index += 1
                    continue
                self.current_leg_index = len(self.mission_plan) - 1
            return

    def _leg_segment_bounds(self, leg: dict) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
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

    def _distance_to_leg(self, pos: Tuple[int, int], leg: Optional[dict]) -> Optional[int]:
        if leg is None:
            return None
        bounds = self._leg_segment_bounds(leg)
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

    def _distance_to_leg_end(self, pos: Tuple[int, int], leg: Optional[dict]) -> Optional[int]:
        if leg is None:
            return None
        target_vec = leg.get("leg_end")
        if not target_vec:
            return None
        tx, ty = target_vec
        return int(round(math.hypot(pos[0] - tx, pos[1] - ty)))

    def _build_initial_mission_plan(self) -> List[dict]:
        return self._build_mission_plan_for_sector(self.assigned_sector)

    def _build_mission_plan_for_sector(self, sector: Optional[Dict[str, Any]]) -> List[dict]:
        """Plan waypoints: current position -> nearest corner -> other corners -> rendezvous."""
        bounds = self._sector_bounds(sector)
        if not bounds:
            sector = self._default_sector_assignment()
            bounds = self._sector_bounds(sector)
            self.assigned_sector = sector
        min_x, max_x, min_y, max_y = bounds

        plan: List[dict] = []
        leg_id = 1
        current_turn = 1
        start_pos = tuple(self.position)

        corners = [
            (min_x, max_y),
            (max_x, max_y),
            (max_x, min_y),
            (min_x, min_y),
        ]
        nearest_idx = min(range(len(corners)), key=lambda i: chebyshev_distance(start_pos, corners[i]))
        ordered_corners = corners[nearest_idx:] + corners[:nearest_idx]

        max_rounds = max(1, CONFIG["simulation"].get("max_rounds", 1))
        rv_turn = max_rounds - 1
        rv_cart = None
        if getattr(self, "rendezvous_directive", None):
            rv_cart = self.rendezvous_directive.get("target_cartesian")
            if rv_cart is None and self.rendezvous_directive.get("target"):
                rv_cart = list(chess_to_cartesian(self.rendezvous_directive["target"]))
            rv_turn = max(1, self.rendezvous_directive.get("turn") or rv_turn)
        if rv_cart is None:
            rv_cart = ordered_corners[-1] if ordered_corners else start_pos

        waypoints = [start_pos] + ordered_corners + [tuple(rv_cart)]
        legs = list(zip(waypoints[:-1], waypoints[1:]))
        distances = [chebyshev_distance(a, b) for a, b in legs]
        total_dist = max(1, sum(distances))
        available_turns = max(1, rv_turn - current_turn)

        cumulative = 0
        for (wp_start, wp_end), dist in zip(legs, distances):
            cumulative += dist
            arrival_turn = current_turn + int(round(available_turns * (cumulative / total_dist)))
            duration_turns = (max_rounds - arrival_turn + 1) if wp_end == tuple(rv_cart) else 0
            plan.append({
                "leg_id": leg_id,
                "turn": arrival_turn,
                "leg_start": [wp_start[0], wp_start[1]],
                "leg_end": [wp_end[0], wp_end[1]],
                "duration_turns": duration_turns,
            })
            leg_id += 1

        self.current_leg_index = 0
        self.mission_plan = plan
        self._sanitize_leg_index()
        self._advance_leg_progress()
        return plan

    def _next_mission_waypoint(self) -> Optional[dict]:
        return self._current_mission_leg()

    
    # --- Decision support ---
    def _decision_support_snapshot(self) -> Dict[str, object]:
        return self.decision_support.snapshot()

    def _format_decision_support_lines(self, snapshot: Dict[str, object]) -> Tuple[List[str], List[str]]:
        return self.decision_support.format_lines(snapshot)

    def _identify_edges(self):
        # Identify edges according to current local board knowledge
        directions = []
        for bx in range(CONFIG["board"]["width"]):
            for by in range(CONFIG["board"]["height"]):
                board_chess = Waypoint((bx,by)).to_chess()
                # Identify figure
                figure_type = self.local_board[board_chess]["type"]
                figure_color = self.local_board[board_chess]["color"]
                if not figure_type in FIGURE_TYPES: continue # Only consider known figures
                if figure_type in ("queen", "rook", "bishop"): # Sliding pieces
                    is_slider = True
                    if figure_type == "rook":
                        directions = [(1,0),(-1,0),(0,1),(0,-1)]
                    elif figure_type == "bishop":
                        directions = [(1,1),(-1,-1),(1,-1),(-1,1)]
                    else:
                        directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]
                else:
                    is_slider = False
                    if figure_type == "knight":
                        directions = [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]
                    elif figure_type == "king":
                        directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]
                    elif figure_type == "pawn":
                        directions = [(1,1),(-1,1)] if figure_color == "white" else [(1,-1),(-1,-1)]

                # Virtually move figure
                for dx, dy in directions:
                    tx, ty = bx, by
                    while True:
                        tx += dx
                        ty += dy
                        target_chess = Waypoint((tx,ty)).to_chess()
                        if not on_board(tx, ty): break
                        if self.local_board[target_chess]["type"] == "unknown": # a possible target (prioritize investigation)
                            self.local_board[target_chess]["type"] = "a possible target"
                        if self.local_board[target_chess]["type"] == "a possible target": # Moving beyond this is pointless for sliders
                            break
                        if self.local_board[target_chess]["type"] in FIGURE_TYPES \
                            or self.local_board[target_chess]["type"] == "any figure": # Found an edge
                            edge = format_edge(
                                figure_type,
                                self.local_board[board_chess]["color"],
                                self.local_board[target_chess]["color"],
                                ((bx,by),(tx,ty)))
                            if edge not in self.identified_edges and (bx,by)!=(tx,ty):
                                self.identified_edges.append(edge)
                                if edge not in self.sim.reported_edges:
                                    if edge in self.sim.gt_edges:
                                        correct_marker = "- CORRECT: "
                                    else:
                                        correct_marker = "- FALSE: "
                                    edge_identified_message = f"{correct_marker}{edge}"
                                    LOGGER.log(edge_identified_message)
                                    if self.sim.gui:
                                        self.sim.gui.post_info(edge_identified_message)
                            break
                        if not is_slider: break

    def update_board_report(self):
        # Current position (see color and type)
        sx, sy = self.position
        tile = self.sim.board[sx][sy]
        if tile.figure:
            self.local_board[cartesian_to_chess((sx,sy))] = {"color": tile.figure.color, "type": tile.figure.figure_type}
        else:
            self.local_board[cartesian_to_chess((sx,sy))] = {"color": "n/a", "type": "n/a"}

        # Neighboring tiles (see color, no type)
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if dx==0 and dy==0: continue
                nx, ny = sx+dx, sy+dy
                if not on_board(nx, ny): continue
                neighbor_tile = self.sim.board[nx][ny]
                if neighbor_tile.figure:
                    if self.local_board[cartesian_to_chess((nx,ny))]["type"] == "unknown":
                        self.local_board[cartesian_to_chess((nx,ny))] = {"color": neighbor_tile.figure.color, "type": "any figure"}
                else:
                    self.local_board[cartesian_to_chess((nx,ny))] = {"color": "n/a", "type": "n/a"}

        # Identify edges according to board knowledge
        self._identify_edges()

        # Report to simulation
        self.sim.report_edges(self.identified_edges)

    def _apply_plan_directive(self, payload: Dict[str, object]) -> None:
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
                if str(drone_ref).strip() == str(self.id):
                    sector = assignment.get("sector")
                    if sector is not None:
                        self._set_sector_assignment(sector)
                    break

        rendezvous = plan.get("rendezvous")
        if isinstance(rendezvous, dict):
            tile = rendezvous.get("tile")
            turn = rendezvous.get("turn")
            if tile:
                try:
                    cart = chess_to_cartesian(str(tile))
                    if not on_board(*cart):
                        LOGGER.log(f"Drone {self.id} received out-of-bounds rendezvous '{tile}'")
                        return
                    max_rounds = max(1, CONFIG["simulation"].get("max_rounds", 1))
                    requested_turn = int(turn) if turn is not None else max_rounds - 1
                    rv_turn = max(1, max_rounds - 1)
                    if rv_turn != requested_turn:
                        LOGGER.log(f"Drone {self.id} set rendezvous turn to {rv_turn} to reserve the final round for intel broadcast")
                    waypoint = {
                        "turn": rv_turn,
                        "target": cartesian_to_chess(cart),
                        "target_cartesian": [cart[0], cart[1]],
                        "distance_steps": chebyshev_distance(self.position, cart),
                        "tolerance_steps": 0,
                        "notes": "Rendezvous directive."
                    }
                    self.rendezvous_directive = waypoint

                    # rebuild mission plan to include rendezvous waypoint
                    try:
                        self._build_mission_plan_for_sector(self.assigned_sector)
                    except Exception as exc:
                        LOGGER.log(f"Drone {self.id} failed to rebuild mission plan with rendezvous: {exc}")
                    LOGGER.log(f"Drone {self.id} adopted rendezvous {waypoint['target']} by turn {rv_turn}")
                except Exception as exc:
                    LOGGER.log(f"Drone {self.id} failed to apply rendezvous directive: {exc}")
        self._sanitize_leg_index()

    def _provide_intelligence_to(self, target_drone: '_Drone'):
        # Share local board knowledge
        for pos, info in self.local_board.items():
            tgt_info = target_drone.local_board.get(pos, {"color": "unknown", "type": "unknown"})
            if tgt_info["color"] == "unknown" and info["color"] != "unknown":
                target_drone.local_board[pos]["color"] = info["color"]
            if tgt_info["type"] == "unknown" and info["type"] != "unknown":
                target_drone.local_board[pos]["type"] = info["type"]

        # Share identified edges
        for edge in self.identified_edges:
            if edge not in target_drone.identified_edges:
                target_drone.identified_edges.append(edge)

        current_round = self.sim.round
        self.info_exchange_rounds[target_drone.id] = current_round
        target_drone.info_exchange_rounds[self.id] = current_round
        # Update their board report (in case new edges imply new "a possible target" statuses)
        target_drone.update_board_report()

    def generate_full_model_response(self) -> List[dict]:
        temperature = CONFIG["simulation"].get("temperature", 0.7)
        situation = self._determine_situation_description()

        # Append cues
        promt_requests = CONFIG.get("prompt_requests", {})
        cues = "\n".join([
            promt_requests.get("rationale",""), 
            promt_requests.get("action",""),
            promt_requests.get("action_move",""), 
            promt_requests.get("action_broadcast",""), 
            promt_requests.get("memory_update","")
        ]).strip()
        user_content = situation if not cues else situation + "\n\n" + cues

        messages = [
            {"role": "system", "content": self.rules},
            {"role": "user", "content": user_content}
        ]
        prompt_char_len = len(user_content) + len(self.rules)
        print(f"Context length (chars): {prompt_char_len}")
        return self._generate_single_model_response(
            messages=messages,
            model=self.model,
            temperature=temperature,
            prompt_char_len=prompt_char_len,
        )

    # --- Movement ---
    def _legal_movement_steps(self) -> List[dict]:
        sx, sy = self.position
        reachable_tiles = []
        for name, (dx, dy) in DIRECTION_MAP.items():
            nx, ny = sx + dx, sy + dy
            if on_board(nx, ny):
                reachable_tiles.append({
                    "direction": name,
                    "new_position": (nx, ny)
                    })
        return reachable_tiles

    def _move(self, direction: str) -> bool:
        direction = (direction or "").lower()
        if direction in DIRECTION_MAP:
            dx, dy = DIRECTION_MAP[direction]
            nx, ny = self.position[0] + dx, self.position[1] + dy
            if on_board(nx, ny):
                self.sim.board[self.position[0]][self.position[1]].remove_drone(self)
                self.position = (nx, ny)
                self.sim.board[nx][ny].add_drone(self)
                self.mission_report.append(self.position)
                self.update_board_report()
                self._advance_leg_progress()
                LOGGER.log(f"Drone {self.id} moved to {cartesian_to_chess(self.position)}.")
                return True
            else:
                LOGGER.log(f"ERROR: Drone {self.id} attempted to move to {(nx,ny)}.")
        else:
            LOGGER.log(f"ERROR: Drone {self.id} attempted invalid direction '{direction}'.")
        return False

    # --- Situation & model I/O ---
    def _visible_neighbor_figures(self) -> str:
        """Return visibility summary of neighboring figures by direction."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = self.position[0] + dx, self.position[1] + dy
                if not on_board(nx, ny):
                    continue
                tile = self.sim.board[nx][ny]
                if tile.figure:
                    neighbors.append(f"{direction_from_vector((dx, dy))}: {tile.figure.color}")
        return ", ".join(neighbors)

    def _collected_figure_information_text(self) -> str:
        """Summarize known figure intel from the local board."""
        entries = []
        for bx in range(CONFIG["board"]["width"]):
            for by in range(CONFIG["board"]["height"]):
                info = self.local_board[cartesian_to_chess((bx, by))]
                if info["type"] in FIGURE_TYPES:
                    entries.append(f"{cartesian_to_chess((bx, by))}: {info['color']} {info['type']}")
                elif info["type"] == "a possible target":
                    entries.append(f"{cartesian_to_chess((bx, by))}: {info['type']}")
        return ", ".join(entries)

    def _determine_situation_description(self) -> str:
        # Only local, directly observable info + own memory + received broadcasts + gathered intelligence
        same_tile_drones = [
            f"Drone {drone.id}" for drone in self.sim.board[self.position[0]][self.position[1]].drones if drone.id != self.id
        ]

        fig_here = "None"
        if self.sim.board[self.position[0]][self.position[1]].figure:
            fig_here = self.sim.board[self.position[0]][self.position[1]].figure.figure_type

        # neighbor figure colors (visibility model)
        neighbor_figures = self._visible_neighbor_figures()

        legal_movements = ", ".join([f"{lms['direction']} to {cartesian_to_chess(lms['new_position'])}" for lms in self._legal_movement_steps()])

        collected_figure_information = self._collected_figure_information_text()

        s = []
        s.append(f"Current round number: {self.sim.round} of {CONFIG['simulation']['max_rounds']} rounds.")
        s.append(f"Current position: {cartesian_to_chess(self.position)}")
        s.append(f"Legal movements: {legal_movements}")
        # rest of standard context
        s.append(f"Visible drones at position: {', '.join(same_tile_drones) if same_tile_drones else 'None'}")
        s.append(f"Visible figure at position: {fig_here}")
        s.append(f"Visible neighbor figures: {neighbor_figures or 'None'}")
        s.append(f"Memory: {self.memory}")
        s.append(f"Collected figure information: {collected_figure_information}")
        if getattr(self, 'assigned_sector', None):
            s.append(f"Assigned coverage sector: {self.sector_summary()}")
        if getattr(self, 'rendezvous_directive', None):
            rv = self.rendezvous_directive
            s.append(f"Rendezvous directive: {rv['target']} on turn {rv['turn']}")
        s.append(f"Broadcast Rx Buffer: {self.rx_buffer}")
        if self.id == 1 and getattr(self.sim, "round", None) == 1 and getattr(self.sim, "turn", None) == 1:
            total_drones = CONFIG["simulation"].get("num_drones", len(getattr(self.sim, "drones", [])))
            s.append("Special directive: As Drone 1 on the opening turn, broadcast a coverage plan assigning every drone their sector before taking other actions.")
            s.append(f"Ensure the broadcast is valid JSON with plan->assignments entries for all {total_drones} drones (including yourself) and describe each sector clearly.")

        snapshot = self._decision_support_snapshot()
        ds_lines, ledger_lines = self._format_decision_support_lines(snapshot)
        if ds_lines:
            s.append("Decision Support:")
            s.extend([f"  {entry}" for entry in ds_lines])
        if ledger_lines:
            s.append("Intel Share Ledger:")
            s.extend([f"  {entry}" for entry in ledger_lines])
        self.rx_buffer = ""  # drain the inbox each turn
        situation_description = "\n".join(s)
        LOGGER.log(f"Drone {self.id} Situation:\n{situation_description}")
        return situation_description

    def _generate_single_model_response(self, messages: List[dict], model: str, temperature: float, prompt_char_len: Optional[int] = None) -> List[dict]:
        if model == "manual":
            try:
                pyperclip.copy(messages[-1]["content"])
            except Exception:
                pass
            content = input("Paste model result: ")
            messages.append({"role": "assistant", "content": content})
            if prompt_char_len is not None:
                approx_tokens = max(1, math.ceil(prompt_char_len / 4))
                print(f"Context length: ~{approx_tokens} tokens ({prompt_char_len} chars)")
            return messages

        else: # Ollama
            request_started = time.perf_counter()
            response = ollama_chat(model=model, messages=messages, stream=False, format="json",
                               options={"temperature": float(temperature)})
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

            if isinstance(completion_tokens, (int, float)) and completion_tokens > 0 and duration_seconds and duration_seconds > 0:
                tokens_per_second = completion_tokens / duration_seconds
                print(f"Tokens per second: {tokens_per_second:.2f} tok/s ({completion_tokens} completion tokens)")
            return messages