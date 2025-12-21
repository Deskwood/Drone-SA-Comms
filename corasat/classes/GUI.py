# GUI
# =========================
class _SimulationGUI:
    """Pygame GUI for rendering board state, overlays, and logs."""
    def __init__(self, sim):
        self.sim = sim
        self.grid_size = (CONFIG["board"]["width"], CONFIG["board"]["height"])
        self.info_lines: List[str] = []
        self.info_max_lines = 10000
        self.info_scroll = 0

        pygame.init()

        gui = CONFIG["gui"]
        self.sidebar_width = gui.get("sidebar_width", 480)
        flags = pygame.HWSURFACE | pygame.DOUBLEBUF

        cell = gui["cell_size"]; m = gui["margin"]
        board_w = self.grid_size[0]*(cell+m) + m
        board_h = self.grid_size[1]*(cell+m) + m
        total_w = board_w + self.sidebar_width
        total_h = board_h

        self.screen = pygame.display.set_mode((total_w, total_h), flags)
        pygame.display.set_caption(self.sim.progress_caption(round_num=1, turn_num=1))
        self.clock = pygame.time.Clock()

        # Fonts
        try:
            self._font = pygame.font.SysFont(None, 18)
            self._font_small = pygame.font.SysFont(None, 16)
        except Exception:
            pygame.font.init()
            self._font = pygame.font.Font(None, 18)
            self._font_small = pygame.font.Font(None, 16)

        # Images
        global FIGURE_IMAGES
        if not FIGURE_IMAGES:
            FIGURE_IMAGES = load_figure_images()
        for k, surf in list(FIGURE_IMAGES.items()):
            try:
                FIGURE_IMAGES[k] = surf.convert_alpha()
            except Exception:
                pass
        self._image_cache = {}

    # ---- utilities
    def post_info(self, text: str):
        for line in text.splitlines():
            self.info_lines.append(line)
        if len(self.info_lines) > self.info_max_lines:
            self.info_lines = self.info_lines[-self.info_max_lines:]
        self.info_scroll = 0

    def _tile_center_px(self, x: int, y: int) -> Tuple[int,int]:
        gui = CONFIG["gui"]
        cell = gui["cell_size"]; m = gui["margin"]
        _, gh = self.grid_size
        y_flip = gh - 1 - y
        cx = x*(cell+m) + m + cell//2
        cy = y_flip*(cell+m) + m + cell//2
        return (cx, cy)

    def save_screenshot(self, path: Optional[str] = None):
        out_dir = "screenshots"
        os.makedirs(out_dir, exist_ok=True)
        fname = path or os.path.join(out_dir, f"last_run.png")
        try:
            pygame.image.save(self.screen, fname)
            LOGGER.log(f"Saved GUI screenshot: {fname}")
        except Exception as e:
            LOGGER.log(f"Failed to save screenshot: {e}")

    # ---- sidebar
    def _draw_score_panel(self, x0: int, y0: int, w: int) -> int:
        s = self.screen
        pad = 12
        y = y0

        s.blit(self._font.render("Score", True, (220, 220, 220)), (x0 + pad, y)); y += 24
        items = [
            ("True edges", len(self.sim.gt_edges)),
            ("Reported edges", len(self.sim.reported_edges)),
            ("- Correct", self.sim.correct_edge_counter),
            ("- False", self.sim.false_edge_counter),
            ("Score", self.sim.score)
        ]
        for lbl, val in items:
            s.blit(self._font_small.render(f"{lbl}: {val}", True, (200, 200, 200)), (x0 + pad, y))
            y += 18

        y += 8
        s.blit(self._font_small.render("Latest at bottom", True, (160, 160, 160)), (x0 + w - 150, y0 + 2))
        pygame.draw.line(s, (60,60,60), (x0, y), (x0 + w, y), 1)
        return y + 10
    def _draw_sidebar(self):
        gui = CONFIG["gui"]
        s = self.screen

        cell = gui["cell_size"]; m = gui["margin"]
        board_w = self.grid_size[0]*(cell+m) + m
        x0 = board_w
        y0 = 0
        w = self.sidebar_width
        h = s.get_height()

        pygame.draw.rect(s, (25,25,25), pygame.Rect(x0, y0, w, h))
        pygame.draw.line(s, (60,60,60), (x0, 0), (x0, h), 1)

        y_log_top = self._draw_score_panel(x0, y0 + 8, w)

        pad = 12
        tx = x0 + pad
        ty = y_log_top + 8
        tw = w - 2 * pad
        th = h - ty - pad

        pygame.draw.rect(s, (32,32,32), pygame.Rect(tx-4, ty-4, tw+8, th+8))

        line_h = self._font_small.get_height() + 4
        max_lines_fit = max(1, th // line_h)
        start_idx = max(0, len(self.info_lines) - max_lines_fit - self.info_scroll)
        end_idx = len(self.info_lines) - self.info_scroll

        y = ty
        for line in self.info_lines[start_idx:end_idx]:
            if y > ty + th - line_h:
                break
            s.blit(self._font_small.render(line, True, (230,230,230)), (tx, y))
            y += line_h

    # ---- board
    def _draw_dashed_line(self, color: Tuple[int, int, int], start_pos: Tuple[int, int], end_pos: Tuple[int, int], width: int = 2, dash_len: int = 8, gap_len: int = 6):
        sx, sy = start_pos
        ex, ey = end_pos
        dx = ex - sx
        dy = ey - sy
        dist = math.hypot(dx, dy)
        if dist <= 0:
            return
        step_x = dx / dist
        step_y = dy / dist
        pos = 0.0
        while pos < dist:
            seg_start = (sx + step_x * pos, sy + step_y * pos)
            seg_end = (sx + step_x * min(pos + dash_len, dist), sy + step_y * min(pos + dash_len, dist))
            pygame.draw.line(self.screen, color, seg_start, seg_end, width)
            pos += dash_len + gap_len

    def _draw_plan_legs(self):
        gui = CONFIG["gui"]
        color_default = gui["drone_color"]
        current_round = max(1, int(getattr(self.sim, "round", 1) or 1))
        def _to_cart(value):
            if value is None:
                return None
            if isinstance(value, str):
                try:
                    return chess_to_cartesian(value)
                except Exception:
                    return None
            if isinstance(value, (list, tuple)) and len(value) == 2:
                try:
                    return (int(value[0]), int(value[1]))
                except (TypeError, ValueError):
                    return None
            return None

        for drone in getattr(self.sim, "drones", []):
            color = getattr(drone, "render_color", color_default)
            current_idx = getattr(drone, "current_leg_index", 0)
            for idx, leg in enumerate(getattr(drone, "mission_plan", [])):
                if idx < current_idx:
                    continue
                start_vec = _to_cart(leg.get("leg_start") or leg.get("start_cartesian") or leg.get("start"))
                end_vec = _to_cart(leg.get("leg_end") or leg.get("target_cartesian") or leg.get("target"))
                if not start_vec or not end_vec:
                    continue
                start_px = self._tile_center_px(*start_vec)
                end_px = self._tile_center_px(*end_vec)
                if idx == current_idx:
                    pygame.draw.line(self.screen, color, start_px, end_px, 3)
                else:
                    self._draw_dashed_line(color, start_px, end_px, width=2)

                leg_turn = leg.get("turn")
                try:
                    remaining = max(0, int(leg_turn) - current_round) if leg_turn is not None else None
                except (TypeError, ValueError):
                    remaining = None
                if remaining is not None:
                    label = f"{remaining}t"
                    text = self._font_small.render(label, True, gui["text_color"], (0, 0, 0))
                    rect = text.get_rect()
                    rect.center = (end_px[0] + 6, end_px[1] - 6)
                    self.screen.blit(text, rect)
    def _draw_sector_boxes(self):
        gui = CONFIG["gui"]
        cell = gui["cell_size"]
        m = gui["margin"]
        gw, gh = self.grid_size
        color_default = gui["drone_color"]
        for drone in getattr(self.sim, "drones", []):
            try:
                bounds = drone._sector_bounds(getattr(drone, "assigned_sector", None))
            except Exception:
                bounds = None
            if not bounds:
                continue
            min_x, max_x, min_y, max_y = bounds
            rect_x = min_x * (cell + m) + m
            rect_y = (gh - 1 - max_y) * (cell + m) + m
            rect_w = (max_x - min_x + 1) * (cell + m) - m
            rect_h = (max_y - min_y + 1) * (cell + m) - m
            color = getattr(drone, "render_color", color_default)
            pygame.draw.rect(self.screen, color, pygame.Rect(rect_x, rect_y, rect_w, rect_h), 2)
    def draw_field(self):
        gui = CONFIG["gui"]
        cell = gui["cell_size"]
        m = gui["margin"]
        gw, gh = self.grid_size

        # clear
        self.screen.fill(gui["background_color"])

        # 1) draw tiles and figures first (paths must overlap tiles)
        for x in range(gw):
            for y in range(gh):
                y_flip = gh - 1 - y
                rect = pygame.Rect(x * (cell + m) + m, y_flip * (cell + m) + m, cell, cell)
                pygame.draw.rect(self.screen, gui["grid_color"], rect)

                tile = self.sim.board[x][y]

                # figure image or fallback
                if tile.figure:
                    img = FIGURE_IMAGES.get((tile.figure.color, tile.figure.figure_type))
                    if img:
                        key = (id(img), cell)
                        if key not in self._image_cache:
                            self._image_cache[key] = pygame.transform.scale(img, (cell, cell))
                        self.screen.blit(self._image_cache[key], rect.topleft)
                    else:
                        pygame.draw.circle(self.screen, (200, 200, 200), rect.center, cell // 3)

                # attacked/defended overlay
                if tile.figure:
                    fig = tile.figure
                    overlay = f"D{fig.defended_by} A{fig.attacked_by}"
                    surf = self._font_small.render(overlay, True, gui["text_color"])
                    pad = 3
                    tx = rect.left + pad
                    ty = rect.bottom - surf.get_height() - pad
                    pygame.draw.rect(self.screen, (40, 40, 40),
                                    pygame.Rect(tx - 2, ty - 1, surf.get_width() + 4, surf.get_height() + 2))
                    self.screen.blit(surf, (tx, ty))

        # 2) draw planned overlays (sectors + legs)
        self._draw_sector_boxes()
        self._draw_plan_legs()

        # 3) draw drone paths
        def _path_offset_vec(drone) -> Tuple[int, int]:
            n = max(1, len(self.sim.drones))
            k = (drone.id - 1) % n
            ang = math.radians(30.0 + 360.0 * k / n)  # diagonal base, evenly spaced by ID
            r = max(2, int(cell * 0.12))              # small pixel offset within tile
            return (int(r * math.cos(ang)), int(r * math.sin(ang)))

        for drone in self.sim.drones:
            if len(drone.mission_report) >= 2:
                base_pts = [ self._tile_center_px(x, y) for (x, y) in drone.mission_report ]
                ox, oy = _path_offset_vec(drone)
                pts = [(x + ox, y + oy) for (x, y) in base_pts]
                color = getattr(drone, "render_color", gui["drone_color"])
                try:
                    pygame.draw.lines(self.screen, color, False, pts, 2)
                except Exception:
                    pass
                # start marker
                sx, sy = pts[0]
                pygame.draw.circle(self.screen, color, (sx, sy), 3)

        # 4) draw drones on top of paths
        for x in range(gw):
            for y in range(gh):
                y_flip = gh - 1 - y
                rect = pygame.Rect(x * (cell + m) + m, y_flip * (cell + m) + m, cell, cell)
                tile = self.sim.board[x][y]

                total = len(tile.drones)
                if total > 0:
                    angle_step = 360 / total if total > 1 else 0
                    radius = cell // 6
                    for d_idx, drone in enumerate(tile.drones):
                        # radial offset for multiple drones on same tile
                        offset = pygame.math.Vector2(0, 0)
                        if total > 1:
                            offset = pygame.math.Vector2(1, 0).rotate(angle_step * d_idx) * (cell // 4)
                        center = (rect.centerx + int(offset.x), rect.centery + int(offset.y))

                        circle_color = getattr(drone, "render_color", gui["drone_color"])
                        pygame.draw.circle(self.screen, circle_color, center, radius)

                        # opaque ID text (prevents path bleed-through)
                        label = str(drone.id)
                        shadow = self._font.render(label, True, (0, 0, 0), circle_color)
                        text_surf = self._font.render(label, True, gui["text_color"], circle_color)
                        shadow_rect = shadow.get_rect(center=(center[0] + 1, center[1] + 1))
                        text_rect = text_surf.get_rect(center=center)
                        self.screen.blit(shadow, shadow_rect)
                        self.screen.blit(text_surf, text_rect)

        # 5) highlight current drone while thinking
        if getattr(self.sim, "_thinking", False):
            current_drone = self.sim.drones[self.sim.turn - 1] # 0-based turn index
            if current_drone:
                cx, cy = current_drone.position
                y_flip = gh - 1 - cy
                hi_rect = pygame.Rect(cx * (cell + m) + m, y_flip * (cell + m) + m, cell, cell)
                pygame.draw.rect(self.screen, (255, 215, 0), hi_rect, 2)

        # 6) sidebar and flip
        self._draw_sidebar()
        pygame.display.flip()