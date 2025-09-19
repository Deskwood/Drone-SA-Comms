#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import logging
import os
import pprint
import time
import random
import re
from datetime import datetime
from typing import Tuple, List, Optional, Literal, Dict, Any, Set
import json
import pygame
import pyperclip
from concurrent.futures import ThreadPoolExecutor, Future
import nbformat
from nbconvert import PythonExporter
from pydantic import BaseModel
try:
    from ollama import chat as ollama_chat
    _OLLAMA_AVAILABLE = True
except Exception:
    _OLLAMA_AVAILABLE = False

# ---------------- Constants ----------------
COLORS = ["white", "black"]
FIGURE_TYPES = ["king", "queen", "rook", "bishop", "knight", "pawn"]
DIRECTION_MAP: Dict[str, Tuple[int, int]] = {
    "north": (0, 1),
    "south": (0, -1),
    "east": (1, 0),
    "west": (-1, 0),
    "northeast": (1, 1),
    "northwest": (-1, 1),
    "southeast": (1, -1),
    "southwest": (-1, -1)
}
VALID_DIRECTIONS = set(DIRECTION_MAP.keys())
PLAN_PREFIX = "PLAN:"

# -------------- Optional: export notebook --------------
try:
    nb = nbformat.read("run_simulation.ipynb", as_version=4)
    body, _ = PythonExporter().from_notebook_node(nb)
    with open("run_simulation.py", "w", encoding="utf-8") as f:
        f.write(body)
except Exception:
    pass

# ---------------- Clean logger ----------------
class TimestampedLogger:
    def __init__(self, log_dir='logs', log_file='simulation.log'):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_file)

        root = logging.getLogger()
        for h in list(root.handlers):
            try: h.close()
            except Exception: pass
            root.removeHandler(h)

        try:
            if os.path.exists(self.log_path):
                os.remove(self.log_path)
        except Exception:
            pass

        fh = logging.FileHandler(self.log_path, mode='w', encoding='utf-8', delay=False)
        ch = logging.StreamHandler()

        fmt = logging.Formatter(fmt='%(levelname)s:%(name)s:%(message)s')
        fh.setFormatter(fmt); ch.setFormatter(fmt)
        root.setLevel(logging.INFO)
        root.addHandler(fh); root.addHandler(ch)

        logging.getLogger('httpx').setLevel(logging.INFO)

        self.start_time = time.time()
        self.last_time = self.start_time
        self.log("Logger initialized.")

    def _now(self): return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _duration(self):
        current_time = time.time()
        d = current_time - self.last_time
        self.last_time = current_time
        return f"{d:.3f}s"

    def log(self, message):
        logging.info(f"[{self._now()}] (+{self._duration()}) {message}")

LOGGER = TimestampedLogger()

# ---------------- Config ----------------
def load_config(config_path: str = "config.json") -> dict:
    LOGGER.log(f"Load Config: {config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing config file: {config_path}")
    cfg.setdefault("prompt_requests", {})
    cfg.setdefault("simulation", {})
    cfg.setdefault("board", {"width": 8, "height": 8})
    cfg.setdefault("gui", {
        "cell_size": 64,
        "margin": 2,
        "background_color": (30,30,30),
        "grid_color": (80,80,80),
        "drone_color": (200,200,50),
        "text_color": (20,20,20),
        "figure_image_dir": "figures",
        "sidebar_width": 480
    })
    sim = cfg["simulation"]
    sim.setdefault("max_rounds", 10)
    sim.setdefault("num_drones", 4)
    sim.setdefault("models", ["manual"])
    sim.setdefault("model_index", 0)
    sim.setdefault("temperature", 0.7)
    sim.setdefault("use_gui", True)
    sim.setdefault("headless", False)
    sim.setdefault("rules_path", cfg.get("rules_path", "rules.txt"))
    # generous budgets with cap
    sim.setdefault("max_tokens_for_rationale", 1024)
    sim.setdefault("max_tokens_for_action", 1024)
    sim.setdefault("max_tokens_for_action_move", 1024)
    sim.setdefault("max_tokens_for_action_broadcast", 1024)
    sim.setdefault("max_tokens_for_memory", 1024)
    sim.setdefault("max_tokens_total_cap", 4096)
    # planning + enforcement
    sim.setdefault("planning_rounds", 3)
    sim.setdefault("enforce_plan", True)
    # figure randomization
    sim.setdefault("randomize_figures", False)
    sim.setdefault("random_seed", None)

    cfg.setdefault("figures", {c: {t: [] for t in FIGURE_TYPES} for c in COLORS})
    return cfg

CONFIG = load_config("config.json")

# ---------------- Images ----------------
def load_figure_images() -> dict:
    images = {}
    base_path = CONFIG["gui"]["figure_image_dir"]

    def try_load(path):
        return pygame.image.load(path) if os.path.exists(path) else None

    for color in COLORS:
        for figure_type in FIGURE_TYPES:
            candidates = [
                f"{color}{figure_type}.png",
                f"{color.capitalize()}{figure_type}.png",
                f"{color}{figure_type.capitalize()}.png",
                f"{color.capitalize()}{figure_type.capitalize()}.png"
            ]
            img = None
            for name in candidates:
                p = os.path.join(base_path, name)
                img = try_load(p)
                if img:
                    LOGGER.log(f"Loaded image: {p}")
                    break
            if img:
                images[(color, figure_type)] = img
            else:
                LOGGER.log(f"Warning: Image not found for {color} {figure_type} in {base_path}")
    return images

FIGURE_IMAGES = {}

def direction_from_vector(vector: Tuple[int, int]) -> str:
    for direction, vec in DIRECTION_MAP.items():
        if vec == vector:
            return direction
    return str(vector)

# ---------------- TurnResult + JSON helpers ----------------
class TurnResult(BaseModel):
    rationale: str
    action: Literal["wait", "move", "broadcast"]
    direction: Optional[str] = None
    message: Optional[str] = None
    memory: str

def _extract_first_json_block(text: str) -> str:
    start = text.find('{')
    if start == -1: return text
    stack = 0
    for i in range(start, len(text)):
        if text[i] == '{': stack += 1
        elif text[i] == '}':
            stack -= 1
            if stack == 0:
                return text[start:i+1]
    return text

def safe_parse_turnresult(payload: str) -> dict:
    try:
        candidate = _extract_first_json_block(payload)
        data = json.loads(candidate)
        return TurnResult.model_validate(data).model_dump()
    except Exception as e:
        return {"rationale": f"Parse/validate error: {e}", "action": "wait", "direction": None, "message": None, "memory": ""}

# ---------------- Plan parsing ----------------
PLAN_RE = re.compile(r'(?i)\bplan\s*:\s*.*?\bpath\s*=\s*([^;|.\n\r]+)')

def parse_plan_from_text(text: str) -> List[str]:
    if not text:
        return []
    m = PLAN_RE.search(text)
    if not m:
        return []
    raw = m.group(1)
    alias = {
        "n": "north", "s": "south", "e": "east", "w": "west",
        "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest"
    }
    kept, dropped = [], []
    for tok in [t.strip().lower() for t in raw.split(",") if t.strip()]:
        tok = alias.get(tok, tok)
        if tok in DIRECTION_MAP:
            kept.append(tok)
        else:
            dropped.append(tok)
    if dropped:
        LOGGER.log(f"Plan parser dropped tokens: {dropped}")
    return kept

# ---------------- Randomize figures ----------------
def _randomize_figures_layout(figures_cfg: dict, board_w: int, board_h: int, seed: Optional[int] = None) -> dict:
    requests = []
    for color, types in figures_cfg.items():
        for ftype, positions in types.items():
            cnt = len(positions)
            if cnt > 0:
                requests.append((color, ftype, cnt))

    all_tiles = [(x, y) for x in range(board_w) for y in range(board_h)]
    rng = random.Random(seed)
    rng.shuffle(all_tiles)

    cursor = 0
    out = {c: {t: [] for t in FIGURE_TYPES} for c in COLORS}
    for color, ftype, cnt in requests:
        if cursor + cnt > len(all_tiles):
            raise ValueError("Not enough tiles to place all requested figures.")
        picks = all_tiles[cursor:cursor+cnt]
        cursor += cnt
        out[color][ftype] = [list(p) for p in picks]
    return out

# ---------------- Board & edges ----------------
class _Figure:
    def __init__(self, position: Tuple[int, int], color: str, figure_type: str):
        self.position = position
        self.color = color
        self.figure_type = figure_type
        self.defended_by = 0
        self.attacked_by = 0
        self.target_positions: List[Tuple[int, int]] = []

    def calculate_figure_targets(self, board: List[List['_Tile']]):
        self.target_positions = []
        W, H = CONFIG["board"]["width"], CONFIG["board"]["height"]

        def on_board(x, y): return 0 <= x < W and 0 <= y < H

        if self.figure_type in ("queen", "rook", "bishop"):
            if self.figure_type == "rook":
                directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            elif self.figure_type == "bishop":
                directions = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
            else:  # queen
                directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                              (1, 1), (-1, -1), (1, -1), (-1, 1)]
            for dx, dy in directions:
                x, y = self.position
                while True:
                    x += dx; y += dy
                    if not on_board(x, y): break
                    self.target_positions.append((x, y))
                    if board[x][y].figure is not None:
                        break

        elif self.figure_type == "knight":
            for dx, dy in [(2, 1), (2, -1), (-2, 1), (-2, -1),
                           (1, 2), (1, -2), (-1, 2), (-1, -2)]:
                x = self.position[0] + dx
                y = self.position[1] + dy
                if on_board(x, y):
                    self.target_positions.append((x, y))

        elif self.figure_type == "king":
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                x = self.position[0] + dx
                y = self.position[1] + dy
                if on_board(x, y):
                    self.target_positions.append((x, y))

        elif self.figure_type == "pawn":
            diagonals = [(1, 1), (-1, 1)] if self.color == "white" else [(1, -1), (-1, -1)]
            for dx, dy in diagonals:
                x = self.position[0] + dx
                y = self.position[1] + dy
                if on_board(x, y):
                    self.target_positions.append((x, y))

class _Tile:
    def __init__(self, x: int, y: int):
        self.x = x; self.y = y
        self.targeted_by = {"white": 0, "black": 0}
        self.figure: Optional[_Figure] = None
        self.drones: List['_Drone'] = []

    def set_figure(self, figure: _Figure): self.figure = figure
    def add_drone(self, drone: '_Drone'):
        if drone not in self.drones:
            self.drones.append(drone)
    def remove_drone(self, drone: '_Drone'):
        if drone in self.drones:
            self.drones.remove(drone)
    def reset_targeted_by_amounts(self): self.targeted_by = {"white": 0, "black": 0}
    def add_targeted_by_amount(self, color: str, amount: int = 1): self.targeted_by[color] += amount

def _compute_edges_for(figures: List[_Figure], board: List[List[_Tile]]) -> Set[Tuple[Tuple[int,int], Tuple[int,int]]]:
    edges: Set[Tuple[Tuple[int,int], Tuple[int,int]]] = set()
    for f in figures:
        for (tx, ty) in f.target_positions:
            if board[tx][ty].figure is not None:
                edges.add((f.position, board[tx][ty].figure.position))
    return edges

# ---------------- Drone ----------------
class _Drone:
    def __init__(self, id: int, position: Tuple[int, int], model: str, rules: str, sim, color: str = "white"):
        self.id = id
        self.position = position
        self.color = color
        self.model = model
        self.sim = sim
        self.rules = rules.replace("DRONE_ID", str(self.id))\
                          .replace("NUMBER_OF_DRONES", str(CONFIG["simulation"]["num_drones"]))\
                          .replace("NUMBER_OF_ROUNDS", str(CONFIG["simulation"]["max_rounds"]))
        self.memory = ""     # per-drone memory (LLM-owned string)
        self.rx_buffer = ""  # per-drone inbox; only filled by co-located broadcasts

    def _move(self, direction: str) -> bool:
        direction = (direction or "").lower()
        if direction in DIRECTION_MAP:
            dx, dy = DIRECTION_MAP[direction]
            nx, ny = self.position[0] + dx, self.position[1] + dy
            if 0 <= nx < CONFIG["board"]["width"] and 0 <= ny < CONFIG["board"]["height"]:
                self.sim.board[self.position[0]][self.position[1]].remove_drone(self)
                self.position = (nx, ny)
                self.sim.board[nx][ny].add_drone(self)
                LOGGER.log(f"Drone {self.id} moved to {self.position}.")
                return True
            else:
                LOGGER.log(f"Drone {self.id} attempted OOB move to {(nx,ny)}.")
        else:
            LOGGER.log(f"Drone {self.id} attempted invalid direction '{direction}'.")
        return False

    def _allowed_directions(self) -> List[str]:
        x, y = self.position
        W, H = CONFIG["board"]["width"], CONFIG["board"]["height"]
        allowed = []
        for name, (dx, dy) in DIRECTION_MAP.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H:
                allowed.append(name)
        return allowed

    def _phase(self) -> str:
        return "Planning" if self.sim.round <= self.sim.planning_rounds else "Execution"

    def _determine_situation_description(self) -> str:
        # Only local, directly observable info + own memory + received broadcasts
        same_tile_drones = [
            f"Drone {d.id}" for d in self.sim.board[self.position[0]][self.position[1]].drones if d.id != self.id
        ]

        fig_here = "None"
        if self.sim.board[self.position[0]][self.position[1]].figure:
            # Co-located tile reveals full (per rules), but we only name the type here as an observation
            fig_here = self.sim.board[self.position[0]][self.position[1]].figure.figure_type

        neigh = ""
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if dx==0 and dy==0: 
                    continue
                nx, ny = self.position[0]+dx, self.position[1]+dy
                if 0 <= nx < CONFIG["board"]["width"] and 0 <= ny < CONFIG["board"]["height"]:
                    t = self.sim.board[nx][ny]
                    if t.figure:
                        # Adjacent tiles: visible color only (per rules)
                        neigh += f"{direction_from_vector((dx,dy))}: {t.figure.color}, "
        neigh = neigh.strip(", ")

        allowed = self._allowed_directions()

        s = []
        s.append(f"Phase: {self._phase()}")
        s.append(f"Current round number: {self.sim.round}")
        s.append(f"Board size: {CONFIG['board']['width']}x{CONFIG['board']['height']} (x=0..{CONFIG['board']['width']-1}, y=0..{CONFIG['board']['height']-1})")
        s.append(f"My grid coords: x={self.position[0]}, y={self.position[1]}")
        s.append(f"Current position: {self.position}")
        s.append(f"AllowedDirections: {allowed}")
        s.append("Reminder: You MUST pick 'direction' only from AllowedDirections when action=='move'.")
        s.append(f"Visible drones at position: {', '.join(same_tile_drones) if same_tile_drones else 'None'}")
        s.append(f"Visible figure at position: {fig_here}")
        s.append(f"Visible neighboring figures: {neigh or 'None'}")
        s.append(f"Memory: {self.memory}")
        s.append(f"Broadcast Rx Buffer: {self.rx_buffer}")
        self.rx_buffer = ""  # drain the inbox each turn
        return "\n".join(s)

    # LLM interface (unchanged besides using only local situation above)
    def _token_budget_total(self) -> int:
        sim = CONFIG["simulation"]
        est = (int(sim.get("max_tokens_for_rationale",2048)) +
               int(sim.get("max_tokens_for_action",2048)) +
               max(int(sim.get("max_tokens_for_action_move",2048)),
                   int(sim.get("max_tokens_for_action_broadcast",2048))) +
               int(sim.get("max_tokens_for_memory",2048)))
        cap = int(sim.get("max_tokens_total_cap", 4096))
        return max(512, min(est, cap))

    def _generate_single_model_response(self, messages: List[dict], model: str, temperature: float) -> List[dict]:
        def _store(text: str) -> List[dict]:
            validated = safe_parse_turnresult(text)
            messages.append({"role": "assistant", "content": validated})
            return messages

        max_tokens_total = self._token_budget_total()
        num_predict = max(128, max_tokens_total)

        if model == "manual":
            try: pyperclip.copy(messages[-1]["content"])
            except Exception: pass
            return _store(input("Paste pure JSON TurnResult: "))

        if not _OLLAMA_AVAILABLE:
            return _store(json.dumps({"rationale": "Ollama not installed; wait.", "action": "wait", "direction": None, "message": None, "memory": ""}))

        def _ollama(extra_hint: Optional[str] = None, np: int = num_predict):
            mm = messages if not extra_hint else messages + [{"role": "user", "content": extra_hint}]
            resp = ollama_chat(model=model, messages=mm, stream=False, format="json",
                               options={"temperature": float(temperature), "num_predict": int(np)})
            content = getattr(resp, "message", None)
            if content and hasattr(content, "content"):
                return content.content or ""
            if isinstance(resp, dict):
                return resp.get("message", {}).get("content", "")
            return str(resp)

        raw = _ollama()
        parsed = safe_parse_turnresult(raw)
        if parsed["action"] == "wait" and parsed["rationale"].startswith("Parse/validate error"):
            raw2 = _ollama("REMINDER: Output ONLY a single valid JSON object exactly matching the schema. No prose.",
                           np=int(num_predict * 2))
            return _store(raw2)
        return _store(raw)

    def generate_full_model_response(self) -> List[dict]:
        temperature = CONFIG["simulation"].get("temperature", 0.7)
        situation = self._determine_situation_description()
        pr = CONFIG.get("prompt_requests", {})
        cues = "\n".join([
            pr.get("rationale",""),
            pr.get("action",""),
            pr.get("action_move",""),
            pr.get("action_broadcast",""),
            pr.get("memory_update","")
        ]).strip()
        user_content = situation if not cues else situation + "\n\n" + cues
        messages = [
            {"role": "system", "content": self.rules},
            {"role": "user", "content": user_content}
        ]
        return self._generate_single_model_response(messages=messages, model=self.model, temperature=temperature)

# ---------------- GUI (with scoring & plan preview) ----------------
class _SimulationGUI:
    def __init__(self, sim):
        self.sim = sim
        self.grid_size = (CONFIG["board"]["width"], CONFIG["board"]["height"])

        pygame.init()
        gui = CONFIG["gui"]
        self.sidebar_width = gui.get("sidebar_width", 480)
        flags = pygame.HWSURFACE | pygame.DOUBLEBUF

        board_w = self.grid_size[0]*(gui["cell_size"]+gui["margin"]) + gui["margin"]
        board_h = self.grid_size[1]*(gui["cell_size"]+gui["margin"]) + gui["margin"]
        total_w = board_w + self.sidebar_width
        total_h = board_h

        self.screen = pygame.display.set_mode((total_w, total_h), flags)
        pygame.display.set_caption(
            f'Simulation - Round {1}.1/{CONFIG["simulation"]["max_rounds"]}.{CONFIG["simulation"]["num_drones"]}'
        )
        self.clock = pygame.time.Clock()

        global FIGURE_IMAGES
        if not FIGURE_IMAGES:
            FIGURE_IMAGES = load_figure_images()
        for k, surf in list(FIGURE_IMAGES.items()):
            try: FIGURE_IMAGES[k] = surf.convert_alpha()
            except pygame.error: pass

        self._image_cache = {}
        try:
            self._font = pygame.font.SysFont(None, 18)
            self._font_small = pygame.font.SysFont(None, 16)
        except Exception:
            pygame.font.init()
            self._font = pygame.font.Font(None, 18)
            self._font_small = pygame.font.Font(None, 16)

        self.info_lines: List[str] = []
        self.info_max_lines = 400
        self.info_scroll = 0

    def _get_scaled(self, img):
        key = (id(img), CONFIG["gui"]["cell_size"])
        if key not in self._image_cache:
            self._image_cache[key] = pygame.transform.scale(img, (CONFIG["gui"]["cell_size"], CONFIG["gui"]["cell_size"]))
        return self._image_cache[key]

    def _draw_score_panel(self, x0: int, y0: int, w: int):
        s = self.screen
        pad = 12
        y = y0

        s.blit(self._font.render("Score", True, (220, 220, 220)), (x0 + pad, y)); y += 24
        stats = self.sim.score_stats()
        items = [
            ("Phase", self.sim.phase_label()),
            ("Identified nodes", stats["identified_nodes"]),
            ("Discovered edges", stats["discovered_edges"]),
            ("GT edges", stats["gt_edges"]),
            ("Correct edges", stats["correct_edges"]),
            ("False edges", stats["false_edges"]),
            ("Score", stats["score"]),
            ("Precision", f'{stats["precision"]:.2f}'),
            ("Recall", f'{stats["recall"]:.2f}')
        ]
        for lbl, val in items:
            s.blit(self._font_small.render(f"{lbl}: {val}", True, (200, 200, 200)), (x0 + pad, y))
            y += 18

        y += 6
        s.blit(self._font.render("Plans", True, (220, 220, 220)), (x0 + pad, y)); y += 20
        for d in self.sim.drones:
            nxt = self.sim._next_planned_step(d.id)
            q = self.sim.plans.get(d.id, [])
            preview = f"next={nxt or '-'} | queue={q}" if q else "next=- | queue=[]"
            s.blit(self._font_small.render(f"Drone {d.id}: {preview}", True, (200, 200, 200)), (x0 + pad, y))
            y += 18

        y += 8
        s.blit(self._font_small.render("Latest at bottom", True, (160, 160, 160)), (x0 + w - 150, y0 + 2))
        pygame.draw.line(s, (60,60,60), (x0, y), (x0 + w, y), 1)
        return y + 10

    def _draw_sidebar(self):
        gui = CONFIG["gui"]
        s = self.screen

        board_w = self.grid_size[0]*(gui["cell_size"]+gui["margin"]) + gui["margin"]
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
        max_lines_fit = th // line_h
        start_idx = max(0, len(self.info_lines) - max_lines_fit - self.info_scroll)
        end_idx = len(self.info_lines) - self.info_scroll

        y = ty
        for line in self.info_lines[start_idx:end_idx]:
            if y > ty + th - line_h:
                break
            s.blit(self._font_small.render(line, True, (230,230,230)), (tx, y))
            y += line_h

    def post_info(self, text: str):
        for line in text.splitlines():
            self.info_lines.append(line)
        if len(self.info_lines) > self.info_max_lines:
            self.info_lines = self.info_lines[-self.info_max_lines:]
        self.info_scroll = 0

    def draw_field(self):
        gui = CONFIG["gui"]
        cell = gui["cell_size"]
        m = gui["margin"]
        gw, gh = self.grid_size

        self.screen.fill(gui["background_color"])

        for x in range(gw):
            for y in range(gh):
                y_flip = gh - 1 - y
                rect = pygame.Rect(x*(cell+m)+m, y_flip*(cell+m)+m, cell, cell)
                pygame.draw.rect(self.screen, gui["grid_color"], rect)

                tile = self.sim.board[x][y]

                if tile.figure:
                    img = FIGURE_IMAGES.get((tile.figure.color, tile.figure.figure_type))
                    if img:
                        self.screen.blit(self._get_scaled(img), rect.topleft)
                    else:
                        pygame.draw.circle(self.screen, (200,200,200), rect.center, cell//3)

                if tile.figure:
                    fig = tile.figure
                    overlay = f"D{fig.defended_by} A{fig.attacked_by}"
                    surf = self._font.render(overlay, True, gui["text_color"])
                    pad = 3
                    tx = rect.left + pad
                    ty = rect.bottom - surf.get_height() - pad
                    pygame.draw.rect(self.screen, (40,40,40), pygame.Rect(tx-2, ty-1, surf.get_width()+4, surf.get_height()+2))
                    self.screen.blit(surf, (tx, ty))

                total = len(tile.drones)
                if total > 0:
                    angle_step = 360/total if total > 1 else 0
                    radius = cell//6
                    for d_idx, drone in enumerate(tile.drones):
                        offset = pygame.math.Vector2(0,0)
                        if total > 1:
                            offset = pygame.math.Vector2(1,0).rotate(angle_step*d_idx) * (cell//4)
                        center = (rect.centerx + int(offset.x), rect.centery + int(offset.y))
                        pygame.draw.circle(self.screen, gui["drone_color"], center, radius)
                        t = self._font.render(str(drone.id), True, gui["text_color"])
                        self.screen.blit(t, t.get_rect(center=center))

        if self.sim._thinking:
            cur = self.sim.current_drone()
            if cur:
                x, y = cur.position
                y_flip = gh - 1 - y
                rect = pygame.Rect(x*(cell+m)+m, y_flip*(cell+m)+m, cell, cell)
                pygame.draw.rect(self.screen, (255,215,0), rect, 2)

        self._draw_sidebar()
        pygame.display.flip()

# ---------------- Simulation (planning coach, coordination, targeting, enforcement, scoring, metrics) ----------------
class Simulation:
    def __init__(self):
        if CONFIG["simulation"].get("headless", False):
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        self.turn = 1
        self.round = 1
        rules_path = CONFIG["simulation"].get("rules_path", CONFIG.get("rules_path", "rules.txt"))
        with open(rules_path, "r", encoding="utf-8") as f:
            self.rules = f.read().replace("NUMBER_OF_DRONES", str(CONFIG["simulation"]["num_drones"]))

        self.grid_size = (CONFIG["board"]["width"], CONFIG["board"]["height"])
        self.max_rounds = CONFIG["simulation"]["max_rounds"]
        self.num_drones = CONFIG["simulation"]["num_drones"]
        self.models = CONFIG["simulation"]["models"]
        self.model_index = CONFIG["simulation"]["model_index"]
        self.model = self.models[self.model_index]
        LOGGER.log(f"Using model: {self.model}")

        self.planning_rounds = int(CONFIG["simulation"].get("planning_rounds", 3))
        self.enforce_plan = bool(CONFIG["simulation"].get("enforce_plan", True))
        self.plans: Dict[int, List[str]] = {}  # optional plan-queue if drones put PLAN:path=... in memory

        # World
        self.board = [[_Tile(x, y) for y in range(self.grid_size[1])] for x in range(self.grid_size[0])]
        self.figures: List[_Figure] = []
        self.drones: List[_Drone] = []

        # Scoring (team-level truth; does not leak to drones)
        self.gt_edges: Set[Tuple[Tuple[int,int], Tuple[int,int]]] = set()
        self.identified_positions: Set[Tuple[int,int]] = set()

        self._edge_log_seen: Set[Tuple[Tuple[int,int], Tuple[int,int]]] = set()

        self._create_figures()
        self._rebuild_threat_map()
        self._compute_attack_defense_per_figure()
        self._compute_gt_edges()

        if self.figures:
            wk = next((f for f in self.figures if f.color == "white" and f.figure_type == "king"), self.figures[0])
            self.drone_base = wk.position
        else:
            self.drone_base = (0, 0)
        self._create_drones()

        if CONFIG["simulation"]["use_gui"]:
            self.gui = _SimulationGUI(self)

        self.executor = ThreadPoolExecutor(max_workers=1)
        self._current_future: Optional[Future] = None
        self._thinking = False

    # ---- Phase helpers
    def phase_label(self) -> str:
        return "Planning" if self.round <= self.planning_rounds else "Execution"

    # ---- Figures & edges (unchanged)
    def _create_figures(self):
        LOGGER.log("Creating figures based on configuration.")
        self.figures = []
        figures_cfg = CONFIG.get("figures", {})
        sim_cfg = CONFIG.get("simulation", {})
        W, H = self.grid_size

        rand_flag = bool(sim_cfg.get("randomize_figures", False))
        seed_val = sim_cfg.get("random_seed", None)
        should_randomize = rand_flag or (seed_val is not None)

        if should_randomize:
            try:
                seed = None
                if seed_val is not None:
                    s = str(seed_val).strip()
                    if s.lstrip("-").isdigit():
                        seed = int(s)
                randomized = _randomize_figures_layout(figures_cfg, W, H, seed)
                figures_cfg = randomized
                LOGGER.log(f"Figure positions RANDOMIZED (seed={seed if seed is not None else '<none>'}).")
                sample_lines = []
                for color in ("white", "black"):
                    for ftype in ("king", "queen", "rook", "bishop", "knight", "pawn"):
                        pos_list = randomized.get(color, {}).get(ftype, [])
                        if pos_list:
                            sample_lines.append(f"{color} {ftype}: {pos_list[:3]}{'...' if len(pos_list)>3 else ''}")
                if sample_lines:
                    LOGGER.log("Randomized sample: " + " | ".join(sample_lines))
            except Exception as e:
                LOGGER.log(f"Randomization failed ({e}); FALLING BACK to configured positions.")

        for color in COLORS:
            for figure_type in FIGURE_TYPES:
                for position in figures_cfg.get(color, {}).get(figure_type, []):
                    self.figures.append(_Figure(tuple(position), color, figure_type))

        for f in self.figures:
            self.board[f.position[0]][f.position[1]].set_figure(f)

    def _create_drones(self):
        LOGGER.log(f"Creating {self.num_drones} drones.")
        for i in range(self.num_drones):
            d = _Drone(id=i+1, position=self.drone_base, model=self.model, rules=self.rules, sim=self)
            self.drones.append(d)
        base = self.board[self.drone_base[0]][self.drone_base[1]]
        for d in self.drones:
            base.add_drone(d)

    def _rebuild_threat_map(self):
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                self.board[x][y].reset_targeted_by_amounts()
        for f in self.figures:
            f.calculate_figure_targets(self.board)
        for f in self.figures:
            for (tx, ty) in f.target_positions:
                self.board[tx][ty].add_targeted_by_amount(f.color, 1)

    def _compute_attack_defense_per_figure(self):
        for f in self.figures:
            tile = self.board[f.position[0]][f.position[1]]
            if f.color == "white":
                f.defended_by = tile.targeted_by["white"]
                f.attacked_by = tile.targeted_by["black"]
            else:
                f.defended_by = tile.targeted_by["black"]
                f.attacked_by = tile.targeted_by["white"]

    def _compute_gt_edges(self):
        self._rebuild_threat_map()
        self.gt_edges = _compute_edges_for(self.figures, self.board)
        LOGGER.log(f"GT Edges computed: {len(self.gt_edges)}")

    # ---- Identification & scoring (team-level, not exposed)
    def _update_identifications_from_drone_tile(self, drone: _Drone):
        tile = self.board[drone.position[0]][drone.position[1]]
        if tile.figure:
            self.identified_positions.add(tile.figure.position)

    def discovered_edges(self) -> Set[Tuple[Tuple[int,int], Tuple[int,int]]]:
        identified_figs = [f for f in self.figures if f.position in self.identified_positions]
        edges = _compute_edges_for(identified_figs, self.board)
        idpos = self.identified_positions
        return {(src, dst) for (src, dst) in edges if dst in idpos}

    def score_stats(self) -> Dict[str, Any]:
        disc = self.discovered_edges()
        gt = self.gt_edges
        correct = disc & gt
        false = disc - gt
        prec = (len(correct) / len(disc)) if disc else 0.0
        rec = (len(correct) / len(gt)) if gt else 0.0
        return {
            "identified_nodes": len(self.identified_positions),
            "discovered_edges": len(disc),
            "gt_edges": len(gt),
            "correct_edges": len(correct),
            "false_edges": len(false),
            "score": len(correct) - len(false),
            "precision": prec,
            "recall": rec
        }

    # ---- Plan parsing/queue (optional; only if drones themselves keep PLAN in memory)
    def _maybe_update_plan_from_text(self, drone_id: int, text: str):
        steps = parse_plan_from_text(text or "")
        if steps:
            self.plans[drone_id] = steps
            self.post_info(f"[Plan] Drone {drone_id} plan set: {steps}")

    def _next_planned_step(self, drone_id: int) -> Optional[str]:
        q = self.plans.get(drone_id, [])
        return q[0] if q else None

    def _advance_plan(self, drone_id: int):
        q = self.plans.get(drone_id, [])
        if q:
            q.pop(0)

    # ---- Edge logging
    def _log_edge_line(self, edge: Tuple[Tuple[int,int], Tuple[int,int]]):
        src, dst = edge
        is_correct = edge in self.gt_edges
        tag = "CORRECT" if is_correct else "FALSE"
        msg = f"Discovered edge: {src} -> {dst} [{tag}]"
        LOGGER.log(msg)
        self.post_info(msg)

    def _log_discovered_edges_incremental(self):
        current = self.discovered_edges()
        new_edges = current - self._edge_log_seen
        if not new_edges:
            return
        for e in sorted(new_edges):
            self._log_edge_line(e)
        self._edge_log_seen |= new_edges

    def _log_final_summary(self):
        disc = self.discovered_edges()
        correct = disc & self.gt_edges
        false = disc - self.gt_edges
        prec = (len(correct) / len(disc)) if disc else 0.0
        rec = (len(correct) / len(self.gt_edges)) if self.gt_edges else 0.0

        LOGGER.log("#" * 60)
        LOGGER.log("FINAL EDGE SUMMARY")
        LOGGER.log(f"Identified nodes: {len(self.identified_positions)}")
        LOGGER.log(f"GT edges:         {len(self.gt_edges)}")
        LOGGER.log(f"Discovered edges: {len(disc)}")
        LOGGER.log(f"  - Correct:      {len(correct)}")
        LOGGER.log(f"  - False:        {len(false)}")
        LOGGER.log(f"Score (correct - false): {len(correct) - len(false)}")
        LOGGER.log(f"Precision: {prec:.3f}  |  Recall: {rec:.3f}")
        if false:
            LOGGER.log(f"False edges list ({len(false)}):")
            for e in sorted(false):
                LOGGER.log(f"  {e[0]} -> {e[1]}")

        self.post_info("=== FINAL EDGE SUMMARY ===")
        self.post_info(f"Disc:{len(disc)}  Corr:{len(correct)}  False:{len(false)}  "
                       f"Score:{len(correct)-len(false)}  P:{prec:.2f} R:{rec:.2f}")

    # ---- GUI/log helper
    def post_info(self, msg: str):
        if hasattr(self, "gui"):
            self.gui.post_info(msg)

    def current_drone(self) -> Optional[_Drone]:
        return self.drones[self.turn - 1] if 1 <= self.turn <= len(self.drones) else None

    # ---- Turn orchestration (no coaching, no global nudges, no auto-retargets)
    def _start_drone_turn(self, drone: _Drone):
        self._thinking = True
        self._current_future = self.executor.submit(drone.generate_full_model_response)

    def _try_finish_drone_turn(self, drone: _Drone) -> bool:
        if self._current_future is None or not self._current_future.done():
            return False

        try:
            messages = self._current_future.result()
            result = messages[-1]["content"]
            LOGGER.log(f"Drone {drone.id} response:\n{pprint.pformat(result, indent=4, width=200)}")

            # Accept plan updates only from the drone's own memory/message
            self._maybe_update_plan_from_text(drone.id, result.get("memory", ""))
            self._maybe_update_plan_from_text(drone.id, result.get("message", ""))

            self.post_info(f"Drone {drone.id}:")
            self.post_info(f"Rationale: {result.get('rationale','')}")
            action = result.get("action", "wait")
            phase = self.phase_label()

            # Planning: enforce no movement, but do not inject hints
            if phase == "Planning" and action == "move":
                self.post_info("Planning phase: movement disabled. Waiting.")
                action = "wait"

            if action == "move":
                direction = (result.get("direction") or "").lower()
                allowed = drone._allowed_directions()
                if direction not in allowed:
                    self.post_info(f"Invalid/OOB direction '{direction}' (allowed={allowed}). Waiting.")
                else:
                    # Optional plan enforcement: if drone published a PLAN, we can require matching head
                    if phase == "Execution" and self.enforce_plan:
                        expected = self._next_planned_step(drone.id)
                        if expected and direction != expected:
                            self.post_info(f"Deviation from plan: expected '{expected}', got '{direction}'. Waiting.")
                        else:
                            if drone._move(direction):
                                self.post_info(f"Move {direction} to {drone.position}")
                                if expected == direction:
                                    self._advance_plan(drone.id)
                    else:
                        if drone._move(direction):
                            self.post_info(f"Move {direction} to {drone.position}")

            elif action == "broadcast":
                msg = (result.get("message") or "").strip()
                if not msg:
                    self.post_info("Invalid broadcast with empty message. Waiting.")
                else:
                    self.post_info("Broadcast")
                    self.post_info(msg)
                    # Only co-located drones receive (already enforced)
                    tile = self.board[drone.position[0]][drone.position[1]]
                    for d in tile.drones:
                        if d.id != drone.id:
                            d.rx_buffer += f"Drone {drone.id} broadcasted: {msg}\n"

            else:
                self.post_info("Wait")

            # Persist memory (do not clobber with empty)
            mem_txt = (result.get("memory") or "").strip()
            if mem_txt:
                drone.memory = mem_txt

            # Identification and incremental edge logging
            self._update_identifications_from_drone_tile(drone)
            self._log_discovered_edges_incremental()

            self.post_info("\n")
        except Exception as e:
            LOGGER.log(f"Error finishing Drone {drone.id}'s turn: {e}")

        self._thinking = False
        self._current_future = None
        return True

    # ---- Main loop
    def run_simulation(self):
        max_rounds = CONFIG["simulation"].get("max_rounds", 10)
        use_gui = CONFIG["simulation"].get("use_gui", True)

        running = True
        clock = pygame.time.Clock()

        if use_gui and hasattr(self, "gui"):
            pygame.display.set_caption(f"Simulation - Round {1}.1/{max_rounds}.{self.num_drones}")
            self.gui.draw_field()

        current_round = 1
        drone_index = 0
        pending = False

        try:
            while running:
                if use_gui and hasattr(self, "gui"):
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT: running = False
                        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running = False

                if not pending:
                    if current_round > max_rounds:
                        break
                    self.round = current_round
                    self.turn = drone_index + 1
                    caption = f"Simulation - Round {current_round}.{self.turn}/{max_rounds}.{self.num_drones}"
                    LOGGER.log('#'*50); LOGGER.log(caption + f" | Phase: {self.phase_label()}")
                    if use_gui and hasattr(self, "gui"):
                        pygame.display.set_caption(caption)

                    self._start_drone_turn(self.drones[drone_index])
                    pending = True

                if pending:
                    d = self.drones[drone_index]
                    if self._try_finish_drone_turn(d):
                        LOGGER.log(f"Round {self.round}.{self.turn} completed.")
                        drone_index += 1
                        if drone_index >= self.num_drones:
                            drone_index = 0
                            current_round += 1
                        pending = False

                if use_gui and hasattr(self, "gui"):
                    self.gui.draw_field()

                clock.tick(60)

        except KeyboardInterrupt:
            LOGGER.log("KeyboardInterrupt received — shutting down gracefully.")
            running = False
        finally:
            try:
                self._log_final_summary()
            except Exception:
                pass
            self.shutdown()

    def shutdown(self):
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
            if hasattr(self, "gui"):
                pygame.display.quit()
            pygame.quit()
        except Exception:
            pass
        LOGGER.log("Clean shutdown complete.")

# ---------------- Entry ----------------
if __name__ == "__main__":
    try:
        LOGGER.log("Launching simulation.")
        SIM = Simulation()
        SIM.run_simulation()
    except KeyboardInterrupt:
        LOGGER.log("Interrupted by user (Ctrl+C).")
        try: SIM.shutdown()
        except Exception: pass

