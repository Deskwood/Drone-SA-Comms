#!/usr/bin/env python
# coding: utf-8

# In[13]:


#!/usr/bin/env python
# coding: utf-8

# =========================
# Imports
# =========================
import os, re, json, time, random, pprint, logging, colorsys, math, nbformat, pygame, pyperclip
from typing import Tuple, List, Optional, Literal, Dict, Any, Set, TypedDict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
from nbconvert.exporters import PythonExporter
from ollama import chat as ollama_chat
from pydantic import BaseModel, field_validator, model_validator


# In[14]:


# Logger
# =========================
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


# In[15]:


# --- Auto-export to .py when running inside a Jupyter notebook ---
def _running_in_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False

def _find_notebook_path() -> Optional[str]:
    try:
        import ipynbname
        return str(ipynbname.path())
    except Exception:
        pass
    try:
        candidates = [nb_path for nb_path in os.listdir(".") if nb_path.endswith(".ipynb")]
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]
    except Exception:
        pass
    return None

def _export_notebook_to_py(nb_path: str, out_py: str = "run_simulation.py") -> None:
    try:
        nb = nbformat.read(nb_path, as_version=4)
        body, _ = PythonExporter().from_notebook_node(nb)
        with open(out_py, "w", encoding="utf-8") as f:
            f.write(body)
        LOGGER.log(f"ERROR: Exported notebook '{nb_path}' -> '{out_py}'")
    except Exception as e:
        LOGGER.log(f"ERROR: Notebook export failed: {e}")

try:
    if _running_in_notebook():
        nbp = _find_notebook_path()
        if nbp:
            _export_notebook_to_py(nbp, out_py="run_simulation.py")
except Exception:
    pass


# In[16]:


# Constants / Config
# =========================
COLORS = ["white", "black"]
FIGURE_TYPES = ["king", "queen", "rook", "bishop", "knight", "pawn"]
DIRECTION_MAP: Dict[str, Tuple[int, int]] = {
    "north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0),
    "northeast": (1, 1), "northwest": (-1, 1), "southeast": (1, -1), "southwest": (-1, -1)
}
VALID_DIRECTIONS = set(DIRECTION_MAP.keys())

def direction_from_vector(vector: Tuple[int, int]) -> str:
    for direction, vec in DIRECTION_MAP.items():
        if vec == vector:
            return direction
    return str(vector)

def hsv_to_rgb255(h_deg: float, s: float, v: float) -> Tuple[int,int,int]:
    """h in degrees [0,360), s,v in [0,1] -> (r,g,b) in 0..255"""
    r, g, b = colorsys.hsv_to_rgb(h_deg/360.0, max(0,min(1,s)), max(0,min(1,v)))
    return (int(r*255), int(g*255), int(b*255))

def load_config(config_path: str = "config.json") -> dict:
    LOGGER.log(f"Load Config: {config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing config file: {config_path}")

    cfg.setdefault("prompt_requests", {})
    pr = cfg["prompt_requests"]
    pr.setdefault("schema",
        ("OUTPUT FORMAT: Return a SINGLE JSON object with keys "
         "rationale, action, direction, message, memory.")
    )

    cfg.setdefault("simulation", {})
    cfg.setdefault("board", {"width": 8, "height": 8})
    cfg.setdefault("gui", {
        "use_gui": True, "cell_size": 64, "margin": 2,
        "background_color": (30,30,30), "grid_color": (80,80,80),
        "drone_color": (200,200,50), "text_color": (20,20,20),
        "sidebar_width": 480
    })
    sim = cfg["simulation"]
    sim.setdefault("max_rounds", 32)
    sim.setdefault("num_drones", 2)
    sim.setdefault("models", ["manual"])
    sim.setdefault("model_index", 0)
    sim.setdefault("temperature", 0.7)
    sim.setdefault("use_gui", True)
    sim.setdefault("headless", False)
    sim.setdefault("rules_path", cfg.get("rules_path", "rules.txt"))
    sim.setdefault("max_tokens_for_rationale", 1024)
    sim.setdefault("max_tokens_for_action", 512)
    sim.setdefault("max_tokens_for_action_move", 512)
    sim.setdefault("max_tokens_for_action_broadcast", 512)
    sim.setdefault("max_tokens_for_memory", 512)
    sim.setdefault("max_tokens_total_cap", 4096)
    sim.setdefault("planning_rounds", 2)
    sim.setdefault("randomize_figures", True)
    sim.setdefault("random_seed", 456)

    # Decision Support defaults
    cfg.setdefault("decision_support", {})
    ds = cfg["decision_support"]
    ds.setdefault("enabled", True)
    ds.setdefault("max_depth", 2)
    ds.setdefault("max_branching", 8)
    ds.setdefault("beam_width", 8)
    ds.setdefault("timeout_ms", 100)
    ds.setdefault("weights", {
        "recall": 0.6, "exploration": 0.15, "plan_adherence": 0.15,
        "move_validity": 0.05, "comm_opportunity": 0.03, "precision": 0.02
    })
    ds.setdefault("prefer_top_recommendation", True)
    ds.setdefault("include_in_prompt", True)
    ds.setdefault("deterministic", True)

    cfg.setdefault("figures", {c: {t: [] for t in FIGURE_TYPES} for c in COLORS})
    return cfg

CONFIG = load_config("config.json")


# In[17]:


# Figure Images
# =========================
FIGURE_IMAGES: Dict[Tuple[str,str], "pygame.Surface"] = {}

def load_figure_images() -> dict:
    images = {}
    base_path = CONFIG["gui"].get("figure_image_dir", "figures")

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
                    break
            if img:
                images[(color, figure_type)] = img
            else:
                LOGGER.log(f"ERROR: Image not found for {color} {figure_type} in {base_path}")
    return images


# In[18]:


# Domain: Figures/Tiles
# =========================
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
                directions = [(1,0),(-1,0),(0,1),(0,-1)]
            elif self.figure_type == "bishop":
                directions = [(1,1),(-1,-1),(1,-1),(-1,1)]
            else:
                directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]
            for dx, dy in directions:
                x, y = self.position
                while True:
                    x += dx; y += dy
                    if not on_board(x, y): break
                    self.target_positions.append((x, y))
                    if board[x][y].figure is not None:
                        break

        elif self.figure_type == "knight":
            for dx, dy in [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]:
                x = self.position[0] + dx
                y = self.position[1] + dy
                if on_board(x, y): self.target_positions.append((x, y))

        elif self.figure_type == "king":
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]:
                x = self.position[0] + dx
                y = self.position[1] + dy
                if on_board(x, y): self.target_positions.append((x, y))

        elif self.figure_type == "pawn":
            diagonals = [(1,1),(-1,1)] if self.color == "white" else [(1,-1),(-1,-1)]
            for dx, dy in diagonals:
                x = self.position[0] + dx
                y = self.position[1] + dy
                if on_board(x, y): self.target_positions.append((x, y))

class _Tile:
    def __init__(self, x: int, y: int):
        self.x = x; self.y = y
        self.targeted_by = {"white": 0, "black": 0}
        self.figure: Optional[_Figure] = None
        self.drones: List['_Drone'] = []

    def set_figure(self, figure: _Figure): self.figure = figure
    def add_drone(self, drone: '_Drone'):
        if drone not in self.drones: self.drones.append(drone)
    def remove_drone(self, drone: '_Drone'):
        if drone in self.drones: self.drones.remove(drone)
    def reset_targeted_by_amounts(self): self.targeted_by = {"white": 0, "black": 0}
    def add_targeted_by_amount(self, color: str, amount: int = 1): self.targeted_by[color] += amount



# In[19]:


# Chess helpers
# =========================
def cartesian_to_chess(pos: Tuple[int,int]) -> str:
    x, y = pos
    col = chr(ord('a') + x)
    row = str(y + 1)
    return f"{col}{row}"

def chess_to_cartesian(s: str) -> Tuple[int,int]:
    s = s.strip().lower()
    if len(s) < 2: raise ValueError(f"Invalid chess pos: {s}")
    col, row = s[0], s[1:]
    x = ord(col) - ord('a')
    y = int(row) - 1
    return (x, y)

def format_edge(source_type: str, source_color: str, target_color: str, edge: Tuple[Tuple[int,int], Tuple[int,int]]) -> str:
    src, dst = edge
    piece_symbol = {
        "king": "K", "queen": "Q", "rook": "R",
        "bishop": "B", "knight": "N", "pawn": ""
    }.get(source_type, "?")
    capture_symbol = "x" if source_color != target_color else "-"
    return f"{piece_symbol}{cartesian_to_chess(src)}{capture_symbol}{cartesian_to_chess(dst)}"

def on_board(x, y):
    return 0 <= x < CONFIG["board"]["width"] and 0 <= y < CONFIG["board"]["height"]


# In[20]:


# Decision Support
# =========================

# //TODO: Implement Decision Support logic here


# In[21]:


# Drone
# =========================
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
        self.memory = ""
        self.rx_buffer = ""
        self.mission_report: List = [self.position]
        self.mission_plan: List[dict] = [] # //TODO: Fill mission plan
        self.local_board = {}
        for bx in range(CONFIG["board"]["width"]):
            for by in range(CONFIG["board"]["height"]):
                self.local_board[cartesian_to_chess((bx, by))] = {"color": "unknown", "type": "unknown"}
        # color may be "unknown", "n/a", "white", "black"
        # type may be "unknown", "n/a", "any figure", "king", "queen", "rook", "bishop", "knight", "pawn", "a possible target"
        # possible combinations: (unknown,unknown), (n/a,n/a), (any figure,white/black), (king/queen/rook/bishop/knight/pawn,white/black), (a possible target,unknown)
        self.identified_edges = []
        self.update_board_report()

    def _identify_edges(self):
        # Identify edges according to current local board knowledge
        directions = []
        for bx in range(CONFIG["board"]["width"]):
            for by in range(CONFIG["board"]["height"]):
                # Identify figure
                figure_type = self.local_board[cartesian_to_chess((bx,by))]["type"]
                figure_color = self.local_board[cartesian_to_chess((bx,by))]["color"]
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
                        if not on_board(tx, ty): break
                        if self.local_board[cartesian_to_chess((tx,ty))]["type"] == "unknown": # a possible target (prioritize investigation)
                            self.local_board[cartesian_to_chess((tx,ty))]["type"] = "a possible target"
                        if self.local_board[cartesian_to_chess((tx,ty))]["type"] == "a possible target": # Moving beyond this is pointless for sliders
                            break
                        if self.local_board[cartesian_to_chess((tx,ty))]["type"] in FIGURE_TYPES \
                            or self.local_board[cartesian_to_chess((tx,ty))]["type"] == "any figure": # Found an edge
                            edge = format_edge(
                                figure_type,
                                self.local_board[cartesian_to_chess((bx,by))]["color"],
                                self.local_board[cartesian_to_chess((tx,ty))]["color"],
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
        print(f"Context length: {len(user_content)+len(self.rules)} chars")
        return self._generate_single_model_response(messages=messages, model=self.model, temperature=temperature)

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
                LOGGER.log(f"Drone {self.id} moved to {cartesian_to_chess(self.position)}.")
                return True
            else:
                LOGGER.log(f"ERROR: Drone {self.id} attempted to move to {(nx,ny)}.")
        else:
            LOGGER.log(f"ERROR: Drone {self.id} attempted invalid direction '{direction}'.")
        return False

    def _determine_situation_description(self) -> str:
        # Only local, directly observable info + own memory + received broadcasts + gathered intelligence
        same_tile_drones = [
            f"Drone {drone.id}" for drone in self.sim.board[self.position[0]][self.position[1]].drones if drone.id != self.id
        ]

        fig_here = "None"
        if self.sim.board[self.position[0]][self.position[1]].figure:
            fig_here = self.sim.board[self.position[0]][self.position[1]].figure.figure_type

        # neighbor figure colors (visibility model)
        neighbor_figures = ""
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if dx==0 and dy==0:
                    continue
                nx, ny = self.position[0]+dx, self.position[1]+dy
                if on_board(nx, ny):
                    tile = self.sim.board[nx][ny]
                    if tile.figure:
                        neighbor_figures += f"{direction_from_vector((dx,dy))}: {tile.figure.color}, "
        neighbor_figures = neighbor_figures.strip(", ")

        legal_movements = ", ".join([f"{lms['direction']} to {cartesian_to_chess(lms['new_position'])}" for lms in self._legal_movement_steps()])

        collected_figure_information = ""
        for bx in range(CONFIG["board"]["width"]):
            for by in range(CONFIG["board"]["height"]):
                info = self.local_board[cartesian_to_chess((bx,by))]
                if info["type"] in FIGURE_TYPES:
                    collected_figure_information += f"{cartesian_to_chess((bx,by))}: {info['color']} {info['type']}, "
                elif info["type"] == "a possible target":
                    collected_figure_information += f"{cartesian_to_chess((bx,by))}: {info['type']}, "
        collected_figure_information = collected_figure_information.strip(", ")
        LOGGER.log(collected_figure_information)

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
        s.append(f"Broadcast Rx Buffer: {self.rx_buffer}")
        self.rx_buffer = ""  # drain the inbox each turn
        return "\n".join(s)

    def _generate_single_model_response(self, messages: List[dict], model: str, temperature: float) -> List[dict]:
        if model == "manual":
            try:
                pyperclip.copy(messages[-1]["content"])
            except Exception:
                pass
            content = input("Paste model result: ")
            messages.append({"role": "assistant", "content": content})
            return messages

        else: # Ollama
            response = ollama_chat(model=model, messages=messages, stream=False, format="json",
                               options={"temperature": float(temperature)})
            content = response["message"]["content"]
            messages.append({"role": "assistant", "content": content})
            return messages


# In[ ]:


# GUI

class _SimulationGUI:
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
        pygame.display.set_caption(
            f'Simulation - Round {1}.1/{CONFIG["simulation"]["max_rounds"]}.{CONFIG["simulation"]["num_drones"]}'
        )
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
    def _get_scaled(self, img):
        key = (id(img), CONFIG["gui"]["cell_size"])
        if key not in self._image_cache:
            self._image_cache[key] = pygame.transform.scale(
                img, (CONFIG["gui"]["cell_size"], CONFIG["gui"]["cell_size"])
            )
        return self._image_cache[key]

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

        y += 6
        s.blit(self._font.render("Plans", True, (220, 220, 220)), (x0 + pad, y)); y += 20
        for drone in self.sim.drones:
            # draw a small color swatch
            sw = pygame.Rect(x0 + pad, y+2, 10, 10)
            pygame.draw.rect(self.screen, getattr(drone, "render_color", (200,200,50)), sw)
            s.blit(self._font_small.render(f"  Drone {drone.id}", True, (200,200,200)), (x0 + pad + 14, y))
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

        # 2) draw drone paths
        def _path_offset_vec(drone) -> Tuple[int, int]:
            n = max(1, len(self.sim.drones))
            k = (drone.id - 1) % n
            ang = math.radians(45.0 + 360.0 * k / n)  # diagonal base, evenly spaced by ID
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

        # 3) draw drones on top of paths
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

        # 4) highlight current drone while thinking
        if getattr(self.sim, "_thinking", False):
            current_drone = self.sim.drones[self.sim.turn - 1] # 0-based turn index
            if current_drone:
                cx, cy = current_drone.position
                y_flip = gh - 1 - cy
                hi_rect = pygame.Rect(cx * (cell + m) + m, y_flip * (cell + m) + m, cell, cell)
                pygame.draw.rect(self.screen, (255, 215, 0), hi_rect, 2)

        # 5) sidebar and flip
        self._draw_sidebar()
        pygame.display.flip()


# In[23]:


# Simulation
class Simulation:
    def __init__(self):
        if CONFIG["simulation"].get("headless", False):
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        self.gui = None
        if CONFIG["simulation"].get("use_gui", True):
            self.gui = _SimulationGUI(self)

        self.turn = 1
        self.round = 1
        rules_path = CONFIG.get("rules_path", CONFIG.get("rules_path", "rules.txt"))
        with open(rules_path, "r", encoding="utf-8") as f:
            self.rules = f.read().replace("NUMBER_OF_DRONES", str(CONFIG["simulation"]["num_drones"]))

        self.grid_size = (CONFIG["board"]["width"], CONFIG["board"]["height"])
        self.max_rounds = CONFIG["simulation"]["max_rounds"]
        self.num_drones = CONFIG["simulation"]["num_drones"]
        self.models = CONFIG["simulation"]["models"]
        self.model_index = CONFIG["simulation"]["model_index"]
        self.model = self.models[self.model_index]
        LOGGER.log(f"Using model: {self.model}")

        self.planning_rounds = int(CONFIG["simulation"].get("planning_rounds", 2))
        self.plans: Dict[int, List[str]] = {}

        self.board = [[_Tile(x, y) for y in range(self.grid_size[1])] for x in range(self.grid_size[0])]
        self.figures: List[_Figure] = []
        self.drones: List[_Drone] = []

        self.gt_edges = []
        self.reported_edges = []
        self.correct_edge_counter = 0
        self.false_edge_counter = 0
        self.score = 0.0

        self._create_figures()
        if self.figures:
            white_king = next((figure for figure in self.figures if figure.color == "white" and figure.figure_type == "king"), self.figures[0])
            self.drone_base = white_king.position
        else:
            self.drone_base = (0, 0)

        self._create_drones()

        self.executor = ThreadPoolExecutor(max_workers=1)
        self._current_future: Optional[Future] = None
        self._thinking = False

    # Figures
    def _create_figures(self):
        LOGGER.log("Creating figures based on configuration.")
        self.figures = []
        figures_cfg = CONFIG.get("figures", {})
        sim_cfg = CONFIG.get("simulation", {})
        W, H = self.grid_size
        randomize = bool(sim_cfg.get("randomize_figures", False) or sim_cfg.get("random_seed", None) is not None)
        rng = random.Random(sim_cfg.get("random_seed", None))
        if randomize:
            try:
                all_tiles = [(x,y) for x in range(W) for y in range(H)]
                rng.shuffle(all_tiles)
                cursor = 0
                req = []
                for color in COLORS:
                    for ftype in FIGURE_TYPES:
                        lst = figures_cfg.get(color, {}).get(ftype, [])
                        cnt = len(lst) if lst else (1 if ftype in ("king", "queen") else (2 if ftype in ("rook","bishop","knight") else 3))
                        req.append((color, ftype, cnt))
                out = {c: {t: [] for t in FIGURE_TYPES} for c in COLORS}
                for color, ftype, cnt in req:
                    picks = all_tiles[cursor:cursor+cnt]; cursor += cnt
                    out[color][ftype] = [list(p) for p in picks]
                figures_cfg = out
                LOGGER.log(f"Figure positions RANDOMIZED (seed={sim_cfg.get('random_seed', None)}).")
            except Exception as e:
                LOGGER.log(f"Randomization failed ({e}); using configured positions.")
        for color in COLORS:
            for figure_type in FIGURE_TYPES:
                for position in figures_cfg.get(color, {}).get(figure_type, []):
                    self.figures.append(_Figure(tuple(position), color, figure_type))
        for f in self.figures:
            self.board[f.position[0]][f.position[1]].set_figure(f)

        # Calculate targeting map
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                self.board[x][y].reset_targeted_by_amounts()
        for figure in self.figures:
            figure.calculate_figure_targets(self.board)
        for figure in self.figures:
            for (tx, ty) in figure.target_positions:
                self.board[tx][ty].add_targeted_by_amount(figure.color, 1)

        # Compute defended_by / attacked_by counts
        for figure in self.figures:
            tile = self.board[figure.position[0]][figure.position[1]]
            if figure.color == "white":
                figure.defended_by = tile.targeted_by["white"]; figure.attacked_by = tile.targeted_by["black"]
            else:
                figure.defended_by = tile.targeted_by["black"]; figure.attacked_by = tile.targeted_by["white"]

        # Compute GT Edges
        self.gt_edges = []
        for figure in self.figures:
            for (x_tgt, y_tgt) in figure.target_positions:
                target_figure = self.board[x_tgt][y_tgt].figure
                if target_figure is not None:
                    edge = format_edge(figure.figure_type, figure.color, target_figure.color, [figure.position, target_figure.position])
                    self.gt_edges.append(edge)
        LOGGER.log(f"GT Edges computed: {len(self.gt_edges)}")

    # Drones
    def _create_drones(self):
        LOGGER.log(f"Creating {self.num_drones} drones.")
        for i in range(self.num_drones):
            d = _Drone(id=i+1, position=self.drone_base, model=self.model, rules=self.rules, sim=self)
            # unique color by rotating hue
            hue_deg = (i / max(1, self.num_drones)) * 360.0
            d.render_color = hsv_to_rgb255(hue_deg, 0.85, 0.95)  # vivid, bright
            vx, vy = d.position
            d.memory = f"VISITED:{vx},{vy}"
            self.drones.append(d)
        base = self.board[self.drone_base[0]][self.drone_base[1]]
        for d in self.drones:
            base.add_drone(d)

    # Edges/score
    def report_edges(self, edges: List[str]):
        LOGGER.log(f"Reported edges: {edges}")
        for edge in edges:
            if edge not in self.reported_edges:
                self.reported_edges.append(edge)
        correct_edge_counter = 0; false_edge_counter = 0
        for edge in self.reported_edges:
            if edge in self.gt_edges:
                correct_edge_counter += 1
            else:
                false_edge_counter += 1
        self.correct_edge_counter = correct_edge_counter
        self.false_edge_counter = false_edge_counter
        self.score = correct_edge_counter - false_edge_counter

    def _log_final_summary(self):
        LOGGER.log("#"*60)
        LOGGER.log("FINAL EDGE SUMMARY")
        LOGGER.log(f"Identified nodes: {len({n for e in self.reported_edges for n in e})}")
        LOGGER.log(f"GT edges:         {len(self.gt_edges)}")
        LOGGER.log(f"Reported edges:   {len(self.reported_edges)}")
        LOGGER.log(f"  - Correct:      {self.correct_edge_counter}")
        LOGGER.log(f"  - False:        {self.false_edge_counter}")
        LOGGER.log(f"Score:            {self.score}")

        LOGGER.log("\nEdge summary:")
        correct_edges = []; false_edges = []
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
        self.post_info(f"Disc:{len(self.reported_edges)}  Corr:{self.correct_edge_counter}  False:{self.false_edge_counter}  Score:{self.score}")

    # GUI/log helper
    def post_info(self, msg: str):
        if hasattr(self, "gui") and self.gui:
            self.gui.post_info(msg)

    # Turn orchestration
    def _try_finish_drone_turn(self, drone: _Drone) -> bool:
        if self._current_future is None or not self._current_future.done():
            return False

        try:
            messages = self._current_future.result()
            result = json.loads(messages[-1]["content"])
            LOGGER.log(f"Drone {drone.id} response:\n{pprint.pformat(result, indent=4, width=200)}")

            self.post_info(f"Drone {drone.id}:")
            self.post_info(f"Rationale: {result.get('rationale','')}")
            action = result.get("action", "wait")

            if action == "move":
                direction = (result.get("direction") or "").lower()
                legal_directions = [lms["direction"] for lms in drone._legal_movement_steps()]
                if direction not in legal_directions:
                    self.post_info(f"Invalid direction: '{direction}', allowed={legal_directions} -> Aborted movement")
                elif drone._move(direction):
                    self.post_info(f"Move {direction} to {cartesian_to_chess(drone.position)}")

            elif action == "broadcast":
                msg = (result.get("message") or "").strip()
                if not msg:
                    self.post_info("ERROR: Empty broadcast.")
                else:
                    self.post_info("Broadcast")
                    self.post_info(msg)
                    tile = self.board[drone.position[0]][drone.position[1]]
                    for target_drone in tile.drones:
                        if target_drone.id != drone.id:
                            target_drone.rx_buffer += f"Drone {drone.id} broadcasted: {msg}\n"
                            drone._provide_intelligence_to(target_drone)
            else:
                self.post_info("Wait")

            # persist memory + mark visited
            mem_txt = (result.get("memory") or "").strip()
            if mem_txt: drone.memory = mem_txt
            vx, vy = drone.position
            token = f"VISITED:{vx},{vy}"
            if token not in drone.memory:
                drone.memory += ("" if drone.memory.endswith("\n") else "\n") + token

            # edge logging
            self.post_info("\n")
        except Exception as e:
            LOGGER.log(f"Error finishing Drone {drone.id}'s turn: {e}")

        self._thinking = False
        self._current_future = None
        return True

    # Main loop
    def run_simulation(self):
        max_rounds = CONFIG["simulation"].get("max_rounds", 10)
        use_gui = CONFIG["simulation"].get("use_gui", True) and hasattr(self, "gui")

        running = True
        clock = None
        if use_gui:
            clock = pygame.time.Clock()
            pygame.display.set_caption(f"Simulation - Round {1}.1/{max_rounds}.{self.num_drones}")
            self.gui.draw_field()

        current_round = 1
        drone_index = 0
        pending = False

        try:
            while running:
                if use_gui:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT: running = False
                        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running = False

                current_drone = self.drones[drone_index]
                if not pending:
                    if current_round > max_rounds: break
                    self.round = current_round
                    self.turn = drone_index + 1
                    caption = f"Simulation - Round {current_round}.{self.turn}/{max_rounds}.{self.num_drones}"
                    LOGGER.log('#'*50); LOGGER.log(caption)
                    if use_gui:
                        pygame.display.set_caption(caption)
                    self._thinking = True
                    self._current_future = self.executor.submit(current_drone.generate_full_model_response)
                    pending = True

                if pending:
                    if self._try_finish_drone_turn(current_drone):
                        drone_index += 1
                        if drone_index >= self.num_drones:
                            drone_index = 0
                            current_round += 1
                        pending = False

                if use_gui:
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
            if self.gui:
                # draw one last frame to ensure latest state is shown
                self.gui.draw_field()
                self.gui.save_screenshot()  # default path in ./screenshots
        except Exception as ex:
            LOGGER.log(f"Error during final GUI draw: {ex}")
        pygame.display.quit()
        pygame.quit()
        LOGGER.log("Clean shutdown complete.")


# In[24]:


# Main
# =========================
if __name__ == "__main__":
    try:
        LOGGER.log("Launching simulation.")
        SIM = Simulation()
        SIM.run_simulation()
    except KeyboardInterrupt:
        LOGGER.log("Interrupted by user (Ctrl+C).")
        try: SIM.shutdown()
        except Exception: pass

