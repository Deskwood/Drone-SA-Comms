#!/usr/bin/env python
# coding: utf-8

# In[29]:


# Cell 1: Setup environment, imports, and global constants

import logging
import os
import pprint
import time
from datetime import datetime
from typing import Tuple, List, Optional
import json
import pygame
import requests
import pyperclip
import re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, Future
import nbformat
from nbconvert import PythonExporter

# --- Constants ------------------------------------------------
COLORS = ["white", "black"]
FIGURE_TYPES = ["king", "queen", "rook", "bishop", "knight", "pawn"]
DIRECTION_MAP = {
    "north": (0, 1),
    "south": (0, -1),
    "east": (1, 0),
    "west": (-1, 0),
    "northeast": (1, 1),
    "northwest": (-1, 1),
    "southeast": (1, -1),
    "southwest": (-1, -1)
}

# --- Convert Jupyter Notebook to Python Script ----------------
try:
    nb = nbformat.read("run_simulation.ipynb", as_version=4)
    body, _ = PythonExporter().from_notebook_node(nb)
    with open("run_simulation.py", "w", encoding="utf-8") as f:
        f.write(body)
except Exception as e:
    pass  # Ignore because this should only run in Jupyter


# In[30]:


# Cell 2: Logging and config

class TimestampedLogger:
    def __init__(self, log_dir='logs', log_file='simulation.log'):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_file)
        logging.basicConfig(filename=self.log_path, level=logging.INFO, filemode='w', encoding='utf-8')
        self.start_time = time.time()
        self.last_time = self.start_time
        self.log("Logger initialized.")

    def _now(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _duration(self):
        current_time = time.time()
        duration = current_time - self.last_time
        self.last_time = current_time
        return f"{duration:.3f}s"

    def log(self, message):
        timestamp = self._now()
        duration = self._duration()
        log_message = f"[{timestamp}] (+{duration}) {message}"
        logging.info(log_message)

LOGGER = TimestampedLogger()


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration with safe defaults for missing keys."""
    LOGGER.log(f"Load Config: {config_path}")
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing config file: {config_path}")

    # minimal defaults
    cfg.setdefault("prompt_requests", {})
    cfg.setdefault("simulation", {})
    cfg.setdefault("board", {"width": 8, "height": 8})
    cfg.setdefault("gui", {"cell_size": 64, "margin": 2, "background_color": (30,30,30),
                           "grid_color": (80,80,80), "drone_color": (200,200,50), "text_color": (20,20,20),
                           "figure_image_dir": "figures"})
    return cfg

CONFIG = load_config("config.json")


# In[31]:


# Cell 3: Image loading (with robust filename casings)

def load_figure_images() -> dict:
    """
    Loads figure images from disk and returns {(color, type): Surface}.
    We don't convert here because display may not be initialized yet.
    We'll convert surfaces after creating the display for maximal blit speed.
    """
    images = {}
    base_path = CONFIG["gui"]["figure_image_dir"]

    def try_load(path):
        return pygame.image.load(path) if os.path.exists(path) else None

    for color in COLORS:
        for figure_type in FIGURE_TYPES:
            candidates = [
                f"{color}{figure_type}.png",                          # whiteking.png
                f"{color.capitalize()}{figure_type}.png",             # Whiteking.png
                f"{color}{figure_type.capitalize()}.png",             # whiteKing.png
                f"{color.capitalize()}{figure_type.capitalize()}.png" # WhiteKing.png
            ]
            img = None
            for name in candidates:
                p = os.path.join(base_path, name)
                img = try_load(p)
                if img: break
            if img:
                images[(color, figure_type)] = img
            else:
                LOGGER.log(f"Warning: Image not found for {color} {figure_type} in {base_path}")
    return images

FIGURE_IMAGES = load_figure_images()


def direction_from_vector(vector: Tuple[int, int]) -> str:
    """Return direction string from vector; otherwise return the tuple as str."""
    for direction, vec in DIRECTION_MAP.items():
        if vec == vector:
            return direction
    return str(vector)


# In[32]:


# Cell 4: Board Entities: Figure, Drone, Tile

class _Figure:
    """Represents a chess figure on the board and its threat counters."""
    def __init__(self, position: Tuple[int, int], color: str, figure_type: str):
        self.position = position
        self.color = color
        self.figure_type = figure_type
        self.defended_by = 0   # friendly pieces targeting this piece's square
        self.attacked_by = 0   # hostile pieces targeting this piece's square
        self.target_positions = []  # squares this piece attacks/defends

    def calculate_figure_targets(self, board: List[List['_Tile']]):
        """
        Populate self.target_positions with squares this figure attacks/defends.

        Sliders (Q/R/B) ray-cast and stop after the first occupied square (included).
        Knight/King/Pawn are discrete targets only (no ray-cast).
        """
        self.target_positions = []
        W, H = CONFIG["board"]["width"], CONFIG["board"]["height"]

        def on_board(x, y):
            return 0 <= x < W and 0 <= y < H

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
                        # include the first hit, then stop this ray
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

class _Drone:
    """Represents a drone (LLM-controlled agent) in the simulation."""
    def __init__(self, id: int, position: Tuple[int, int], model: str, rules: str, sim, color: str = "white"):
        self.id = id
        self.position = position
        self.color = color
        self.model = model
        self.rules = rules\
            .replace("DRONE_ID", str(self.id))\
            .replace("NUMBER_OF_DRONES", str(CONFIG["simulation"]["num_drones"]))\
            .replace("NUMBER_OF_ROUNDS", str(CONFIG["simulation"]["max_rounds"]))
        self.memory = ""   # Memory of past actions or observations
        self.rx_buffer = ""  # Buffer for received broadcasts
        self.sim = sim

    # ---------------- Movement ----------------
    def _move(self, direction: str):
        """Moves the drone in the specified direction if in bounds."""
        if direction in DIRECTION_MAP:
            dx, dy = DIRECTION_MAP[direction]
            new_position = (self.position[0] + dx, self.position[1] + dy)
            if 0 <= new_position[0] < CONFIG["board"]["width"] and 0 <= new_position[1] < CONFIG["board"]["height"]:
                # Remove drone from old tile
                old_tile = self.sim.board[self.position[0]][self.position[1]]
                old_tile.remove_drone(self)
                # Move
                self.position = new_position
                # Add to new tile
                new_tile = self.sim.board[self.position[0]][self.position[1]]
                new_tile.add_drone(self)
                LOGGER.log(f"Drone {self.id} moved to {self.position}.")
            else:
                LOGGER.log(f"Drone {self.id} attempted to move out of bounds to {new_position}. Action ignored.")
        else:
            LOGGER.log(f"Drone {self.id} attempted invalid direction '{direction}'. Action ignored.")

    # ---------------- Situation description ----------------
    def _determine_situation_description(self) -> str:
        """Creates a compact state summary for prompting the LLM."""
        # Identify co-located drones
        other_drones_at_position = ""
        for drone in self.sim.board[self.position[0]][self.position[1]].drones:
            if drone.id != self.id:
                other_drones_at_position += f"Drone {drone.id}, "
        other_drones_at_position = other_drones_at_position.strip(", ")

        # Identify co-located figure
        figure_at_position = "None"
        if self.sim.board[self.position[0]][self.position[1]].figure:
            f0 = self.sim.board[self.position[0]][self.position[1]].figure
            figure_at_position = f0.figure_type

        # Identify neighboring figures with direction and type
        neighboring_figure_colors = ""
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx = self.position[0] + dx
                ny = self.position[1] + dy
                if 0 <= nx < CONFIG["board"]["width"] and 0 <= ny < CONFIG["board"]["height"]:
                    neighbor_tile = self.sim.board[nx][ny]
                    if neighbor_tile.figure:
                        f = neighbor_tile.figure
                        neighboring_figure_colors += f"{direction_from_vector((dx, dy))}: {f.color}, "
        neighboring_figure_colors = neighboring_figure_colors.strip(", ")

        # Generate a rationale
        situation = f"Current round number: {self.sim.round}\n"
        situation += f"Current position: {self.position}\n"
        situation += f"Visible drones at position: {other_drones_at_position}\n"
        situation += f"Visible figure at position: {figure_at_position}\n"
        situation += f"Visible neighboring figures: {neighboring_figure_colors}\n"
        situation += f"Memory: {self.memory}\n"
        situation += f"Broadcast Rx Buffer: {self.rx_buffer}"
        self.rx_buffer = ""  # Clear the buffer after reading
        return situation

    # ---------------- LLM interface ----------------
    def _generate_single_model_response(self, messages:List, model: str = "qwen3:30b", temperature: float = 0.7, remove_thinking: bool = True) -> List[dict]:
        """Generate a response from the selected model and append to messages."""
        if model == "manual":  # Manual input mode
            pyperclip.copy(messages[-1]["content"])
            llm_response = input("Model prompt is on clipboard, paste model's response: ")
        elif "gpt" in model:   # OpenAI GPT models
            client = OpenAI()
            llm_response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            ).choices[0].message.content
        else:  # Ollama (local)
            url = "http://localhost:11434/api/chat"
            payload = {"model": model, "messages": messages, "temperature": temperature, "stream": False}
            try:
                response = requests.post(url, json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()
                llm_response = data.get("message", {}).get("content", "")
            except Exception as e:
                llm_response = f"ERROR contacting local model '{model}': {e}"
            if remove_thinking:
                # Remove think sections for models that emit them
                llm_response = re.sub(r"<think>.*?</think>", "", llm_response, flags=re.DOTALL|re.IGNORECASE)
                llm_response = re.sub(r"\[thinking\].*?\[/thinking\]", "", llm_response, flags=re.DOTALL|re.IGNORECASE)
                llm_response = llm_response.strip()

        messages.append({"role": "assistant", "content": llm_response})
        return messages

    def _generate_full_model_response(self) -> dict:
        """Blocking call that drives a full turn decision (rationale -> action -> specifics -> memory)."""
        temperature = CONFIG["simulation"].get("temperature", 0.7)

        # 1) Rationale
        situation = self._determine_situation_description()
        messages = [
            {"role": "system", "content": self.rules},
            {"role": "user", "content": situation},
            {"role": "assistant", "content": "Situation clear, what do you want me to do?"},
            {"role": "user", "content": CONFIG["prompt_requests"]["rationale"]}
        ]
        messages = self._generate_single_model_response(messages=messages, model=self.model, temperature=temperature)

        # 2) Action (move, wait, broadcast)
        messages.append({"role": "user", "content": CONFIG["prompt_requests"]["action"]})
        messages = self._generate_single_model_response(messages=messages, model=self.model, temperature=temperature)

        raw = messages[-1]["content"]

        def pick_action(text: str) -> str:
            t = text.lower()
            best = None
            best_pos = 10**9
            for tok in ("move", "wait", "broadcast"):
                m = re.search(r'\b' + re.escape(tok) + r'\b', t)
                if m and m.start() < best_pos:
                    best, best_pos = tok, m.start()
            return best or "wait"

        action = pick_action(raw)
        messages[-1]["content"] = f'{action}: "{raw}"'

        # 3) Action-specific details
        content = ""
        if action == "move":
            # Ask for direction explicitly (2nd prompt)
            messages.append({"role": "user", "content": CONFIG["prompt_requests"]["action_move"]})
            messages = self._generate_single_model_response(messages=messages, model=self.model, temperature=temperature)
            dir_match = re.search(r'\b(north|south|east|west|northeast|northwest|southeast|southwest)\b', messages[-1]["content"].lower())
            if dir_match:
                content = dir_match.group(1)
                messages[-1]["content"] = f'{content}: "{messages[-1]["content"]}"'
            else:
                action, content = "wait", ""
                messages[-1]["content"] = f'Invalid direction, action reverted to wait: "{messages[-1]["content"]}"'

        if action == "broadcast":
            messages.append({"role": "user", "content": CONFIG["prompt_requests"]["action_broadcast"]})
            messages = self._generate_single_model_response(messages=messages, model=self.model, temperature=temperature)
            content = messages[-1]["content"]

        # 4) Memory update
        messages.append({"role": "user", "content": CONFIG["prompt_requests"]["memory_update"]})
        messages = self._generate_single_model_response(messages=messages, model=self.model, temperature=temperature)
        self.memory = messages[-1]["content"]

        # 5) Compile
        response = {
            "action": action,
            "content": content,
            "messages": messages[1:]  # exclude system prompt for logging brevity
        }
        return response

class _Tile:
    """Represents one board square (tile)."""
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.targeted_by = {"white": 0, "black": 0}  # number of rays hitting this tile per color
        self.figure: Optional[_Figure] = None
        self.drones: List[_Drone] = []

    def set_figure(self, figure: _Figure):
        self.figure = figure

    def add_drone(self, drone: _Drone):
        assert drone not in self.drones
        self.drones.append(drone)

    def remove_drone(self, drone: _Drone):
        if drone in self.drones:
            self.drones.remove(drone)

    def reset_targeted_by_amounts(self):
        self.targeted_by = {"white": 0, "black": 0}

    def add_targeted_by_amount(self, color: str, amount: int = 1):
        self.targeted_by[color] += amount


# In[33]:


# Cell 5: GUI (responsive, double-buffered, cached)

class _SimulationGUI:
    def __init__(self, sim):
        """Initialize the GUI. We accept `sim` to avoid global coupling."""
        self.sim = sim
        self.grid_size = (CONFIG["board"]["width"], CONFIG["board"]["height"])

        pygame.init()
        gui_config = CONFIG["gui"]

        self.sidebar_width = gui_config.get("sidebar_width", 1600)

        # Double-buffer for smoothness (and hardware surface when available).
        flags = pygame.HWSURFACE | pygame.DOUBLEBUF

        board_w = self.grid_size[0] * (gui_config["cell_size"] + gui_config["margin"]) + gui_config["margin"]
        board_h = self.grid_size[1] * (gui_config["cell_size"] + gui_config["margin"]) + gui_config["margin"]
        total_w = board_w + self.sidebar_width
        total_h = board_h

        self.screen = pygame.display.set_mode((total_w, total_h), flags)
        pygame.display.set_caption(
            f'Simulation - Round {1}.1/{CONFIG["simulation"]["max_rounds"]}.{CONFIG["simulation"]["num_drones"]}'
        )
        self.clock = pygame.time.Clock()

        # Speed up blits by converting images to display format.
        global FIGURE_IMAGES
        for k, surf in list(FIGURE_IMAGES.items()):
            try:
                FIGURE_IMAGES[k] = surf.convert_alpha()
            except pygame.error:
                pass

        self._image_cache = {}  # cache scaled images by (id, cell_size)
        self._font = pygame.font.SysFont(None, 18)
        self._font_small = pygame.font.SysFont(None, 16)

        self.info_lines = []
        self.info_max_lines = 200
        self.info_scroll = 0

    def _get_scaled(self, img):
        key = (id(img), CONFIG["gui"]["cell_size"])
        if key not in self._image_cache:
            self._image_cache[key] = pygame.transform.scale(img, (CONFIG["gui"]["cell_size"], CONFIG["gui"]["cell_size"]))
        return self._image_cache[key]

    def _draw_sidebar(self):
        gui = CONFIG["gui"]
        s = self.screen

        # Geometry
        board_w = self.grid_size[0] * (gui["cell_size"] + gui["margin"]) + gui["margin"]
        x0 = board_w
        y0 = 0
        w = self.sidebar_width
        h = s.get_height()

        # Background + border
        sidebar_bg = (25, 25, 25)
        border = (60, 60, 60)
        pygame.draw.rect(s, sidebar_bg, pygame.Rect(x0, y0, w, h))
        pygame.draw.line(s, border, (x0, 0), (x0, h), 1)

        # Title
        title = "Simulation Info"
        title_surf = self._font.render(title, True, (220, 220, 220))
        s.blit(title_surf, (x0 + 12, 12))

        # Scroll hint (optional)
        hint = "Latest at bottom"
        hint_surf = self._font_small.render(hint, True, (160, 160, 160))
        s.blit(hint_surf, (x0 + w - hint_surf.get_width() - 12, 14))

        # Text area
        pad = 12
        tx = x0 + pad
        ty = 40
        tw = w - 2 * pad
        th = h - ty - pad

        # Text background
        text_bg = (32, 32, 32)
        pygame.draw.rect(s, text_bg, pygame.Rect(tx - 4, ty - 4, tw + 8, th + 8))

        # Render last N lines to fit
        line_h = self._font_small.get_height() + 4
        max_lines_fit = th // line_h
        start_idx = max(0, len(self.info_lines) - max_lines_fit - self.info_scroll)
        end_idx = len(self.info_lines) - self.info_scroll

        y = ty
        for line in self.info_lines[start_idx:end_idx]:
            # simple clipping
            if y > ty + th - line_h:
                break
            surf = self._font_small.render(line, True, (230, 230, 230))
            s.blit(surf, (tx, y))
            y += line_h

    def post_info(self, text: str):
        # Split on \n so callers can post multiline
        for line in text.splitlines():
            self.info_lines.append(line)
        # Trim history
        if len(self.info_lines) > self.info_max_lines:
            self.info_lines = self.info_lines[-self.info_max_lines:]
        # Auto-scroll to bottom when new text arrives
        self.info_scroll = 0

    def draw_field(self):
        """Draw the entire board, pieces, threat overlays, and drones."""
        gui_config = CONFIG["gui"]
        cell_size = gui_config["cell_size"]
        margin = gui_config["margin"]
        grid_width, grid_height = self.grid_size

        self.screen.fill(gui_config["background_color"])

        for x in range(grid_width):
            for y in range(grid_height):
                y_flip = grid_height - 1 - y  # flip Y so (0,0) is bottom-left like chess
                rect = pygame.Rect(
                    x * (cell_size + margin) + margin,
                    y_flip * (cell_size + margin) + margin,
                    cell_size,
                    cell_size
                )
                pygame.draw.rect(self.screen, gui_config["grid_color"], rect)

                tile = self.sim.board[x][y]

                # Draw figure, if present
                if tile.figure:
                    figure_image = FIGURE_IMAGES.get((tile.figure.color, tile.figure.figure_type))
                    if figure_image:
                        self.screen.blit(self._get_scaled(figure_image), rect.topleft)
                    else:
                        pygame.draw.circle(self.screen, (200, 200, 200), rect.center, cell_size // 3)

                # Overlay "D# A#" (defended/attacked) with a dark background, bottom-left
                if tile.figure:
                    fig = tile.figure
                    overlay = f"D{fig.defended_by} A{fig.attacked_by}"
                    text_surf = self._font.render(overlay, True, gui_config["text_color"])

                    # Position: bottom-left with padding
                    pad = 3
                    tx = rect.left + pad
                    ty = rect.bottom - text_surf.get_height() - pad

                    # Background rectangle (dark grey) for contrast
                    bg_color = (40, 40, 40)  # tweak if needed
                    bg_rect = pygame.Rect(
                        tx - 2,                      # tiny margin around text
                        ty - 1,
                        text_surf.get_width() + 4,
                        text_surf.get_height() + 2
                    )
                    pygame.draw.rect(self.screen, bg_color, bg_rect)

                    # Blit text on top
                    self.screen.blit(text_surf, (tx, ty))

                # Draw drones (multiple per tile supported)
                total = len(tile.drones)
                if total > 0:
                    angle_step = 360 / total if total > 1 else 0
                    radius = cell_size // 6
                    for d_idx, drone in enumerate(tile.drones):
                        angle = angle_step * d_idx
                        offset = pygame.math.Vector2(0, 0)
                        if total > 1:
                            offset = pygame.math.Vector2(1, 0).rotate(angle) * (cell_size // 4)
                        center = (rect.centerx + int(offset.x), rect.centery + int(offset.y))

                        pygame.draw.circle(self.screen, gui_config["drone_color"], center, radius)
                        text = self._font.render(str(drone.id), True, gui_config["text_color"])
                        text_rect = text.get_rect(center=center)
                        self.screen.blit(text, text_rect)

        # Optional: if current drone is thinking, draw a highlight over its cell and a small indicator
        if self.sim._thinking:
            current = self.sim.current_drone()
            if current:
                x, y = current.position
                y_flip = grid_height - 1 - y
                rect = pygame.Rect(
                    x * (cell_size + margin) + margin,
                    y_flip * (cell_size + margin) + margin,
                    cell_size,
                    cell_size
                )
                pygame.draw.rect(self.screen, (255, 215, 0), rect, 2)  # gold outline

        self._draw_sidebar()
        pygame.display.flip()


# In[34]:


# Cell 6: Simulation core

class Simulation:
    """Holds game state, figures, tiles, drones, and runs the main loop."""
    def __init__(self):
        # Game state
        self.turn = 1   # Which drone's turn it is (1-based)
        self.round = 1  # Which round we are in (1-based)
        self.rules = ""
        with open(CONFIG["simulation"]["rules_path"], "r") as f:
            self.rules = f.read().replace("NUMBER_OF_DRONES", str(CONFIG["simulation"]["num_drones"]))

        self.grid_size = (CONFIG["board"]["width"], CONFIG["board"]["height"])
        self.max_rounds = CONFIG["simulation"]["max_rounds"]
        self.num_drones = CONFIG["simulation"]["num_drones"]
        self.models = CONFIG["simulation"]["models"]
        self.model_index = CONFIG["simulation"]["model_index"]
        self.model = self.models[self.model_index]
        LOGGER.log(f"Using model: {self.model}")

        # Board and entity lists
        self.board = [[_Tile(x, y) for y in range(self.grid_size[1])] for x in range(self.grid_size[0])]
        self.figures: List[_Figure] = []
        self.drones: List[_Drone] = []

        # Figures
        self._create_figures()

        # Threat map and per-figure attack/defense
        self._rebuild_threat_map()
        self._compute_attack_defense_per_figure()

        # Drones start at white king pos (first figure assumed white king)
        self.drone_base = self.figures[0].position
        self._create_drones()

        # GUI
        if CONFIG["simulation"]["use_gui"]:
            self.gui = _SimulationGUI(self)

        # Async inference control
        self.executor = ThreadPoolExecutor(max_workers=1)  # run one LLM turn at a time
        self._current_future: Optional[Future] = None
        self._thinking = False

    # ------------- Setup helpers -------------
    def _create_figures(self):
        """Generate all figures from CONFIG and place them on the board."""
        LOGGER.log("Creating figures based on configuration.")
        self.figures = []
        for color in COLORS:
            for figure_type in FIGURE_TYPES:
                for position in CONFIG["figures"][color][figure_type]:
                    self.figures.append(_Figure(position, color, figure_type))

        # Place figures on the board
        for figure in self.figures:
            self.board[figure.position[0]][figure.position[1]].set_figure(figure)

    def _create_drones(self):
        """Create drones and place them at the base."""
        LOGGER.log(f"Creating {self.num_drones} drones.")
        for i in range(self.num_drones):
            drone = _Drone(id=i+1, position=self.drone_base, model=self.model, rules=self.rules, sim=self)
            self.drones.append(drone)
        for drone in self.drones:
            tile = self.board[drone.position[0]][drone.position[1]]
            tile.add_drone(drone)

    # ------------- Threat map computation -------------
    def _rebuild_threat_map(self):
        """Recompute tile-level targeted_by counts from scratch."""
        # Reset
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                self.board[x][y].reset_targeted_by_amounts()

        # Rebuild targets for each figure
        for f in self.figures:
            f.calculate_figure_targets(self.board)

        # Accumulate targeted_by on tiles
        for f in self.figures:
            for (tx, ty) in f.target_positions:
                self.board[tx][ty].add_targeted_by_amount(f.color, 1)

    def _compute_attack_defense_per_figure(self):
        """Set each figure's defended_by and attacked_by counters from its tile."""
        for f in self.figures:
            x, y = f.position
            tile = self.board[x][y]
            if f.color == "white":
                f.defended_by = tile.targeted_by["white"]
                f.attacked_by = tile.targeted_by["black"]
            else:
                f.defended_by = tile.targeted_by["black"]
                f.attacked_by = tile.targeted_by["white"]

    def post_info(self, msg: str):
        """Convenient relay to the GUI sidebar."""
        if hasattr(self, "gui"):
            self.gui.post_info(msg)

    def threat_summary(self):
        """Log a compact summary of attack/defense per figure."""
        for f in self.figures:
            LOGGER.log(f"{f.color} {f.figure_type} at {f.position} — defended_by: {f.defended_by}, attacked_by: {f.attacked_by}")

    # ------------- Turn orchestration (async) -------------
    def _start_drone_turn(self, drone: _Drone):
        """Kick off LLM inference for this drone without blocking the GUI."""
        self._thinking = True
        self._current_future = self.executor.submit(drone._generate_full_model_response)

    def _try_finish_drone_turn(self, drone: _Drone) -> bool:
        """If the LLM finished, apply its action and advance. Return True when applied."""
        if self._current_future is None or not self._current_future.done():
            return False

        try:
            response = self._current_future.result()
            LOGGER.log(f"Drone {drone.id} response:\n{pprint.pformat(response, indent=4, width=200)}")
            self.post_info(f"Drone {drone.id}:")
            if response["action"] == "move":
                drone._move(response["content"])
                self.post_info(f"Move {response['content']} to {drone.position}")
            elif response["action"] == "broadcast":
                self.post_info(f"Broadcast")
                self.post_info(f"{pprint.pformat(response['content'], indent=4, width=250)}")
                tile = self.board[drone.position[0]][drone.position[1]]
                for d in tile.drones:
                    if d.id != drone.id:
                        d.rx_buffer += f"Drone {drone.id} broadcasted: {response['content']}\n"
            self.post_info("\n")
        except Exception as e:
            LOGGER.log(f"Error finishing Drone {drone.id}'s turn: {e}")

        self._thinking = False
        self._current_future = None
        return True

    def current_drone(self) -> Optional[_Drone]:
        """Return the drone whose turn it is (for GUI highlighting)."""
        return self.drones[self.turn - 1] if 1 <= self.turn <= len(self.drones) else None

    # ------------- Responsive main loop -------------
    def run_simulation(self):
        """Main loop that keeps GUI responsive while turns compute in the background."""
        max_rounds = CONFIG["simulation"].get("max_rounds", 10)
        use_gui = CONFIG["simulation"].get("use_gui", True)

        running = True
        clock = pygame.time.Clock()

        if use_gui:
            pygame.display.set_caption(f"Simulation - Round {1}.1/{max_rounds}.{self.num_drones}")
            self.gui.draw_field()

        current_round = 1
        drone_index = 0  # 0..num_drones-1
        pending_turn = False

        try:
            while running:
                # 1) Handle events every frame (never block)
                if use_gui:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                            running = False

                # 2) Start a new turn if none pending
                if not pending_turn:
                    if current_round > max_rounds:
                        break
                    self.round = current_round
                    self.turn = drone_index + 1
                    caption = f"Simulation - Round {current_round}.{self.turn}/{max_rounds}.{self.num_drones}"
                    LOGGER.log('#' * 50); LOGGER.log(caption)
                    if use_gui:
                        pygame.display.set_caption(caption)

                    drone = self.drones[drone_index]
                    self._start_drone_turn(drone)
                    pending_turn = True

                # 3) If current LLM finished, apply result and advance indices
                if pending_turn:
                    drone = self.drones[drone_index]
                    if self._try_finish_drone_turn(drone):
                        LOGGER.log(f"Round {self.round}.{self.turn} completed.")
                        drone_index += 1
                        if drone_index >= self.num_drones:
                            drone_index = 0
                            current_round += 1
                        pending_turn = False

                # 4) Redraw every frame
                if use_gui:
                    self.gui.draw_field()

                # 5) Limit FPS
                clock.tick(60)

        except KeyboardInterrupt:
            LOGGER.log("KeyboardInterrupt received — shutting down gracefully.")
            running = False
        finally:
            # Ensure resources are always released
            self.shutdown()


    def shutdown(self):
        """Cleanly stop background work and close the GUI."""
        # Cancel any running LLM future (best-effort)
        try:
            if getattr(self, "_current_future", None) and not self._current_future.done():
                self._current_future.cancel()
        except Exception:
            pass

        # Stop the executor
        try:
            if getattr(self, "executor", None):
                # don't wait on long model calls; cancel if supported
                self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

        # Close Pygame cleanly
        try:
            if hasattr(self, "gui"):
                pygame.display.quit()
            pygame.quit()
        except Exception:
            pass

        LOGGER.log("Clean shutdown complete.")


# In[35]:


# Cell 7: Entry point

if __name__ == "__main__":
    try:
        LOGGER.log("Launching simulation.")
        SIM = Simulation()
        SIM.run_simulation()
    except KeyboardInterrupt:
        LOGGER.log("Interrupted by user (Ctrl+C).")
        try:
            SIM.shutdown()
        except Exception:
            pass


