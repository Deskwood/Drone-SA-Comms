# Domain: Figures/Tiles
# =========================
from __future__ import annotations


def direction_from_vector(vector: Tuple[int, int]) -> str:
    """Get direction name from vector, or return string representation of vector if not found."""
    for direction, vec in DIRECTION_MAP.items():
        if vec == vector:
            return direction
    return str(vector)

def hsv_to_rgb255(h_deg: float, s: float, v: float) -> Tuple[int,int,int]:
    """h in degrees [0,360), s,v in [0,1] -> (r,g,b) in 0..255"""
    r, g, b = colorsys.hsv_to_rgb(h_deg/360.0, max(0,min(1,s)), max(0,min(1,v)))
    return (int(r*255), int(g*255), int(b*255))

def set_global_seed(seed: int):
    """Set global random seed for reproducibility."""
    if seed is None:
        print("No seed set.")
        return
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    print(f"Global seed set to {seed}.")

def chebyshev_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Calculate Chebyshev distance between two points a and b."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

def format_edge(source_type: str, source_color: str, target_color: str, edge: Tuple[Waypoint, Waypoint]) -> str:
    """Format an edge in chess notation."""
    src, dst = edge
    piece_symbol = {
        "king": "K", "queen": "Q", "rook": "R",
        "bishop": "B", "knight": "N", "pawn": ""
    }.get(source_type, "?")
    capture_symbol = "x" if source_color != target_color else "-"
    return f"{piece_symbol}{src.to_chess()}{capture_symbol}{dst.to_chess()}"

def on_board(x, y):
    """Check if coordinates (x,y) are within board boundaries."""
    return 0 <= x < CONFIG["board"]["width"] and 0 <= y < CONFIG["board"]["height"]

def load_figure_images() -> dict:
    """Load figure images from the configured directory."""
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

class _Tile:
    def __init__(self, x: int, y: int):
        self.x = x; self.y = y
        self.targeted_by = {"white": 0, "black": 0}
        self.figure = None
        self.drones = []

    def set_figure(self, figure): self.figure = figure
    def add_drone(self, drone):
        if drone not in self.drones: self.drones.append(drone)
    def remove_drone(self, drone):
        if drone in self.drones: self.drones.remove(drone)
    def reset_targeted_by_amounts(self): self.targeted_by = {"white": 0, "black": 0}
    def add_targeted_by_amount(self, color: str, amount: int = 1): self.targeted_by[color] += amount

class _Figure:
    def __init__(self, position: Tuple[int, int], color: str, figure_type: str):
        self.position = position
        self.color = color
        self.figure_type = figure_type
        self.defended_by = 0
        self.attacked_by = 0
        self.target_positions: List[Tuple[int, int]] = []

    def calculate_figure_targets(self, board: List[List[_Tile]]):
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

class _Local_Tile:
    """This class represents local knowledge of a tile on the board."""

    # color may be "unknown", "n/a", "white", "black"
    # type may be "unknown", "n/a", "any figure", "king", "queen", "rook", "bishop", "knight", "pawn"
    # possible combinations: (unknown,unknown), (n/a,n/a), (any figure,white/black), (king/queen/rook/bishop/knight/pawn,white/black)

    def __init__(self, true_figure: _Figure | None):
        self.true_figure = true_figure
        self.figure_type = "unknown"
        self.figure_color = "unknown"
        self.confirmed_targeter_count = 0

    def identify_true_figure_type_and_color(self):
        """If a figure is present on this tile, store its type and color, else set to "n/a"."""
        if self.true_figure is not None:
            self.figure_type = self.true_figure.figure_type
            self.figure_color = self.true_figure.color
        else:
            self.figure_type = "n/a"
            self.figure_color = "n/a"

    def identify_true_figure_color(self):
        """If a figure is present on this tile, store its color without figure type, else set to "n/a"."""
        if self.true_figure is not None:
            if self.figure_type == "unknown":
                self.figure_type = "any figure"
            self.figure_color = self.true_figure.color
        else:
            self.figure_type = "n/a"
            self.figure_color = "n/a"

    def clear_targeter_count(self):
        """Clear confirmed targeter count."""
        self.confirmed_targeter_count = 0

    def increase_targeter_count(self):
        """Increase confirmed targeter count."""
        self.confirmed_targeter_count += 1

class Waypoint:
    """Defines a waypoint on the board. It comprises x,y coordinates, optional turn and wait time."""

    def __init__(self, coordinate, turn: Optional[int] = None, wait: Optional[int] = None):
        """Initialize Waypoint from chess notation (e.g., 'e4') or cartesian tuple (x,y)."""
        self.x = None
        self.y = None
        self.turn = turn
        self.wait = wait

        # Determine initialization type
        if isinstance(coordinate, str): # Chess notation
            s = coordinate.strip().lower()
            if len(s) < 2:
                raise ValueError(f"Invalid chess coordinate: {coordinate}")
            col, row = s[0], s[1:]
            try:
                self.x = ord(col) - ord('a')
                self.y = int(row) - 1
            except Exception as e:
                raise ValueError(f"Invalid chess coordinate: {coordinate}")
        elif isinstance(coordinate, (tuple, list)): # Cartesian tuple/list
            if len(coordinate) != 2:
                raise ValueError("Cartesian coordinate must be a 2-tuple/list")
            self.x, self.y = int(x[0]), int(x[1])
        else: # Error
            raise ValueError("y coordinate must be provided for cartesian initialization")

    def to_chess(self) -> str:
        """Returns chess notation of the waypoint."""
        if self.x is None or self.y is None:
            raise ValueError("Waypoint coordinates are not set")
        col = chr(ord('a') + self.x)
        row = str(self.y + 1)
        return f"{col}{row}"

class Sector:
    """Defines a sector based on 2 waypoints."""
    
    def __init__(self, upper_left: Optional[Waypoint] = None, lower_right: Optional[Waypoint] = None):
        """Initialize sector with upper-left and lower-right waypoints. Defaults to full board if not provided."""
        if upper_left is None:
            upper_left = Waypoint((0, CONFIG["board"]["height"] - 1))
        if lower_right is None:
            lower_right = Waypoint((CONFIG["board"]["width"] - 1, 0))
        self.upper_left = upper_left
        self.lower_right = lower_right

    def equals(self, other: Sector) -> bool:
        """Check if two sectors are equal."""
        return (self.upper_left.x == other.upper_left.x and
                self.upper_left.y == other.upper_left.y and
                self.lower_right.x == other.lower_right.x and
                self.lower_right.y == other.lower_right.y)

    def change(self, upper_left: Optional[Waypoint] = None, lower_right: Optional[Waypoint] = None):
        """Change sector boundaries."""
        if upper_left is not None:
            self.upper_left = upper_left
        if lower_right is not None:
            self.lower_right = lower_right