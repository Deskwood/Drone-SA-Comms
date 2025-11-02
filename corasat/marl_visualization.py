"""
Visualization helpers for the CORASAT MARL environment.

Generates a static matplotlib figure summarising a completed episode,
including drone trajectories, final positions, and board layout. Designed to
mirror the feel of the GUI used in the language-model simulation while keeping
dependencies light-weight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

from marl_env import cartesian_to_chess, load_config
from marl_training import EpisodePlayback

try:
    import pygame
except ImportError:  # pragma: no cover - pygame optional at import time
    pygame = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from marl_training import EvaluationResult, TrainingStats

DEFAULT_CELL_COLORS = ("#f0d9b5", "#b58863")
DRONE_COLORS = [
    "#ff5555",
    "#55aaff",
    "#66dd66",
    "#ffcc33",
    "#bb66ff",
    "#ff66aa",
]

PIECE_SYMBOL = {
    "king": "K",
    "queen": "Q",
    "rook": "R",
    "bishop": "B",
    "knight": "N",
    "pawn": "P",
}


def _hex_to_rgb(value: str, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Parse a hex string to an RGB tuple, returning `fallback` on failure."""
    try:
        value = value.lstrip("#")
        if len(value) == 6:
            return tuple(int(value[i : i + 2], 16) for i in range(0, 6, 2))
    except Exception:
        pass
    return fallback


def _ensure_color(value: object, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Coerce a loose RGB-like value into a strict integer tuple."""
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return tuple(
                int(max(0, min(255, round(float(component))))) for component in value  # type: ignore[arg-type]
            )
        except Exception:
            return fallback
    return fallback


def _resolve_gui_settings(
    config_path: Optional[str],
    cell_size_override: Optional[int],
    margin_override: Optional[int],
    sidebar_override: Optional[int],
) -> Dict[str, object]:
    """Load GUI colours and sizing, falling back to sensible defaults."""
    settings: Dict[str, object] = {
        "cell_size": int(cell_size_override) if cell_size_override else 72,
        "margin": int(margin_override) if margin_override else 6,
        "sidebar_width": int(sidebar_override) if sidebar_override else 360,
        "background": (30, 30, 30),
        "panel_background": (22, 22, 22),
        "text": (235, 235, 235),
        "accent": (70, 70, 70),
    }
    try:
        config = load_config(config_path)
        gui = config.get("gui", {})
        settings["cell_size"] = int(cell_size_override or gui.get("cell_size", settings["cell_size"]))
        settings["margin"] = int(margin_override or gui.get("margin", settings["margin"]))
        settings["sidebar_width"] = int(sidebar_override or gui.get("sidebar_width", settings["sidebar_width"]))
        settings["background"] = _ensure_color(gui.get("background_color"), settings["background"])  # type: ignore[arg-type]
        settings["panel_background"] = _ensure_color(
            gui.get("sidebar_color"), settings["panel_background"]  # type: ignore[arg-type]
        )
        settings["text"] = _ensure_color(gui.get("text_color"), settings["text"])  # type: ignore[arg-type]
        settings["accent"] = _ensure_color(gui.get("grid_color"), settings["accent"])  # type: ignore[arg-type]
    except Exception:
        # Fall back to defaults if config is missing or malformed.
        pass
    return settings


def _cycle_color(index: int) -> str:
    """Pick a palette colour for the given drone index."""
    return DRONE_COLORS[index % len(DRONE_COLORS)]


def _plot_board(ax, width: int, height: int) -> None:
    """Render the checkerboard grid with coordinate labels."""
    for x in range(width):
        for y in range(height):
            color = DEFAULT_CELL_COLORS[(x + y) % 2]
            rect = Rectangle((x, y), 1, 1, facecolor=color, edgecolor="black", linewidth=0.5)
            ax.add_patch(rect)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.set_xticks([i + 0.5 for i in range(width)])
    ax.set_xticklabels([chr(ord("a") + i) for i in range(width)])
    ax.set_yticks([i + 0.5 for i in range(height)])
    ax.set_yticklabels([str(i + 1) for i in range(height)])
    ax.tick_params(left=False, bottom=False)
    ax.set_xlabel("File")
    ax.set_ylabel("Rank")


def _plot_figures(ax, figures: Iterable[Dict[str, object]]) -> None:
    """Draw chess pieces using stylised text markers."""
    for fig in figures:
        position = fig["position"]
        ftype = str(fig["type"])
        color = str(fig["color"])
        symbol = PIECE_SYMBOL.get(ftype, "?")
        text_color = "black" if color == "black" else "white"
        edge_color = "white" if color == "black" else "black"
        ax.text(
            position[0] + 0.5,
            position[1] + 0.5,
            symbol,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color=text_color,
            path_effects=[],
        )
        circle = plt.Circle(
            (position[0] + 0.5, position[1] + 0.5),
            0.38,
            fill=False,
            linewidth=1.8,
            edgecolor=edge_color,
        )
        ax.add_patch(circle)


def _plot_trajectories(
    ax,
    trajectories: Dict[int, List[Tuple[int, int]]],
    upto_step: Optional[int] = None,
    current_positions: Optional[Dict[int, Tuple[int, int]]] = None,
) -> None:
    """Overlay drone paths, highlighting the latest known position."""
    for drone_index, (drone_id, path) in enumerate(sorted(trajectories.items())):
        if not path:
            continue
        limit = len(path)
        if upto_step is not None:
            limit = min(limit, max(1, upto_step + 1))
        trimmed = path[:limit]
        if not trimmed:
            continue
        xs = [x + 0.5 for x, _ in trimmed]
        ys = [y + 0.5 for _, y in trimmed]
        color = _cycle_color(drone_index)
        ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.8)
        if current_positions and drone_id in current_positions:
            px, py = current_positions[drone_id]
            current_x = px + 0.5
            current_y = py + 0.5
        else:
            current_x = xs[-1]
            current_y = ys[-1]
        ax.scatter(current_x, current_y, color=color, s=120, edgecolor="black", zorder=5)
        ax.text(
            current_x,
            current_y,
            f"D{drone_id}",
            fontsize=10,
            ha="center",
            va="center",
            color="black",
            zorder=6,
            bbox=dict(facecolor="white", edgecolor=color, boxstyle="round,pad=0.2"),
        )


def _annotate_edges(ax, edges: List[str], board_height: int) -> None:
    """Summarise reported edges inside the board area."""
    if not edges:
        return
    snippet = "\n".join(edges[:8])
    if len(edges) > 8:
        snippet += "\n…"
    ax.text(
        0.02,
        0.98,
        f"Reported edges ({len(edges)}):\n{snippet}",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )


def _add_summary(fig, playback: EpisodePlayback) -> None:
    """Display aggregated episode statistics in the sidebar."""
    summary = playback.score_summary
    text_lines = [
        f"Score: {summary['score']:.0f} / {summary['gt_edges']}",
        f"Correct edges: {summary['correct_edges']}",
        f"False edges: {summary['false_edges']}",
        f"Reported edges: {summary['reported_edges']}",
    ]
    text_lines.append("")
    text_lines.append("Trajectories length:")
    for drone_id, path in sorted(playback.trajectories.items()):
        text_lines.append(f"  D{drone_id}: {len(path) - 1} steps")
    fig.text(
        0.78,
        0.75,
        "\n".join(text_lines),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f5f5", edgecolor="#888888"),
    )


def render_episode_snapshot(
    playback: EpisodePlayback,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> str:
    """Render a matplotlib visualization of an episode.

    Args:
        playback: Episode information as returned by `play_episode`.
        output_path: Optional path to save the image. When omitted, a PNG is
            created inside the project’s `screenshots` directory.
        figsize: Size of the matplotlib figure in inches.

    Returns:
        The filesystem path of the generated image.
    """
    width, height = playback.board_size
    fig, ax = plt.subplots(figsize=figsize)
    _plot_board(ax, width, height)
    _plot_figures(ax, playback.figures)
    _plot_trajectories(ax, playback.trajectories)
    _annotate_edges(ax, playback.reported_edges, height)
    _add_summary(fig, playback)
    ax.set_title("MARL Episode Overview", fontsize=16, pad=12)

    output_dir = Path("Code/corasat/screenshots")
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = output_dir / "marl_episode.png"
    else:
        output_path = Path(output_path)
        if output_path.is_dir():
            output_path = output_path / "marl_episode.png"

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return str(output_path)


def _figure_to_rgb_array(fig) -> np.ndarray:
    """Convert a matplotlib figure to an RGB numpy array."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width_px, height_px = canvas.get_width_height()
    buffer = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape((height_px, width_px, 4))
    return buffer[..., :3].copy()


def render_training_progress_video(
    training_stats: Iterable["TrainingStats"],
    output_path: Optional[str] = None,
    fps: float = 4.0,
    max_episodes: Optional[int] = 50,
    episode_stride: int = 1,
    figsize: Tuple[int, int] = (8, 8),
) -> str:
    """Create an animated GIF showing recorded training episodes.

    Each frame visualises the full drone trajectories of one recorded episode,
    mirroring the look of the main simulation GUI.
    """
    if Image is None:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "Pillow is required to export GIF animations. Install it with `pip install pillow`."
        )
    if episode_stride <= 0:
        raise ValueError("episode_stride must be a positive integer.")
    recorded: List[Tuple[int, float, EpisodePlayback]] = []
    for stat in training_stats:
        playback = getattr(stat, "playback", None)
        if playback is None:
            continue
        recorded.append((int(getattr(stat, "episode", len(recorded) + 1)), float(stat.score), playback))

    if not recorded:
        raise ValueError(
            "No recorded playbacks found in training history. "
            "Pass `record_playbacks_every` to `train_marl` to capture episodes."
        )

    recorded = recorded[::episode_stride]
    if max_episodes is not None and max_episodes > 0 and len(recorded) > max_episodes:
        recorded = recorded[-max_episodes:]

    frames: List[Image.Image] = []
    for episode_number, score, playback in recorded:
        fig, ax = plt.subplots(figsize=figsize)
        width, height = playback.board_size
        _plot_board(ax, width, height)
        _plot_figures(ax, playback.figures)
        _plot_trajectories(ax, playback.trajectories)
        _annotate_edges(ax, playback.reported_edges, height)

        title = f"Episode {episode_number} — Score {score:.0f}"
        ax.set_title(title, fontsize=14, pad=12)
        info_lines = []
        for drone_id, path in sorted(playback.trajectories.items()):
            if not path:
                continue
            coord = cartesian_to_chess(tuple(path[-1]))
            info_lines.append(f"D{drone_id}: {coord} ({len(path) - 1} steps)")
        if info_lines:
            ax.text(
                0.02,
                0.02,
                "\n".join(info_lines),
                transform=ax.transAxes,
                fontsize=9,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        rgb = _figure_to_rgb_array(fig)
        frames.append(Image.fromarray(rgb))
        plt.close(fig)

    output_dir = Path("Code/corasat/screenshots")
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = output_dir / "marl_training_progress.gif"
    else:
        output_path = Path(output_path)
        if output_path.is_dir():
            output_path = output_path / "marl_training_progress.gif"

    if not frames:
        raise ValueError("No frames generated for the training progress video.")

    duration_ms = max(1, int(1000 / max(fps, 0.1)))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return str(output_path)


class MarlEpisodeViewer:
    """Interactive pygame GUI to inspect MARL episode playbacks."""

    def __init__(
        self,
        playbacks: List[EpisodePlayback],
        *,
        config_path: Optional[str] = None,
        cell_size: Optional[int] = None,
        margin: Optional[int] = None,
        sidebar_width: Optional[int] = None,
        title: str = "MARL Episode Viewer",
        autoplay_fps: float = 2.0,
    ) -> None:
        if pygame is None:
            raise RuntimeError("pygame is required for the interactive viewer. Install it with `pip install pygame`.")  # pragma: no cover - defensive
        if not playbacks:
            raise ValueError("At least one EpisodePlayback is required to start the viewer.")

        self.playbacks = playbacks
        self.settings = _resolve_gui_settings(config_path, cell_size, margin, sidebar_width)
        self.cell_size = int(self.settings["cell_size"])
        self.margin = int(self.settings["margin"])
        self.sidebar_width = int(self.settings["sidebar_width"])
        self.background_color = tuple(self.settings["background"])  # type: ignore[arg-type]
        self.panel_color = tuple(self.settings["panel_background"])  # type: ignore[arg-type]
        self.text_color = tuple(self.settings["text"])  # type: ignore[arg-type]
        self.accent_color = tuple(self.settings["accent"])  # type: ignore[arg-type]
        self.light_square = _hex_to_rgb(DEFAULT_CELL_COLORS[0], (240, 217, 181))
        self.dark_square = _hex_to_rgb(DEFAULT_CELL_COLORS[1], (181, 136, 99))
        self.drone_colors = [_hex_to_rgb(color, (255, 180, 0)) for color in DRONE_COLORS]

        self.title = title
        self.autoplay = True
        self.autoplay_fps = max(0.25, float(autoplay_fps))
        self.step_duration = 1.0 / self.autoplay_fps
        self._autoplay_timer = 0.0
        self.episode_index = 0
        self.step_index = 0

        self.board_cols, self.board_rows = self.playbacks[0].board_size
        for playback in self.playbacks:
            if playback.board_size != (self.board_cols, self.board_rows):
                raise ValueError("All playbacks must share the same board size.")
        self.board_pixel_width = self.board_cols * self.cell_size + (self.board_cols + 1) * self.margin
        self.board_pixel_height = self.board_rows * self.cell_size + (self.board_rows + 1) * self.margin
        self.total_width = self.board_pixel_width + self.sidebar_width
        self.total_height = self.board_pixel_height

        pygame.init()
        self.screen = pygame.display.set_mode((self.total_width, self.total_height))
        pygame.display.set_caption(self.title)
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("Segoe UI", 22)
            self.font_small = pygame.font.SysFont("Segoe UI", 18)
            self.font_large = pygame.font.SysFont("Segoe UI", 28)
        except Exception:
            self.font = pygame.font.Font(None, 22)
            self.font_small = pygame.font.Font(None, 18)
            self.font_large = pygame.font.Font(None, 28)

    # ------------------------------------------------------------------ event loop
    def run(self) -> None:
        """Main pygame loop that handles input, autoplay, and drawing."""
        running = True
        while running:
            delta = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)
            if not running:
                break
            self._update_autoplay(delta)
            self._draw()
        pygame.quit()

    def _handle_key(self, key: int) -> bool:
        """React to keyboard shortcuts; return False to exit the viewer."""
        if key in (pygame.K_ESCAPE, pygame.K_q):
            return False
        if key in (pygame.K_RIGHT, pygame.K_d):
            self._advance_step(1)
        elif key in (pygame.K_LEFT, pygame.K_a):
            self._advance_step(-1)
        elif key in (pygame.K_DOWN, pygame.K_s):
            self._change_episode(1)
        elif key in (pygame.K_UP, pygame.K_w):
            self._change_episode(-1)
        elif key == pygame.K_SPACE:
            self.autoplay = not self.autoplay
            self._autoplay_timer = 0.0
        elif key == pygame.K_HOME:
            self.step_index = 0
            self._autoplay_timer = 0.0
        elif key == pygame.K_END:
            self.step_index = self._last_step_index()
            self._autoplay_timer = 0.0
        return True

    def _advance_step(self, delta: int) -> None:
        """Move within the current episode playback snapshot list."""
        snapshots = self._current_snapshots()
        if not snapshots:
            self.step_index = 0
            return
        self.step_index = max(0, min(self.step_index + delta, len(snapshots) - 1))
        self._autoplay_timer = 0.0

    def _change_episode(self, delta: int) -> None:
        """Switch to another recorded episode while resetting step state."""
        new_index = max(0, min(self.episode_index + delta, len(self.playbacks) - 1))
        if new_index != self.episode_index:
            self.episode_index = new_index
            self.step_index = 0
            self._autoplay_timer = 0.0

    def _update_autoplay(self, delta: float) -> None:
        """Advance playback automatically when autoplay mode is enabled."""
        if not self.autoplay:
            return
        snapshots = self._current_snapshots()
        if not snapshots:
            return
        if self.step_index >= len(snapshots) - 1:
            if self.episode_index < len(self.playbacks) - 1:
                self._change_episode(1)
            else:
                self.autoplay = False
            return
        self._autoplay_timer += delta
        while self._autoplay_timer >= self.step_duration:
            self._autoplay_timer -= self.step_duration
            if self.step_index < len(snapshots) - 1:
                self.step_index += 1
            else:
                break

    def _last_step_index(self) -> int:
        snapshots = self._current_snapshots()
        return len(snapshots) - 1 if snapshots else 0

    # ------------------------------------------------------------------ state helpers
    def _current_playback(self) -> EpisodePlayback:
        return self.playbacks[self.episode_index]

    def _current_snapshots(self) -> List[Dict[str, object]]:
        return self._current_playback().snapshots

    def _current_snapshot(self) -> Optional[Dict[str, object]]:
        snapshots = self._current_snapshots()
        if not snapshots:
            return None
        index = min(self.step_index, len(snapshots) - 1)
        return snapshots[index]

    def _current_positions(self, playback: EpisodePlayback) -> Dict[int, Tuple[int, int]]:
        snapshot = self._current_snapshot()
        if snapshot:
            return {int(agent_id): tuple(position) for agent_id, position in snapshot["positions"].items()}  # type: ignore[dict-item]
        return {agent_id: path[-1] for agent_id, path in playback.trajectories.items() if path}

    def _drone_color(self, index: int) -> Tuple[int, int, int]:
        return self.drone_colors[index % len(self.drone_colors)]

    def _path_offset(self, index: int, total: int) -> Tuple[int, int]:
        if total <= 1:
            return (0, 0)
        angle = math.radians(45.0 + 360.0 * (index % total) / total)
        radius = max(2, int(self.cell_size * 0.14))
        return int(radius * math.cos(angle)), int(radius * math.sin(angle))

    # ------------------------------------------------------------------ drawing
    def _draw(self) -> None:
        """Render the current frame (board, figures, paths, HUD) to the screen."""
        playback = self._current_playback()
        self.screen.fill(self.background_color)
        self._draw_board()
        self._draw_figures(playback)
        self._draw_paths(playback)
        self._draw_drones(playback)
        self._draw_sidebar(playback)
        pygame.display.flip()

    def _draw_board(self) -> None:
        """Fill the background with alternating light/dark squares."""
        for x in range(self.board_cols):
            for y in range(self.board_rows):
                rect = self._tile_rect(x, y)
                color = self.light_square if (x + y) % 2 == 0 else self.dark_square
                pygame.draw.rect(self.screen, color, rect)

    def _tile_rect(self, x: int, y: int) -> "pygame.Rect":
        margin = self.margin
        cell = self.cell_size
        y_flip = self.board_rows - 1 - y
        left = margin + x * (cell + margin)
        top = margin + y_flip * (cell + margin)
        return pygame.Rect(left, top, cell, cell)

    def _tile_center(self, x: int, y: int) -> Tuple[int, int]:
        rect = self._tile_rect(x, y)
        return rect.centerx, rect.centery

    def _draw_figures(self, playback: EpisodePlayback) -> None:
        """Overlay the static chess pieces on the board."""
        for figure in playback.figures:
            x, y = figure["position"]  # type: ignore[misc]
            rect = self._tile_rect(x, y)
            color = str(figure.get("color", "white")).lower()
            symbol = PIECE_SYMBOL.get(str(figure.get("type")), "?")
            base_color = (230, 230, 230) if color == "white" else (40, 40, 40)
            text_color = (30, 30, 30) if color == "white" else (230, 230, 230)
            pygame.draw.circle(self.screen, base_color, rect.center, max(8, self.cell_size // 2 - 6), 2)
            label = self.font_large.render(symbol, True, text_color)
            self.screen.blit(label, label.get_rect(center=rect.center))

    def _draw_paths(self, playback: EpisodePlayback) -> None:
        """Render historical trajectories with slight offsets to avoid overlap."""
        snapshots = self._current_snapshots()
        step_limit = len(snapshots) + 1 if snapshots else None
        total_drones = max(1, len(playback.trajectories))

        for index, (drone_id, path) in enumerate(sorted(playback.trajectories.items())):
            color = self._drone_color(index)
            if not path:
                continue
            if step_limit is not None:
                limit = min(len(path), max(1, self.step_index + 2))
                path_segment = path[:limit]
            else:
                path_segment = path
            centres = [self._tile_center(px, py) for (px, py) in path_segment]
            offset = self._path_offset(index, total_drones)
            adjusted = [(cx + offset[0], cy + offset[1]) for (cx, cy) in centres]
            if len(adjusted) >= 2:
                pygame.draw.lines(self.screen, color, False, adjusted, 3)
            if adjusted:
                pygame.draw.circle(self.screen, color, adjusted[0], 5)

    def _draw_drones(self, playback: EpisodePlayback) -> None:
        """Draw the drones at their current positions using coloured discs."""
        positions = self._current_positions(playback)
        for index, (drone_id, (x, y)) in enumerate(sorted(positions.items())):
            rect = self._tile_rect(x, y)
            color = self._drone_color(index)
            center = rect.center
            radius = max(12, self.cell_size // 3)
            pygame.draw.circle(self.screen, color, center, radius)
            pygame.draw.circle(self.screen, (15, 15, 15), center, radius, 2)
            label = self.font_small.render(f"D{drone_id}", True, (0, 0, 0))
            self.screen.blit(label, label.get_rect(center=center))

    def _draw_sidebar(self, playback: EpisodePlayback) -> None:
        """Populate the sidebar with scoreboard stats and control hints."""
        panel_rect = pygame.Rect(self.board_pixel_width, 0, self.sidebar_width, self.total_height)
        pygame.draw.rect(self.screen, self.panel_color, panel_rect)
        pygame.draw.line(
            self.screen,
            self.accent_color,
            (panel_rect.left, 0),
            (panel_rect.left, panel_rect.bottom),
            2,
        )

        x = panel_rect.left + 18
        y = panel_rect.top + 20
        line_height = self.font_small.get_linesize() + 2

        header = self.font.render(f"Episode {self.episode_index + 1}/{len(self.playbacks)}", True, self.text_color)
        self.screen.blit(header, (x, y))
        y += line_height + 6

        snapshot = self._current_snapshot()
        if snapshot:
            step_text = f"Step {self.step_index + 1}/{len(self._current_snapshots())} • Round {snapshot['round']}"
            edges = snapshot.get("reported_edges", [])
        else:
            step_text = "Step 0/0"
            edges = playback.reported_edges
        status = "Mode: auto" if self.autoplay else "Mode: manual"
        step_surface = self.font_small.render(step_text, True, self.text_color)
        status_surface = self.font_small.render(status, True, self.text_color)
        self.screen.blit(step_surface, (x, y))
        y += line_height
        self.screen.blit(status_surface, (x, y))
        y += line_height + 4

        summary = playback.score_summary
        for line in (
            f"Score {summary['score']:.0f}/{summary['gt_edges']}",
            f"Correct {summary['correct_edges']}",
            f"False {summary['false_edges']}",
            f"Reported {summary['reported_edges']}",
        ):
            self.screen.blit(self.font_small.render(line, True, self.text_color), (x, y))
            y += line_height
        y += line_height // 2

        positions = self._current_positions(playback)
        for index, (drone_id, path) in enumerate(sorted(playback.trajectories.items())):
            position = positions.get(drone_id, path[-1])
            coord = cartesian_to_chess(tuple(position))
            steps = max(0, len(path) - 1)
            color = self._drone_color(index)
            pygame.draw.rect(self.screen, color, pygame.Rect(x, y + 5, 10, 10))
            label = f"D{drone_id}: {coord} ({steps} steps)"
            self.screen.blit(self.font_small.render(label, True, self.text_color), (x + 16, y))
            y += line_height
        y += line_height // 2

        self.screen.blit(self.font_small.render("Reported edges:", True, self.text_color), (x, y))
        y += line_height
        max_edges = 10
        for edge in list(edges)[:max_edges]:
            self.screen.blit(self.font_small.render(f"- {edge}", True, self.text_color), (x + 12, y))
            y += line_height
        if len(edges) > max_edges:
            self.screen.blit(self.font_small.render("... more edges omitted", True, self.text_color), (x + 12, y))
            y += line_height
        y += line_height

        for line in (
            "Controls:",
            "Space  toggle auto",
            "←/→    step",
            "↑/↓    episode",
            "Home   start",
            "End    finish",
            "Esc/Q  exit",
        ):
            self.screen.blit(self.font_small.render(line, True, self.text_color), (x, y))
            y += line_height


def launch_episode_viewer(
    playbacks: List[EpisodePlayback],
    **viewer_kwargs: object,
) -> None:
    """Convenience helper to launch the interactive MARL viewer."""
    viewer = MarlEpisodeViewer(playbacks, **viewer_kwargs)
    viewer.run()


def launch_evaluation_viewer(result: "EvaluationResult", **viewer_kwargs: object) -> None:
    """Launch the viewer for an EvaluationResult returned by `evaluate_policy`."""
    if not getattr(result, "playbacks", None):
        raise ValueError("The evaluation result does not contain stored playbacks.")
    launch_episode_viewer(list(result.playbacks), **viewer_kwargs)


__all__ = [
    "render_episode_snapshot",
    "render_training_progress_video",
    "MarlEpisodeViewer",
    "launch_episode_viewer",
    "launch_evaluation_viewer",
]
