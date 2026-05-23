"""Matplotlib animation for visible wound geometry over time."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def _column(results, name: str) -> np.ndarray:
    if hasattr(results, "__getitem__") and not isinstance(results, list):
        return np.asarray(results[name], dtype=float)
    return np.asarray([row[name] for row in results], dtype=float)


def animate_wound(results, save_path: str | None = None):
    """Animate wound closure from top and side views.

    Args:
        results: DataFrame or list of dictionaries from simulate_wound.
        save_path: Optional .gif or .mp4 path. GIF uses Pillow; MP4 uses the
            Matplotlib ffmpeg writer when available.

    Returns:
        The matplotlib FuncAnimation instance.
    """

    time = _column(results, "time")
    width = _column(results, "width")
    length = _column(results, "length")
    depth = _column(results, "depth")

    if len(time) == 0:
        raise ValueError("results must contain at least one time point.")

    max_width = max(float(width[0]), 1.0)
    max_length = max(float(length[0]), 1.0)
    max_depth = max(float(depth[0]), 1.0)

    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(11, 5))
    fig.subplots_adjust(top=0.82)

    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    top_fill = ax_top.fill([], [], color="#b84a3a", alpha=0.72)[0]
    top_outline, = ax_top.plot([], [], color="#6f241d", linewidth=2)

    side_fill = ax_side.fill([], [], color="#8c4a32", alpha=0.75)[0]
    side_line, = ax_side.plot([], [], color="#4f2417", linewidth=2)
    surface_line = ax_side.axhline(0.0, color="#333333", linewidth=1)

    ax_top.set_title("Top view")
    ax_top.set_xlabel("Length (mm)")
    ax_top.set_ylabel("Width (mm)")
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.set_xlim(-max_length * 0.6, max_length * 0.6)
    ax_top.set_ylim(-max_width * 0.6, max_width * 0.6)
    ax_top.grid(True, alpha=0.25)

    ax_side.set_title("Side view")
    ax_side.set_xlabel("Length (mm)")
    ax_side.set_ylabel("Depth (mm)")
    ax_side.set_xlim(-max_length * 0.6, max_length * 0.6)
    ax_side.set_ylim(-max_depth * 1.15, max_depth * 0.2)
    ax_side.grid(True, alpha=0.25)

    title = fig.suptitle("")

    def update(frame: int):
        W = max(width[frame], 0.0)
        L = max(length[frame], 0.0)
        D = max(depth[frame], 0.0)

        x_top = (L / 2.0) * np.cos(theta)
        y_top = (W / 2.0) * np.sin(theta)
        top_fill.set_xy(np.column_stack([x_top, y_top]))
        top_outline.set_data(x_top, y_top)

        if L > 0.0 and D > 0.0:
            x_side = np.linspace(-L / 2.0, L / 2.0, 240)
            profile = -D * np.sqrt(np.maximum(0.0, 1.0 - (2.0 * x_side / L) ** 2))
            fill_x = np.concatenate([x_side, x_side[::-1]])
            fill_y = np.concatenate([np.zeros_like(x_side), profile[::-1]])
        else:
            x_side = np.array([0.0])
            profile = np.array([0.0])
            fill_x = np.array([0.0, 0.0])
            fill_y = np.array([0.0, 0.0])

        side_fill.set_xy(np.column_stack([fill_x, fill_y]))
        side_line.set_data(x_side, profile)
        surface_line.set_ydata([0.0, 0.0])

        title.set_text(
            f"Day {time[frame]:.2f} | Width {W:.2f} mm | "
            f"Length {L:.2f} mm | Depth {D:.2f} mm"
        )
        return top_fill, top_outline, side_fill, side_line, surface_line, title

    interval_ms = 60
    if len(time) > 1:
        interval_ms = max(20, int(1000 * min(np.diff(time).mean(), 0.2)))

    anim = FuncAnimation(
        fig,
        update,
        frames=len(time),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    if save_path:
        path = Path(save_path)
        if path.suffix.lower() == ".gif":
            anim.save(path, writer=PillowWriter(fps=15))
        else:
            anim.save(path)
        plt.close(fig)
    else:
        plt.show()

    return anim
