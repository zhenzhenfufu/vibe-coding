"""Animation routines for the wound-healing model."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from utils import compute_anisotropy, compute_orientation_angle


def _save_animation(anim: FuncAnimation, save_path: str, fps: int) -> None:
    path = Path(save_path)
    suffix = path.suffix.lower()
    if suffix == ".mp4" and shutil.which("ffmpeg"):
        anim.save(path, fps=fps, dpi=140, writer="ffmpeg")
    else:
        if suffix not in {".gif", ".mp4"}:
            path = path.with_suffix(".gif")
        elif suffix == ".mp4":
            path = path.with_suffix(".gif")
        anim.save(path, dpi=120, writer=PillowWriter(fps=fps))
        print(f"Saved GIF animation to {path}")


def animate_wound(snapshots, metrics_df, save_path: str | None = None, fps: int = 15):
    """Animate top-view wound geometry, tensor orientation, depth, and metrics."""
    first = snapshots[0]
    X, Y = np.meshgrid(first.x, first.y)
    extent = [first.x.min(), first.x.max(), first.y.min(), first.y.max()]
    max_depth = max(float(metrics_df["depth"].max()), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    ax_top, ax_depth, ax_series = axes

    depth_image = ax_top.imshow(
        first.Z,
        origin="lower",
        extent=extent,
        cmap="magma_r",
        vmin=0.0,
        vmax=max_depth,
        alpha=0.92,
    )
    boundary = ax_top.contour(X, Y, first.psi, levels=[0.0], colors="cyan", linewidths=2.0)
    orientation_lines = []
    ax_top.set_title("Top view: depth, boundary, orientation")
    ax_top.set_xlabel("x (mm)")
    ax_top.set_ylabel("y (mm)")
    ax_top.set_aspect("equal")
    fig.colorbar(depth_image, ax=ax_top, fraction=0.046, pad=0.04, label="Depth (mm)")

    center_y_index = len(first.y) // 2
    (depth_line,) = ax_depth.plot(first.x, first.Z[center_y_index, :], color="tab:red", lw=2)
    ax_depth.fill_between(first.x, 0, first.Z[center_y_index, :], color="tab:red", alpha=0.25)
    ax_depth.set_xlim(first.x.min(), first.x.max())
    ax_depth.set_ylim(0.0, max_depth * 1.05)
    ax_depth.set_title("Centerline depth profile")
    ax_depth.set_xlabel("x (mm)")
    ax_depth.set_ylabel("Depth (mm)")
    ax_depth.grid(alpha=0.25)

    time = metrics_df["time"].to_numpy()
    ax_series.plot(time, metrics_df["width"], label="width", color="tab:blue")
    ax_series.plot(time, metrics_df["length"], label="length", color="tab:green")
    ax_series.plot(time, metrics_df["depth"], label="depth", color="tab:red")
    if "mean_collagen" in metrics_df:
        collagen_scaled = metrics_df["mean_collagen"] * max(metrics_df["length"].max(), 1.0)
        ax_series.plot(time, collagen_scaled, label="mean collagen (scaled)", color="tab:purple", alpha=0.7)
    marker = ax_series.axvline(first.time, color="black", lw=1.5, alpha=0.8)
    ax_series.set_xlim(time.min(), time.max())
    ax_series.set_ylim(0.0, max(metrics_df[["width", "length", "depth"]].max().max(), 1.0) * 1.08)
    ax_series.set_title("Geometry over time")
    ax_series.set_xlabel("Time (days)")
    ax_series.grid(alpha=0.25)
    ax_series.legend(loc="upper right")

    sample_step = max(6, min(first.x.size, first.y.size) // 22)
    segment_base = 0.55 * sample_step * min(first.dx, first.dy)

    def draw_orientation(state):
        nonlocal orientation_lines
        for line in orientation_lines:
            line.remove()
        orientation_lines = []

        theta = compute_orientation_angle(state.A_xx, state.A_xy, state.A_yy)
        anisotropy = compute_anisotropy(state.A_xx, state.A_xy, state.A_yy)
        visible = (state.C > 0.05) | (state.F > 0.05) | (state.psi < 0.0)
        for iy in range(0, len(state.y), sample_step):
            for ix in range(0, len(state.x), sample_step):
                if not visible[iy, ix]:
                    continue
                strength = float(anisotropy[iy, ix])
                if strength < 0.04:
                    continue
                half_len = segment_base * (0.25 + strength)
                u = np.cos(theta[iy, ix])
                v = np.sin(theta[iy, ix])
                x0 = state.x[ix]
                y0 = state.y[iy]
                (line,) = ax_top.plot(
                    [x0 - half_len * u, x0 + half_len * u],
                    [y0 - half_len * v, y0 + half_len * v],
                    color="black",
                    alpha=min(0.85, 0.18 + 0.85 * strength),
                    lw=0.8,
                    solid_capstyle="round",
                )
                orientation_lines.append(line)

    def update(frame_index):
        nonlocal boundary
        state = snapshots[frame_index]
        depth_image.set_data(state.Z)
        boundary.remove()
        boundary = ax_top.contour(X, Y, state.psi, levels=[0.0], colors="cyan", linewidths=2.0)
        draw_orientation(state)

        depth_line.set_ydata(state.Z[center_y_index, :])
        for coll in list(ax_depth.collections):
            coll.remove()
        ax_depth.fill_between(state.x, 0, state.Z[center_y_index, :], color="tab:red", alpha=0.25)
        marker.set_xdata([state.time, state.time])

        row_index = int(np.argmin(np.abs(metrics_df["time"].to_numpy() - state.time)))
        row = metrics_df.iloc[row_index]
        fig.suptitle(
            f"t={row.time:.2f} days | width={row.width:.2f} mm | "
            f"length={row.length:.2f} mm | depth={row.depth:.2f} mm",
            fontsize=13,
        )
        return [depth_image, depth_line, marker, *orientation_lines]

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=1000 / fps, blit=False)
    update(0)

    if save_path:
        _save_animation(anim, save_path, fps)
        plt.close(fig)
    else:
        plt.show()
    return anim
