"""Command-line entry point for the 2D wound-healing simulation."""

from __future__ import annotations

import argparse

from animation import animate_wound
from wound_model import SimulationConfig, simulate_wound_spatial, validate_initial_wound


def _prompt_float(label: str) -> float:
    while True:
        raw = input(f"{label}: ").strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if value <= 0.0:
            print("Please enter a positive value.")
            continue
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate and animate simplified 2D wound healing.")
    parser.add_argument("--width", type=float, help="Initial wound width W0 in mm.")
    parser.add_argument("--length", type=float, help="Initial wound length L0 in mm.")
    parser.add_argument("--depth", type=float, help="Initial wound depth D0 in mm.")
    parser.add_argument("--dt", type=float, help="Requested time step in days.")
    parser.add_argument("--days", type=float, help="Total simulation time in days.")
    parser.add_argument("--save-animation", type=str, default=None, help="Optional output path for MP4 or GIF.")
    parser.add_argument("--gif", action="store_true", help="Save wound_animation.gif if no path is provided.")
    parser.add_argument("--nx", type=int, default=160, help="Number of grid points in x.")
    parser.add_argument("--ny", type=int, default=160, help="Number of grid points in y.")
    parser.add_argument("--snapshot-every", type=int, default=5, help="Store every N internal steps for animation.")
    parser.add_argument("--fps", type=int, default=15, help="Animation frames per second when saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    width = args.width if args.width is not None else _prompt_float("Initial wound width W0 (mm)")
    length = args.length if args.length is not None else _prompt_float("Initial wound length L0 (mm)")
    depth = args.depth if args.depth is not None else _prompt_float("Initial wound depth D0 (mm)")
    dt = args.dt if args.dt is not None else _prompt_float("Time step dt (days)")
    days = args.days if args.days is not None else _prompt_float("Total simulation time (days)")

    try:
        validate_initial_wound(width, length, depth, dt, days)
    except ValueError as exc:
        raise SystemExit(f"Invalid input: {exc}") from exc

    config = SimulationConfig(Nx=args.nx, Ny=args.ny, snapshot_every=args.snapshot_every)
    print("Running wound-healing simulation...")
    snapshots, metrics_df = simulate_wound_spatial(width, length, depth, dt, days, config=config)
    metrics_df.to_csv("wound_results.csv", index=False)

    final = metrics_df.iloc[-1]
    print("Saved metrics to wound_results.csv")
    print(
        "Final wound dimensions: "
        f"width={final.width:.3f} mm, length={final.length:.3f} mm, depth={final.depth:.3f} mm"
    )

    save_path = args.save_animation
    if args.gif and not save_path:
        save_path = "wound_animation.gif"
    animate_wound(snapshots, metrics_df, save_path=save_path, fps=args.fps)


if __name__ == "__main__":
    main()
