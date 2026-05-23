"""Command-line entry point for the wound-healing simulation."""

from __future__ import annotations

import argparse

from animation import animate_wound
from wound_model import simulate_wound


def _positive_or_zero_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate and visualize simplified wound healing over time."
    )
    parser.add_argument("--width", type=_positive_or_zero_float, help="Initial wound width in mm.")
    parser.add_argument("--length", type=_positive_or_zero_float, help="Initial wound length in mm.")
    parser.add_argument("--depth", type=_positive_or_zero_float, help="Initial wound depth in mm.")
    parser.add_argument("--dt", type=_positive_float, help="Time step in days.")
    parser.add_argument("--days", type=_positive_or_zero_float, help="Total simulation time in days.")
    parser.add_argument(
        "--save-animation",
        help="Optional output path for animation, for example wound.gif or wound.mp4.",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Run the simulation and CSV export without displaying an animation.",
    )
    return parser.parse_args()


def _prompt_missing(label: str, current_value: float | None, positive: bool) -> float:
    if current_value is not None:
        return current_value

    while True:
        raw = input(f"{label}: ").strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a numeric value.")
            continue

        if positive and value <= 0:
            print("Please enter a value greater than 0.")
            continue
        if not positive and value < 0:
            print("Please enter a non-negative value.")
            continue
        return value


def _write_results(results, path: str) -> None:
    if hasattr(results, "to_csv"):
        results.to_csv(path, index=False)
        return

    import csv

    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def _last_row(results):
    if hasattr(results, "iloc"):
        return results.iloc[-1]
    return results[-1]


def main() -> None:
    args = parse_args()

    width = _prompt_missing("Initial wound width W0 (mm)", args.width, positive=False)
    length = _prompt_missing("Initial wound length L0 (mm)", args.length, positive=False)
    depth = _prompt_missing("Initial wound depth D0 (mm)", args.depth, positive=False)
    dt = _prompt_missing("Time step dt (days)", args.dt, positive=True)
    days = _prompt_missing("Total simulation time (days)", args.days, positive=False)

    results = simulate_wound(width, length, depth, dt, days)
    _write_results(results, "wound_results.csv")

    final = _last_row(results)
    print("\nFinal wound dimensions")
    print(f"  Width:  {float(final['width']):.3f} mm")
    print(f"  Length: {float(final['length']):.3f} mm")
    print(f"  Depth:  {float(final['depth']):.3f} mm")
    print("Saved time-series data to wound_results.csv")

    if args.save_animation:
        animate_wound(results, save_path=args.save_animation)
        print(f"Saved animation to {args.save_animation}")
    elif not args.no_animation:
        animate_wound(results)


if __name__ == "__main__":
    main()
