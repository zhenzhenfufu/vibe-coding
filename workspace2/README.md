# 2D Wound Healing Simulation

This project is a practical, simplified spatial wound-healing model inspired by
the paper "A mathematical model of wound healing and subsequent scarring":
https://pmc.ncbi.nlm.nih.gov/articles/PMC2839370/

It preserves the main computational ideas requested here: evolving wound
geometry, spatial biological fields, changing wound depth, and a local 2x2
orientation tensor for tissue/collagen alignment. It is a research/demo model,
not a clinical prediction tool.

The command-line model is intentionally scoped to wounds small enough to
plausibly heal naturally without closure procedures:

- initial width must be `13 mm` or less (`1.3 cm`)
- initial depth must be less than `6 mm` (`0.6 cm`)

## Files

- `main.py` - command-line interface.
- `wound_model.py` - model dataclasses, initialization, time stepping, and simulation loop.
- `animation.py` - Matplotlib animation with top view, orientation overlay, depth profile, and time series.
- `utils.py` - finite differences, level-set reinitialization, tensor helpers, and geometry metrics.
- `README.md` - this file.

## Model Summary

The wound boundary is represented by a signed field `psi(x, y, t)`:

- `psi < 0` inside the wound
- `psi = 0` at the wound edge
- `psi > 0` outside the wound

The initial wound is an ellipse centered at the origin:

```text
psi(x, y, 0) = sqrt((2x / L0)^2 + (2y / W0)^2) - 1
```

The wound depth map `Z(x, y, t)` starts as a bowl-shaped crater and fills as
collagen matures.

The normalized biological fields are:

- `Q` - fibrin clot / provisional matrix
- `P` - tPA activity
- `M` - macrophage activity
- `G` - TGF-beta signal
- `F` - fibroblast activity
- `C` - collagen / granulation tissue maturity

These fields use explicit Euler updates on reaction-diffusion equations with
zero-flux boundaries.

## Orientation Tensor

Local tissue/collagen orientation is stored at each grid position as a symmetric
2x2 tensor:

```text
A = [[A_xx, A_xy],
     [A_xy, A_yy]]
```

The dominant local orientation angle is:

```text
theta = 0.5 * atan2(2 * A_xy, A_xx - A_yy)
```

The tensor relaxes toward tangential alignment near the wound edge using the
local level-set normal and tangent. The animation draws short bidirectional line
segments whose angle comes from `theta` and whose strength is controlled by the
tensor anisotropy.

## Installation

Use Python 3.10 or newer. Install dependencies:

```bash
pip install numpy matplotlib pandas scipy pillow
```

For MP4 export, install `ffmpeg` and make sure it is on your PATH. Without
`ffmpeg`, MP4 requests fall back to GIF output.

## Run

Interactive animation:

```bash
python main.py --width 10 --length 25 --depth 4 --dt 0.05 --days 30
```

Save a GIF:

```bash
python main.py --width 10 --length 25 --depth 4 --dt 0.05 --days 30 --gif
```

Save an animation to a chosen path:

```bash
python main.py --width 10 --length 25 --depth 4 --dt 0.05 --days 30 --save-animation wound_healing.mp4
```

Use a smaller grid for quick tests:

```bash
python main.py --width 10 --length 25 --depth 4 --dt 0.05 --days 5 --nx 80 --ny 80 --gif
```

## Outputs

Each run writes `wound_results.csv` with:

- `time`
- `width`
- `length`
- `depth`
- mean values of fibrin, tPA, macrophage, TGF-beta, fibroblast, and collagen

The animation includes:

- top-view wound depth heatmap
- wound boundary contour
- local tensor-derived orientation field
- centerline depth profile
- time-series curves for width, length, depth, and scaled mean collagen

## Notes

This implementation intentionally uses a robust simplified approximation rather
than the full hybrid model in the paper. The reaction and transport terms are
kept simple, the wound front is evolved from local healing activity, and the
level set is periodically rebuilt from the wound mask using SciPy's distance
transform.
