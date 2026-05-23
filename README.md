# Wound Healing Geometry Simulator

This project simulates and visualizes simplified wound healing over time. It is
inspired by the six-species wound-healing model from "A mathematical model of
wound healing and subsequent scarring":

https://pmc.ncbi.nlm.nih.gov/articles/PMC2839370/

The user-facing model focuses on wound geometry:

- wound width `W(t)`, in mm
- wound length `L(t)`, in mm
- wound depth `D(t)`, in mm

Hidden normalized biological variables drive the geometric closure:

- `Q`: fibrin clot / provisional matrix
- `P`: tissue plasminogen activator, tPA
- `M`: macrophage activity
- `G`: TGF-beta signal
- `F`: fibroblast activity
- `C`: collagen / granulation tissue maturity

This is a simplified research-style toy model, not a clinical prediction tool.
It should not be used for medical diagnosis, treatment decisions, or prognosis.

## Install

Create a virtual environment if desired, then install dependencies:

```bash
pip install numpy matplotlib pandas
```

`pandas` is optional for the core simulation, but it is recommended because the
program exports results directly to CSV.

To save GIF animations, install Pillow:

```bash
pip install pillow
```

To save MP4 animations, Matplotlib needs an available ffmpeg installation.

## Run

Run with command-line arguments:

```bash
python main.py --width 20 --length 40 --depth 5 --dt 0.1 --days 30
```

Save an animation:

```bash
python main.py --width 20 --length 40 --depth 5 --dt 0.1 --days 30 --save-animation wound.gif
```

Run without opening the animation window:

```bash
python main.py --width 20 --length 40 --depth 5 --dt 0.1 --days 30 --no-animation
```

If any required input is omitted, `main.py` prompts for it interactively.

## Output

The program prints the final wound dimensions and saves the full time series to
`wound_results.csv`.

The animation has two panels:

- top view: the wound shrinks as an ellipse using width and length
- side view: depth is shown as a curved basin that becomes shallower over time

## Model

The hidden variables are integrated with explicit Euler steps and clamped to the
range `[0, 1]`. Healing activity is defined as:

```text
A(t) = F(t) * C(t) * G(t) / (K_G + G(t))
```

The visible geometry evolves as:

```text
dW/dt = -2 v_W A(t)
dL/dt = -2 v_L A(t)
dD/dt = -v_D C(t)
```

Width, length, and depth are clamped at zero so the simulated wound cannot
become negative in size.
