"""Simplified wound-healing ODE model and geometry integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover - fallback for minimal environments
    pd = None


@dataclass
class WoundParameters:
    """Model parameters for the hidden biology and visible wound geometry."""

    k_Q: float = 1.0
    s_P: float = 0.8
    delta_P: float = 1.0

    s_M: float = 0.9
    delta_M: float = 0.35

    s_G: float = 0.8
    s_G0: float = 0.3
    delta_G: float = 0.5

    r_F: float = 0.45
    K_F: float = 1.0
    alpha_F: float = 0.25
    delta_F: float = 0.08

    r_C: float = 0.35
    delta_C: float = 0.01
    K_G: float = 0.3

    v_W: float = 0.35
    v_L: float = 0.25
    v_D: float = 0.25


@dataclass
class WoundState:
    """Current visible geometry and normalized hidden biological state."""

    width: float
    length: float
    depth: float
    fibrin: float = 1.0
    tPA: float = 0.0
    macrophage: float = 0.0
    TGF_beta: float = 0.2
    fibroblast: float = 0.05
    collagen: float = 0.0


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _validate_inputs(W0: float, L0: float, D0: float, dt: float, days: float) -> None:
    if W0 < 0 or L0 < 0 or D0 < 0:
        raise ValueError("Initial width, length, and depth must be non-negative.")
    if dt <= 0:
        raise ValueError("dt must be greater than 0.")
    if days < 0:
        raise ValueError("days must be non-negative.")


def _record(time: float, state: WoundState, healing_activity: float) -> dict[str, float]:
    return {
        "time": time,
        "width": state.width,
        "length": state.length,
        "depth": state.depth,
        "fibrin": state.fibrin,
        "tPA": state.tPA,
        "macrophage": state.macrophage,
        "TGF_beta": state.TGF_beta,
        "fibroblast": state.fibroblast,
        "collagen": state.collagen,
        "healing_activity": healing_activity,
    }


def simulate_wound(
    W0: float,
    L0: float,
    D0: float,
    dt: float,
    days: float,
    params: WoundParameters | None = None,
):
    """Simulate wound healing with explicit Euler integration.

    Args:
        W0: Initial wound width in mm.
        L0: Initial wound length in mm.
        D0: Initial wound depth in mm.
        dt: Time step in days.
        days: Total simulation time in days.
        params: Optional model parameter dataclass.

    Returns:
        A pandas DataFrame when pandas is installed, otherwise a list of row
        dictionaries. Columns include visible geometry, hidden biological
        variables, and healing activity.
    """

    _validate_inputs(W0, L0, D0, dt, days)
    p = params or WoundParameters()
    state = WoundState(width=float(W0), length=float(L0), depth=float(D0))

    n_steps = int(np.ceil(days / dt))
    rows: list[dict[str, float]] = []

    for step in range(n_steps + 1):
        time = min(step * dt, days)
        healing_activity = (
            state.fibroblast
            * state.collagen
            * state.TGF_beta
            / (p.K_G + state.TGF_beta)
            if p.K_G + state.TGF_beta > 0
            else 0.0
        )
        rows.append(_record(time, state, healing_activity))

        if step == n_steps:
            break

        step_dt = min(dt, days - time)
        if step_dt <= 0:
            break

        Q = state.fibrin
        P = state.tPA
        M = state.macrophage
        G = state.TGF_beta
        F = state.fibroblast
        C = state.collagen

        dQ = -p.k_Q * P * Q
        dP = p.s_P * Q - p.delta_P * P
        dM = p.s_M * Q - p.delta_M * M
        dG = p.s_G * M + p.s_G0 * Q - p.delta_G * G
        dF = (
            p.r_F * F * (1.0 - F / p.K_F)
            + p.alpha_F * G / (p.K_G + G)
            - p.delta_F * F
        )
        dC = p.r_C * F * G / (p.K_G + G) * (1.0 - C) - p.delta_C * C

        state = WoundState(
            width=max(0.0, state.width - 2.0 * p.v_W * healing_activity * step_dt),
            length=max(0.0, state.length - 2.0 * p.v_L * healing_activity * step_dt),
            depth=max(0.0, state.depth - p.v_D * C * step_dt),
            fibrin=_clamp01(Q + step_dt * dQ),
            tPA=_clamp01(P + step_dt * dP),
            macrophage=_clamp01(M + step_dt * dM),
            TGF_beta=_clamp01(G + step_dt * dG),
            fibroblast=_clamp01(F + step_dt * dF),
            collagen=_clamp01(C + step_dt * dC),
        )

    if pd is not None:
        return pd.DataFrame(rows)
    return rows
