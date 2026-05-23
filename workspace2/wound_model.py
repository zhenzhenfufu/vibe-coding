"""Simplified spatial wound-healing model with tensor orientation."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from utils import (
    apply_neumann_boundary,
    compute_anisotropy,
    compute_geometry_from_mask,
    compute_orientation_angle,
    gradient,
    laplacian,
    reinitialize_level_set,
)

MAX_NATURAL_HEALING_WIDTH_MM = 13.0
MAX_NATURAL_HEALING_DEPTH_MM = 6.0


@dataclass
class WoundParameters:
    D_Q: float = 0.02
    D_P: float = 0.03
    D_M: float = 0.03
    D_G: float = 0.03
    D_F: float = 0.02
    D_C: float = 0.01
    D_A: float = 0.01
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
    v0: float = 0.6
    v_D: float = 0.25
    k_align: float = 1.0
    a_t: float = 0.8
    a_n: float = 0.2
    eps: float = 1e-8


@dataclass
class SimulationConfig:
    Nx: int = 160
    Ny: int = 160
    snapshot_every: int = 5
    reinit_every: int = 0
    band_width: float | None = None
    cfl_safety: float = 0.20


@dataclass
class WoundState:
    x: np.ndarray
    y: np.ndarray
    dx: float
    dy: float
    time: float
    step: int
    psi: np.ndarray
    Z: np.ndarray
    Q: np.ndarray
    P: np.ndarray
    M: np.ndarray
    G: np.ndarray
    F: np.ndarray
    C: np.ndarray
    A_xx: np.ndarray
    A_xy: np.ndarray
    A_yy: np.ndarray


def validate_initial_wound(W0: float, L0: float, D0: float, dt: float | None = None, days: float | None = None) -> None:
    """Validate inputs for naturally healing wounds.

    The model is scoped to wounds that are small enough to plausibly heal
    without closure procedures: width <= 13 mm and depth < 6 mm.
    """
    required = {"width": W0, "length": L0, "depth": D0}
    if dt is not None:
        required["dt"] = dt
    if days is not None:
        required["days"] = days
    invalid_positive = [name for name, value in required.items() if value <= 0.0]
    if invalid_positive:
        joined = ", ".join(invalid_positive)
        raise ValueError(f"{joined} must be positive.")
    if W0 > MAX_NATURAL_HEALING_WIDTH_MM:
        raise ValueError(
            f"initial width must be <= {MAX_NATURAL_HEALING_WIDTH_MM:g} mm "
            "(1.3 cm) for this natural-healing model."
        )
    if D0 >= MAX_NATURAL_HEALING_DEPTH_MM:
        raise ValueError(
            f"initial depth must be < {MAX_NATURAL_HEALING_DEPTH_MM:g} mm "
            "(0.6 cm) for this natural-healing model."
        )


def _copy_state(state: WoundState) -> WoundState:
    arrays = {
        name: getattr(state, name).copy()
        for name in ("psi", "Z", "Q", "P", "M", "G", "F", "C", "A_xx", "A_xy", "A_yy")
    }
    return replace(state, x=state.x.copy(), y=state.y.copy(), **arrays)


def initialize_simulation(
    W0: float, L0: float, D0: float, config: SimulationConfig, params: WoundParameters
) -> WoundState:
    validate_initial_wound(W0, L0, D0)
    x = np.linspace(-1.2 * L0, 1.2 * L0, config.Nx)
    y = np.linspace(-1.2 * W0, 1.2 * W0, config.Ny)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    X, Y = np.meshgrid(x, y)

    ellipse_radius_sq = (2.0 * X / L0) ** 2 + (2.0 * Y / W0) ** 2
    psi = np.sqrt(ellipse_radius_sq) - 1.0
    mask = psi < 0.0
    Z = D0 * np.sqrt(np.maximum(0.0, 1.0 - ellipse_radius_sq))
    Z[~mask] = 0.0

    Q = mask.astype(float)
    P = np.zeros_like(Q)
    M = np.zeros_like(Q)
    G = 0.2 * mask.astype(float)
    F = np.full_like(Q, 0.05)
    C = np.zeros_like(Q)
    A_xx = np.full_like(Q, 0.5)
    A_xy = np.zeros_like(Q)
    A_yy = np.full_like(Q, 0.5)

    return WoundState(x, y, dx, dy, 0.0, 0, psi, Z, Q, P, M, G, F, C, A_xx, A_xy, A_yy)


def _stable_substep_dt(dt: float, state: WoundState, params: WoundParameters, config: SimulationConfig) -> float:
    max_diffusion = max(
        params.D_Q,
        params.D_P,
        params.D_M,
        params.D_G,
        params.D_F,
        params.D_C,
        params.D_A,
        params.eps,
    )
    diffusion_limit = config.cfl_safety / (max_diffusion * (1.0 / state.dx**2 + 1.0 / state.dy**2))
    return min(dt, diffusion_limit)


def _record_metrics(state: WoundState) -> dict[str, float]:
    mask = state.psi < 0.0
    width, length, depth = compute_geometry_from_mask(mask, state.x, state.y, state.Z)
    return {
        "time": state.time,
        "width": width,
        "length": length,
        "depth": depth,
        "mean_fibrin": float(np.mean(state.Q)),
        "mean_tPA": float(np.mean(state.P)),
        "mean_macrophage": float(np.mean(state.M)),
        "mean_TGF_beta": float(np.mean(state.G)),
        "mean_fibroblast": float(np.mean(state.F)),
        "mean_collagen": float(np.mean(state.C)),
    }


def step_simulation(
    state: WoundState, params: WoundParameters, config: SimulationConfig, dt: float
) -> WoundState:
    dx, dy, eps = state.dx, state.dy, params.eps
    Q, P, M, G, F, C = state.Q, state.P, state.M, state.G, state.F, state.C

    dQ = params.D_Q * laplacian(Q, dx, dy) - params.k_Q * P * Q
    dP = params.D_P * laplacian(P, dx, dy) + params.s_P * Q - params.delta_P * P
    dM = params.D_M * laplacian(M, dx, dy) + params.s_M * Q - params.delta_M * M
    dG = params.D_G * laplacian(G, dx, dy) + params.s_G * M + params.s_G0 * Q - params.delta_G * G
    dF = (
        params.D_F * laplacian(F, dx, dy)
        + params.r_F * F * (1.0 - F / params.K_F)
        + params.alpha_F * G / (params.K_G + G + eps)
        - params.delta_F * F
    )
    dC = (
        params.D_C * laplacian(C, dx, dy)
        + params.r_C * F * G / (params.K_G + G + eps) * (1.0 - C)
        - params.delta_C * C
    )

    Qn = np.clip(Q + dt * dQ, 0.0, 1.0)
    Pn = np.clip(P + dt * dP, 0.0, 1.0)
    Mn = np.clip(M + dt * dM, 0.0, 1.0)
    Gn = np.clip(G + dt * dG, 0.0, 1.0)
    Fn = np.clip(F + dt * dF, 0.0, 1.0)
    Cn = np.clip(C + dt * dC, 0.0, 1.0)
    for field in (Qn, Pn, Mn, Gn, Fn, Cn):
        apply_neumann_boundary(field)

    H = Fn * Cn * Gn / (params.K_G + Gn + eps)

    dpsi_dx, dpsi_dy = gradient(state.psi, dx, dy)
    grad_norm = np.sqrt(dpsi_dx**2 + dpsi_dy**2)
    n_x = dpsi_dx / (grad_norm + eps)
    n_y = dpsi_dy / (grad_norm + eps)
    tau_x = -n_y
    tau_y = n_x

    S_xx = params.a_t * tau_x * tau_x + params.a_n * n_x * n_x
    S_xy = params.a_t * tau_x * tau_y + params.a_n * n_x * n_y
    S_yy = params.a_t * tau_y * tau_y + params.a_n * n_y * n_y

    A_xx = state.A_xx + dt * (
        params.D_A * laplacian(state.A_xx, dx, dy) + params.k_align * H * (S_xx - state.A_xx)
    )
    A_xy = state.A_xy + dt * (
        params.D_A * laplacian(state.A_xy, dx, dy) + params.k_align * H * (S_xy - state.A_xy)
    )
    A_yy = state.A_yy + dt * (
        params.D_A * laplacian(state.A_yy, dx, dy) + params.k_align * H * (S_yy - state.A_yy)
    )
    for field in (A_xx, A_xy, A_yy):
        apply_neumann_boundary(field)

    trace = A_xx + A_yy
    valid = trace > eps
    A_xx[valid] /= trace[valid]
    A_xy[valid] /= trace[valid]
    A_yy[valid] /= trace[valid]
    A_xx[:] = np.clip(A_xx, 0.0, 1.0)
    A_yy[:] = np.clip(A_yy, 0.0, 1.0)
    A_xy[:] = np.clip(A_xy, -0.5, 0.5)

    band_width = config.band_width if config.band_width is not None else 2.0 * min(dx, dy)
    band = np.abs(state.psi) < band_width
    # With psi < 0 inside, adding positive speed moves the zero contour inward.
    psi = state.psi.copy()
    psi[band] = psi[band] + dt * params.v0 * H[band] * grad_norm[band]

    if config.reinit_every > 0 and (state.step + 1) % config.reinit_every == 0:
        psi = reinitialize_level_set(psi < 0.0, dx, dy)

    wound_mask = psi < 0.0
    Z = np.maximum(0.0, state.Z - dt * params.v_D * Cn * wound_mask)
    Z[~wound_mask] = 0.0

    return WoundState(
        state.x,
        state.y,
        dx,
        dy,
        state.time + dt,
        state.step + 1,
        psi,
        Z,
        Qn,
        Pn,
        Mn,
        Gn,
        Fn,
        Cn,
        A_xx,
        A_xy,
        A_yy,
    )


def simulate_wound_spatial(
    W0: float,
    L0: float,
    D0: float,
    dt: float,
    days: float,
    params: WoundParameters | None = None,
    config: SimulationConfig | None = None,
) -> tuple[list[WoundState], pd.DataFrame]:
    validate_initial_wound(W0, L0, D0, dt, days)
    params = params or WoundParameters()
    config = config or SimulationConfig()
    state = initialize_simulation(W0, L0, D0, config, params)
    sub_dt = _stable_substep_dt(dt, state, params, config)
    if sub_dt < dt:
        print(f"Using internal substep dt={sub_dt:.4g} days for explicit stability.")

    snapshots = [_copy_state(state)]
    metrics = [_record_metrics(state)]
    target_steps = int(np.ceil(days / sub_dt))

    for _ in range(target_steps):
        actual_dt = min(sub_dt, days - state.time)
        if actual_dt <= 0.0:
            break
        state = step_simulation(state, params, config, actual_dt)
        metrics.append(_record_metrics(state))
        if state.step % max(1, config.snapshot_every) == 0 or state.time >= days:
            snapshots.append(_copy_state(state))

    return snapshots, pd.DataFrame(metrics)


__all__ = [
    "SimulationConfig",
    "WoundParameters",
    "WoundState",
    "MAX_NATURAL_HEALING_DEPTH_MM",
    "MAX_NATURAL_HEALING_WIDTH_MM",
    "compute_anisotropy",
    "compute_orientation_angle",
    "initialize_simulation",
    "simulate_wound_spatial",
    "step_simulation",
    "validate_initial_wound",
]
