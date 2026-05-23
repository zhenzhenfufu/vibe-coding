"""Numerical helpers for the 2D wound-healing demo model."""

from __future__ import annotations

import numpy as np


def apply_neumann_boundary(field: np.ndarray) -> np.ndarray:
    """Copy edge-adjacent values to impose zero-flux boundaries in-place."""
    field[0, :] = field[1, :]
    field[-1, :] = field[-2, :]
    field[:, 0] = field[:, 1]
    field[:, -1] = field[:, -2]
    return field


def laplacian(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Five-point Laplacian with zero-flux Neumann boundary conditions."""
    f = np.pad(field, 1, mode="edge")
    return (
        (f[1:-1, 2:] - 2.0 * f[1:-1, 1:-1] + f[1:-1, :-2]) / dx**2
        + (f[2:, 1:-1] - 2.0 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / dy**2
    )


def gradient(field: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference gradient returning d/dx and d/dy arrays."""
    d_dy, d_dx = np.gradient(field, dy, dx, edge_order=1)
    return d_dx, d_dy


def reinitialize_level_set(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Build an approximate signed-distance field from a wound mask.

    The returned field is negative inside the wound and positive outside.
    SciPy's Euclidean distance transform is used when available.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:  # pragma: no cover - only used on minimal installs
        raise RuntimeError("scipy is required for level-set reinitialization") from exc

    mask = mask.astype(bool)
    if not np.any(mask):
        return np.ones(mask.shape, dtype=float) * min(dx, dy)
    if np.all(mask):
        return -np.ones(mask.shape, dtype=float) * min(dx, dy)

    outside_distance = distance_transform_edt(~mask, sampling=(dy, dx))
    inside_distance = distance_transform_edt(mask, sampling=(dy, dx))
    return outside_distance - inside_distance


def compute_orientation_angle(
    A_xx: np.ndarray, A_xy: np.ndarray, A_yy: np.ndarray
) -> np.ndarray:
    """Dominant axis angle for a symmetric 2x2 orientation tensor."""
    return 0.5 * np.arctan2(2.0 * A_xy, A_xx - A_yy)


def compute_anisotropy(
    A_xx: np.ndarray, A_xy: np.ndarray, A_yy: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Normalized eigenvalue contrast of a symmetric 2x2 tensor."""
    trace = A_xx + A_yy
    discriminant = np.sqrt((A_xx - A_yy) ** 2 + 4.0 * A_xy**2)
    lambda1 = 0.5 * (trace + discriminant)
    lambda2 = 0.5 * (trace - discriminant)
    return np.clip((lambda1 - lambda2) / (lambda1 + lambda2 + eps), 0.0, 1.0)


def compute_geometry_from_mask(
    mask: np.ndarray, x: np.ndarray, y: np.ndarray, Z: np.ndarray
) -> tuple[float, float, float]:
    """Return wound width, length, and maximum depth from the current mask."""
    if not np.any(mask):
        return 0.0, 0.0, 0.0

    X, Y = np.meshgrid(x, y)
    width = float(Y[mask].max() - Y[mask].min())
    length = float(X[mask].max() - X[mask].min())
    depth = float(np.max(Z[mask])) if np.any(mask) else 0.0
    return width, length, depth
