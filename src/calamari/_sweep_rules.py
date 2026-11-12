"""Backend-free numeric rules shared by sweep consumers; callers own selection policy."""

from numbers import Real

import numpy as np

# Existing tolerance distinguishing small setpoint jitter from a retraced sweep span.
BACKTRACK_FRACTION = 0.01


def frequency_clusters(values: np.ndarray, *, rtol: float) -> list[float]:
    """Cluster prepared frequencies relative to each cluster's first value."""
    clusters: list[float] = []
    for value in sorted(float(v) for v in np.unique(values)):
        if not clusters or value > clusters[-1] * (1 + rtol):
            clusters.append(value)
    return clusters


def sweep_backtracking(x: np.ndarray) -> tuple[float, float]:
    """Return total span and travel against the endpoint direction (at least two points)."""
    span = float(np.max(x) - np.min(x))
    direction = 1.0 if x[-1] >= x[0] else -1.0
    steps = np.diff(x) * direction
    return span, float(-steps[steps < 0].sum())


def validate_frequency(value, error_type) -> None:
    """Require a finite positive real frequency request."""
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0):
        raise error_type(f"frequency must be positive and finite, got {value!r}")


def select_frequency(clusters, requested, *, rtol, error_type) -> float:
    """Resolve exactly one recorded cluster using a purely relative tolerance."""
    validate_frequency(requested, error_type)
    matches = [c for c in clusters if abs(requested - c) <= rtol * abs(c)]
    if len(matches) != 1:
        available = ", ".join(f"{c:g}" for c in clusters) or "(none)"
        reason = "matches none" if not matches else "ambiguously matches several"
        raise error_type(
            f"frequency={requested:g} Hz {reason} of the recorded frequencies "
            f"({available} Hz); choose exactly one within {100 * rtol:g}%."
        )
    return matches[0]
