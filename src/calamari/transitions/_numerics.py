"""Array preparation, smoothing, noise estimation and candidate detection."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from numbers import Real

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_coeffs, savgol_filter

from calamari._provenance import calamari_version as _calamari_version
from calamari._sweep_rules import BACKTRACK_FRACTION as _BACKTRACK_FRACTION
from calamari._sweep_rules import sweep_backtracking as _sweep_backtracking
from calamari.exceptions import CurveInputError, DetectionInputError
from calamari.transitions._metadata import (
    _SECOND_DIFF_TO_SIGMA,
    _VALID_NOISE_STRINGS,
    CRITERION_PRIORITY,
    DEFAULT_PROMINENCE_FRACTION,
    NOISE_RATIO_WARN,
)
from calamari.transitions._results import TransitionResult


def _check_monotonic(x: np.ndarray, label: str) -> None:
    """Refuse a non-monotonic swept axis rather than sorting two branches
    together. Shared backtracking mathematics live in ``_sweep_rules``;
    this boundary owns the detection-specific error and remedy."""
    if len(x) < 2:
        return
    span, backtrack = _sweep_backtracking(x)
    if span == 0.0:
        raise CurveInputError(f"{label}: the swept axis does not vary, so there is nothing to search.")
    if backtrack > _BACKTRACK_FRACTION * span:
        raise CurveInputError(
            f"{label}: the swept axis is not monotonic (it backtracks by "
            f"{backtrack:.4g} over a span of {span:.4g}) -- a hysteresis loop or "
            "re-swept range. Detecting on it would mix branches that are "
            "different measurements. Take one branch first "
            "(calamari.sweeps.increasing_field_branch, or pass "
            "branch='increasing' to detect_transitions), or aggregate repeat "
            "sweeps with calamari.repeats before detecting."
        )


#: Fraction of a sweep's points allowed to duplicate an earlier swept-axis
#: value before the sweep is refused as a repeats file. A genuine
#: repeat-measurement file (2-3 readings per setpoint, calamari.repeats
#: territory) is ~50%+ duplicates and must be aggregated, not searched; an
#: isolated duplicate (a settle re-read at the same reported field, a
#: segment join) is routine in real sweeps and refusing a whole sweep over
#: one pair would be obstructive. 5% sits between the regimes. Below it,
#: duplicates are dropped keep-first -- NOT silently: the count is
#: recorded in ``params['n_duplicates_dropped']``.
_MAX_DUPLICATE_FRACTION = 0.05


def _prepare(
    x: np.ndarray, y: np.ndarray, y_err: np.ndarray | None, label: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    """Finite-mask, monotonicity-check (on the sweep's real order, before
    any sort) and orient ascending. Returns strictly increasing x plus the
    number of duplicate-x points dropped (see
    :data:`_MAX_DUPLICATE_FRACTION`) for the params record."""
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if y_err is not None:
        y_err = np.asarray(y_err, dtype=float)[finite]
    if len(x) < 5:
        raise CurveInputError(
            f"{label}: only {len(x)} finite points -- too few to search for "
            "transitions (at least 5 are needed for smoothing and grouping)."
        )
    _check_monotonic(x, label)
    if x[-1] < x[0]:
        x, y = x[::-1], y[::-1]
        if y_err is not None:
            y_err = y_err[::-1]
    # Any residual non-increase is setpoint jitter (the monotonic check
    # passed); a stable sort plus strict-increase keep makes the axis usable.
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    if y_err is not None:
        y_err = y_err[order]
    keep = np.ones(len(x), dtype=bool)
    keep[1:] = np.diff(x) > 0
    n_dropped = int((~keep).sum())
    if len(x) - n_dropped < (1 - _MAX_DUPLICATE_FRACTION) * len(x):
        raise CurveInputError(
            f"{label}: {n_dropped} of {len(x)} points repeat the same "
            "swept-axis value -- systematically repeated setpoints. Aggregate "
            "them with calamari.repeats before detecting, so their scatter "
            "informs the noise estimate instead of being discarded."
        )
    if y_err is not None:
        y_err = y_err[keep]
    return x[keep], y[keep], y_err, n_dropped


def _resample(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Linear resample onto a uniform grid at the median sample spacing.

    Savitzky-Golay assumes uniform spacing. The median spacing (rather
    than span/n) keeps a window-in-points meaning what the physical window
    says wherever sampling is typical; across a sparse interior region the
    interpolation *manufactures* values -- documented, and the grid spacing
    is recorded in ``params`` so the manufactured resolution is auditable.
    """
    dx = float(np.median(np.diff(x)))
    n = int(round((x[-1] - x[0]) / dx)) + 1
    n = max(len(x) // 2, min(n, 4 * len(x)), 5)  # bound cost and degeneracy
    xu = np.linspace(x[0], x[-1], n)
    return xu, np.interp(xu, x, y), float(xu[1] - xu[0])


def _check_smooth_window(smooth_window):
    """A physical smoothing window is None or a finite positive number."""
    if smooth_window is not None and (
        isinstance(smooth_window, bool) or not isinstance(smooth_window, Real)
        or not np.isfinite(smooth_window) or smooth_window <= 0
    ):
        raise CurveInputError(f"smooth_window must be positive and finite, got {smooth_window!r}")


def _window_points(smooth_window: float | None, dx: float, n: int) -> int:
    """Physical smoothing window -> a valid odd point count on the grid."""
    _check_smooth_window(smooth_window)
    if smooth_window is not None:
        pts = int(round(smooth_window / dx))
    else:
        pts = 11  # Default window when no physical width is supplied.
    pts = max(5, min(pts, n if n % 2 == 1 else n - 1))
    if pts % 2 == 0:
        pts += 1
    return pts


def _kernel_norm(window_pts: int, polyorder: int, deriv: int, dx: float) -> float:
    """L2 norm of the Savitzky-Golay impulse response: uncorrelated input
    noise sigma becomes sigma * norm on the filtered signal, which is what
    lets prominence and noise be compared on the SAME representation."""
    coeffs = savgol_coeffs(window_pts, polyorder, deriv=deriv, delta=dx)
    return float(np.sqrt(np.sum(coeffs**2)))


def _instrument_sigma(y_err: np.ndarray | None) -> float | None:
    """Median of the instrument's usable per-point errors, or None when
    the column does not carry at least 3 finite, positive values -- the
    usability floor below which an error column tells us nothing."""
    if y_err is None:
        return None
    finite = y_err[np.isfinite(y_err)]
    finite = finite[finite > 0]
    if finite.size < 3:
        return None
    return float(np.median(finite))


def _scatter_sigma(y: np.ndarray) -> float:
    """Corrected second-difference estimate of the point-to-point scatter
    sigma (0.0 when there are too few finite points to estimate it)."""
    finite = y[np.isfinite(y)]
    if finite.size < 5:
        return 0.0
    return float(np.median(np.abs(np.diff(finite, n=2)))) / _SECOND_DIFF_TO_SIGMA


def _check_noise(noise):
    """Validate noise configuration without requiring an instrument error column."""
    if isinstance(noise, str):
        if noise not in _VALID_NOISE_STRINGS:
            raise DetectionInputError(
                f"noise must be one of {_VALID_NOISE_STRINGS} or a positive float, got {noise!r}"
            )
    elif isinstance(noise, bool):
        raise DetectionInputError(f"noise must be a positive finite float, got {noise!r}")
    else:
        try:
            value = float(noise)
        except (TypeError, ValueError):
            raise DetectionInputError(
                f"noise must be one of {_VALID_NOISE_STRINGS} or a positive float, got {noise!r}"
            ) from None
        if not math.isfinite(value) or value <= 0:
            raise DetectionInputError(f"noise must be a positive finite float, got {noise!r}")


def _resolve_noise(
    y: np.ndarray, y_err: np.ndarray | None, noise: str | float, label: str
) -> dict:
    """Resolve ``noise=`` into the sigma to use, plus its full provenance.

    One function rather than inlining the branches in :func:`find_transitions`
    so the same resolution -- and the same warning -- cannot drift out of
    step between the array API and the Measurement API (both funnel through
    here). Returns the keys that land directly in ``params``: ``noise``,
    ``noise_source`` (``'instrument'``/``'scatter'``/``'caller'``),
    ``noise_sigma``, ``noise_sigma_instrument`` (float or None),
    ``noise_sigma_scatter`` (float), ``noise_ratio`` (instrument/scatter,
    or None when either side is unavailable or the scatter is zero).

    Raises
    ------
    ValueError
        ``noise='instrument'`` with no usable error column; an unrecognised
        string; or a non-positive/non-finite number.
    """
    _check_noise(noise)
    sigma_instrument = _instrument_sigma(y_err)
    sigma_scatter = _scatter_sigma(y)
    ratio = (
        sigma_instrument / sigma_scatter
        if sigma_instrument is not None and sigma_scatter > 0
        else None
    )

    if isinstance(noise, str):
        if noise == "instrument":
            if sigma_instrument is None:
                raise CurveInputError(
                    f"{label}: noise='instrument' was requested, but this sweep "
                    "carries no usable per-point errors (fewer than 3 finite, "
                    "positive values). Use noise='scatter' to fall back to the "
                    "second-difference estimate."
                )
            sigma, source = sigma_instrument, "instrument"
        elif noise == "scatter":
            sigma, source = sigma_scatter, "scatter"
        elif noise == "auto":
            if sigma_instrument is not None:
                sigma, source = sigma_instrument, "instrument"
                if ratio is not None and ratio > NOISE_RATIO_WARN:
                    warnings.warn(
                        f"{label}: the instrument's per-point error column gives "
                        f"noise sigma {sigma_instrument:.3g}, {ratio:.3g}x the "
                        f"second-difference scatter estimate ({sigma_scatter:.3g}). "
                        f"An error column more than {NOISE_RATIO_WARN:g}x away from "
                        "the measured scatter is not describing the same noise, so "
                        "candidates on this channel are being scored against a "
                        "sigma that looks wrong. Use noise='scatter' to score "
                        "against the scatter estimate instead.",
                        stacklevel=3,
                    )
            else:
                sigma, source = sigma_scatter, "scatter"
    else:
        sigma, source = float(noise), "caller"

    return {
        "noise": noise,
        "noise_source": source,
        "noise_sigma": sigma,
        "noise_sigma_instrument": sigma_instrument,
        "noise_sigma_scatter": sigma_scatter,
        "noise_ratio": ratio,
    }


def _extreme_value_factor(n: int, window_pts: int) -> float:
    """Extreme-value-inspired heuristic for prominence normalisation.

    The formula uses m = max(2, n // window_pts) effective elements and a
    peak-to-peak factor of two. It does not establish a null distribution
    for smoothed, correlated or selected extrema; calibrate the complete
    workflow before assigning a false-alarm probability.
    """
    m = max(2, n // max(window_pts, 1))
    return 2.0 * math.sqrt(2.0 * math.log(m))


def _vertex(
    xu: np.ndarray, s: np.ndarray, i: int, half: int, sigma_rep: float
) -> tuple[float, float]:
    """Sub-grid extremum location and its standard error from a local
    quadratic fit.

    The SE is the fitted vertex's, propagated from ``sigma_rep`` (noise of
    the representation the extremum was found in) -- it responds to noise,
    unlike the parabola half-width, which is a feature width. Smoothing
    correlates neighbouring residuals, so this is somewhat optimistic;
    documented rather than corrected. Floored at the grid spacing.
    """
    dx = xu[1] - xu[0]
    lo, hi = max(0, i - half), min(len(xu), i + half + 1)
    xs, ys = xu[lo:hi] - xu[i], s[lo:hi]
    if len(xs) < 4:
        return float(xu[i]), float(dx)
    design = np.vander(xs, 3)  # columns x^2, x, 1
    beta, *_ = np.linalg.lstsq(design, ys, rcond=None)
    a, b = beta[0], beta[1]
    if a == 0.0:
        return float(xu[i]), float(xu[hi - 1] - xu[lo])
    x0 = -b / (2.0 * a)
    if not (xs[0] <= x0 <= xs[-1]):  # fit did not describe a local extremum here
        return float(xu[i]), float(dx)
    cov = np.linalg.inv(design.T @ design) * sigma_rep**2
    grad = np.array([b / (2.0 * a**2), -1.0 / (2.0 * a), 0.0])
    se = float(np.sqrt(max(grad @ cov @ grad, 0.0)))
    return float(xu[i] + x0), max(se, float(dx))


def _line_fit(
    xs: np.ndarray, ys: np.ndarray, sigma: float
) -> tuple[float, float, np.ndarray]:
    """OLS line fit returning (slope, intercept, 2x2 covariance)."""
    design = np.vander(xs, 2)
    beta, *_ = np.linalg.lstsq(design, ys, rcond=None)
    cov = np.linalg.inv(design.T @ design) * sigma**2
    return float(beta[0]), float(beta[1]), cov


def smooth_derivative(
    x: Sequence[float],
    y: Sequence[float],
    *,
    order: int = 1,
    smooth_window: float | None = None,
    polyorder: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Noise-robust Savitzky-Golay derivative of ``y`` with respect to ``x``.

    Kept public because downstream analysis reuses it directly. The sweep
    is resampled onto a uniform grid at its own median spacing (see
    :func:`_resample` for what that manufactures across sparse regions);
    ``smooth_window`` is in x-units, not points.

    Returns ``(x_uniform, y_smooth, d)`` where ``d`` is the ``order``-th
    derivative (``order=0`` returns the smoothed signal twice).

    Raises
    ------
    CurveInputError
        Mismatched shapes, insufficient finite points, a non-monotonic or
        systematically repeated swept axis, or a non-positive smoothing
        window. Also a ``ValueError``. NumPy/SciPy exceptions retain their types.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape:
        raise CurveInputError(f"x and y must have the same shape, got {x_arr.shape} vs {y_arr.shape}")
    x_arr, y_arr, _, _ = _prepare(x_arr, y_arr, None, "smooth_derivative")
    xu, yu, dx = _resample(x_arr, y_arr)
    pts = _window_points(smooth_window, dx, len(xu))
    poly = min(polyorder, pts - 1)
    y_smooth = savgol_filter(yu, window_length=pts, polyorder=poly, deriv=0)
    if order == 0:
        return xu, y_smooth, y_smooth
    d = savgol_filter(yu, window_length=pts, polyorder=poly, deriv=order, delta=dx)
    return xu, y_smooth, d


# ---------------------------------------------------------------------------
# Detectors. Each returns a list of detection dicts on the uniform grid:
# {method, feature, sign, x, x_err, raw_strength, significance, index,
#  edge, group_lo, group_hi (peak, derivative and kink), parent (onset only)}.
# ---------------------------------------------------------------------------


def _detect_peak(xu, y_smooth, sigma0, evf, prominence_fraction, window_pts):
    hits = []
    ptp = float(np.nanmax(y_smooth) - np.nanmin(y_smooth))
    scale = float(np.nanmax(np.abs(y_smooth))) if len(y_smooth) else 0.0
    if ptp <= max(1e-12, 1e-9 * scale):  # flat to round-off: no features
        return hits
    floor = prominence_fraction * ptp
    half = window_pts // 2
    # The edge zone is the terminal HALF-window: those are the points
    # savgol_filter's boundary polynomial refits, and the region where the
    # vertex fit (half a window each side) runs out of room. Flagging a
    # full window would drop genuinely interior candidates.
    for sign_label, signed in (("max", y_smooth), ("min", -y_smooth)):
        idx, props = find_peaks(signed, prominence=floor)
        for k, i in enumerate(idx):
            prom = float(props["prominences"][k])
            x0, se = _vertex(xu, signed, int(i), half, sigma0)
            hits.append(
                {
                    "method": "peak",
                    "feature": "extremum",
                    "sign": sign_label,
                    "x": x0,
                    "x_err": se,
                    "raw_strength": prom,
                    "significance": prom / (sigma0 * evf) if sigma0 > 0 else np.inf,
                    "index": int(i),
                    "edge": i < half or i >= len(xu) - half,
                    "group_lo": int(props["left_bases"][k]),
                    "group_hi": int(props["right_bases"][k]),
                }
            )
    return hits


def _detect_derivative(
    xu,
    dydx,
    sigma1,
    evf,
    prominence_fraction,
    window_pts,
    *,
    method="derivative",
    feature="step",
    signs=("rise", "fall"),
):
    """Extrema of dy/dx, with their prominence bases kept.

    ``group_lo``/``group_hi`` are the prominence bases, which grouping
    uses to host a cusp's derivative flanks. ``contour_lo``/``contour_hi``
    are the extremum's footprint at its own prominence reference level --
    where dy/dx first falls back to the higher of the two base values on
    each side (``scipy.signal.peak_widths`` at ``rel_height=1``) -- and
    they are what decides whether a curvature extremum is this step's
    shoulder or a slope change of its own (rule 3 of
    :func:`_group_detections`). The footprint lies inside the base
    interval, and on the side of the higher base it ends exactly at that
    base, but on the other side it can be much shorter. That difference
    is the point: a weak dy/dx bump on a slope plateau has a base
    interval reaching past wherever dy/dx next drops, while its footprint
    is the bump itself. Both are exact geometry, not tolerances.

    ``method``, ``feature`` and ``signs`` (the labels for a maximum and a
    minimum of ``dydx``) exist so the Fisher criterion reuses exactly this
    extremum search on d(yT)/dT under its own vocabulary: there a maximum
    is the specific-heat anomaly, not a rising step, and a shared label
    would say otherwise.
    """
    hits = []
    ptp = float(np.nanmax(dydx) - np.nanmin(dydx))
    scale = float(np.nanmax(np.abs(dydx))) if len(dydx) else 0.0
    if ptp <= max(1e-12, 1e-9 * scale):
        return hits
    floor = prominence_fraction * ptp
    half = window_pts // 2
    for sign_label, signed in ((signs[0], dydx), (signs[1], -dydx)):
        idx, props = find_peaks(signed, prominence=floor)
        if len(idx):
            # Interpolated sample positions where each extremum's footprint at
            # its prominence reference level (rel_height=1) begins and ends.
            _, _, ips_lo, ips_hi = peak_widths(
                signed,
                idx,
                rel_height=1.0,
                prominence_data=(props["prominences"], props["left_bases"], props["right_bases"]),
            )
        for k, i in enumerate(idx):
            prom = float(props["prominences"][k])
            x0, se = _vertex(xu, signed, int(i), half, sigma1)
            hits.append(
                {
                    "method": method,
                    "feature": feature,
                    "sign": sign_label,
                    "x": x0,
                    "x_err": se,
                    "raw_strength": prom,
                    "significance": prom / (sigma1 * evf) if sigma1 > 0 else np.inf,
                    "index": int(i),
                    "edge": i < half or i >= len(xu) - half,
                    "group_lo": int(props["left_bases"][k]),
                    "group_hi": int(props["right_bases"][k]),
                    "contour_lo": float(np.interp(ips_lo[k], np.arange(len(xu)), xu)),
                    "contour_hi": float(np.interp(ips_hi[k], np.arange(len(xu)), xu)),
                }
            )
    return hits


def _detect_kink(xu, d2ydx2, sigma2, evf, prominence_fraction, window_pts):
    """Curvature extrema: a slope change reported at its centre.

    Mirrors :func:`_detect_peak` on the second-derivative representation.
    A one-sided kink -- a change of slope rather than a cusp -- is a
    *step* in dy/dx, so the derivative detector finds it on one of the
    step's flanks; d²y/dx² has its extremum at the slope change itself.
    ``'up'`` is a d² maximum (the slope increases through the kink),
    ``'down'`` a d² minimum (the slope decreases). The prominence floor,
    the extreme-value correction and the edge zone are exactly those of
    the other detectors, applied to this representation's own noise
    (``sigma2``), so the significance stays comparable across methods.

    ``group_lo``/``group_hi`` (the prominence bases) are recorded for
    symmetry with the other detectors and to keep a hit's own extent
    auditable. Grouping does NOT host anything inside them: a curvature
    extremum's base interval on a cusp can span most of a sweep, and
    hosting on it was exactly the defect that let a cusp swallow a
    distant slope change (see :func:`_group_detections`).
    """
    hits = []
    ptp = float(np.nanmax(d2ydx2) - np.nanmin(d2ydx2))
    scale = float(np.nanmax(np.abs(d2ydx2))) if len(d2ydx2) else 0.0
    if ptp <= max(1e-12, 1e-9 * scale):  # flat to round-off: no curvature features
        return hits
    floor = prominence_fraction * ptp
    half = window_pts // 2
    for sign_label, signed in (("up", d2ydx2), ("down", -d2ydx2)):
        idx, props = find_peaks(signed, prominence=floor)
        for k, i in enumerate(idx):
            prom = float(props["prominences"][k])
            x0, se = _vertex(xu, signed, int(i), half, sigma2)
            hits.append(
                {
                    "method": "kink",
                    "feature": "kink",
                    "sign": sign_label,
                    "x": x0,
                    "x_err": se,
                    "raw_strength": prom,
                    "significance": prom / (sigma2 * evf) if sigma2 > 0 else np.inf,
                    "index": int(i),
                    "edge": i < half or i >= len(xu) - half,
                    "group_lo": int(props["left_bases"][k]),
                    "group_hi": int(props["right_bases"][k]),
                }
            )
    return hits


def _detect_onset(xu, y_smooth, dydx, derivative_hits, sigma0, window_pts):
    """Two-tangent onsets, one per derivative extremum where the geometry
    supports it: baseline line (the flatter adjacent side) intersected
    with the steepest-slope tangent. The onset inherits its parent's
    significance -- it is the same feature located by a different, cited
    convention, not independent evidence."""
    hits = []
    half = max(window_pts // 2, 2)
    dx = xu[1] - xu[0]
    n = len(xu)
    for parent_k, hit in enumerate(derivative_hits):
        i = hit["index"]
        s_lo, s_hi = max(0, i - half), min(n, i + half + 1)
        left_lo, left_hi = max(0, i - 3 * half), max(0, i - half)
        right_lo, right_hi = min(n, i + half + 1), min(n, i + 3 * half + 1)
        if left_hi - left_lo < 3 and right_hi - right_lo < 3:
            continue
        mean_abs = lambda lo, hi: (  # noqa: E731 - tiny local helper
            float(np.mean(np.abs(dydx[lo:hi]))) if hi - lo >= 3 else np.inf
        )
        if mean_abs(left_lo, left_hi) <= mean_abs(right_lo, right_hi):
            b_lo, b_hi = left_lo, left_hi
        else:
            b_lo, b_hi = right_lo, right_hi
        if b_hi - b_lo < 3:
            continue
        m1, c1, cov1 = _line_fit(xu[b_lo:b_hi], y_smooth[b_lo:b_hi], sigma0)
        m2, c2, cov2 = _line_fit(xu[s_lo:s_hi], y_smooth[s_lo:s_hi], sigma0)
        if m2 == m1:
            continue
        x_star = (c1 - c2) / (m2 - m1)
        if not (xu[i] - 3 * half * dx <= x_star <= xu[i] + 3 * half * dx):
            continue  # unstable intersection: the shape is not baseline-plus-rise
        # Delta-method SE from the two independent fits' covariances.
        dm = m2 - m1
        g1 = np.array([x_star / dm, 1.0 / dm])  # d/dm1, d/dc1
        g2 = np.array([-x_star / dm, -1.0 / dm])  # d/dm2, d/dc2
        var = float(g1 @ cov1 @ g1 + g2 @ cov2 @ g2)
        hits.append(
            {
                "method": "onset",
                "feature": "onset",
                "sign": hit["sign"],
                "x": float(x_star),
                "x_err": max(math.sqrt(max(var, 0.0)), dx),
                "raw_strength": abs(m2 - m1),
                "significance": hit["significance"],
                "index": int(np.clip(round((x_star - xu[0]) / dx), 0, n - 1)),
                "edge": hit["edge"],
                "parent": parent_k,
            }
        )
    return hits


# ---------------------------------------------------------------------------
# Grouping: one physical feature -> one candidate row.
# ---------------------------------------------------------------------------


#: Which curvature sign is a rounded step's shoulder on each side of its
#: derivative extremum. A rising step curves upwards as it leaves the
#: lower baseline ('up' below the midpoint) and back down as it joins the
#: upper one ('down' above); a falling step is the mirror image. Written
#: down as a table because it is the whole of the step rule: a curvature
#: extremum on the wrong side for its sign is not this step's shoulder.
_STEP_SHOULDER_SIGN = {
    ("rise", "below"): "up",
    ("rise", "above"): "down",
    ("fall", "below"): "down",
    ("fall", "above"): "up",
}


def _group_detections(hits: list[dict], xu: np.ndarray, priority: tuple[str, ...]) -> list[dict]:
    """Group detections that belong to one physical feature.

    Grouping deduplicates geometric companions, not independent evidence:
    a peak's derivative flanks are geometric companions
    (at +-sigma for a Gaussian), so agreement between them carries no
    independent evidence. Every rule below is exact geometry -- a
    prominence base interval, a side, a nearest neighbour -- and never a
    distance tolerance (which would have to be tuned, and would merge
    genuinely separate features on a dense sweep) and never a comparison
    of confidences across methods.

    The shapes the rules encode
    ---------------------------
    Under Savitzky-Golay smoothing:

    * a *cusp* has a derivative rise below it and a fall above it (mirrored
      for a minimum), a curvature extremum AT the cusp (a minimum for a
      maximum) and curvature extrema of the opposite sign on its flanks,
      just beyond the derivative extrema;
    * a rounded *step* has ONE derivative extremum, at its midpoint, and
      two curvature shoulders of opposite sign either side of it (see
      :data:`_STEP_SHOULDER_SIGN`);
    * a one-sided *kink* has one curvature extremum and no genuine
      derivative extremum at all -- dy/dx steps through a sigmoid there,
      so any derivative hit near it is noise or a curvature artefact.

    In order
    --------
    1. Peaks each start a group. A derivative extremum joins a peak by the
       side rule: within that peak's base interval, a rise below and a
       fall above a maximum (mirrored for a minimum); nearest such peak.
    2. Kink companions of a peak. A kink joins a peak's group when it lies
       within that peak's base interval AND either (a) lies between the
       peak's grouped derivative flanks -- if only one flank exists,
       on the peak's side of it; if the peak has no grouped flank at all,
       the nearest kink to the peak on each side of it -- or (b) is the
       FIRST kink beyond a grouped flank on that side, which is where a
       cusp's own flanking curvature extremum sits. Any further kink out
       there is a separate feature. Nearest peak wins if several qualify.
    3. Step shoulders. A derivative extremum NOT claimed by a peak claims,
       on each side, a still-unclaimed kink of the matching shoulder sign
       that is (a) ADJACENT to it -- no other detected extremum of y,
       dy/dx or d²y/dx² lies between them -- and (b) inside its
       FOOTPRINT, where dy/dx stands above the level the extremum's
       prominence is measured from (see :func:`_detect_derivative`). It
       forms a step group only if it finds a shoulder on BOTH sides, since
       one shoulder is not a step. Claims are resolved by kink (each kink
       joins the nearest qualifying derivative extremum). The footprint
       limits the claim to the bump; adjacency prevents it from reaching
       past other detected extrema. Neither is a distance tolerance: a true
       step's shoulders are the curvature extrema either side of its
       midpoint, with the midpoint's own dy/dx standing above its
       reference level all the way out to them.
    4. Nothing else absorbs a kink: a derivative extremum never joins a
       kink's group, and a kink claimed by nobody stands alone as its own
       candidate.
    5. An onset joins its parent derivative's group. With ``'onset'``
       searched but not ``'derivative'``, that parent is not among the
       reported hits, so the onset starts its own group -- an onset row
       without the derivative row it was built from.

    The reported coordinate comes from the highest-*priority* member (see
    :data:`CRITERION_PRIORITY`), never from comparing confidences across
    methods. Significance breaks ties only within one method, and a kink
    is eligible to report only where it is the group's own slope change: a
    lone kink (rule 4), or a cusp's central curvature extremum -- sign
    ``'down'`` under a maximum, ``'up'`` over a minimum, between the
    flanks of rule 2(a), the most significant if there are several.
    A cusp's flank companions and a step's shoulders never report; with
    ``criterion='kink'`` such a group falls through to the next method in
    the order, which is exactly what it reports under the default. Without
    that rule the most significant kink reported, and on a cusp whose feet
    are sharper than its top that is a flank: off the cusp, and with the
    opposite sign.
    """
    peaks = [h for h in hits if h["method"] == "peak"]
    kinks = [h for h in hits if h["method"] == "kink"]
    derivs = [h for h in hits if h["method"] == "derivative"]
    onsets = [h for h in hits if h["method"] == "onset"]

    # Peak i is group i; everything ungrouped appends its own group after
    # them, which is what lets deriv_group index into `groups`.
    groups: list[list[dict]] = [[p] for p in peaks]

    def contains(host: dict, hit: dict) -> bool:
        """Does ``hit`` fall inside ``host``'s prominence base interval?"""
        return xu[host["group_lo"]] <= hit["x"] <= xu[host["group_hi"]]

    def in_footprint(d: dict, hit: dict) -> bool:
        """Does ``hit`` lie where derivative extremum ``d`` stands above the
        level its prominence is measured from (see :func:`_detect_derivative`)?"""
        return d["contour_lo"] <= hit["x"] <= d["contour_hi"]

    # Every detected extremum of y, dy/dx and d²y/dx² (onsets are constructions
    # on a derivative extremum, not extrema, so they do not count).
    extrema_x = sorted(h["x"] for h in (*peaks, *derivs, *kinks))

    def adjacent(a: dict, b: dict) -> bool:
        """Is no other detected extremum strictly between ``a`` and ``b``?"""
        lo, hi = sorted((a["x"], b["x"]))
        return not any(lo < x < hi for x in extrema_x)

    # 1. Derivative flanks of a peak, by the side rule.
    deriv_group: dict[int, int] = {}
    for j, d in enumerate(derivs):
        matches = []
        for gi, p in enumerate(peaks):
            if not contains(p, d):
                continue
            before = d["x"] < p["x"]
            wants_rise_before = p["sign"] == "max"
            side_ok = (d["sign"] == "rise") == (before == wants_rise_before)
            if side_ok:
                matches.append(gi)
        if matches:
            gi = min(matches, key=lambda g: abs(peaks[g]["x"] - d["x"]))
            groups[gi].append(d)
            deriv_group[j] = gi

    def flanks_of(gi: int) -> tuple[dict | None, dict | None]:
        """The peak's grouped derivative flanks, nearest one per side.

        Nearest rather than outermost: the flank of the cusp itself is the
        derivative extremum closest to it, and it is that flank -- not some
        further hit that also passed the side rule -- which bounds the
        cusp's own curvature companions.
        """
        p = peaks[gi]
        mine = [d for j, d in enumerate(derivs) if deriv_group.get(j) == gi]
        below = [d for d in mine if d["x"] < p["x"]]
        above = [d for d in mine if d["x"] >= p["x"]]
        lo = min(below, key=lambda d: p["x"] - d["x"]) if below else None
        hi = min(above, key=lambda d: d["x"] - p["x"]) if above else None
        return lo, hi

    def own_kink_of(gi: int, members: list[dict]) -> dict | None:
        """The one kink in peak group ``gi`` that is the cusp's own slope change.

        A cusp group also holds its flank companions -- curvature extrema of
        the opposite sign, just beyond the derivative flanks -- and on a
        cusp with sharp feet and a rounded top those are the *stronger*
        curvature extrema. Letting the most significant kink report under
        ``criterion='kink'`` therefore moved the coordinate to a flank, with
        the flank's sign. The cusp's own slope change is the curvature
        extremum of the central sign (``'down'`` under a maximum, ``'up'``
        over a minimum) lying between the flanks, by exactly the
        between-the-flanks test of rule 2(a) on ``flanks_of``, so the
        two cannot drift apart: with one flank, on the peak's side of it;
        with none, any central-sign member. Several such kinks go to the
        most significant, then the nearest to the peak, then the lowest
        ``x``. Significance first, because these are all one method's
        curvature extrema and the comparison is commensurable; nearest
        first would let a noise wiggle hugging the maximum outrank the
        slope change itself, and pull the reported kink onto the peak --
        the very drift off the slope change that ``criterion='kink'`` is
        chosen to escape. ``None`` when the group has no such kink, so the
        peak reports.
        """
        p = peaks[gi]
        lo, hi = flanks_of(gi)
        central = "down" if p["sign"] == "max" else "up"
        own = [
            m
            for m in members
            if m["method"] == "kink"
            and m["sign"] == central
            and (lo is None or m["x"] >= lo["x"])
            and (hi is None or m["x"] <= hi["x"])
        ]
        if not own:
            return None
        return min(own, key=lambda k: (-k["significance"], abs(k["x"] - p["x"]), k["x"]))

    def may_report(gi: int, members: list[dict]) -> list[dict]:
        """The members allowed to supply group ``gi``'s coordinate.

        Every member except a kink that is not the group's own slope
        change. A kink reports only a lone slope change (no peak and no
        derivative in its group, rule 4) or a cusp's central curvature
        extremum (``own_kink_of``). A step's kinks are its two
        shoulders, half a window either side of the midpoint, and a cusp's
        other kinks are its flanks: neither is where the feature is. Under
        the default priority the kink is last and every group holding an
        ineligible kink also holds a peak or a derivative, so this changes
        nothing there; it matters only when ``criterion='kink'`` promotes
        the kink, and then a group without a kink of its own reports
        exactly what it would under the default order.
        """
        if gi < len(peaks):
            own = own_kink_of(gi, members)
            return [m for m in members if m["method"] != "kink" or m is own]
        if any(m["method"] == "derivative" for m in members):
            return [m for m in members if m["method"] != "kink"]
        return members

    def companions_of(gi: int) -> set[int]:
        """Indices of the kinks that are geometric companions of peak ``gi``."""
        p = peaks[gi]
        lo, hi = flanks_of(gi)
        companions: set[int] = set()
        if lo is None and hi is None:
            below = [kj for kj, k in enumerate(kinks) if k["x"] < p["x"]]
            above = [kj for kj, k in enumerate(kinks) if k["x"] >= p["x"]]
            if below:
                companions.add(max(below, key=lambda kj: kinks[kj]["x"]))
            if above:
                companions.add(min(above, key=lambda kj: kinks[kj]["x"]))
            return companions
        for kj, k in enumerate(kinks):  # (a) between the flanks
            if lo is not None and k["x"] < lo["x"]:
                continue
            if hi is not None and k["x"] > hi["x"]:
                continue
            companions.add(kj)
        if lo is not None:  # (b) the first kink beyond each flank
            beyond = [kj for kj, k in enumerate(kinks) if k["x"] < lo["x"]]
            if beyond:
                companions.add(max(beyond, key=lambda kj: kinks[kj]["x"]))
        if hi is not None:
            beyond = [kj for kj, k in enumerate(kinks) if k["x"] > hi["x"]]
            if beyond:
                companions.add(min(beyond, key=lambda kj: kinks[kj]["x"]))
        return companions

    # 2. Kink companions of a peak.
    companions = [companions_of(gi) for gi in range(len(peaks))]
    kink_group: dict[int, int] = {}
    for kj, k in enumerate(kinks):
        matches = [
            gi for gi in range(len(peaks)) if kj in companions[gi] and contains(peaks[gi], k)
        ]
        if matches:
            gi = min(matches, key=lambda g: abs(peaks[g]["x"] - k["x"]))
            groups[gi].append(k)
            kink_group[kj] = gi

    # 3. Step shoulders: an unclaimed derivative extremum plus one
    # unclaimed kink of the matching sign on EACH side of it, adjacent and
    # inside its footprint. Claims are resolved by kink so that two nearby
    # steps cannot share a shoulder.
    free_derivs = [j for j in range(len(derivs)) if j not in deriv_group]
    claims: dict[int, list[int]] = {j: [] for j in free_derivs}
    for kj, k in enumerate(kinks):
        if kj in kink_group:
            continue
        qualifying = []
        for j in free_derivs:
            d = derivs[j]
            side = "below" if k["x"] < d["x"] else "above"
            if (
                k["sign"] == _STEP_SHOULDER_SIGN[(d["sign"], side)]
                and in_footprint(d, k)
                and adjacent(d, k)
            ):
                qualifying.append(j)
        if qualifying:
            j = min(qualifying, key=lambda jj: (abs(derivs[jj]["x"] - k["x"]), jj))
            claims[j].append(kj)
    for j in free_derivs:
        d = derivs[j]
        below = [kj for kj in claims[j] if kinks[kj]["x"] < d["x"]]
        above = [kj for kj in claims[j] if kinks[kj]["x"] >= d["x"]]
        groups.append([d])
        gi = len(groups) - 1
        deriv_group[j] = gi
        if not below or not above:
            continue  # one shoulder is not a step
        lo_kj = max(below, key=lambda kj: kinks[kj]["x"])
        hi_kj = min(above, key=lambda kj: kinks[kj]["x"])
        for kj in (lo_kj, hi_kj):
            groups[gi].append(kinks[kj])
            kink_group[kj] = gi

    # 4. A kink nobody claimed is a feature in its own right.
    for kj, k in enumerate(kinks):
        if kj not in kink_group:
            groups.append([k])
            kink_group[kj] = len(groups) - 1

    # 5. Onsets follow their parent derivative -- or stand alone when
    # 'onset' was searched without 'derivative' and there is no parent row.
    for o in onsets:
        gi = deriv_group.get(o["parent"])
        if gi is None:
            groups.append([o])
        else:
            groups[gi].append(o)

    ranked = []
    for gi, members in enumerate(groups):
        rep = min(
            may_report(gi, members),
            key=lambda m: (
                priority.index(m["method"]) if m["method"] in priority else len(priority),
                -m["significance"],  # tie-break within a method only; -inf sorts first
            ),
        )
        ranked.append({"rep": rep, "members": members})
    ranked.sort(key=lambda g: g["rep"]["x"])
    return ranked


# ---------------------------------------------------------------------------
# Array-level API.
# ---------------------------------------------------------------------------

#: The methods whose hits :func:`_group_detections` may merge into one
#: candidate, and so the only valid ``criterion=`` values.
_GROUPED_METHODS = ("peak", "derivative", "kink", "onset")

#: Every accepted ``methods=`` key. ``'fisher'`` is accepted but never
#: grouped (see :func:`find_transitions`), and it is never a scan-kind
#: default (see :func:`_default_methods`).
_VALID_METHODS = (*_GROUPED_METHODS, "fisher")


def _check_detection_options(
    methods, criterion, smooth_window, polyorder, prominence_fraction, noise
):
    """Validate data-independent detection settings and normalise method names."""
    if methods is None:
        normalised = None
    elif isinstance(methods, str):
        normalised = (methods,)
    else:
        try:
            normalised = tuple(methods)
        except TypeError:
            raise DetectionInputError(f"methods must be a string or sequence, got {methods!r}") from None
    if normalised is not None and (
        not normalised or any(m not in _VALID_METHODS for m in normalised)
    ):
        raise DetectionInputError(
            f"methods must be a non-empty subset of {_VALID_METHODS}, got {methods!r}"
        )
    if criterion == "fisher":
        raise DetectionInputError(
            "criterion='fisher' has nothing to rank: Fisher hits are never grouped "
            "with the other methods (their d(yT)/dT extremum sits systematically "
            "below the chi cusp), so every Fisher candidate already reports "
            f"itself. criterion must be one of {_GROUPED_METHODS}."
        )
    if criterion is not None and criterion not in _GROUPED_METHODS:
        raise DetectionInputError(f"criterion must be one of {_GROUPED_METHODS}, got {criterion!r}")
    _check_smooth_window(smooth_window)
    if isinstance(polyorder, bool) or not isinstance(polyorder, (int, np.integer)) or polyorder < 0:
        raise DetectionInputError(f"polyorder must be a non-negative integer, got {polyorder!r}")
    if isinstance(prominence_fraction, bool) or not isinstance(prominence_fraction, Real):
        raise DetectionInputError("prominence_fraction must be a finite non-negative number")
    if not np.isfinite(prominence_fraction) or prominence_fraction < 0:
        raise DetectionInputError("prominence_fraction must be a finite non-negative number")
    _check_noise(noise)
    return normalised


def find_transitions(
    x: Sequence[float],
    y: Sequence[float],
    *,
    methods: str | Sequence[str],
    x_name: str = "x",
    x_unit: str = "",
    y_err: Sequence[float] | None = None,
    smooth_window: float | None = None,
    polyorder: int = 3,
    prominence_fraction: float = DEFAULT_PROMINENCE_FRACTION,
    criterion: str | None = None,
    label: str = "find_transitions",
    noise: str | float = "auto",
) -> TransitionResult:
    """Detect transition candidates in one 1-D sweep given as arrays.

    The array-level core under :func:`detect_transitions`; use that for
    :class:`~calamari.measurement.Measurement` inputs (it resolves the
    swept axis, the fixed coordinate, units, channels and refusals). This
    level exists for arbitrary ``(x, y)`` pairs.

    Parameters
    ----------
    x, y : array-like
        The swept variable and the signal. The sweep must be monotonic in
        ``x`` up to setpoint jitter -- a hysteresis loop is refused by
        name (take a branch first).
    methods : str or sequence of str
        Which detectors to run: ``'peak'``, ``'derivative'``, ``'kink'``,
        ``'onset'``, ``'fisher'`` (see the module docstring and
        :data:`METHODS`). No default here: the scan-kind defaults live in
        :func:`detect_transitions`, which knows what the data is. Each
        method has its own ``sign`` vocabulary in the output -- ``'max'``
        / ``'min'`` for a peak, ``'rise'`` / ``'fall'`` for a derivative
        extremum, ``'up'`` / ``'down'`` for a kink (the slope increasing
        or decreasing through it), ``'max'`` / ``'min'`` of d(yT)/dT for
        ``'fisher'`` -- because a shared one would imply
        that a maximum and a rising slope are the same statement about
        the curve. Which hits are then merged into one candidate is exact
        geometry, described in full in :func:`_group_detections`: a cusp
        keeps its derivative flanks and the curvature extrema those flanks
        bound, a rounded step keeps the two curvature shoulders either
        side of its midpoint, and a slope change with neither is reported
        once, on its own. ``'onset'`` searched WITHOUT ``'derivative'``
        reports the onsets alone: they are built from derivative extrema,
        but those parent rows are not in the output, so each onset is its
        own candidate.

        ``'fisher'`` is Fisher's criterion: extrema of d(yT)/dT, which for
        a simple antiferromagnet tracks the magnetic specific heat (Fisher,
        Philos. Mag. 7, 1731 (1962)). It forms ``z = x*y`` on the
        prepared points, resamples and differentiates ``z`` with the same
        Savitzky-Golay window as everything else, and searches ``dz/dx``
        exactly as ``'derivative'`` searches dy/dx. It needs absolute
        temperature, so it requires ``x_unit='K'`` and every ``x > 0``,
        and raises otherwise. Its hits are **never grouped** with the
        other methods: each is its own candidate with ``n_methods = 1``,
        listed after the grouped candidates, in order of location. That is
        deliberate. The d(yT)/dT maximum sits systematically below the chi
        cusp, by an amount that depends on the material and on the
        smoothing (a smoothed, lopsided peak and the vertex fit that
        locates it both lean away from a steep drop at T_N), so merging the
        two under :data:`CRITERION_PRIORITY` would hide that offset, or
        report one criterion's temperature under the other's name -- the
        failure the priority exists to prevent. It is a different criterion, not a
        more sensitive one: nothing here claims it finds a boundary the
        cusp misses. Its noise is resolved separately, for ``z`` (see
        ``noise``), and recorded in ``params['fisher_noise']``; its
        ``sweep_range_over_noise`` is ``z``'s range over that sigma, and
        its ``raw_strength_unit`` is ``y·K/K`` (the prominence of
        d(yT)/dT, in ``y``'s own unit once the Kelvins cancel).
    x_name, x_unit : str
        Name the output's location columns (e.g. ``x_name='T'``,
        ``x_unit='K'`` gives ``T_K``/``T_K_err``): DM-2's the-unit-is-in-
        the-name rule applied to a caller-defined axis.
    y_err : array-like, optional
        Per-point 1-sigma errors on ``y`` (the instrument's, where the
        file carries them). Absent, the noise level falls back to a
        second-difference estimate; either way ``params['noise_source']``
        records which was used -- see ``noise=`` for how that choice is
        made and overridden.
    smooth_window : float, optional
        Savitzky-Golay window in x-units (not points). Default: 11 grid
        points, recorded as the resolved physical width in ``params``.
    polyorder : int, optional
        Savitzky-Golay polynomial order (default 3).
    prominence_fraction : float, optional
        Detection floor -- see :data:`DEFAULT_PROMINENCE_FRACTION`.
    criterion : str, optional
        Method whose member reports a grouped feature's coordinate,
        promoted to the front of :data:`CRITERION_PRIORITY`.
        ``criterion='kink'`` reports a group's *own* slope change: a lone
        kink, or a cusp's central curvature extremum (``'down'`` under a
        maximum, ``'up'`` over a minimum, between its derivative flanks).
        It never reports a cusp's flank companions or a rounded step's
        shoulders, which are curvature extrema too but sit off the feature,
        and on a cusp with sharp feet are stronger than its centre. A group
        with no kink of its own reports what it would under the default
        order: the peak for a cusp, the midpoint (or its onset) for a step.
    label : str, optional
        Name used in error messages (the measurement path, at the
        measurement level).
    noise : {'auto', 'instrument', 'scatter'} or float, optional
        Which noise estimate scores ``significance`` against. The
        instrument's reported per-point errors and the observed point-to-point
        scatter can describe different uncertainty contributions. A discrepancy
        affects the diagnostic candidate score; neither source is automatically
        a calibrated noise model. ``'auto'`` (default)
        prefers the instrument column when
        usable (>= 3 finite, positive values), else the second-difference
        scatter estimate; if it picks the instrument column and that
        column's sigma exceeds :data:`NOISE_RATIO_WARN` times the scatter
        estimate, it warns rather than using it silently.
        ``'instrument'`` requires a usable error column and raises
        ``ValueError`` naming ``noise='scatter'`` as the fix when there is
        none. ``'scatter'`` always uses the second-difference estimate,
        ignoring any error column. A positive float supplies the sigma
        directly, in ``y``'s own units. Recorded in ``params`` as
        ``noise``, ``noise_source`` (``'instrument'``/``'scatter'``/
        ``'caller'``), ``noise_sigma``, ``noise_sigma_instrument``,
        ``noise_sigma_scatter`` and ``noise_ratio``.

        With ``'fisher'`` the same choice is resolved a second time, for
        ``z = x*y``, and recorded with the same keys under
        ``params['fisher_noise']``; the signal's own record above is left
        exactly as it would be without ``'fisher'``. The instrument
        estimate is the median of ``y_err*x`` (each point's error scales
        with its own temperature) and the scatter estimate is the
        second-difference estimate of ``z`` itself. A caller's float is a
        sigma in ``y``'s units, so for ``z`` it becomes
        ``noise * median(x)`` -- the same median the instrument estimate
        takes -- rather than being read, wrongly, as a sigma in ``y·K``.
        The record keeps the float as passed under ``noise`` and the
        converted value under ``noise_sigma``.

    Returns
    -------
    TransitionResult
        With location columns named from ``x_name``/``x_unit``. With
        ``'fisher'``, ``curves`` also carries ``'yT'`` (the smoothed
        ``x*y``) and ``'dyT_dT'`` (its derivative, the representation the
        Fisher extrema were found in).

    Raises
    ------
    ValueError
        For unknown ``methods`` or ``criterion`` (``criterion='fisher'``
        included: Fisher hits are never grouped, so there is nothing for
        it to take priority over); for ``'fisher'`` on an axis that is not
        absolute temperature; and for a sweep that cannot be searched (too
        few points, not monotonic, systematically repeated setpoints).
    """
    methods = _check_detection_options(
        methods, criterion, smooth_window, polyorder, prominence_fraction, noise
    )
    if methods is None:
        raise DetectionInputError("methods must name a non-empty subset of the detection methods")
    fisher = "fisher" in methods
    if fisher and x_unit != "K":
        raise CurveInputError(
            f"{label}: Fisher's criterion needs absolute temperature. 'fisher' "
            "differentiates y*T with respect to T, which is only meaningful with T "
            f"in Kelvin, but x_unit is {x_unit!r}. Pass a temperature sweep with "
            "x_name='T', x_unit='K'."
        )
    priority = CRITERION_PRIORITY
    if criterion is not None:
        priority = (criterion, *[m for m in CRITERION_PRIORITY if m != criterion])

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape:
        raise CurveInputError(f"x and y must have the same shape, got {x_arr.shape} vs {y_arr.shape}")
    err_arr = None if y_err is None else np.asarray(y_err, dtype=float)
    x_arr, y_arr, err_arr, n_duplicates_dropped = _prepare(x_arr, y_arr, err_arr, label)
    if fisher and float(np.min(x_arr)) <= 0:
        raise CurveInputError(
            f"{label}: Fisher's criterion needs absolute temperature, so every x "
            f"must be positive, but the smallest is {float(np.min(x_arr)):g}. A "
            "temperature at or below 0 K is not one; check the axis and its unit."
        )

    xu, yu, dx = _resample(x_arr, y_arr)
    pts = _window_points(smooth_window, dx, len(xu))
    poly = min(polyorder, pts - 1)
    y_smooth = savgol_filter(yu, window_length=pts, polyorder=poly, deriv=0)
    dydx = savgol_filter(yu, window_length=pts, polyorder=poly, deriv=1, delta=dx)
    # Always computed, not only for methods including 'kink': the curvature is
    # the representation a caller audits a slope change against, and curves[]
    # is that audit trail (a second savgol pass on an already-resampled grid is
    # negligible next to the loading it took to get here).
    d2ydx2 = savgol_filter(yu, window_length=pts, polyorder=poly, deriv=2, delta=dx)

    noise_info = _resolve_noise(y_arr, err_arr, noise, label)
    sigma = noise_info["noise_sigma"]
    evf = _extreme_value_factor(len(xu), pts)
    sigma0 = sigma * _kernel_norm(pts, poly, 0, dx)
    sigma1 = sigma * _kernel_norm(pts, poly, 1, dx)
    sigma2 = sigma * _kernel_norm(pts, poly, 2, dx)
    range_over_noise = (
        float((np.max(y_arr) - np.min(y_arr)) / sigma) if sigma > 0 else np.inf
    )

    # Fisher's d(yT)/dT: the same pipeline on z = x*y, with z's own noise.
    # Resolved after the signal's noise, so an invalid noise= has already
    # raised there and the signal's record is untouched by what follows.
    fisher_hits: list[dict] = []
    fisher_noise = None
    fisher_curves = {}
    fisher_range_over_noise = np.nan
    if fisher:
        z = x_arr * y_arr
        z_err = None if err_arr is None else err_arr * x_arr
        _, zu, _ = _resample(x_arr, z)  # the same grid: it depends on x alone
        z_smooth = savgol_filter(zu, window_length=pts, polyorder=poly, deriv=0)
        dzdx = savgol_filter(zu, window_length=pts, polyorder=poly, deriv=1, delta=dx)
        # A caller's float is a sigma in y's units; z = x*y carries it scaled
        # point by point, summarised by the same median the instrument
        # estimate takes. Reading it as a sigma in y*K would be a unit error.
        z_noise = noise if isinstance(noise, str) else float(noise) * float(np.median(x_arr))
        fisher_noise = _resolve_noise(z, z_err, z_noise, f"{label} [fisher: y*T]")
        fisher_noise["noise"] = noise
        sigma_z = fisher_noise["noise_sigma"]
        fisher_hits = _detect_derivative(
            xu,
            dzdx,
            sigma_z * _kernel_norm(pts, poly, 1, dx),
            evf,
            prominence_fraction,
            pts,
            method="fisher",
            feature="fisher",
            signs=("max", "min"),
        )
        fisher_range_over_noise = (
            float((np.max(z) - np.min(z)) / sigma_z) if sigma_z > 0 else np.inf
        )
        fisher_curves = {"yT": z_smooth, "dyT_dT": dzdx}

    hits: list[dict] = []
    if "peak" in methods:
        hits += _detect_peak(xu, y_smooth, sigma0, evf, prominence_fraction, pts)
    deriv_hits = []
    if "derivative" in methods or "onset" in methods:
        deriv_hits = _detect_derivative(xu, dydx, sigma1, evf, prominence_fraction, pts)
    if "derivative" in methods:
        hits += deriv_hits
    if "kink" in methods:
        hits += _detect_kink(xu, d2ydx2, sigma2, evf, prominence_fraction, pts)
    if "onset" in methods:
        hits += _detect_onset(xu, y_smooth, dydx, deriv_hits, sigma0, pts)

    loc = f"{x_name}_{x_unit}" if x_unit else x_name
    groups = _group_detections(hits, xu, priority)
    # Fisher hits never enter grouping (see the docstring): each is its own
    # candidate, appended after the grouped ones so their ids and order are
    # exactly what they would be without 'fisher'.
    groups += [
        {"rep": h, "members": [h]} for h in sorted(fisher_hits, key=lambda h: h["x"])
    ]

    cand_rows, det_rows = [], []
    for cid, grp in enumerate(groups):
        rep, members = grp["rep"], grp["members"]
        cand_rows.append(
            {
                "candidate_id": cid,
                loc: rep["x"],
                f"{loc}_err": rep["x_err"],
                "method": rep["method"],
                "n_methods": len({m["method"] for m in members}),
                "feature": rep["feature"],
                "sign": rep["sign"],
                "significance": rep["significance"],
                "sweep_range_over_noise": (
                    fisher_range_over_noise if rep["method"] == "fisher" else range_over_noise
                ),
                "edge": bool(rep["edge"]),
            }
        )
        for m in members:
            # raw_strength lives in the searched signal's own units for a
            # peak, in signal-per-x units for the slope-based methods, in
            # signal-per-x-squared for the curvature and in y*K/K for Fisher's
            # d(yT)/dT; the unit column keeps the per-row unit explicit (DM-2)
            # since one column would otherwise silently mix them.
            per_x = x_unit or x_name
            if m["method"] == "peak":
                unit = "y"
            elif m["method"] == "kink":
                unit = f"y/{per_x}²"
            elif m["method"] == "fisher":
                unit = f"y·{per_x}/{per_x}"
            else:
                unit = f"y/{per_x}"
            det_rows.append(
                {
                    "candidate_id": cid,
                    "method": m["method"],
                    loc: m["x"],
                    "raw_strength": m["raw_strength"],
                    "raw_strength_unit": unit,
                    "significance": m["significance"],
                    "feature": m["feature"],
                    "sign": m["sign"],
                    "edge": bool(m["edge"]),
                }
            )

    cand_cols = [
        "candidate_id", loc, f"{loc}_err", "method", "n_methods", "feature",
        "sign", "significance", "sweep_range_over_noise", "edge",
    ]
    det_cols = [
        "candidate_id", "method", loc, "raw_strength", "raw_strength_unit",
        "significance", "feature", "sign", "edge",
    ]
    params = {
        "methods": methods,
        "criterion_priority": priority,
        "smooth_window": pts * dx,
        "smooth_points": pts,
        "grid_dx": dx,
        "polyorder": poly,
        "prominence_fraction": prominence_fraction,
        **noise_info,
        "n_points": len(x_arr),
        "n_duplicates_dropped": n_duplicates_dropped,
        "calamari_version": _calamari_version(),
    }
    if fisher:
        params["fisher_noise"] = fisher_noise
    return TransitionResult(
        candidates=pd.DataFrame(cand_rows, columns=cand_cols),
        detections=pd.DataFrame(det_rows, columns=det_cols),
        params=params,
        curves={
            ("", ""): {
                "x": xu,
                "y_smooth": y_smooth,
                "dydx": dydx,
                "d2ydx2": d2ydx2,
                **fisher_curves,
            }
        },
    )
