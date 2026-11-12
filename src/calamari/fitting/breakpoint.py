"""Fit a sharp or rounded change of slope and compare it with a straight line.

:func:`fit_breakpoint` fits ``segmented_linear`` or ``bent_cable`` within a
caller-specified range. It reports model comparisons, HESSE errors and a profile
of the breakpoint location. The straight-line comparison helps assess whether
the data support a break; information-criterion differences are not p-values.

A sharp-corner fit can depend on the chosen range when the feature is rounded.
The bent-cable model allows symmetric rounding, but its inferred centre can
still depend on model mismatch or asymmetric rounding. Compare results over
physically justified ranges with :func:`~calamari.fitting.windows.fit_windows`.
The function reports unresolved rounding rather than selecting a model for you.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar

from calamari._provenance import calamari_version as _calamari_version
from calamari.exceptions import FitInputError
from calamari.fitting.engine import _EXACT_FIT_RTOL, FitResult, fit
from calamari.fitting.models import (
    MIN_POINTS_PER_SEGMENT,
    _bent_cable_gamma_limits,
    _bent_cable_solve,
    _segmented_candidates,
    _segmented_profile,
    _segmented_solve,
)

#: The fewest points :func:`fit_breakpoint` accepts:
#: :data:`~calamari.fitting.models.MIN_POINTS_PER_SEGMENT` strictly on each side of a
#: corner that is itself a sample, so the profile over sample positions has at least one
#: candidate. Seven points also leave the four-parameter fit three degrees of freedom.
MIN_POINTS = 2 * MIN_POINTS_PER_SEGMENT + 1

#: Numbers of parameters compared by the information criteria: the corner,
#: its value and two slopes, against a slope and an intercept. The Gaussian
#: noise level, unknown in both, adds one to each and cancels from the
#: difference.
_K_SEGMENTED = 4
_K_LINEAR = 2

#: Parameters of the bent cable in the same count: the four of the sharp
#: corner and the half-width of the bend.
_K_BENT_CABLE = 5

#: The corner models :func:`fit_breakpoint` fits, by their registry names; the
#: first is the default.
_BREAKPOINT_MODELS = ("segmented_linear", "bent_cable")

#: Grid density, in points per decade, of the logarithmic search over the
#: bend's half-width that brackets its profiled minimum before a bounded
#: scalar minimiser refines it. Twenty per decade puts neighbouring widths 12 %
#: apart, so the bracket is two such steps wide: fine enough that a second
#: basin would have to be narrower than that to be missed, and the refinement
#: inside the bracket is exact to :data:`_INTERVAL_XTOL`. The range is
#: ``log10(2 (n - 1))`` decades for ``n`` evenly spaced points (from half a
#: spacing to the span of ``x``; two decades at 51 points), so the grid is a
#: few dozen widths, evaluated in one batch.
_GAMMA_GRID_PER_DECADE = 20

#: Bisection stopping width for the ends of the profile interval, as a
#: fraction of the width of the search range. 1e-9 locates an end far more
#: finely than any HESSE error it is shown beside, and needs about 30 halvings.
_INTERVAL_XTOL = 1e-9

#: Hard cap on bisection steps, a guard only: halving from the full range
#: reaches :data:`_INTERVAL_XTOL` in about 30 steps.
_INTERVAL_MAX_STEPS = 200


@dataclass(frozen=True)
class BreakpointResult:
    """The outcome of :func:`fit_breakpoint`: a corner, and whether the data want one.

    Read :attr:`identified` (and ``delta_bic``) before ``x_break``.
    A corner fitted to data with no break still has a value and a small HESSE
    error; neither means anything unless the comparison favours the break.

    The corner model is :attr:`model`. Every field describes that model's
    fit, except :attr:`segmented` and ``rss_segmented``, which always
    describe the sharp corner. The fields from :attr:`bent_cable` onwards are
    filled only for ``model="bent_cable"``, and keep their defaults (``None``
    or NaN) for the default ``"segmented_linear"``.

    Attributes
    ----------
    segmented : FitResult
        The ``segmented_linear`` fit, the sharp corner. Draw it with
        :func:`calamari.plot.plot_fit`, as any other fit. With
        ``model="bent_cable"`` it is the sharp-corner comparison.
    linear : FitResult
        The ``linear`` fit to the same points, the no-break alternative.
    x_break, x_break_err : float
        The fitted corner and its HESSE error, the canonical uncertainty
        (FT-1). For the bent cable, the centre of the bend, where its two
        arms meet when extended. NaN if the fit of :attr:`model` failed.
    slope_below, slope_below_err : float
        Slope below the corner, with its HESSE error.
    slope_above, slope_above_err : float
        Slope above the corner, ``slope_below + slope_change``. Its error is
        propagated through the covariance by
        :meth:`~calamari.fitting.engine.FitResult.derive`, so the
        correlation between the two fitted slopes counts.
    slope_change, slope_change_err : float
        The change of slope across the corner, with its HESSE error.
    delta_aic, delta_bic : float
        Information criterion of the straight line minus that of the broken
        one (the :attr:`model` fitted), from ordinary least squares (see
        :func:`fit_breakpoint`). **Positive means the break is preferred.**
        Descriptive, not a test.
    rss_linear, rss_segmented : float
        The ordinary (unweighted) least-squares residual sums of squares the
        criteria are computed from, of the line and of the sharp corner.
        ``rss_segmented`` is minimised exactly over the corner within the
        limits, not only on the sample grid.
    profile : pandas.DataFrame
        Columns ``x_break`` and ``rss``: the unweighted residual sum of
        squares with the corner held at each sample position inside the
        limits (the grid the model's guess searches), the other parameters
        solved exactly. For display: plot ``rss`` against ``x_break`` with
        :attr:`profile_threshold` as a horizontal line. For the bent cable,
        the half-width is profiled too (minimised at each corner, not held),
        and a third column ``gamma`` holds where that minimum fell.
    profile_threshold : float
        The RSS level that defines :attr:`profile_interval`,
        ``rss * exp(1 / n_points)``, where ``rss`` is the minimum for
        :attr:`model` (``rss_segmented`` or ``rss_bent_cable``).
    profile_interval : tuple of float
        ``(lower, upper)``: the smallest interval holding every corner whose
        profiled RSS is at or under :attr:`profile_threshold`, found exactly
        between sample positions. An end that does not come back above the
        threshold inside the limits is ``-inf`` or ``+inf``: the data do not
        bound the corner on that side. **Display-only**, with the status
        MINOS has under FT-1: it is not a calibrated confidence interval
        (see :func:`fit_breakpoint`), and ``x_break_err`` stays
        canonical.
    n_points, n_below, n_above : int
        Points fitted, and how many lie strictly below and strictly above the
        fitted corner (zero each if the segmented fit failed).
    identified : bool
        ``True`` only when the fit of :attr:`model` and the line both
        converged, ``delta_bic > 0``, and the corner is clear of its limits by
        more than its HESSE error. For the bent cable, also only when each
        straight arm keeps at least three points outside the bend. When
        ``False``, :attr:`warnings` says why. A bend whose width is not
        resolved (see ``gamma``) warns but does not unset it: the corner
        is still located, and the sharp one describes it as well.
    warnings : tuple of str
        The messages of the warnings :func:`fit_breakpoint` raised, in order.
    params : dict
        The recipe: ``'model'``, ``'limits'`` (the ``(lower, upper)`` bounds
        used on the corner), ``'limits_given'`` (whether the caller set them),
        ``'yerr'`` (``'none'``, ``'scalar'`` or ``'array'``),
        ``'information_criteria'`` and ``'profile_interval'`` (text naming the
        construction of each) and ``'calamari_version'``. For the bent cable,
        also ``'gamma_limits'``, the bounds used on the half-width.
    model : str
        ``'segmented_linear'`` (the default) or ``'bent_cable'``.
    bent_cable : FitResult or None
        The ``bent_cable`` fit; ``None`` unless it was the model. Draw it with
        :func:`calamari.plot.plot_fit` beside :attr:`segmented`.
    gamma, gamma_err : float
        The half-width of the bend and its HESSE error, in the units of
        ``x``. It is bounded below at half the median spacing of ``x``, the
        narrowest bend the sampling resolves (see
        :data:`~calamari.fitting.models.BENT_CABLE_GAMMA_FLOOR_SPACINGS`), and
        a value within its error of that bound means the corner is sharp at
        this resolution. NaN unless the bent cable was fitted and converged.
    rss_bent_cable : float
        The bent cable's unweighted least-squares residual sum of squares, the
        smaller of the profiled minimum and the fitted curve's own RSS.
    delta_aic_vs_segmented, delta_bic_vs_segmented : float
        Information criterion of the sharp corner minus that of the bent
        cable, on the same points, as for ``delta_bic``. **Positive means
        the rounded corner is preferred to the sharp one.** Descriptive, not a
        test; NaN unless the bent cable was fitted.
    """

    segmented: FitResult = field(repr=False)
    linear: FitResult = field(repr=False)
    x_break: float
    x_break_err: float
    slope_below: float
    slope_below_err: float
    slope_above: float
    slope_above_err: float
    slope_change: float
    slope_change_err: float
    delta_aic: float
    delta_bic: float
    rss_linear: float
    rss_segmented: float
    profile: pd.DataFrame = field(repr=False)
    profile_threshold: float
    profile_interval: tuple[float, float]
    n_points: int
    n_below: int
    n_above: int
    identified: bool
    warnings: tuple[str, ...]
    params: dict[str, Any] = field(repr=False)
    # Added with the bent cable. Defaults keep the default model's result, and
    # any code building one, unchanged.
    model: str = "segmented_linear"
    bent_cable: FitResult | None = field(default=None, repr=False)
    gamma: float = float("nan")
    gamma_err: float = float("nan")
    rss_bent_cable: float = float("nan")
    delta_aic_vs_segmented: float = float("nan")
    delta_bic_vs_segmented: float = float("nan")


def _is_real(value: Any) -> bool:
    """True for a real number (numpy scalars included), False for a bool."""
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_))


def _validate_data(
    x: ArrayLike, y: ArrayLike, yerr: ArrayLike | None
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None, str]:
    """Check the points and errors, before anything is fitted.

    Bad input is the caller's error, not a fit the data could not support,
    so it raises here, naming the fix, rather than coming back as a failed
    fit that would be indistinguishable from data with no break.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.shape != y_arr.shape:
        raise FitInputError(
            f"x and y must be 1-D arrays of the same length; got shapes {x_arr.shape} "
            f"and {y_arr.shape}. Pass one sweep's values, e.g. m.temperature and m.chi_real."
        )
    bad = ~(np.isfinite(x_arr) & np.isfinite(y_arr))
    if bad.any():
        raise FitInputError(
            f"x and y must be finite; {int(bad.sum())} point(s) are NaN or infinite "
            f"(first at index {int(np.flatnonzero(bad)[0])}). Drop them first, e.g. "
            "keep = np.isfinite(x) & np.isfinite(y)."
        )
    if x_arr.size < MIN_POINTS:
        raise FitInputError(
            f"fit_breakpoint needs at least {MIN_POINTS} points ({MIN_POINTS_PER_SEGMENT} "
            f"strictly on each side of a corner at a sample); got {x_arr.size}. Widen the "
            "range of x passed in."
        )
    if _segmented_candidates(x_arr).size == 0:
        raise FitInputError(
            f"No sample position has {MIN_POINTS_PER_SEGMENT} distinct-x points strictly on "
            f"each side (x has {np.unique(x_arr).size} distinct values among {x_arr.size} "
            "points). Widen the range of x passed in."
        )

    if yerr is None:
        return x_arr, y_arr, None, "none"
    yerr_arr = np.asarray(yerr, dtype=float)
    kind = "scalar" if yerr_arr.ndim == 0 else "array"
    if yerr_arr.ndim != 0 and yerr_arr.shape != x_arr.shape:
        raise FitInputError(
            f"yerr must be a scalar or a 1-D array the length of x; got shape "
            f"{yerr_arr.shape} for {x_arr.size} points. Pass one error per point, or "
            "yerr=None to estimate the scatter from the residuals."
        )
    if not (np.all(np.isfinite(yerr_arr)) and np.all(yerr_arr > 0)):
        raise FitInputError(
            "yerr must be finite and positive at every point; got "
            f"min {float(np.nanmin(yerr_arr)):g}, with "
            f"{int(np.sum(~np.isfinite(yerr_arr)))} non-finite. Drop those points, or pass "
            "yerr=None to estimate the scatter from the residuals."
        )
    return x_arr, y_arr, np.broadcast_to(yerr_arr, x_arr.shape).copy(), kind


def _validate_limits(limits: Any, x: NDArray[np.float64]) -> tuple[float, float]:
    """Resolve the corner's bounds: the default, or the caller's, checked.

    The default is the closed range from the 3rd to the 3rd-from-last point
    in ascending ``x``, so a corner strictly inside keeps
    :data:`~calamari.fitting.models.MIN_POINTS_PER_SEGMENT` points strictly on
    each side. A caller's limits must lie inside that same range, for the
    same reason, and must hold at least one sample position for the profile.
    """
    xs = np.sort(x)
    k = MIN_POINTS_PER_SEGMENT
    allowed = (float(xs[k - 1]), float(xs[-k]))
    if limits is None:
        return allowed
    if isinstance(limits, np.ndarray):
        limits = limits.tolist()
    if (
        isinstance(limits, (str, bytes))
        or not hasattr(limits, "__len__")
        or len(limits) != 2
        or not all(_is_real(v) for v in limits)
    ):
        raise FitInputError(
            f"limits must be a (lower, upper) pair of numbers for x_break; got {limits!r}. "
            "Pass e.g. limits=(8.0, 12.0), or omit it for the default range."
        )
    lower, upper = float(limits[0]), float(limits[1])
    if not (np.isfinite(lower) and np.isfinite(upper)) or lower >= upper:
        raise FitInputError(f"limits must be finite with lower < upper; got ({lower:g}, {upper:g}).")
    if lower < allowed[0] or upper > allowed[1]:
        raise FitInputError(
            f"limits ({lower:g}, {upper:g}) reach outside ({allowed[0]:g}, {allowed[1]:g}), "
            f"the range from the {k}rd to the {k}rd-from-last point, which keeps "
            f"{k} points strictly on each side of the corner. Narrow the limits to "
            "that range, or widen the range of x passed in."
        )
    grid = _segmented_candidates(x)
    if not np.any((grid >= lower) & (grid <= upper)):
        raise FitInputError(
            f"limits ({lower:g}, {upper:g}) hold no sample position of x, so there is "
            "nothing to profile the corner over. Widen the limits to include at least one "
            "point."
        )
    return lower, upper


def _two_line_corner(
    x: NDArray[np.float64], y: NDArray[np.float64], left: float, right: float
) -> tuple[float, float] | None:
    """The best corner strictly between two adjacent knots, if there is one there.

    With the corner anywhere in ``(left, right)`` the points split the same
    way, so the least-squares problem is two independent lines, one through
    each side, constrained to meet. The unconstrained pair already meets
    somewhere; if that is inside ``(left, right)`` the constraint costs
    nothing and that is the exact minimum for this split. Otherwise the
    profiled RSS on the interval is monotone or has only an interior maximum
    (it is a squared linear function of the corner over a positive quadratic),
    so its minimum is at a knot, which the caller evaluates anyway. Returns
    ``(corner, rss)`` or ``None``.
    """
    below = x < right  # x <= left, as no sample lies strictly between the knots
    above = ~below
    if below.sum() < 2 or above.sum() < 2:
        return None
    b1, a1 = np.polyfit(x[below], y[below], 1)
    b2, a2 = np.polyfit(x[above], y[above], 1)
    if b1 == b2:
        return None
    corner = float((a2 - a1) / (b1 - b2))
    if not (left < corner < right):
        return None
    rss = float(
        np.sum((y[below] - (a1 + b1 * x[below])) ** 2)
        + np.sum((y[above] - (a2 + b2 * x[above])) ** 2)
    )
    return corner, rss


def _profile_minimum(
    x: NDArray[np.float64], y: NDArray[np.float64], lower: float, upper: float
) -> tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:
    """The exact unweighted least-squares corner in ``[lower, upper]``.

    Evaluates the profiled RSS at every knot (the sample positions inside the
    limits, and the limits themselves) and at each interval's two-line corner
    where that lies inside the interval (:func:`_two_line_corner`). Between
    these points the profile has no minimum, so the smallest value found is
    the exact minimum over the continuous range. Returns ``(corner, rss,
    points, rss_at_points)`` with the points sorted, for the interval search.
    """
    knots = np.unique(np.concatenate([[lower, upper], x[(x > lower) & (x < upper)]]))
    points = list(knots)
    for left, right in zip(knots[:-1], knots[1:]):
        found = _two_line_corner(x, y, float(left), float(right))
        if found is not None:
            points.append(found[0])
    points_arr = np.unique(np.asarray(points, dtype=float))
    rss = _segmented_profile(x, y, points_arr)
    best = int(np.argmin(rss))
    return float(points_arr[best]), float(rss[best]), points_arr, rss


def _crossing(
    rss_at: Callable[[float], float], inside: float, outside: float, level: float
) -> float:
    """Bisect for where the profiled RSS, ``rss_at(corner)``, crosses *level* between two points.

    ``inside`` is at or under the level and ``outside`` above it. For the
    sharp corner the evaluation points bracketing a crossing are adjacent,
    and between adjacent points the profile crosses a level at most once in
    this direction, so the bisection finds the unique crossing. For the bent
    cable the points are the sample positions, and the bisection finds a
    crossing between them.
    """
    tol = _INTERVAL_XTOL * abs(outside - inside)
    for _ in range(_INTERVAL_MAX_STEPS):
        if abs(outside - inside) <= tol:
            break
        middle = 0.5 * (inside + outside)
        if rss_at(middle) <= level:
            inside = middle
        else:
            outside = middle
    return 0.5 * (inside + outside)


def _profile_interval(
    rss_at: Callable[[float], float],
    points: NDArray[np.float64],
    rss: NDArray[np.float64],
    level: float,
) -> tuple[float, float]:
    """The hull of corners whose profiled RSS is at or under *level* (see the result)."""
    under = np.flatnonzero(rss <= level)
    first, last = int(under[0]), int(under[-1])
    lower = -np.inf if first == 0 else _crossing(rss_at, points[first], points[first - 1], level)
    upper = (
        np.inf
        if last == points.size - 1
        else _crossing(rss_at, points[last], points[last + 1], level)
    )
    return float(lower), float(upper)


def _bounded_minimum(
    cost: Callable[[float], float], lower: float, upper: float
) -> tuple[float, float]:
    """A bounded scalar minimiser's answer on ``[lower, upper]``, as ``(argument, value)``.

    Deterministic (Brent's method on a bracket), stopping at
    :data:`_INTERVAL_XTOL` of the bracket's width, as the interval ends do.
    """
    if not upper > lower:
        return lower, cost(lower)
    found = minimize_scalar(
        cost,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": _INTERVAL_XTOL * (upper - lower)},
    )
    return float(found.x), float(found.fun)


def _gamma_profiled(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    corner: float,
    gamma_limits: tuple[float, float],
) -> tuple[float, float]:
    """The bent cable's least RSS with the centre held at *corner*: ``(gamma, rss)``.

    The other three parameters are solved exactly at each half-width. The
    half-width is found on a logarithmic grid between its limits
    (:data:`_GAMMA_GRID_PER_DECADE`), then refined by a bounded minimiser
    between the best grid point's neighbours. The grid's best is kept if the
    refinement does not beat it, so the answer is never worse than the grid.

    With the centre held, the line part (a constant and ``x - corner``) is the
    same at every half-width, so it is projected out once and each half-width
    costs one projection of its bend column: the exact least-squares RSS, as
    :func:`~calamari.fitting.models._bent_cable_solve` gives it, for a whole
    grid at once.
    """
    offset = x - corner
    line_basis, _ = np.linalg.qr(np.column_stack([np.ones_like(x), offset]))
    y_rest = y - line_basis @ (line_basis.T @ y)

    def rss_at(gammas: NDArray[np.float64]) -> NDArray[np.float64]:
        g = gammas[:, None]
        bend = np.where(
            np.abs(offset) < g, (offset + g) ** 2 / (4.0 * g), np.maximum(offset, 0.0)
        )
        bend_rest = bend - (bend @ line_basis) @ line_basis.T
        norm = np.einsum("ij,ij->i", bend_rest, bend_rest)
        # A bend column inside the line's span to round-off (every point on
        # one arm) adds nothing; projecting onto its round-off remainder would
        # "fit" noise. The boundary is the exact-fit tolerance used elsewhere.
        usable = norm > _EXACT_FIT_RTOL**2 * np.einsum("ij,ij->i", bend, bend)
        coef = np.zeros_like(norm)
        coef[usable] = (bend_rest[usable] @ y_rest) / norm[usable]
        residual = y_rest - coef[:, None] * bend_rest
        return np.einsum("ij,ij->i", residual, residual)

    lower, upper = gamma_limits
    steps = int(np.ceil(_GAMMA_GRID_PER_DECADE * np.log10(upper / lower)))
    grid = np.geomspace(lower, upper, max(steps, 1) + 1)
    rss = rss_at(grid)
    best = int(np.argmin(rss))
    gamma, value = _bounded_minimum(
        lambda g: float(rss_at(np.array([g]))[0]),
        float(grid[max(best - 1, 0)]),
        float(grid[min(best + 1, grid.size - 1)]),
    )
    if value < rss[best]:
        return gamma, value
    return float(grid[best]), float(rss[best])


def _bent_cable_profile_minimum(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    lower: float,
    upper: float,
    gamma_limits: tuple[float, float],
) -> tuple[
    float, float, float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """The bent cable's unweighted least-squares centre in ``[lower, upper]``, half-width free.

    Profiles the RSS over the centre, the half-width minimised at each
    (:func:`_gamma_profiled`), at the sample positions inside the limits and
    the limits themselves, then refines the centre between the best point's
    neighbours with a bounded minimiser. Unlike the sharp corner's
    (:func:`_profile_minimum`) this is numerical, not exact: the profile has
    no closed form between samples. Returns ``(corner, gamma, rss, points,
    rss_at_points, gamma_at_points)``, the points sorted and including the
    refined centre.
    """
    knots = np.unique(np.concatenate([[lower, upper], x[(x > lower) & (x < upper)]]))
    profiled = np.array([_gamma_profiled(x, y, float(c), gamma_limits) for c in knots])
    gamma_knots, rss_knots = profiled[:, 0], profiled[:, 1]
    best = int(np.argmin(rss_knots))
    corner, rss = _bounded_minimum(
        lambda c: _gamma_profiled(x, y, c, gamma_limits)[1],
        float(knots[max(best - 1, 0)]),
        float(knots[min(best + 1, knots.size - 1)]),
    )
    if not rss < rss_knots[best] or np.any(knots == corner):
        return (
            float(knots[best]),
            float(gamma_knots[best]),
            float(rss_knots[best]),
            knots,
            rss_knots,
            gamma_knots,
        )
    gamma = _gamma_profiled(x, y, corner, gamma_limits)[0]
    order = np.argsort(np.append(knots, corner), kind="stable")
    return (
        corner,
        gamma,
        rss,
        np.append(knots, corner)[order],
        np.append(rss_knots, rss)[order],
        np.append(gamma_knots, gamma)[order],
    )


def _information_gain(rss_linear: float, rss_segmented: float, n: int, floor: float) -> float:
    """``n * ln(rss_linear / rss_segmented)``, with round-off-level RSS held at *floor*.

    For noise-free data both sums of squares are round-off, and their ratio
    is noise: a straight line computed exactly can come out "preferring" a
    break by a factor of ten. Holding each sum at the exact-fit floor makes
    an exact line score a ratio of one, and an exact kink a large gain.
    """
    return float(n * np.log(max(rss_linear, floor) / max(rss_segmented, floor)))


def fit_breakpoint(
    x: ArrayLike,
    y: ArrayLike,
    *,
    yerr: ArrayLike | None = None,
    limits: tuple[float, float] | None = None,
    model: str = "segmented_linear",
) -> BreakpointResult:
    """Fit two lines meeting at a corner, and compare them with one straight line.

    Fits ``segmented_linear`` (see
    :func:`~calamari.fitting.models.segmented_linear`) and ``linear`` to the
    same points, and reports the corner with the evidence for it. The corner
    is ``x_break +/- x_break_err`` (HESSE, canonical under FT-1), and the
    evidence is ``delta_aic`` / ``delta_bic``. **Read the evidence first:**
    with no break in the data, the corner still gets a value and a small
    error bar, and both are meaningless. So this warns, and sets
    ``identified=False``, whenever ``delta_bic <= 0``. That threshold is not
    tuned: it is where the comparison itself changes sign. It warns too when
    the corner ends within its error of a limit, because then the limit, not
    the data, is placing it. The result is returned either way (warn, don't
    fix).

    With ``model="bent_cable"`` the corner is rounded by a parabolic bend
    whose half-width ``gamma`` is fitted too
    (:func:`~calamari.fitting.models.bent_cable`), and ``x_break`` is the
    centre of the bend. The sharp corner is still fitted, on the same points,
    so the result compares all three: ``delta_bic`` against the straight line
    as before, and ``delta_bic_vs_segmented`` against the sharp corner. Use
    it when the sharp corner moves with the fitting range (caveat 2 below).
    Two more warnings apply. One is raised when ``gamma`` ends within its
    error of its lower bound, half a sample spacing: the rounding is not
    resolved, and the sharp corner describes the data as well. The other is
    raised when the bend leaves fewer than three points on a straight arm:
    ``x_break`` is where the two arms meet, and it is poorly defined when an
    arm is not established. That second one sets ``identified=False``.

    Parameters
    ----------
    x, y : array_like
        One sweep's points, 1-D and finite: pass only the range around the
        feature, since both lines must describe the data on their side of
        the corner. Order does not matter.
    yerr : array_like or float, optional
        Per-point errors, used to weight the *parameter* fits, as in
        :func:`~calamari.fitting.engine.fit` (absolute errors). ``None``
        (default) estimates one scatter from the residuals. The model
        comparison and the profile ignore ``yerr`` (see Notes).
    limits : (float, float), optional
        Bounds on ``x_break``. The default is the range from the 3rd to the
        3rd-from-last point in ascending ``x``, so the corner keeps at least
        three points strictly on each side: two points fix a line exactly,
        and three is the fewest that leaves each line a residual degree of
        freedom. Given limits must lie within that range and contain at
        least one sample position.
    model : {'segmented_linear', 'bent_cable'}, optional
        The corner: sharp (the default) or rounded. For the bent cable ``gamma`` is bounded
        to ``[s / 2, span]``, with ``s`` the median spacing of the distinct
        ``x`` and ``span`` their full range: below half a spacing a bend holds
        at most one sample and cannot be resolved, and beyond the span every
        point is already inside it. The bounds are recorded in
        ``params['gamma_limits']``.

    Returns
    -------
    BreakpointResult
        The fits, the corner and slopes with their errors, the comparisons,
        the RSS profile with its display-only interval, ``identified``, the
        warnings raised, and the recipe in ``params``.

    Raises
    ------
    ValueError
        For ``x`` and ``y`` that are not 1-D, of equal length and finite; for
        fewer than seven points; for a ``yerr`` of the wrong shape or not
        finite and positive; for ``limits`` that are not a finite
        ``(lower, upper)`` pair inside the default range, or that hold no
        sample position; and for a ``model`` other than the two above. A fit
        that fails is not an error: it comes back with ``identified=False``
        and a warning.

    Warns
    -----
    UserWarning
        When ``delta_bic <= 0`` (no break preferred), when the corner lies
        within its HESSE error of a limit, and when any fit fails. For the
        bent cable, also when ``gamma`` lies within its HESSE error of its
        lower bound, and when the bend leaves fewer than three points on
        either arm.

    Notes
    -----
    **Why the comparison uses the plain residual sum of squares.** Both
    criteria come from a Gaussian likelihood with one unknown noise level,
    maximised by ordinary least squares: ``-2 ln L = n ln(RSS / n) +
    const``. Hence ``delta_aic = n ln(RSS_linear / RSS_segmented) - 2 (4 -
    2)`` and ``delta_bic = n ln(RSS_linear / RSS_segmented) - (4 - 2) ln n``.
    They use the unweighted RSS whatever ``yerr`` is, because a
    chi-squared built on instrument error columns would inherit their scale,
    and such columns have been seen 130 to 500 times the real scatter. The
    comparison then asks only whether the data's own scatter is reduced
    enough by a corner. ``RSS_segmented`` is minimised exactly over the
    corner inside the limits (knots and each interval's two-line solution),
    so with ``yerr=None`` it is the minimum the segmented fit reaches.

    **The profile interval.** Holding the corner at ``c`` and solving the
    rest exactly gives the profiled ``RSS(c)``. With the noise level
    maximised out, the likelihood-ratio statistic for the corner is ``n
    ln(RSS(c) / RSS_segmented)``. The usual one-parameter 68.3 % (1 sigma)
    construction keeps the corners where that is at most 1, the 68.3 %
    point of chi-squared with one degree of freedom. So the interval is
    every ``c`` with ``RSS(c) <= RSS_segmented * exp(1 / n)``
    (:attr:`~BreakpointResult.profile_threshold`). It is the natural picture
    for a corner, because it shows asymmetry and a second basin that HESSE
    cannot. But it is **display-only**, for the reason in the first caveat
    below: the chi-squared calibration is asymptotic, and it fails exactly
    when the break is weak. HESSE stays canonical.

    **The bent cable.** The same construction applies with five parameters:
    ``delta_bic = n ln(RSS_linear / RSS_bent_cable) - (5 - 2) ln n``, and
    against the sharp corner ``delta_bic_vs_segmented = n ln(RSS_segmented /
    RSS_bent_cable) - (5 - 4) ln n`` (and the AIC forms with ``2 k``). The
    sharp corner is the bent cable at ``gamma = 0``, a value at the edge of
    the allowed range, so that comparison has no chi-squared calibration
    either, and is descriptive too. The bent cable has no closed-form
    profile, so ``RSS_bent_cable`` is found numerically. The centre is held
    at each sample position, and the half-width is minimised by a
    logarithmic grid refined by a bounded minimiser, with the three linear
    parameters exact. The best centre is then refined between samples, and
    MIGRAD's own minimum is taken if it is lower. The profile and its
    interval therefore hold the centre and minimise the half-width at each
    value: the honest analogue of the sharp corner's, not a slice at the
    fitted ``gamma``. Where a bend would reach past an end of the data, only
    the far end of the bend is constrained, and the profile goes flat. That
    is the regime of the arms warning.

    **Limitations**

    Under the no-break hypothesis, the breakpoint location is undefined, so
    the likelihood-ratio statistic does not follow the usual one-degree-of-freedom
    chi-squared law :footcite:p:`davies1987`. ``delta_aic`` and ``delta_bic``
    are descriptive comparisons, not significance tests.

    Rounding and model mismatch can shift the fitted location beyond its HESSE
    interval. Repeat fits over justified ranges for both sharp and bent-cable
    models. Asymmetric rounding need not have a unique physically meaningful
    centre. Correlated residuals are not included in the reported HESSE errors.

    The intersecting-line problem is described by :footcite:p:`hinkley1969`.
    Muggeo :footcite:p:`muggeo2003` describes an iterative linearisation approach;
    Calamari instead seeds MIGRAD from a least-squares profile. The bent-cable
    model and its least-squares theory follow :footcite:p:`chiu2006`.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    >>> import numpy as np
    >>> from calamari.fitting import fit_breakpoint, segmented_linear
    >>> x = np.linspace(4.0, 14.0, 51)
    >>> rng = np.random.default_rng(7)
    >>> y = segmented_linear(x, 8.2, 1.0, 0.2, -0.2) + rng.normal(scale=0.004, size=x.size)
    >>> bp = fit_breakpoint(x, y)
    >>> print(f"x_break = {bp.x_break:.3f} +/- {bp.x_break_err:.3f}, identified={bp.identified}")
    x_break = 8.201 +/- 0.010, identified=True
    """
    if not (isinstance(model, str) and model in _BREAKPOINT_MODELS):
        raise FitInputError(
            f"model must be one of {list(_BREAKPOINT_MODELS)}; got {model!r}. Pass "
            "model='segmented_linear' for a sharp corner (the default) or model='bent_cable' "
            "for a rounded one."
        )
    x_arr, y_arr, yerr_arr, yerr_kind = _validate_data(x, y, yerr)
    limits_given = limits is not None
    lower, upper = _validate_limits(limits, x_arr)
    n = int(x_arr.size)

    # The exact unweighted optimum seeds the weighted fit, inside the limits.
    corner, rss_segmented, points, rss_points = _profile_minimum(x_arr, y_arr, lower, upper)
    (y_break0, slope_below0, slope_change0), _ = _segmented_solve(x_arr, y_arr, corner)
    initial = {
        "x_break": corner,
        "y_break": float(y_break0),
        "slope_below": float(slope_below0),
        "slope_change": float(slope_change0),
    }
    segmented = fit(
        "segmented_linear",
        x_arr,
        y_arr,
        yerr=yerr_arr,
        initial=initial,
        limits={"x_break": (lower, upper)},
    )
    linear = fit("linear", x_arr, y_arr, yerr=yerr_arr)

    slope_line, intercept_line = np.polyfit(x_arr, y_arr, 1)
    residual_line = y_arr - (slope_line * x_arr + intercept_line)
    rss_linear = float(residual_line @ residual_line)
    floor = n * (_EXACT_FIT_RTOL * float(np.max(np.abs(y_arr)))) ** 2
    grid = _segmented_candidates(x_arr)
    grid = grid[(grid >= lower) & (grid <= upper)]

    nan = float("nan")
    bent: FitResult | None = None
    gamma_limits = (nan, nan)
    rss_bent = delta_aic_vs = delta_bic_vs = nan
    if model == "segmented_linear":
        primary = segmented
        gain = _information_gain(rss_linear, rss_segmented, n, floor)
        delta_aic = gain - 2.0 * (_K_SEGMENTED - _K_LINEAR)
        delta_bic = gain - (_K_SEGMENTED - _K_LINEAR) * float(np.log(n))

        profile_threshold = float(rss_segmented * np.exp(1.0 / n))
        profile_interval = _profile_interval(
            lambda c: _segmented_solve(x_arr, y_arr, c)[1], points, rss_points, profile_threshold
        )
        profile = pd.DataFrame({"x_break": grid, "rss": _segmented_profile(x_arr, y_arr, grid)})
    else:
        # The same scheme as the sharp corner: the unweighted optimum, found by
        # profiling, seeds the (possibly weighted) fit. Here it is numerical.
        gamma_limits = _bent_cable_gamma_limits(x_arr)
        centre, gamma0, rss_bent, b_points, b_rss, b_gamma = _bent_cable_profile_minimum(
            x_arr, y_arr, lower, upper, gamma_limits
        )
        (y_break0, slope_below0, slope_change0), _ = _bent_cable_solve(
            x_arr, y_arr, centre, gamma0
        )
        bent = fit(
            "bent_cable",
            x_arr,
            y_arr,
            yerr=yerr_arr,
            initial={
                "x_break": centre,
                "y_break": float(y_break0),
                "slope_below": float(slope_below0),
                "slope_change": float(slope_change0),
                "gamma": gamma0,
            },
            limits={"x_break": (lower, upper), "gamma": gamma_limits},
        )
        primary = bent
        if bent.success:
            # MIGRAD may refine the profiled minimum further; either is an
            # unweighted RSS the bent cable reaches, so the smaller stands.
            residual = y_arr - bent.curve(x_arr)
            rss_bent = min(rss_bent, float(residual @ residual))
        gain = _information_gain(rss_linear, rss_bent, n, floor)
        delta_aic = gain - 2.0 * (_K_BENT_CABLE - _K_LINEAR)
        delta_bic = gain - (_K_BENT_CABLE - _K_LINEAR) * float(np.log(n))
        gain_vs = _information_gain(rss_segmented, rss_bent, n, floor)
        delta_aic_vs = gain_vs - 2.0 * (_K_BENT_CABLE - _K_SEGMENTED)
        delta_bic_vs = gain_vs - (_K_BENT_CABLE - _K_SEGMENTED) * float(np.log(n))

        profile_threshold = float(rss_bent * np.exp(1.0 / n))
        profile_interval = _profile_interval(
            lambda c: _gamma_profiled(x_arr, y_arr, c, gamma_limits)[1],
            b_points,
            b_rss,
            profile_threshold,
        )
        on_grid = np.isin(b_points, grid)
        profile = pd.DataFrame(
            {"x_break": b_points[on_grid], "rss": b_rss[on_grid], "gamma": b_gamma[on_grid]}
        )

    messages: list[str] = []
    if primary.success:
        p, e = primary.parameters, primary.uncertainties
        x_break, x_break_err = float(p["x_break"]), float(e.get("x_break", nan))
        slope_below, slope_below_err = float(p["slope_below"]), float(e.get("slope_below", nan))
        slope_change = float(p["slope_change"])
        slope_change_err = float(e.get("slope_change", nan))
        slope_above, slope_above_err = primary.derive(
            lambda q: q["slope_below"] + q["slope_change"]
        )
        n_below = int(np.sum(x_arr < x_break))
        n_above = int(np.sum(x_arr > x_break))
    else:
        x_break = x_break_err = slope_below = slope_below_err = nan
        slope_above = slope_above_err = slope_change = slope_change_err = nan
        n_below = n_above = 0
        messages.append(
            f"fit_breakpoint: the {model} fit did not converge (success=False), so "
            "there is no corner to report. Check the points passed in, or try "
            f"fit('{model}', x, y) directly to inspect it."
        )
    if bent is not None and not segmented.success:
        messages.append(
            "fit_breakpoint: the segmented_linear fit, kept for comparison, did not converge "
            "(success=False). delta_bic_vs_segmented still uses the exact least-squares sharp "
            "corner, but check the data."
        )
    if not linear.success:
        messages.append(
            "fit_breakpoint: the linear fit did not converge (success=False); the straight "
            "line it is compared against is the exact least-squares one, but check the data."
        )

    if not delta_bic > 0:
        messages.append(
            f"fit_breakpoint: delta_bic = {delta_bic:.3g} <= 0, so the data do not prefer a "
            "change of slope to a straight line. x_break is not identified and its error "
            f"bar (+/- {x_break_err:.3g}) carries no meaning. Report the straight line, or "
            "pass a range of x that brackets the feature with points on both sides."
        )
    at_limit = False
    if primary.success:
        distance = min(x_break - lower, upper - x_break)
        reach = x_break_err if np.isfinite(x_break_err) else 0.0
        at_limit = bool(distance <= reach)
        if at_limit:
            which = lower if (x_break - lower) <= (upper - x_break) else upper
            messages.append(
                f"fit_breakpoint: x_break = {x_break:.6g} is within its HESSE error "
                f"({x_break_err:.3g}) of the limit {which:.6g} (limits {lower:.6g} to "
                f"{upper:.6g}), so the limit, not the data, is placing the corner. Pass a "
                "range of x, or limits=, that leave the feature room on both sides."
            )

    gamma = gamma_err = nan
    arms_established = True
    if bent is not None and bent.success:
        # The curve uses |gamma|; the limits keep it positive, so this is a guard.
        gamma = abs(float(bent.parameters["gamma"]))
        gamma_err = float(bent.uncertainties.get("gamma", nan))
        reach = gamma_err if np.isfinite(gamma_err) else 0.0
        if gamma - gamma_limits[0] <= reach:
            messages.append(
                f"fit_breakpoint: gamma = {gamma:.3g} is within its HESSE error "
                f"({gamma_err:.3g}) of its lower limit {gamma_limits[0]:.3g}, half the median "
                "spacing of x, so the rounding is not resolved: the corner is sharp at this "
                f"resolution (delta_bic_vs_segmented = {delta_bic_vs:.3g}). "
                "model='segmented_linear' describes it as well."
            )
        arm_below = int(np.sum(x_arr < x_break - gamma))
        arm_above = int(np.sum(x_arr > x_break + gamma))
        arms_established = min(arm_below, arm_above) >= MIN_POINTS_PER_SEGMENT
        if not arms_established:
            messages.append(
                f"fit_breakpoint: the bend spans {x_break - gamma:.6g} to "
                f"{x_break + gamma:.6g} (gamma = {gamma:.3g}), leaving {arm_below} point(s) "
                f"below it and {arm_above} above. A straight arm needs "
                f"{MIN_POINTS_PER_SEGMENT} to be judged, so x_break, where the two arms meet, "
                "is poorly defined. Pass a wider range of x around the feature."
            )

    identified = bool(
        primary.success
        and linear.success
        and delta_bic > 0
        and not at_limit
        and arms_established
    )
    for message in messages:
        warnings.warn(message, UserWarning, stacklevel=2)

    params: dict[str, Any] = {
        "model": model,
        "limits": (lower, upper),
        "limits_given": limits_given,
        "yerr": yerr_kind,
    }
    if bent is None:
        params["information_criteria"] = (
            "Gaussian likelihood, noise level unknown, from the unweighted least-squares "
            f"RSS; k = {_K_SEGMENTED} (segmented) vs {_K_LINEAR} (linear)"
        )
        params["profile_interval"] = "RSS(x_break) <= rss_segmented * exp(1/n); display-only"
    else:
        params["gamma_limits"] = gamma_limits
        params["information_criteria"] = (
            "Gaussian likelihood, noise level unknown, from the unweighted least-squares "
            f"RSS; k = {_K_BENT_CABLE} (bent_cable) vs {_K_SEGMENTED} (segmented) vs "
            f"{_K_LINEAR} (linear)"
        )
        params["profile_interval"] = (
            "RSS(x_break), gamma minimised at each, <= rss_bent_cable * exp(1/n); display-only"
        )
    params["calamari_version"] = _calamari_version()
    return BreakpointResult(
        segmented=segmented,
        linear=linear,
        x_break=x_break,
        x_break_err=x_break_err,
        slope_below=slope_below,
        slope_below_err=slope_below_err,
        slope_above=float(slope_above),
        slope_above_err=float(slope_above_err),
        slope_change=slope_change,
        slope_change_err=slope_change_err,
        delta_aic=float(delta_aic),
        delta_bic=float(delta_bic),
        rss_linear=rss_linear,
        rss_segmented=rss_segmented,
        profile=profile,
        profile_threshold=profile_threshold,
        profile_interval=profile_interval,
        n_points=n,
        n_below=n_below,
        n_above=n_above,
        identified=identified,
        warnings=tuple(messages),
        params=params,
        model=model,
        bent_cable=bent,
        gamma=gamma,
        gamma_err=gamma_err,
        rss_bent_cable=float(rss_bent),
        delta_aic_vs_segmented=float(delta_aic_vs),
        delta_bic_vs_segmented=float(delta_bic_vs),
    )
