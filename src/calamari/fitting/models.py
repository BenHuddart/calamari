"""Built-in magnetometry fit functions and the model registry.

Each model is a plain callable ``f(x, *params) -> ndarray`` paired with a
:class:`ModelDefinition` carrying its parameter names, generic starting
values, a provenance-bearing description, and (optionally) a data-driven
``guess`` seed. Models live in the name-keyed :data:`MODELS` registry and are
looked up by name from :func:`calamari.fitting.engine.fit`.

Design follows the asymmetry toolkit's fitting core: a registry of *plain
callables* plus metadata, so a model function stays independently testable and
re-usable (you can call :func:`curie_weiss` directly) while the registry gives
the fit engine a uniform, name-addressable surface. Physics symbol names
(``C``, ``J``, ``T_K``, ``M_s``) are kept against PEP8 deliberately -- the lint
waiver is quarantined to this file in ``pyproject.toml``.

All magnetometry here is CGS-Gaussian (moments in emu, fields in Oe), matching
Quantum Design instruments; the constants used by :func:`brillouin` come from
:mod:`calamari.constants` and nowhere else.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from calamari.constants import K_B_ERG_PER_K, MU_B_EMU
from calamari.exceptions import ModelRegistrationError


@dataclass(frozen=True)
class ModelDefinition:
    """Descriptor for a built-in (or user-registered) fit function.

    A model is a pure callable plus the metadata the fit engine needs to drive
    it: what its parameters are called, where to start them, and how to
    describe it to a human. Kept ``frozen`` so a registered definition cannot
    be mutated out from under the registry.

    Attributes
    ----------
    name : str
        Registry key, e.g. ``'curie_weiss'``. Must be unique within
        :data:`MODELS` (see :func:`register_model`).
    function : Callable
        The model itself, ``f(x, *params) -> numpy.ndarray``, with ``params``
        in the order given by ``param_names``.
    param_names : tuple of str
        Ordered parameter names. The order is load-bearing: it is the order in
        which values are passed to ``function`` and reported by the fit.
    param_defaults : dict of str to float
        Generic starting values, one per parameter. These are *fallback*
        seeds: :func:`calamari.fitting.engine.fit` overlays a data-driven
        ``guess`` (if any) and then the caller's explicit ``initial`` on top of
        them, so a default only survives when nothing better is supplied.
    description : str
        Human-readable description including a literature/provenance note.
    guess : Callable or None, optional
        ``guess(x, y) -> dict`` returning data-driven starting values for some
        or all parameters. Used by the engine to refine ``param_defaults``
        before fitting. ``None`` (the default) means no data-driven seed.
    """

    name: str
    function: Callable[..., NDArray[np.float64]]
    param_names: tuple[str, ...]
    param_defaults: dict[str, float]
    description: str
    guess: Callable[[ArrayLike, ArrayLike], dict[str, float]] | None = None


#: Name-keyed registry of every known model. Populated at import with the
#: built-ins below; extend it via :func:`register_model` (never by mutating
#: this dict directly, so the duplicate-name guard always runs).
MODELS: dict[str, ModelDefinition] = {}


def register_model(defn: ModelDefinition) -> ModelDefinition:
    """Insert *defn* into :data:`MODELS`, rejecting a duplicate name.

    There is exactly one registration path so the uniqueness rule is enforced
    in one place (asymmetry's ``insert_definition`` convention). Returns the
    definition so it can be used as a decorator-style one-liner.

    Parameters
    ----------
    defn : ModelDefinition
        The model to register.

    Returns
    -------
    ModelDefinition
        The same object, for chaining.

    Raises
    ------
    ModelRegistrationError
        If ``defn.name`` is already present in :data:`MODELS`. Re-registering
        an existing name is treated as a programming error, not an overwrite,
        so a typo can never silently shadow a built-in.
    """
    if defn.name in MODELS:
        raise ModelRegistrationError(
            f"Model {defn.name!r} is already registered; choose a different "
            "name (registration never overwrites an existing model)."
        )
    MODELS[defn.name] = defn
    return defn


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------


def curie_weiss(T_K: ArrayLike, C: float, theta: float, chi0: float = 0.0) -> NDArray[np.float64]:
    """Curie-Weiss susceptibility :math:`\\chi(T)=C/(T-\\theta)+\\chi_0`.

    The standard paramagnetic law for a system of local moments with a
    mean-field inter-moment interaction. ``theta`` (the Weiss temperature) is
    positive for net ferromagnetic correlations and negative for
    antiferromagnetic; ``C`` is the Curie constant (from which an effective
    moment follows via ``Sample.mu_eff_from_curie_constant``); ``chi0`` is a
    temperature-independent background (diamagnetic core + Van Vleck).

    Parameters
    ----------
    T_K : array_like
        Temperature in Kelvin. Must avoid the pole at ``T_K == theta``.
    C : float
        Curie constant, in the units of ``chi`` times Kelvin.
    theta : float
        Weiss temperature in Kelvin.
    chi0 : float, optional
        Temperature-independent offset. Default 0.

    Returns
    -------
    numpy.ndarray
        ``chi(T)`` at each ``T_K``.
    """
    T_K = np.asarray(T_K, dtype=float)
    return C / (T_K - theta) + chi0


def inverse_curie_weiss(T_K: ArrayLike, C: float, theta: float) -> NDArray[np.float64]:
    """Inverse Curie-Weiss law ``1/chi(T) = (T - theta) / C``.

    The linearised form fit when working in ``chi^-1`` vs ``T`` (the classic
    Curie-Weiss straight-line analysis): the slope is ``1/C`` and the
    x-intercept is ``theta``. Fitting the inverse rather than ``chi`` directly
    de-weights the low-temperature, large-``chi`` region and is the
    conventional way to read off ``theta`` by eye.

    There is no ``chi0`` term, and there cannot be one: a
    temperature-independent background makes ``1/chi`` curve, so the straight
    line no longer describes the data. Ignoring a real ``chi0`` does not make
    the fit fail; it biases ``theta``, and the bias changes with the fitting
    range, so ``theta`` drifts as ``T_min`` rises. If the sample may carry a
    background (core diamagnetism, Van Vleck, a sample holder), fit ``chi``
    with :func:`curie_weiss`, whose ``chi0`` parameter absorbs it, and use
    :func:`~calamari.fitting.windows.fit_windows` to check that ``theta``
    holds still across fitting ranges.

    Parameters
    ----------
    T_K : array_like
        Temperature in Kelvin.
    C : float
        Curie constant.
    theta : float
        Weiss temperature in Kelvin (the x-intercept).

    Returns
    -------
    numpy.ndarray
        ``1/chi(T)`` at each ``T_K``.
    """
    T_K = np.asarray(T_K, dtype=float)
    return (T_K - theta) / C


def linear(x: ArrayLike, slope: float, intercept: float) -> NDArray[np.float64]:
    """Straight line ``y = slope * x + intercept``.

    A general-purpose baseline model -- e.g. a linear background under a
    feature, or a high-field ``M(H)`` slope used to subtract a paramagnetic or
    diamagnetic contribution before reading a saturation moment.

    Parameters
    ----------
    x : array_like
        Independent variable.
    slope, intercept : float
        Line coefficients.

    Returns
    -------
    numpy.ndarray
        ``y`` at each ``x``.
    """
    x = np.asarray(x, dtype=float)
    return slope * x + intercept


def segmented_linear(
    x: ArrayLike, x_break: float, y_break: float, slope_below: float, slope_change: float
) -> NDArray[np.float64]:
    """Two straight lines joined at a sharp corner (a continuous "broken stick").

    ``y = y_break + slope_below * (x - x_break) + slope_change * max(0, x - x_break)``,
    so the slope is ``slope_below`` for ``x < x_break`` and
    ``slope_below + slope_change`` above it. This is the shape of a kink: a
    susceptibility whose temperature slope changes at an ordering transition
    without a peak, or a magnetisation whose field slope changes at a
    spin-flop.

    It is parametrised *at the break* rather than as ``a + b x + c max(0, x -
    x_break)``. An intercept at ``x = 0`` is an extrapolation far outside the
    data whenever they sit away from the origin, and it is then almost
    perfectly anticorrelated with the slope (a transition at 150 K: the
    correlation rounds to -1 at four decimals). Measured at the corner
    instead, every correlation is set by the data's shape alone and is the
    same wherever the origin of ``x`` lies.

    The corner is **sharp** by construction. A real transition is rounded,
    and a sharp-corner fit to a rounded one places ``x_break`` by an amount
    that depends on where the fitting range sits; see
    :func:`~calamari.fitting.breakpoint.fit_breakpoint` for what that does to
    the error bar and the check that exposes it.

    Parameters
    ----------
    x : array_like
        Independent variable.
    x_break : float
        Location of the corner, in the units of ``x``.
    y_break : float
        Model value at the corner.
    slope_below : float
        Slope for ``x < x_break``.
    slope_change : float
        Change of slope across the corner: the slope above is
        ``slope_below + slope_change``.

    Returns
    -------
    numpy.ndarray
        ``y`` at each ``x``.
    """
    x = np.asarray(x, dtype=float)
    offset = x - x_break
    return y_break + slope_below * offset + slope_change * np.maximum(offset, 0.0)


def _bent_cable_q(offset: NDArray[np.float64], half_width: float) -> NDArray[np.float64]:
    """The bent-cable basis: ``max(0, offset)`` with its corner replaced by a parabola.

    ``(offset + g)^2 / (4 g)`` for ``|offset| < g`` and ``max(0, offset)``
    outside, with ``g = |half_width|``. The two pieces meet with equal value and
    slope at ``offset = -g`` (0 and 0) and ``offset = +g`` (``g`` and 1). The
    parabola is used only strictly inside the join, so at ``g = 0`` no point
    reaches it and the hinge comes back with no division by zero; the
    denominator is replaced by 1 there only to keep the unused branch of
    ``np.where`` finite.
    """
    g = abs(float(half_width))
    inside = np.abs(offset) < g
    denominator = 4.0 * g if g > 0.0 else 1.0
    return np.where(inside, (offset + g) ** 2 / denominator, np.maximum(offset, 0.0))


def bent_cable(
    x: ArrayLike,
    x_break: float,
    y_break: float,
    slope_below: float,
    slope_change: float,
    gamma: float,
) -> NDArray[np.float64]:
    """Two straight lines joined by a parabolic bend of half-width ``gamma`` (a "bent cable").

    ``y = y_break + slope_below * (x - x_break) + slope_change * q(x)``, with

    ``q(x) = (x - x_break + gamma)^2 / (4 gamma)`` for ``|x - x_break| <= gamma``,
    ``q(x) = x - x_break`` above the bend and ``0`` below it.

    This is :func:`segmented_linear` with its corner rounded. A sharp-corner
    fit to a rounded feature can depend on the chosen fitting range. Allowing
    symmetric rounding can reduce this dependence when that shape is appropriate (see
    :func:`~calamari.fitting.breakpoint.fit_breakpoint` with
    ``model="bent_cable"``). The slope changes linearly across the bend, from
    ``slope_below`` at ``x_break - gamma`` to ``slope_below + slope_change`` at
    ``x_break + gamma``. The model and its first derivative are continuous
    everywhere, and at ``gamma = 0`` it is exactly :func:`segmented_linear`.

    The parametrisation is the one :func:`segmented_linear` uses, so the two
    compare parameter by parameter. ``x_break`` is the centre of the bend,
    where the two straight arms, extended, meet, and ``y_break`` is their
    value there. It is *not* the curve's value at ``x_break``, which is
    ``y_break + slope_change * gamma / 4``. This is the bent cable of Chiu,
    Lockhart & Routledge, *J. Am. Stat. Assoc.* 101, 542 (2006), which the
    authors write ``beta0 + beta1 x + beta2 q(x; tau, gamma)`` (``tau`` is
    ``x_break``, ``beta1`` is ``slope_below``, ``beta2`` is
    ``slope_change``, and ``beta0 = y_break - slope_below * x_break``). The
    basis ``q`` is Eq. (2) of Chiu, Lockhart & Routledge, *J. Stat. Plan.
    Inference* 127, 143 (2005).

    **Only the size of** ``gamma`` **matters**: the curve uses ``|gamma|``, so
    a fit without limits (plain :func:`~calamari.fitting.engine.fit` or
    :func:`~calamari.fitting.windows.fit_windows`) may report a negative
    value, meaning the same bend. That also keeps the model defined for every
    value MIGRAD may try. :func:`~calamari.fitting.breakpoint.fit_breakpoint`
    bounds it to be positive.

    Parameters
    ----------
    x : array_like
        Independent variable.
    x_break : float
        Centre of the bend, where the extended arms meet, in the units of ``x``.
    y_break : float
        Value of the extended arms where they meet.
    slope_below : float
        Slope for ``x < x_break - |gamma|``.
    slope_change : float
        Change of slope across the bend: the slope above ``x_break + |gamma|``
        is ``slope_below + slope_change``.
    gamma : float
        Half-width of the bend, in the units of ``x``; ``0`` gives a sharp
        corner.

    Returns
    -------
    numpy.ndarray
        ``y`` at each ``x``.

    Examples
    --------
    >>> import numpy as np
    >>> from calamari.fitting import bent_cable
    >>> x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> print(bent_cable(x, 2.0, 1.0, 0.5, -1.0, 1.0))
    [0.   0.5  0.75 0.5  0. ]
    """
    x = np.asarray(x, dtype=float)
    offset = x - x_break
    return y_break + slope_below * offset + slope_change * _bent_cable_q(offset, gamma)


def brillouin(
    H_oe: ArrayLike, M_s: float, g: float, J: float, T_K: float
) -> NDArray[np.float64]:
    """Brillouin-function magnetisation :math:`M(H)=M_{\\mathrm{s}}B_J(y)`.

    The exact single-ion paramagnetic magnetisation for total angular momentum
    ``J`` in a field, with

    .. math::

        B_J(y) = \\frac{2J+1}{2J}\\coth\\!\\left(\\frac{2J+1}{2J}\\,y\\right)
                 - \\frac{1}{2J}\\coth\\!\\left(\\frac{y}{2J}\\right),
        \\qquad y = \\frac{g\\,J\\,\\mu_{\\mathrm{B}}\\,H}{k_{\\mathrm{B}}\\,T},

    the standard form (e.g. Blundell, *Magnetism in Condensed Matter*, Eq.
    2.34; Ashcroft & Mermin Ch. 31). The argument ``y`` is evaluated in
    **CGS-Gaussian** units so it stays consistent with Quantum Design data:
    ``H`` in Oe, ``mu_B`` in emu (= erg/G) and ``k_B`` in erg/K from
    :mod:`calamari.constants`. Because 1 Oe of applied field corresponds to 1 G
    for the ``mu_B * H`` energy, the product ``mu_B * H_oe`` is in erg and the
    argument is dimensionless.

    ``B_J`` is odd in ``y`` (hence in ``H``), rises linearly at small field
    (recovering the Curie law), and saturates to 1 at large ``y`` -- so
    ``M(H)`` saturates to ``M_s``. In a fit ``g``, ``J`` and ``T_K`` are
    normally *fixed* by the caller (temperature is measured; ``g`` and ``J``
    come from the ion), leaving ``M_s`` -- and thus the moment per ion -- as
    the free quantity of interest.

    Parameters
    ----------
    H_oe : array_like
        Applied field in Oe (may be negative).
    M_s : float
        Saturation moment, in the units of the measured moment (emu).
    g : float
        Lande g-factor of the magnetic ion.
    J : float
        Total angular momentum quantum number (``> 0``).
    T_K : float
        Temperature in Kelvin (``> 0``).

    Returns
    -------
    numpy.ndarray
        ``M(H)`` at each ``H_oe``, same shape as the input.
    """
    H_oe = np.asarray(H_oe, dtype=float)
    y = g * J * MU_B_EMU * H_oe / (K_B_ERG_PER_K * T_K)
    a = (2.0 * J + 1.0) / (2.0 * J)
    b = 1.0 / (2.0 * J)
    # coth diverges at the origin, but B_J(0) = 0 with the two terms cancelling.
    # Evaluate away from y == 0 on a safe surrogate and patch the exact zero,
    # so H = 0 returns 0 cleanly (also enforcing the odd symmetry at the origin)
    # instead of a nan from 1/tanh(0).
    small = y == 0.0
    y_safe = np.where(small, 1.0, y)
    b_j = a / np.tanh(a * y_safe) - b / np.tanh(b * y_safe)
    b_j = np.where(small, 0.0, b_j)
    return M_s * b_j


# ---------------------------------------------------------------------------
# Guess functions
# ---------------------------------------------------------------------------


def _curie_weiss_guess(x: ArrayLike, y: ArrayLike) -> dict[str, float]:
    """Data-driven starting values for a Curie-Weiss ``chi(T)`` fit.

    Uses the linearised inverse law: on a plot of ``1/chi`` vs ``T`` the
    Curie-Weiss form is a straight line of slope ``1/C`` and x-intercept
    ``theta``. A two-point extrapolation through the coldest and hottest
    samples gives ``C`` from the slope and ``theta`` from the intercept -- good
    enough to seed Minuit, which then refines from there. ``chi0`` is seeded at
    0 (no background assumed a priori).

    Degenerate inputs (fewer than two distinct temperatures, a zero or
    non-finite slope) fall back to ``C = 1``, ``theta = 0`` so the caller still
    gets a usable dict rather than a nan seed.
    """
    T_K = np.asarray(x, dtype=float)
    chi = np.asarray(y, dtype=float)
    fallback = {"C": 1.0, "theta": 0.0, "chi0": 0.0}
    if T_K.size < 2:
        return fallback
    i_lo = int(np.argmin(T_K))
    i_hi = int(np.argmax(T_K))
    dT = T_K[i_hi] - T_K[i_lo]
    if dT == 0.0 or chi[i_lo] == 0.0 or chi[i_hi] == 0.0:
        return fallback
    inv_lo = 1.0 / chi[i_lo]
    inv_hi = 1.0 / chi[i_hi]
    slope = (inv_hi - inv_lo) / dT  # = 1 / C
    if slope == 0.0 or not np.isfinite(slope):
        return fallback
    C = 1.0 / slope
    intercept = inv_lo - slope * T_K[i_lo]  # = -theta / C
    theta = -intercept * C
    if not (np.isfinite(C) and np.isfinite(theta)):
        return fallback
    return {"C": float(C), "theta": float(theta), "chi0": 0.0}


#: The fewest points a segment of :func:`segmented_linear` may keep on each
#: side of the corner. Two points fix a line exactly, leaving nothing to judge
#: it by; three is the fewest that leaves each segment's line a residual degree
#: of freedom. Candidate corners in the profile, and the default limits of
#: :func:`~calamari.fitting.breakpoint.fit_breakpoint`, keep at least this many
#: points strictly below and strictly above.
MIN_POINTS_PER_SEGMENT = 3


def _segmented_candidates(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sample positions eligible as a corner: at least three points strictly each side.

    Distinct values of ``x`` only, in ascending order, so tied or unsorted
    input gives the same grid as the sorted, de-duplicated one.
    """
    xs = np.unique(x)
    below = np.searchsorted(np.sort(x), xs, side="left")  # count of x < c
    above = x.size - np.searchsorted(np.sort(x), xs, side="right")  # count of x > c
    keep = (below >= MIN_POINTS_PER_SEGMENT) & (above >= MIN_POINTS_PER_SEGMENT)
    return xs[keep]


def _segmented_solve(
    x: NDArray[np.float64], y: NDArray[np.float64], x_break: float
) -> tuple[NDArray[np.float64], float]:
    """Exact least squares of :func:`segmented_linear` with the corner held at *x_break*.

    With ``x_break`` fixed the model is linear in the other three parameters,
    so their best values and the residual sum of squares follow in one
    ``lstsq`` call. Returns ``((y_break, slope_below, slope_change), rss)``.
    The RSS is summed from the residuals rather than read from ``lstsq``,
    which omits it for a rank-deficient design.
    """
    offset = x - x_break
    design = np.column_stack([np.ones_like(x), offset, np.maximum(offset, 0.0)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    return coef, float(residual @ residual)


def _segmented_profile(
    x: NDArray[np.float64], y: NDArray[np.float64], candidates: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The residual sum of squares at each candidate corner, the others profiled out."""
    return np.array([_segmented_solve(x, y, float(c))[1] for c in candidates], dtype=float)


def _segmented_linear_guess(x: ArrayLike, y: ArrayLike) -> dict[str, float]:
    """Seed :func:`segmented_linear` at the best corner on the data's own sample grid.

    The least-squares cost is multimodal in ``x_break``: noise puts local
    minima in it, and with no break at all it is nearly flat. MIGRAD only
    finds the minimum of the basin it starts in, so a generic seed lands in
    whichever basin is nearest rather than the best one. This guess removes that dependence on luck: it holds
    the corner at every sample position with at least
    :data:`MIN_POINTS_PER_SEGMENT` points strictly on each side, solves the
    remaining three parameters exactly by linear least squares, and returns
    the candidate with the smallest residual sum of squares, together with
    its three exact coefficients. MIGRAD then refines the corner off the grid.

    Deterministic: no random restarts, and a tie goes to the lowest ``x``.
    Non-finite points are left out of the profile (the fit itself will still
    fail on them). With too few finite points to leave one candidate, it
    falls back to a straight line through the data with the corner at the
    median ``x``: a usable seed for a problem ``fit()`` then judges.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        return {}
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size == 0:
        return {}
    candidates = _segmented_candidates(x)
    if candidates.size == 0:
        x_break = float(np.median(x))
        if np.ptp(x) > 0.0 and x.size >= 2:
            slope, intercept = np.polyfit(x, y, 1)
        else:
            slope, intercept = 0.0, float(np.mean(y))
        return {
            "x_break": x_break,
            "y_break": float(intercept + slope * x_break),
            "slope_below": float(slope),
            "slope_change": 0.0,
        }
    rss = _segmented_profile(x, y, candidates)
    best = float(candidates[int(np.argmin(rss))])
    (y_break, slope_below, slope_change), _ = _segmented_solve(x, y, best)
    return {
        "x_break": best,
        "y_break": float(y_break),
        "slope_below": float(slope_below),
        "slope_change": float(slope_change),
    }


#: Where the :func:`bent_cable` guess starts ``gamma``, in median sample
#: spacings of ``x``. A bend narrower than one spacing may hold no sample at
#: all, and then the cost does not change with ``gamma``: MIGRAD's first step
#: sees a flat direction. Two spacings put about four samples inside the bend,
#: so the cost has a gradient in ``gamma`` from the start, whether the true
#: corner is sharp or rounded, while staying a small part of any range with
#: :data:`MIN_POINTS_PER_SEGMENT` points on each arm, so the seed is still
#: close to the sharp-corner solution the rest of the guess comes from.
BENT_CABLE_GAMMA_START_SPACINGS = 2.0

#: The smallest ``gamma`` :func:`~calamari.fitting.breakpoint.fit_breakpoint`
#: allows, in median sample spacings of ``x``. A bend of half-width under
#: half a spacing is narrower than the gap between samples, so it holds at
#: most one of them. Every other point then sees exactly the sharp corner
#: with the same other parameters, and ``gamma`` can do nothing but fit that
#: one point's noise: rounding on that scale is not resolved by the sampling.
#: So half a spacing is the resolution floor, and a ``gamma`` that ends there
#: (within its error) says the corner is sharp *at this resolution*. Keeping
#: the floor positive also keeps ``gamma`` away from zero, where the cost is
#: flat in it and HESSE cannot invert.
BENT_CABLE_GAMMA_FLOOR_SPACINGS = 0.5


def _median_spacing(x: NDArray[np.float64]) -> float:
    """The median gap between distinct values of ``x``; NaN with fewer than two."""
    xs = np.unique(x[np.isfinite(x)])
    if xs.size < 2:
        return float("nan")
    return float(np.median(np.diff(xs)))


def _bent_cable_gamma_limits(x: NDArray[np.float64]) -> tuple[float, float]:
    """The ``(lower, upper)`` range :func:`~calamari.fitting.breakpoint.fit_breakpoint` gives ``gamma``.

    Lower: :data:`BENT_CABLE_GAMMA_FLOOR_SPACINGS` median spacings, the
    resolution floor. Upper: the full span of ``x``. With the centre inside
    the data, a half-width that large already puts every point in the bend,
    after which nothing about the curve on the data changes with ``gamma``
    (a flat direction, not a fit).
    """
    return (
        BENT_CABLE_GAMMA_FLOOR_SPACINGS * _median_spacing(x),
        float(np.ptp(x)),
    )


def _bent_cable_solve(
    x: NDArray[np.float64], y: NDArray[np.float64], x_break: float, gamma: float
) -> tuple[NDArray[np.float64], float]:
    """Exact least squares of :func:`bent_cable` with the centre and half-width held.

    As for :func:`_segmented_solve`: with ``x_break`` and ``gamma`` fixed the
    model is linear in the other three parameters. Returns ``((y_break,
    slope_below, slope_change), rss)``.
    """
    offset = x - x_break
    design = np.column_stack([np.ones_like(x), offset, _bent_cable_q(offset, gamma)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    return coef, float(residual @ residual)


def _bent_cable_guess(x: ArrayLike, y: ArrayLike) -> dict[str, float]:
    """Seed :func:`bent_cable` from the sharp corner, with a bend of two sample spacings.

    The corner, its value and the two slopes come from
    :func:`_segmented_linear_guess`, the exact least-squares profile over the
    sample positions, for the same reason as there: the cost is multimodal in
    ``x_break`` and MIGRAD only finds the basin it starts in. On a rounded
    corner that seed is biased by a fraction of the rounding, which is well
    inside the basin, so MIGRAD moves it. ``gamma`` starts at
    :data:`BENT_CABLE_GAMMA_START_SPACINGS` median spacings (see there for
    why). Deterministic. With fewer than two distinct ``x`` there is no
    spacing, and ``gamma`` is left to the model's default.
    """
    seed = _segmented_linear_guess(x, y)
    x_arr = np.asarray(x, dtype=float).ravel()
    spacing = _median_spacing(x_arr)
    if seed and np.isfinite(spacing) and spacing > 0.0:
        seed["gamma"] = BENT_CABLE_GAMMA_START_SPACINGS * spacing
    return seed


# ---------------------------------------------------------------------------
# Registry population
# ---------------------------------------------------------------------------

register_model(
    ModelDefinition(
        name="curie_weiss",
        function=curie_weiss,
        param_names=("C", "theta", "chi0"),
        param_defaults={"C": 1.0, "theta": 0.0, "chi0": 0.0},
        description=(
            "Curie-Weiss susceptibility C/(T - theta) + chi0. Standard "
            "mean-field paramagnet law; theta is the Weiss temperature, C the "
            "Curie constant, chi0 a temperature-independent background."
        ),
        guess=_curie_weiss_guess,
    )
)

register_model(
    ModelDefinition(
        name="inverse_curie_weiss",
        function=inverse_curie_weiss,
        param_names=("C", "theta"),
        param_defaults={"C": 1.0, "theta": 0.0},
        description=(
            "Inverse Curie-Weiss law (T - theta)/C for chi^-1 vs T straight-"
            "line fits; slope 1/C, x-intercept theta."
        ),
    )
)

register_model(
    ModelDefinition(
        name="linear",
        function=linear,
        param_names=("slope", "intercept"),
        param_defaults={"slope": 1.0, "intercept": 0.0},
        description="Straight line slope*x + intercept (baselines, high-field slopes).",
    )
)

register_model(
    ModelDefinition(
        name="segmented_linear",
        function=segmented_linear,
        param_names=("x_break", "y_break", "slope_below", "slope_change"),
        # Fallback seeds only: the guess always supplies all four for data it
        # can use. slope_change is non-zero so that x_break has a gradient
        # even from these (with no change of slope the corner can sit anywhere).
        param_defaults={"x_break": 0.0, "y_break": 0.0, "slope_below": 1.0, "slope_change": 1.0},
        description=(
            "Two straight lines meeting at a sharp corner: y_break + "
            "slope_below*(x - x_break) + slope_change*max(0, x - x_break). "
            "Continuous two-phase regression (Hinkley, Biometrika 56, 495 "
            "(1969)); seeded from an exact least-squares profile over the "
            "sample positions. See calamari.fitting.fit_breakpoint for the "
            "comparison against no break."
        ),
        guess=_segmented_linear_guess,
    )
)

register_model(
    ModelDefinition(
        name="bent_cable",
        function=bent_cable,
        param_names=("x_break", "y_break", "slope_below", "slope_change", "gamma"),
        # Fallback seeds only, as for segmented_linear: the guess supplies all
        # five for data with at least two distinct x. gamma is non-zero so that
        # it has a gradient even from these.
        param_defaults={
            "x_break": 0.0,
            "y_break": 0.0,
            "slope_below": 1.0,
            "slope_change": 1.0,
            "gamma": 1.0,
        },
        description=(
            "Two straight lines joined by a parabolic bend of half-width gamma: "
            "y_break + slope_below*(x - x_break) + slope_change*q(x), with "
            "q = (x - x_break + gamma)^2/(4 gamma) inside the bend. The bent "
            "cable (Chiu, Lockhart & Routledge, J. Am. Stat. Assoc. 101, 542 "
            "(2006)); segmented_linear at gamma = 0. Seeded from the "
            "segmented_linear profile, gamma at two sample spacings. See "
            "calamari.fitting.fit_breakpoint(model='bent_cable')."
        ),
        guess=_bent_cable_guess,
    )
)

register_model(
    ModelDefinition(
        name="brillouin",
        function=brillouin,
        param_names=("M_s", "g", "J", "T_K"),
        param_defaults={"M_s": 1.0, "g": 2.0, "J": 0.5, "T_K": 2.0},
        description=(
            "Brillouin-function magnetisation M_s * B_J(g J mu_B H / k_B T), "
            "CGS argument (H in Oe). Standard single-ion paramagnet form "
            "(Blundell Eq. 2.34); g, J, T_K normally fixed by the caller."
        ),
    )
)
