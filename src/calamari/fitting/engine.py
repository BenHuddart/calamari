"""Least-squares fitting driver (iminuit) and its result object.

A thin, deterministic wrapper over :class:`iminuit.Minuit` driving
:class:`iminuit.cost.LeastSquares`, matching the asymmetry toolkit's house
pattern: MIGRAD to the minimum, then an explicit HESSE for canonical symmetric
uncertainties, with MINOS available on request as *display-only* asymmetric
errors (HESSE stays authoritative). The single entry point is :func:`fit`; it
returns ``success=False`` for recognised numerical failures; unexpected model or
backend exceptions propagate. That includes a problem with no more
points than free parameters, which is refused before any minimisation: with
zero degrees of freedom the points fix the parameters exactly (or not at all),
so there is no fit to judge and no scatter to estimate uncertainties from.

Error convention (important, and deliberately scipy-compatible)
---------------------------------------------------------------
When the caller supplies per-point ``yerr``, those errors are taken at face
value and the reported uncertainties are absolute (the
``scipy.optimize.curve_fit`` ``absolute_sigma=True`` convention). When ``yerr``
is ``None``, the data are assumed to scatter about the model by an unknown
constant sigma that is estimated from the fit residuals, and the parameter
uncertainties are the unit-error HESSE errors rescaled by ``sqrt(chi2/dof)``.
That is exactly ``curve_fit``'s default ``absolute_sigma=False`` behaviour, and
it is flagged on the result as ``yerr_assumed=True`` so a caller can tell which
regime produced the error bars. ``chi_squared`` is then the raw sum of squared
residuals, in the units of ``y`` squared.

Why assumed errors are not unit errors
--------------------------------------
The obvious implementation -- minimise with ``yerr = 1`` and rescale
afterwards -- is a trap, because MIGRAD's convergence test is *absolute*: it
stops when the estimated distance to the minimum (EDM) falls below ~2e-4 in
units of the cost. With unit errors the cost is the raw sum of squared
residuals, so its size is set by the units ``y`` happens to be in. A
susceptibility in emu/Oe is 1e-5 to 1e-7, the cost at the *starting point* is
already ~1e-9, and MIGRAD declares convergence before it has moved: the fit
reports ``success=True`` with the starting guess, or near it. Noise-free
``curie_weiss(T, 0.5, -20, 2e-4)`` over 50-300 K came back as C = 0.575,
theta = -29.7 K, chi0 = -2.6e-5, with no sign that anything was wrong. Very
large data fail the other way: the goal becomes unreachable below the cost's
own round-off, and a good fit is reported as a failure.

So with assumed errors the fit is minimised against a constant sigma on the
data's own scale, in two passes. The first uses ``std(y)``, which is enough to
bring the cost off the floor. The second uses the first pass's residual
scatter, ``sqrt(sum(r^2)/dof)`` -- by definition the sigma
``absolute_sigma=False`` assumes -- so the cost sits at ~dof and MIGRAD
converges as it would on honestly weighted data. A constant sigma does not
move the least-squares minimum, and HESSE errors scale with sigma while the
rescaling factor scales with its inverse, so the reported uncertainties are
exactly those of the unit-error recipe; only now the minimum they describe
has actually been reached, and nothing about the fit depends on the units of
``y``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from calamari.exceptions import (
    FitConstraintError,
    FitInputError,
    FitParameterError,
    InputTypeError,
    UnknownModelError,
)
from calamari.fitting.models import MODELS, ModelDefinition

#: Residual scatter below this fraction of ``max(|y|)`` is treated as an exact
#: fit, and the assumed-error second pass is skipped. The square root of
#: double-precision machine epsilon (~1.5e-8) is the conventional boundary
#: below which finite differences stop resolving a function: a cost weighted
#: by a sigma much smaller than that is dominated by round-off in ``y - f``.
#: Above it, the first pass has left real scatter to refine against.
_EXACT_FIT_RTOL = float(np.sqrt(np.finfo(float).eps))

#: Central-difference step for :meth:`FitResult.derive`, as a fraction of each
#: parameter's own HESSE sigma. Scaling by sigma rather than by the value
#: makes the step scale-free (``curie_weiss`` parameters span ~1e-4 for chi0
#: to ~1e2 for theta, and a value can be zero) and keeps every probe well
#: inside the region the covariance describes. The size is the usual balance
#: for a central difference: the truncation error is second order in the
#: step, ~(1e-3 sigma / L)^2 relative for a function curving on a scale L,
#: and first-order propagation is only trusted when L >> sigma anyway, so it
#: is below 1e-6; the round-off error in the slope is ~eps |f| / (1e-3 sigma),
#: i.e. ~2e-13 |f| / sigma_f relative to the propagated error, negligible
#: unless the derived quantity is known to better than ~1 part in 1e10.
_DERIVE_STEP_FRACTION = 1e-3


@dataclass
class FitResult:
    """Outcome of a single :func:`fit` call.

    Attributes
    ----------
    success : bool
        Whether MIGRAD converged to a valid minimum. ``False`` is a normal
        return value (e.g. too few points, a degenerate model), not an error.
        "Too few" means no more points than free parameters (``dof <= 0``):
        that problem is never minimised, and ``parameters`` holds the starting
        values.
    model : str
        Name of the fitted model.
    parameters : dict of str to float
        Best-fit value for every parameter (free *and* fixed).
    uncertainties : dict of str to float
        HESSE 1-sigma uncertainties for the free parameters (rescaled when
        ``yerr_assumed``; see the module docstring). Empty if HESSE failed;
        fixed parameters are reported with 0.0.
    covariance : numpy.ndarray or None
        Covariance matrix over the free parameters (rescaled to match
        ``uncertainties``), or ``None`` if unavailable. Row/column order is
        given by ``covariance_parameters``.
    covariance_parameters : tuple of str
        Free-parameter names labelling ``covariance``'s axes.
    chi_squared : float
        Least-squares cost at the minimum. With supplied ``yerr`` this is the
        weighted chi-squared; with assumed errors it is the raw sum of squared
        residuals.
    reduced_chi_squared : float
        ``chi_squared / dof``; NaN when the problem was not minimised.
    dof : int
        Degrees of freedom, ``n_points`` minus the number of *free*
        parameters (fixed parameters do not consume a degree of freedom).
        Reported as is, not clamped: an underdetermined problem shows zero or
        a negative number, which is why it failed. Every successful fit has
        ``dof >= 1``.
    n_points : int
        Number of data points fitted.
    fixed : dict of str to float
        Parameters that were held fixed, and the values they were held at.
    yerr_assumed : bool
        ``True`` when no ``yerr`` was supplied and uncertainties were rescaled
        by ``sqrt(chi2/dof)`` (curve_fit ``absolute_sigma=False`` equivalent).
    minos_errors : dict or None
        Asymmetric ``(lower, upper)`` MINOS offsets per free parameter when
        ``run_minos=True`` and the fit is valid, else ``None``. **Display-only**
        -- HESSE (``uncertainties``) remains the canonical error estimate.
    """

    success: bool
    model: str
    parameters: dict[str, float]
    uncertainties: dict[str, float]
    covariance: NDArray[np.float64] | None
    covariance_parameters: tuple[str, ...]
    chi_squared: float
    reduced_chi_squared: float
    dof: int
    n_points: int
    fixed: dict[str, float]
    yerr_assumed: bool
    minos_errors: dict[str, tuple[float, float]] | None = None
    #: The fitted model callable, kept privately so :meth:`curve` can evaluate
    #: the best fit without re-looking-up the registry. Not part of the public
    #: field set and excluded from repr/eq.
    _function: object | None = field(default=None, repr=False, compare=False)
    #: Ordered parameter names for :meth:`curve`'s positional call.
    _param_order: tuple[str, ...] = field(default=(), repr=False, compare=False)

    def curve(self, x: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the best-fit model at *x*.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the fitted model.

        Returns
        -------
        numpy.ndarray
            The model with its best-fit parameters at each ``x``.

        Raises
        ------
        RuntimeError
            If the result carries no model function (should not happen for a
            result produced by :func:`fit`).
        """
        if self._function is None:
            raise RuntimeError("FitResult has no model function to evaluate.")
        values = [self.parameters[name] for name in self._param_order]
        return np.asarray(self._function(np.asarray(x, dtype=float), *values), dtype=float)

    def derive(self, func: Callable[[Mapping[str, float]], float]) -> tuple[float, float]:
        """A quantity derived from the fitted parameters, with its propagated error.

        The fit reports ``C`` and ``theta``; the physics usually wants
        something built from them, such as the effective moment. Computing
        that by hand drops its uncertainty, and computing the uncertainty by
        hand from ``uncertainties`` alone ignores the correlations between
        parameters. This evaluates ``func`` at the best fit and propagates
        :attr:`covariance` through it, with no refit.

        The quantity is the caller's to define, deliberately. The fit does
        not know the mole basis of the ``chi`` it was given, so a
        "built-in" ``mu_eff`` would happily turn a raw emu/Oe Curie constant
        into a value with the wrong normalisation.
        Convert ``chi`` to the basis you mean (e.g. with
        :meth:`~calamari.sample.Sample.chi_to_chi_molar_per_ion`) before
        fitting, then say what to derive.

        Parameters
        ----------
        func : callable
            Called as ``func(parameters)`` with a fresh ``dict`` of every
            parameter, free and fixed (a copy of :attr:`parameters`, so it
            may be modified freely), and must return a single real number,
            e.g. ``lambda p: mu_eff_from_curie_constant(p["C"])``.

        Returns
        -------
        value : float
            ``func`` at the best fit.
        err : float
            Its first-order (delta-method) 1-sigma uncertainty,
            ``sqrt(J C J^T)``, with ``C`` the :attr:`covariance` over
            :attr:`covariance_parameters` and ``J`` the gradient of ``func``
            with respect to those parameters. It inherits the fit's error
            convention: absolute with supplied ``yerr``, residual-scaled when
            :attr:`yerr_assumed`.

        Raises
        ------
        TypeError
            If ``func`` returns something that is not a single number.
        Exception
            Whatever ``func`` itself raises is propagated unchanged: that is
            a fault in the caller's code, not a fit that failed.

        Notes
        -----
        ``J`` is taken by central differences, stepping each free parameter
        by ``1e-3`` of its own HESSE sigma: scale-free (the step for a
        ``chi0`` of 1e-4 and a ``theta`` of 1e2 are each sized to that
        parameter's precision) and always inside the region the covariance
        describes. A free parameter with zero sigma contributes no variance
        whatever the slope, so it is not stepped; nor is one whose step is
        lost in the rounding of its value (sigma below ~1e-13 of it), whose
        variance is equally beyond representing. Fixed parameters are not in
        the covariance and contribute nothing either: their values are
        exact by assumption.

        What comes back when there is nothing to propagate:

        - an unsuccessful fit gives ``(nan, nan)`` without calling ``func``:
          where MIGRAD stopped is not an estimate to derive anything from,
          matching the NaN failure rows of
          :func:`~calamari.fitting.windows.fit_windows`;
        - a non-finite value (``sqrt`` of a negative ``C``, say) gives
          ``(nan, nan)``. NumPy's own warning about it, if any, is let
          through: it is the caller's function reporting on the best fit;
        - no covariance (HESSE failed) gives ``(value, nan)``;
        - a non-finite ``func`` at a probe point -- the best fit sits within
          a small fraction of a sigma of a domain edge -- gives
          ``(value, nan)``: there is no slope to propagate. Floating-point
          warnings raised at the probe points are silenced, since the
          caller never asked about those points and the NaN says it all.

        **First order only.** The propagation linearises ``func`` over the
        parameters' uncertainty. That is exact for a linear ``func`` and
        good whenever the relative errors are small, but it breaks down when
        they are large *and* ``func`` is strongly non-linear on that scale
        -- ``1/x`` or ``sqrt(x)`` of a parameter known to 30 %, or near
        zero. The symmetric error it returns is then misleading. Use a
        profile-likelihood interval (MINOS, via ``run_minos=True`` on a
        reparametrised model whose parameter *is* the derived quantity) or a
        Monte Carlo over the covariance instead.

        Examples
        --------
        >>> import numpy as np
        >>> from calamari import fit, mu_eff_from_curie_constant
        >>> from calamari.simulate import curie_weiss_chi
        >>> T = np.linspace(20, 300, 57)
        >>> rng = np.random.default_rng(3)
        >>> chi = curie_weiss_chi(T, 0.5, -20.0, 2e-4) * (1 + 0.01 * rng.standard_normal(T.size))
        >>> result = fit("curie_weiss", T, chi)
        >>> mu, mu_err = result.derive(lambda p: mu_eff_from_curie_constant(p["C"]))
        >>> print(f"mu_eff = {mu:.3f} +/- {mu_err:.3f} muB")
        mu_eff = 1.982 +/- 0.014 muB
        """
        if not self.success:
            return float("nan"), float("nan")

        def evaluate(parameters: dict[str, float]) -> float:
            returned = func(dict(parameters))
            try:
                return float(returned)
            except (TypeError, ValueError):
                raise InputTypeError(
                    f"derive(): func must return a single number; it returned {returned!r}."
                ) from None

        parameters = dict(self.parameters)
        value = evaluate(parameters)
        if not np.isfinite(value):
            return float("nan"), float("nan")
        if self.covariance is None:
            return value, float("nan")

        names = self.covariance_parameters
        covariance = np.asarray(self.covariance, dtype=float)
        if not np.all(np.isfinite(covariance)):
            return value, float("nan")
        sigmas = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
        jacobian = np.zeros(len(names))
        with np.errstate(all="ignore"):
            for i, (name, sigma) in enumerate(zip(names, sigmas)):
                if sigma == 0.0:
                    continue  # no variance to carry, and no step to take
                step = _DERIVE_STEP_FRACTION * sigma
                above = parameters[name] + step
                below = parameters[name] - step
                if above == below:
                    # sigma below the value's own floating-point resolution:
                    # a variance too small to represent, treated as exact.
                    continue
                f_above = evaluate({**parameters, name: above})
                f_below = evaluate({**parameters, name: below})
                if not (np.isfinite(f_above) and np.isfinite(f_below)):
                    return value, float("nan")
                # Divide by the step actually taken, (centre + h) - (centre - h),
                # not 2h: when sigma is tiny beside the value the two differ by
                # the rounding of centre + h.
                jacobian[i] = (f_above - f_below) / (above - below)
        variance = float(jacobian @ covariance @ jacobian)
        # A valid HESSE covariance is positive definite, so a negative
        # quadratic form can only be round-off about a true zero.
        return value, float(np.sqrt(max(variance, 0.0)))

    def summary(self) -> str:
        """Return an aligned, human-readable summary block.

        Lists the fit status and goodness-of-fit, then one line per parameter:
        free parameters as ``value +/- uncertainty`` (with the MINOS band
        appended when present), fixed parameters marked ``(fixed)``. Every free
        parameter name appears, so the block doubles as a quick check that a
        fit touched the parameters you expected.
        """
        lines = [
            f"FitResult: model={self.model!r}  "
            f"{'converged' if self.success else 'FAILED'}",
            f"  chi2 = {self.chi_squared:.6g}   "
            f"chi2/dof = {self.reduced_chi_squared:.6g}   "
            f"dof = {self.dof}   n_points = {self.n_points}",
        ]
        if self.yerr_assumed:
            lines.append("  (yerr assumed: uncertainties rescaled by sqrt(chi2/dof))")
        width = max((len(n) for n in self.parameters), default=0)
        for name in self._param_order or tuple(self.parameters):
            value = self.parameters[name]
            if name in self.fixed:
                lines.append(f"  {name:>{width}} = {value:.6g}   (fixed)")
                continue
            err = self.uncertainties.get(name)
            if err is None:
                lines.append(f"  {name:>{width}} = {value:.6g}   +/- (no HESSE)")
            else:
                line = f"  {name:>{width}} = {value:.6g} +/- {err:.6g}"
                if self.minos_errors and name in self.minos_errors:
                    lo, hi = self.minos_errors[name]
                    line += f"   MINOS [{lo:+.3g}, {hi:+.3g}]"
                lines.append(line)
        return "\n".join(lines)


def _resolve_model(model: str | ModelDefinition) -> ModelDefinition:
    """Return the :class:`ModelDefinition` for a name or pass a definition through."""
    if isinstance(model, ModelDefinition):
        return model
    try:
        return MODELS[model]
    except KeyError:
        raise UnknownModelError(
            f"Unknown model {model!r}; registered models are "
            f"{sorted(MODELS)}."
        ) from None


def _named_model(defn: ModelDefinition):
    """Wrap ``defn.function`` with an explicit ``(x, *param_names)`` signature.

    :class:`iminuit.cost.LeastSquares` reads parameter names by introspecting
    the model signature. Our model functions take ``*params`` (so they stay
    directly callable), which is not introspectable, so we attach a synthetic
    ``__signature__`` built from ``param_names`` -- the version-robust way to
    tell iminuit what the free parameters are called.
    """

    def model_call(x, *params):
        return defn.function(x, *params)

    sig_params = [inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    sig_params += [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in defn.param_names
    ]
    model_call.__signature__ = inspect.Signature(sig_params)
    return model_call


def _resolve_initial(
    defn: ModelDefinition,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    initial: dict[str, float] | None,
) -> dict[str, float]:
    """Layer starting values: generic defaults < data guess < caller's initial.

    ``param_defaults`` provides a value for every parameter as the base layer,
    a model's optional ``guess(x, y)`` refines it with data-driven seeds, and
    an explicit ``initial`` from the caller wins over both.
    """
    values = dict(defn.param_defaults)
    if defn.guess is not None:
        guessed = defn.guess(x, y)
        values.update({k: v for k, v in guessed.items() if k in defn.param_defaults})
    if initial:
        unknown = set(initial) - set(defn.param_names)
        if unknown:
            raise FitParameterError(
                f"initial contains unknown parameter(s) {sorted(unknown)} for "
                f"model {defn.name!r}; parameters are {defn.param_names}."
            )
        values.update(initial)
    return values


def _seed_sigma(y: NDArray[np.float64]) -> float:
    """First-pass assumed sigma: the spread of the data themselves.

    Any constant on the data's own scale moves MIGRAD's absolute convergence
    goal to that scale, which is all the first pass needs. The standard
    deviation of ``y`` is that scale for any curve with some variation. It is
    zero for constant data and non-finite for data carrying NaN or inf, and
    neither can weight a cost, so both fall back to 1.0.
    """
    # Non-finite data are handled by the fallback below, and the fit itself
    # then fails (FT-1); numpy's warning on the way would only be noise.
    with np.errstate(invalid="ignore", over="ignore"):
        spread = float(np.std(y))
    if not np.isfinite(spread) or spread <= 0.0:
        return 1.0
    return spread


def _refined_sigma(
    cost_value: float, sigma: float, dof: int, y: NDArray[np.float64]
) -> float | None:
    """Second-pass assumed sigma: the residual scatter of the first pass.

    ``sqrt(sum(r^2) / dof)`` *is* the constant sigma that
    ``absolute_sigma=False`` assumes, so minimising against it puts the cost
    at ~dof and MIGRAD's convergence goal where it would be for honestly
    weighted data. ``std(y)`` alone is not enough for that: it tracks the size
    of the signal, not of the scatter, so a smooth curve with little noise
    still has a cost far below dof and MIGRAD can stop short again.

    Returns ``None`` when there is no scatter to refine against, and the first
    pass should stand: residuals that are not finite, or that sit at the
    floating-point round-off of the data themselves (an exact fit). The
    second case matters: minimising against a sigma of order 1e-16 of the data
    turns the cost into round-off noise, and MIGRAD then reports a perfectly
    good fit as invalid.
    """
    refined = float(np.sqrt(cost_value * sigma**2 / dof))
    finite = y[np.isfinite(y)]
    floor = _EXACT_FIT_RTOL * float(np.max(np.abs(finite))) if finite.size else 0.0
    if not np.isfinite(refined) or refined <= floor:
        return None
    return refined


def fit(
    model: str | ModelDefinition,
    x: ArrayLike,
    y: ArrayLike,
    *,
    yerr: ArrayLike | None = None,
    initial: dict[str, float] | None = None,
    limits: dict[str, tuple[float, float]] | None = None,
    fixed: dict[str, float] | None = None,
    run_minos: bool = False,
) -> FitResult:
    """Least-squares fit of *model* to ``(x, y)`` via iminuit MIGRAD + HESSE.

    Parameters
    ----------
    model : str or ModelDefinition
        A registered model name (see
        :data:`calamari.fitting.MODELS <calamari.fitting.models.MODELS>`)
        or a :class:`~calamari.fitting.models.ModelDefinition` directly.
    x, y : array_like
        Finite one-dimensional independent and dependent data of equal length.
        Recast to float arrays; select valid observations explicitly.
    yerr : array_like, optional
        Per-point 1-sigma errors. If ``None`` (default), the scatter is
        assumed constant and estimated from the residuals, and the reported
        uncertainties are the unit-error ones rescaled by ``sqrt(chi2/dof)``
        (curve_fit ``absolute_sigma=False`` equivalent); the result then has
        ``yerr_assumed=True``. The minimisation itself runs against a sigma on
        the data's own scale, not 1, so that data in small units (a
        susceptibility in emu/Oe) converge -- see the module docstring. A
        scalar broadcasts to every point. Supplied values must be finite and
        strictly positive, with one value per point or a scalar.
    initial : dict, optional
        Starting values overriding the model defaults / data guess.
    limits : dict, optional
        ``{name: (lower, upper)}`` box constraints; a bound may be ``None`` or
        ``-inf`` (lower) or ``+inf`` (upper) to leave that side unconstrained.
        Endpoints are inclusive, including equal bounds. For a parameter also
        in ``fixed``, bounds must be ordered and contain no NaN.
    fixed : dict, optional
        ``{name: value}`` parameters to hold constant. Fixed parameters are
        excluded from the degrees of freedom. A value also constrained by
        ``limits`` must be finite and lie within its inclusive interval.
    run_minos : bool, optional
        When ``True`` and the fit is valid, also run MINOS and report the
        asymmetric bands in ``minos_errors`` (display-only; HESSE stays
        canonical). Default ``False``.

    Returns
    -------
    FitResult
        The fit outcome. On a numerical fit failure (an arithmetic/linear-algebra exception or an
        invalid minimum) this is returned with ``success=False`` rather than
        propagating the error. So is a problem with no more points than free
        parameters (``dof <= 0``): it is not minimised, and the result carries
        the starting values, no uncertainties, and the true, unclamped
        ``dof``. No warning is issued; ``success`` is the signal, as for every
        other failure.

    Notes
    -----
    Unexpected model/backend exceptions propagate, including during MINOS.
    ArithmeticError and numpy.linalg.LinAlgError during MINOS leave the valid
    HESSE result available with ``minos_errors=None``.

    Raises
    ------
    FitInputError
        If observations are non-finite, not one-dimensional or misaligned,
        or supplied uncertainties are non-finite, non-positive or misaligned.
        Raised before minimisation, including for sparse inputs.
    FitConstraintError
        If a parameter in both ``fixed`` and ``limits`` has a fixed value
        outside its permitted interval or is non-finite, or the interval has
        reversed or NaN bounds. This caller error raises before
        minimisation, even for empty or underdetermined data. Correct the
        fixed value or its bounds; it never becomes ``success=False``.
    UnknownModelError
        For an unknown model name (also a ``KeyError``).
    FitParameterError
        For a name in ``initial``, ``limits`` or ``fixed`` that is not one
        of the model's parameters (also a ``KeyError``). These are caller
        errors, not fit failures, so they raise whatever the data -- including
        on a problem too sparse to fit -- rather than coming back as
        ``success=False``.
    """
    from iminuit import Minuit
    from iminuit.cost import LeastSquares

    defn = _resolve_model(model)
    arrays = {}
    for name, value in (("x", x), ("y", y), ("yerr", yerr)):
        if name == "yerr" and value is None:
            continue
        try:
            if np.iscomplexobj(value):
                raise FitInputError(f"{name} must be real-valued; complex values are unsupported.")
            arrays[name] = np.asarray(value, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise FitInputError(f"{name} must contain real numeric values; correct {name}.") from exc
        array = arrays[name]
        expected_ndim = (0, 1) if name == "yerr" else (1,)
        if array.ndim not in expected_ndim or not np.all(np.isfinite(array)):
            raise FitInputError(
                f"{name} must be finite and {'scalar or one-dimensional' if name == 'yerr' else 'one-dimensional'}; "
                f"got shape {array.shape}. Select valid observations explicitly."
            )
        if name == "yerr" and np.any(array <= 0):
            raise FitInputError(f"yerr must be strictly positive; got {value!r}. Supply valid uncertainties.")
    x, y = arrays["x"], arrays["y"]
    if x.shape != y.shape:
        raise FitInputError(f"x and y must have equal lengths; got {x.shape} and {y.shape}.")
    if yerr is not None:
        yerr = arrays["yerr"]
        if yerr.ndim == 1 and yerr.shape != y.shape:
            raise FitInputError(f"yerr must be scalar or match y shape {y.shape}; got {yerr.shape}.")
    n_points = int(x.size)

    fixed = dict(fixed or {})
    unknown_fixed = set(fixed) - set(defn.param_names)
    if unknown_fixed:
        raise FitParameterError(
            f"fixed contains unknown parameter(s) {sorted(unknown_fixed)} for "
            f"model {defn.name!r}; parameters are {defn.param_names}."
        )
    # Checked here, not where the limits are applied: that is inside the FT-1
    # guard below, which turned a misspelt name into success=False -- a caller
    # error indistinguishable from a fit the data could not support.
    unknown_limits = set(limits or {}) - set(defn.param_names)
    if unknown_limits:
        raise FitParameterError(
            f"limits contains unknown parameter(s) {sorted(unknown_limits)} for "
            f"model {defn.name!r}; parameters are {defn.param_names}."
        )

    # Validate the scientific assumption before sparse-data returns or the
    # numerical-failure guard: Minuit otherwise clips fixed values to limits.
    for name, value in fixed.items():
        if name not in (limits or {}):
            continue
        bounds = limits[name]
        lower, upper = bounds
        lower = -np.inf if lower is None else lower
        upper = np.inf if upper is None else upper
        if not np.isfinite(value) or not lower <= value <= upper:
            raise FitConstraintError(
                f"Model {defn.name!r}: fixed parameter {name!r} = {value!r} "
                f"conflicts with limits {bounds!r}; correct the fixed value "
                "or the bounds: use a finite fixed value in an ordered, inclusive "
                "interval without NaN bounds."
            )

    yerr_assumed = yerr is None

    # Report true degrees of freedom, including non-positive values on failure.
    # Underdetermined inputs return before arithmetic requiring positive dof.
    n_free = len(defn.param_names) - len(fixed)
    dof = n_points - n_free

    # Starting values honour any fixed values exactly (a fixed parameter must
    # start at the value it is pinned to).
    start = _resolve_initial(defn, x, y, initial)
    start.update(fixed)

    def _failed() -> FitResult:
        """Best-effort result for an unfittable problem: report seeds, no errors."""
        return FitResult(
            success=False,
            model=defn.name,
            parameters=dict(start),
            uncertainties={},
            covariance=None,
            covariance_parameters=(),
            chi_squared=float("nan"),
            reduced_chi_squared=float("nan"),
            dof=dof,
            n_points=n_points,
            fixed=dict(fixed),
            yerr_assumed=yerr_assumed,
            minos_errors=None,
            _function=defn.function,
            _param_order=defn.param_names,
        )

    # No more points than free parameters: the points fix the parameters
    # exactly or not at all, so there is no fit to judge and no scatter for
    # HESSE to turn into uncertainties. Refused before sigma is seeded, which
    # would warn on an empty array.
    if dof <= 0:
        return _failed()

    # With assumed errors the fit is minimised against a constant sigma on the
    # data's own scale rather than 1 (see "Why assumed errors are not unit
    # errors" in the module docstring). ``sigma`` is the constant in use; it
    # is refined from the residuals after the first MIGRAD pass.
    sigma = _seed_sigma(y) if yerr_assumed else 1.0
    yerror: ArrayLike = sigma if yerr_assumed else yerr

    def _minimiser(yerror_: ArrayLike, values: dict[str, float]):
        """A Minuit over ``yerror_``, seeded at ``values``, with fixed and limits applied."""
        cost = LeastSquares(x, y, yerror_, _named_model(defn))
        m = Minuit(cost, name=defn.param_names, **{n: values[n] for n in defn.param_names})

        for name, value in fixed.items():
            m.values[name] = value
            m.fixed[name] = True

        for name, bounds in (limits or {}).items():
            lower, upper = bounds
            lower = -np.inf if lower is None else lower
            upper = np.inf if upper is None else upper
            m.limits[name] = (lower, upper)
        return m

    try:
        m = _minimiser(yerror, start)
        m.migrad()
        if yerr_assumed:
            refined = _refined_sigma(float(m.fval), sigma, dof, y)
            if refined is not None:
                # A fresh Minuit seeded at the first-pass values, not the same
                # one with its cost re-weighted: carried-over step sizes can
                # prevent a positive-definite seed matrix at small parameter
                # scales. Default steps scale with each parameter's value.
                sigma = refined
                m = _minimiser(sigma, {n: float(m.values[n]) for n in defn.param_names})
                m.migrad()
        m.hesse()
    except (ArithmeticError, np.linalg.LinAlgError):
        return _failed()

    # With assumed errors the cost is sum((r / sigma)^2); multiplying back by
    # sigma^2 reports the raw sum of squared residuals.
    # With supplied errors ``sigma`` is 1 and this is the weighted chi^2.
    chi_squared = float(m.fval) * sigma**2

    if not bool(m.valid):
        # Report where MIGRAD stopped, but flag failure and withhold errors.
        result = _failed()
        result.parameters = {name: float(m.values[name]) for name in defn.param_names}
        result.chi_squared = chi_squared
        result.reduced_chi_squared = chi_squared / dof
        return result

    parameters = {name: float(m.values[name]) for name in defn.param_names}
    free_names = tuple(name for name in defn.param_names if name not in fixed)

    reduced_chi_squared = chi_squared / dof
    # absolute_sigma=False rescaling: the residual scatter estimates the true
    # per-point sigma, so parameter errors scale by sqrt(chi2_sigma/dof) and
    # covariances by chi2_sigma/dof, where chi2_sigma = m.fval is the cost at
    # the sigma actually minimised against. HESSE errors are proportional to
    # sigma and chi2_sigma to 1/sigma^2, so the product is independent of
    # sigma: these are exactly the numbers a unit-sigma fit rescaled by
    # sqrt(chi2/dof) would give, had that fit converged. For noisy data the
    # second pass leaves chi2_sigma ~ dof, so the factor is ~1.
    scale = 1.0
    if yerr_assumed:
        reduced_cost = float(m.fval) / dof
        # An exact fit leaves no scatter to estimate sigma from. The errors
        # then fall back to unit-sigma HESSE errors, as they always have;
        # they carry no statistical meaning, and chi_squared ~ 0 says so.
        scale = np.sqrt(reduced_cost) if reduced_cost > 0 else 1.0 / sigma

    uncertainties: dict[str, float] = {}
    covariance: NDArray[np.float64] | None = None
    covariance_parameters: tuple[str, ...] = ()
    if m.covariance is not None:
        for name in free_names:
            uncertainties[name] = float(m.errors[name]) * scale
        for name in fixed:
            uncertainties[name] = 0.0
        full = np.asarray(m.covariance, dtype=float)
        idx = [defn.param_names.index(name) for name in free_names]
        covariance = full[np.ix_(idx, idx)] * (scale**2)
        covariance_parameters = free_names

    minos_errors: dict[str, tuple[float, float]] | None = None
    if run_minos:
        try:
            m.minos()
            collected: dict[str, tuple[float, float]] = {}
            for name in free_names:
                me = m.merrors.get(name)
                if me is not None and getattr(me, "is_valid", False):
                    collected[name] = (float(me.lower) * scale, float(me.upper) * scale)
            minos_errors = collected or None
        except (ArithmeticError, np.linalg.LinAlgError):
            minos_errors = None

    return FitResult(
        success=True,
        model=defn.name,
        parameters=parameters,
        uncertainties=uncertainties,
        covariance=covariance,
        covariance_parameters=covariance_parameters,
        chi_squared=chi_squared,
        reduced_chi_squared=reduced_chi_squared,
        dof=dof,
        n_points=n_points,
        fixed=dict(fixed),
        yerr_assumed=yerr_assumed,
        minos_errors=minos_errors,
        _function=defn.function,
        _param_order=defn.param_names,
    )
