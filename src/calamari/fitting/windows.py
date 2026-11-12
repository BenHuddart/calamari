"""Fit the same model over several caller-specified ranges.

:func:`fit_windows` records how fitted parameters and derived quantities change
with the fitting range. Each window uses :func:`~calamari.fitting.engine.fit`:
supplied standard errors give absolute uncertainties; otherwise the fit estimates
constant scatter from the residuals. Numerical failures appear as unsuccessful
rows. Invalid configuration and unexpected model errors propagate.

Choose ranges that are physically appropriate for the data. Overlapping windows
are correlated, and stable results do not establish that the model is adequate.
No fitting ranges, stability threshold or preferred window are chosen automatically.

``derived=`` evaluates caller-supplied scalar functions of each fit's parameters
and propagates the full covariance with
:meth:`~calamari.fitting.engine.FitResult.derive`. Convert the input quantities to
the required normalisation before fitting; the fit does not infer a mole basis.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from calamari._provenance import calamari_version as _calamari_version
from calamari._provenance import describe_callable as _describe_callable
from calamari.exceptions import FitInputError, FitParameterError, InputTypeError
from calamari.fitting.engine import FitResult, _resolve_model, fit
from calamari.fitting.models import ModelDefinition

#: The leading columns of :attr:`FitWindowsResult.table`, in order. Each
#: model parameter ``p`` then contributes ``p`` and ``p_err``, in the model's
#: own parameter order, and each derived quantity ``q`` then ``q`` and
#: ``q_err``, in the order of the ``derived`` mapping.
_LEADING_COLUMNS: tuple[str, ...] = (
    "x_min",
    "x_max",
    "n_points",
    "success",
    "dof",
    "reduced_chi_squared",
    "yerr_assumed",
)

#: The :func:`~calamari.fitting.engine.fit` keywords :func:`fit_windows`
#: forwards. Anything else is refused up front, before the first window is
#: fitted, with a message naming what is forwarded.
_FORWARDED_FIT_KWARGS = ("initial", "fixed", "limits", "run_minos")


@dataclass(frozen=True)
class FitWindowsResult:
    """The outcome of :func:`fit_windows`: one fit per window, tabulated.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per window, in the order the windows were given. Columns
        are ``x_min``, ``x_max``, ``n_points``, ``success``, ``dof``,
        ``reduced_chi_squared`` and ``yerr_assumed``, followed by ``<p>`` and ``<p>_err``
        for every model parameter ``p``:

        ``x_min``, ``x_max``
            The window's inclusive bounds, as given.
        ``n_points``
            How many points fell inside it.
        ``success``
            Whether the fit converged to a valid minimum. A window with no
            more points than free parameters is a failure: ``fit()`` refuses
            it rather than interpolating it.
        ``dof``
            ``n_points`` minus the number of free parameters, from the fit.
            Reported as is, so an underdetermined window shows zero or a
            negative number, which is the reason it failed.
        ``reduced_chi_squared``
            From the fit; NaN on a failure row.
        ``yerr_assumed``
            ``True`` when no ``yerr`` was supplied (see
            :class:`~calamari.fitting.engine.FitResult`).
        ``<p>``, ``<p>_err``
            The best-fit value and its HESSE uncertainty. A fixed parameter
            reports its fixed value and an error of 0. A failure row is NaN
            throughout: where MIGRAD stopped is not an estimate, and a
            number in the table would invite plotting it. Drill into
            :attr:`results` for that.
        ``<q>``, ``<q>_err``
            One pair per entry of ``derived``, after the parameter columns
            and in the mapping's order: the quantity at the best fit and its
            first-order propagated uncertainty (see
            :meth:`~calamari.fitting.engine.FitResult.derive`, which also
            says when that approximation fails). NaN on a failure row, like
            the parameters; ``<q>_err`` alone is NaN when the window has no
            covariance to propagate.
    results : list of FitResult
        The per-window fit results, aligned with the rows of :attr:`table`
        (``results[i]`` is row ``i``), for drill-down: covariances, MINOS
        bands, :meth:`~calamari.fitting.engine.FitResult.curve`. A failed
        window's entry is ``fit()``'s own ``success=False`` result.
    params : dict
        The recipe: ``'model'`` (the model name), ``'windows'`` (the bounds
        as float pairs), ``'fit_kwargs'`` (the forwarded ``fit()``
        keywords, copied), ``'derived'`` (each derived quantity's name mapped
        to a qualified function/type or ``partial(function)`` description;
        empty without ``derived``) and
        ``'calamari_version'``. Callable labels do not serialise configured
        state; retain callable source and configuration with the recipe.
    """

    table: pd.DataFrame
    results: list[FitResult] = field(repr=False)
    params: dict[str, Any]



def _is_real(value: Any) -> bool:
    """True for a real number (numpy scalars included), False for a bool.

    ``bool`` subclasses ``int``, so ``(False, 300)`` would otherwise pass as a
    window starting at 0 -- a typo turned into a silently wrong fit.
    """
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_))


def _validate_windows(windows: Any) -> list[tuple[float, float]]:
    """Check every window is a finite ``(lo, hi)`` pair with ``lo < hi``.

    A malformed window is the caller's error, not a fit that failed, so it
    raises here -- before any fitting -- rather than becoming a failure row
    that would be indistinguishable from a window the data could not support.
    """
    if isinstance(windows, np.ndarray):
        windows = windows.tolist()
    if isinstance(windows, (str, bytes)) or not isinstance(windows, Sequence):
        raise FitInputError(
            "windows must be a sequence of (x_lo, x_hi) pairs, e.g. "
            f"[(50, 300), (100, 300)]; got {windows!r}."
        )
    if len(windows) == 0:
        raise FitInputError("windows is empty; give at least one (x_lo, x_hi) pair.")
    resolved = []
    for i, window in enumerate(windows):
        if isinstance(window, np.ndarray):
            window = window.tolist()
        if (
            isinstance(window, (str, bytes))
            or not isinstance(window, Sequence)
            or len(window) != 2
            or not all(_is_real(v) for v in window)
        ):
            raise FitInputError(
                f"windows[{i}] = {window!r} is not an (x_lo, x_hi) pair of numbers. "
                "Give windows as a sequence of pairs, e.g. [(50, 300), (100, 300)] "
                "-- a single window still needs the outer brackets."
            )
        lo, hi = float(window[0]), float(window[1])
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise FitInputError(
                f"windows[{i}] = {window!r} has a non-finite bound; give finite "
                "limits (use the data's own min or max for an open end)."
            )
        if lo >= hi:
            raise FitInputError(
                f"windows[{i}] = {window!r} needs x_lo < x_hi; got {lo:g} >= {hi:g}."
            )
        resolved.append((lo, hi))
    return resolved


def _validate_fit_kwargs(defn: ModelDefinition, fit_kwargs: dict[str, Any]) -> None:
    """Refuse an unknown keyword or parameter name before fitting anything.

    ``fit()`` raises for an unknown keyword and for an unknown name in
    ``initial``, ``fixed`` or ``limits`` itself, on the first window. Checking
    here says so before any fitting, in ``fit_windows``'s own terms.
    """
    unknown = sorted(set(fit_kwargs) - set(_FORWARDED_FIT_KWARGS))
    if unknown:
        raise InputTypeError(
            f"fit_windows() got unexpected keyword argument(s) {unknown}; it "
            f"forwards only {list(_FORWARDED_FIT_KWARGS)} to fit()."
        )
    for key in ("initial", "fixed", "limits"):
        names = set(fit_kwargs.get(key) or {})
        bad = sorted(names - set(defn.param_names))
        if bad:
            raise FitParameterError(
                f"{key} contains unknown parameter(s) {bad} for model "
                f"{defn.name!r}; parameters are {defn.param_names}."
            )


def _validate_derived(
    derived: Any, taken: Sequence[str]
) -> dict[str, Callable[[Mapping[str, float]], float]]:
    """Check ``derived`` is a mapping of new column names to callables.

    Refused here, before any fitting, for the same reason as a bad keyword:
    a malformed request should fail the same way whatever the data, not
    after several windows have been fitted, and not only when some window
    happens to succeed and call the function. ``taken`` is the columns
    already in the table; a derived name may not reuse one, since a
    quantity silently overwriting ``C`` or ``C_err`` would be worse than
    useless. A name ending in ``_err`` is refused outright: it would read as
    the uncertainty of some other column.
    """
    if derived is None:
        return {}
    if not isinstance(derived, Mapping):
        raise InputTypeError(
            "derived must be a mapping of column name to callable, e.g. "
            "{'mu_eff': lambda p: mu_eff_from_curie_constant(p['C'])}; "
            f"got {type(derived).__name__} {derived!r}."
        )
    columns = set(taken)
    resolved: dict[str, Callable[[Mapping[str, float]], float]] = {}
    for name, func in derived.items():
        if not isinstance(name, str) or not name:
            raise InputTypeError(f"derived names must be non-empty strings; got {name!r}.")
        if not callable(func):
            raise InputTypeError(
                f"derived[{name!r}] = {func!r} is not callable; give a function "
                "taking the dict of fitted parameters and returning a number."
            )
        if name.endswith("_err"):
            raise FitInputError(
                f"derived name {name!r} ends in '_err', which marks an "
                "uncertainty column in the table; choose another name."
            )
        clash = [c for c in (name, f"{name}_err") if c in columns]
        if clash:
            raise FitInputError(
                f"derived name {name!r} would add column(s) {clash}, which the "
                "table already has (from the fit, the model's parameters or an "
                "earlier derived quantity); choose another name."
            )
        columns.update((name, f"{name}_err"))
        resolved[name] = func
    return resolved


def fit_windows(
    model: str | ModelDefinition,
    x: ArrayLike,
    y: ArrayLike,
    *,
    windows: Sequence[tuple[float, float]],
    yerr: ArrayLike | None = None,
    derived: Mapping[str, Callable[[Mapping[str, float]], float]] | None = None,
    **fit_kwargs: Any,
) -> FitWindowsResult:
    """Fit *model* separately over each x-window and tabulate the results.

    The check that a fitted parameter does not depend on the fitting range:
    a Curie-Weiss ``theta`` that drifts as ``T_min`` rises is the classic sign
    of a neglected background (fit ``curie_weiss``, whose ``chi0`` term
    absorbs it, rather than ``inverse_curie_weiss``, which cannot). Each
    window is an independent :func:`~calamari.fitting.engine.fit` of the
    points inside it; see the module docstring for why there is no verdict,
    threshold or default window.

    Parameters
    ----------
    model : str or ModelDefinition
        A registered model name or a definition, as for ``fit()``.
    x, y : array_like
        The full dataset, same length. Points are selected per window by
        ``x``; a point whose ``x`` is NaN lies in no window. A non-finite
        ``y`` inside a window is kept, and that window's fit fails -- the
        data are not quietly cleaned.
    windows : sequence of (float, float)
        Inclusive ``(x_lo, x_hi)`` bounds, one fit each, reported in the
        order given. Required: choosing the windows is the physics. Windows
        may overlap and need not be sorted.
    yerr : array_like or float, optional
        Per-point errors. An array (same length as ``x``) is sliced with each
        window, so every point keeps its own error; a scalar applies to every
        point. ``None`` (default) means assumed errors, as for ``fit()``.
    derived : mapping of str to callable, optional
        Quantities to compute from each window's parameters, tabulated with
        their propagated errors as ``<name>`` and ``<name>_err`` columns
        after the parameters, in the mapping's order. Each callable gets a
        dict of every parameter (free and fixed) and returns a number; see
        :meth:`~calamari.fitting.engine.FitResult.derive` for the
        propagation and its limits. It is not forwarded to ``fit()``. Make
        sure the quantity means what its name says: an effective moment
        from ``C`` needs ``chi`` fitted per mole of magnetic ion, which the
        fit cannot check.
    **fit_kwargs
        Forwarded unchanged to every ``fit()`` call: ``initial``, ``fixed``,
        ``limits`` and ``run_minos``. Recorded in ``params['fit_kwargs']``.

    Returns
    -------
    FitWindowsResult
        ``.table`` (one row per window), ``.results`` (the aligned
        ``FitResult`` objects) and ``.params`` (the recipe).

    Raises
    ------
    ValueError
        For a malformed window (not a pair of numbers, a non-finite bound,
        ``x_lo >= x_hi``), an empty ``windows``, ``x`` and ``y`` of different
        lengths, an array ``yerr`` of the wrong length, or a ``derived`` name
        that would collide with another column (or ends in ``_err``). Also
        when a ``derived`` function raises: the error is re-raised naming the
        quantity and the window, chained to the original.
    TypeError
        For a keyword ``fit()`` does not take, or a ``derived`` that is not a
        mapping of names to callables.
    FitConstraintError
        A forwarded fixed value conflicts with its limits; propagated from
        :func:`~calamari.fitting.engine.fit`, including in empty windows.
    KeyError
        For an unknown model, or an unknown parameter name in ``initial``,
        ``fixed`` or ``limits``.

    Notes
    -----
    Unknown models and parameter names raise
    :class:`~calamari.exceptions.UnknownModelError` and
    :class:`~calamari.exceptions.FitParameterError`, respectively (both also
    ``KeyError`` and :class:`~calamari.exceptions.CalamariError`).

    FT-1 holds per window: a window whose fit fails is a ``success=False``
    row, not an exception. That includes a window with no more points than
    free parameters, which ``fit()`` refuses: with zero degrees of freedom
    the points determine the parameters exactly, and there is no fit to
    judge.

    Examples
    --------
    >>> import numpy as np
    >>> from calamari.fitting import curie_weiss, fit_windows
    >>> T = np.linspace(20, 300, 57)
    >>> chi = curie_weiss(T, 0.5, -20.0, 2e-4)
    >>> scan = fit_windows("curie_weiss", T, chi, windows=[(20, 300), (100, 300)])
    >>> scan.table[["x_min", "n_points", "success"]]
       x_min  n_points  success
    0   20.0        57     True
    1  100.0        41     True
    >>> [round(t, 3) for t in scan.table["theta"]]
    [-20.0, -20.0]

    A derived quantity gets a column and an error column of its own:

    >>> from calamari import mu_eff_from_curie_constant
    >>> scan = fit_windows(
    ...     "curie_weiss", T, chi, windows=[(20, 300), (100, 300)],
    ...     derived={"mu_eff": lambda p: mu_eff_from_curie_constant(p["C"])},
    ... )
    >>> list(scan.table.columns[-2:])
    ['mu_eff', 'mu_eff_err']
    >>> [round(v, 3) for v in scan.table["mu_eff"]]
    [2.0, 2.0]
    """
    defn = _resolve_model(model)
    resolved = _validate_windows(windows)
    _validate_fit_kwargs(defn, fit_kwargs)

    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    if x_arr.shape != y_arr.shape:
        raise FitInputError(
            f"x and y must be the same length; got {x_arr.size} and {y_arr.size}."
        )
    yerr_arr: np.ndarray | None = None
    if yerr is not None:
        yerr_arr = np.asarray(yerr, dtype=float)
        if yerr_arr.ndim == 0:
            yerr_arr = np.full_like(x_arr, float(yerr_arr))
        else:
            yerr_arr = yerr_arr.ravel()
            if yerr_arr.shape != x_arr.shape:
                raise FitInputError(
                    f"yerr must be a scalar or the same length as x; got "
                    f"{yerr_arr.size} errors for {x_arr.size} points."
                )

    param_columns = [c for p in defn.param_names for c in (p, f"{p}_err")]
    clash = sorted(set(param_columns) & set(_LEADING_COLUMNS))
    if clash or len(set(param_columns)) != len(param_columns):
        raise FitInputError(
            f"Model {defn.name!r} has parameter names {defn.param_names} whose "
            "table columns collide with each other or with "
            f"{list(_LEADING_COLUMNS)}; rename the parameters to tabulate it."
        )
    derived_funcs = _validate_derived(derived, [*_LEADING_COLUMNS, *param_columns])
    derived_columns = [c for q in derived_funcs for c in (q, f"{q}_err")]

    rows: list[dict[str, Any]] = []
    results: list[FitResult] = []
    for index, (lo, hi) in enumerate(resolved):
        inside = (x_arr >= lo) & (x_arr <= hi)
        result = fit(
            defn,
            x_arr[inside],
            y_arr[inside],
            yerr=None if yerr_arr is None else yerr_arr[inside],
            **fit_kwargs,
        )
        results.append(result)

        row: dict[str, Any] = {
            "x_min": lo,
            "x_max": hi,
            "n_points": result.n_points,
            "success": bool(result.success),
            "dof": result.dof,
            "reduced_chi_squared": (
                float(result.reduced_chi_squared) if result.success else float("nan")
            ),
            "yerr_assumed": result.yerr_assumed,
        }
        for name in defn.param_names:
            if result.success:
                row[name] = float(result.parameters[name])
                err = result.uncertainties.get(name)
                row[f"{name}_err"] = float("nan") if err is None else float(err)
            else:
                row[name] = float("nan")
                row[f"{name}_err"] = float("nan")
        for name, func in derived_funcs.items():
            # derive() returns (nan, nan) for a failed fit without calling func.
            try:
                row[name], row[f"{name}_err"] = result.derive(func)
            except Exception as exc:
                raise FitInputError(
                    f"fit_windows(): derived quantity {name!r} "
                    f"({_describe_callable(func)}) raised {exc!r} in window {index} "
                    f"({lo:g}, {hi:g})."
                ) from exc
        rows.append(row)

    table = pd.DataFrame(rows, columns=[*_LEADING_COLUMNS, *param_columns, *derived_columns])
    table = table.astype({"n_points": int, "dof": int, "success": bool, "yerr_assumed": bool})
    params = {
        "model": defn.name,
        "windows": list(resolved),
        "fit_kwargs": copy.deepcopy(fit_kwargs),
        # Descriptions, not the callables: a closure need not deep-copy, and
        # the record should stay printable and comparable.
        "derived": {name: _describe_callable(func) for name, func in derived_funcs.items()},
        "calamari_version": _calamari_version(),
    }
    return FitWindowsResult(table=table, results=results, params=params)
