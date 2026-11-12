"""Fit an explicit model to selected candidate locations across sweeps.

:func:`fit_boundary` fits :math:`T(H)` or :math:`H(T)` to at most one selected
point per sweep, using one detection criterion. The caller supplies the model
and interprets whether the selected features represent a physical boundary.
Raw sweep data are not fitted jointly.

The result records every sweep, including those without selected points.
Predictions for missing points are separate from observations. Diagnostics include
HESSE uncertainties, a leave-one-sweep-out jackknife, a covariance-derived
one-standard-error band, residuals, pulls and comparisons with caller-listed models.

Several smoothing windows or fitting ranges reuse the same observations; they
must not be treated as independent points. Supply analysis-choice uncertainty
explicitly through ``yerr_sys`` when appropriate. Statistical-only and combined
uncertainty fits are reported separately. Mixed criteria are refused because
they can locate systematically different positions on the same feature.

Monotonic fitting requires a monotonic parametric model. The fit describes
candidate coordinates and does not assign phase labels.
:func:`calamari.plot.plot_boundary` draws the existing result.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from calamari._provenance import calamari_version as _calamari_version
from calamari.exceptions import FitInputError
from calamari.fitting.engine import FitResult, _resolve_model, fit
from calamari.fitting.models import ModelDefinition
from calamari.transitions import TransitionResult

__all__ = [
    "LOW_CHI2_TAIL",
    "BAND_POINTS",
    "POINT_COLUMNS",
    "SWEEP_COLUMNS",
    "BoundaryFit",
    "fit_boundary",
]

#: Lower-tail probability of the chi-squared distribution below which the
#: scatter is flagged as too small for the stated errors
#: (:attr:`BoundaryFit.scatter_below_errors`). The value is Phi(-3), the
#: one-sided tail of a 3-sigma Gaussian excursion, about 0.00135: the flag is
#: a claim that the stated errors are implausible, and 3 sigma is the level at
#: which calamari's neighbours in physics treat a deviation as evidence rather
#: than a fluctuation. With honest errors it fires in 0.135 % of fits,
#: whatever the degrees of freedom, because it is read off the chi-squared
#: distribution for that fit's own dof rather than off chi2/dof, whose
#: spread depends on dof (with 2 dof, chi2/dof below 0.1 happens 10 % of the
#: time by chance alone). It was chosen before looking at any fit and is not
#: tuned; it warns and gates nothing.
LOW_CHI2_TAIL = float(stats.norm.cdf(-3.0))

#: Number of held-coordinate values the stored band is evaluated at, evenly
#: spaced across the used points' range. 201 gives 200 intervals, so a curve
#: drawn from it is smooth at any figure size calamari's styles produce, and
#: the band costs 201 delta-method propagations, each a handful of model
#: evaluations.
BAND_POINTS = 201

#: Per swept axis: the location column (the swept coordinate, where the
#: criterion placed the candidate), the held coordinate, the statistical
#: error, the optional systematic error, the error the fit was weighted by,
#: the prediction and residual columns, and the unit of the location. Names
#: carry their units (DM-2), as the candidates table's do.
_ROLES = {
    "T": {
        "location": "T_K",
        "held": "H_T",
        "err": "T_err_K",
        "sys": "T_err_sys_K",
        "fit_err": "T_err_fit_K",
        "predicted": "predicted_K",
        "residual": "residual_K",
        "unit": "K",
        "held_unit": "T",
    },
    "H": {
        "location": "H_T",
        "held": "T_K",
        "err": "H_err_T",
        "sys": "H_err_sys_T",
        "fit_err": "H_err_fit_T",
        "predicted": "predicted_T",
        "residual": "residual_T",
        "unit": "T",
        "held_unit": "K",
    },
}

#: The columns a DataFrame of points must carry, per swept axis: one row per
#: sweep, its held coordinate, the location with its statistical error, and
#: the criterion that placed it. A ``fit_breakpoint`` result per T-sweep
#: becomes a row ``{sweep_id, H_T, T_K=x_break, T_err_K=x_break_err,
#: criterion='breakpoint:segmented_linear'}``.
POINT_COLUMNS = {
    axis: ("sweep_id", roles["held"], roles["location"], roles["err"], "criterion")
    for axis, roles in _ROLES.items()
}

#: Optional columns of a DataFrame of points: the candidate's significance
#: (shown, never used as a weight), the systematic error (the alternative to
#: ``yerr_sys=``) and the file, for the audit trail.
_OPTIONAL_POINT_COLUMNS = {
    axis: ("significance", roles["sys"], "measurement_file") for axis, roles in _ROLES.items()
}

#: Column order of :attr:`BoundaryFit.sweeps`, per swept axis: one row per
#: sweep of the family. ``predicted_*`` is the model at the held coordinate,
#: kept apart from the location so a prediction can never be read as a
#: point; ``pull`` is the residual over the error the fit was weighted by.
SWEEP_COLUMNS = {
    axis: [
        "sweep_id",
        "status",
        roles["held"],
        roles["location"],
        roles["err"],
        roles["sys"],
        roles["fit_err"],
        "significance",
        roles["predicted"],
        roles["residual"],
        "pull",
        "criterion",
        "measurement_file",
    ]
    for axis, roles in _ROLES.items()
}

#: Status of a sweep whose point entered the fit.
_USED = "used"
#: Status of a sweep whose point the caller left out (``exclude=``). It is
#: still listed, with its residual from the fit made without it.
_EXCLUDED = "excluded by caller"
#: Status of a sweep that had candidates on the swept axis, none kept by the
#: selection: nothing of that criterion in its window, or everything there
#: removed by ``robust()`` (``params['select'][axis]['empty_sweeps']``).
_EMPTY = "no candidate in window"
#: Status of a searched sweep that produced no candidate at all. Its swept
#: axis and held coordinate are unknown, since only a candidate carries them.
_NONE_DETECTED = "no candidate detected"
#: Status of a DataFrame row that names a sweep but gives no location.
_NO_POINT = "no point given"

#: How the information criteria are computed, recorded in ``params``.
_IC_TEXT = (
    "chi-squared with the supplied errors taken as known: AIC = chi2 + 2k, "
    "BIC = chi2 + k ln n; delta = compared - stated, positive favours the stated model"
)


@dataclass(frozen=True, eq=False)
class BoundaryFit:
    """One stated model fitted to one criterion's points, one per sweep, with its diagnostics.

    Read :attr:`sweeps` and :attr:`scatter_below_errors` before the
    parameters: the first says which sweeps the curve rests on and which it
    does not, and the second whether the error bars describe the scatter at
    all. The curve is a model of candidate locations under the caller's
    assumed form, nothing more.

    Attributes
    ----------
    fit : calamari.fitting.engine.FitResult
        The fit (iminuit MIGRAD + HESSE, FT-1), of the location against the
        held coordinate, weighted by each used point's error. A failure is
        ``fit.success == False``, never an exception.
    estimates : pandas.DataFrame
        One row per model parameter: ``value``, ``hesse_err`` (canonical),
        ``fixed``, ``jackknife_se`` (the leave-one-sweep-out standard error;
        NaN unless every refit succeeded), ``max_abs_shift_sigma`` and
        ``most_influential_sweep_id`` (from :attr:`jackknife`), and
        ``criterion``.
    sweeps : pandas.DataFrame
        One row per sweep of the family (:data:`SWEEP_COLUMNS`), sorted by
        ``sweep_id``. ``status`` is ``'used'``, ``'excluded by caller'``,
        ``'no candidate in window'``, ``'no candidate detected'``, ``'no
        point given'`` or ``'skipped: <reason>'``. The location and its
        errors are filled only where the sweep gave a point; nothing is
        imputed. ``predicted_*`` is the model at every known held
        coordinate, including the sweeps without a point, in its own column.
        ``residual_*`` is location minus prediction and ``pull`` is that
        over the fit's weight (``*_err_fit_*``). An excluded point's
        residual is from the fit made without it.
    band : pandas.DataFrame
        The model with its 1-sigma band on :data:`BAND_POINTS` values of the
        held coordinate spanning the used points (no extrapolation): the held
        column, ``model_*``, ``model_err_*``, ``lower_*``, ``upper_*`` (unit
        suffixes as the location's), and ``criterion``. It is a band on the
        MODEL, under its assumptions: the form is right and the errors are
        right. It is not a band on where the points will fall, and it does
        not include the form's own uncertainty. See :meth:`band_at` for other
        values.
    jackknife : pandas.DataFrame
        One row per used sweep: that sweep left out and the model refitted
        (seeded at the full fit). ``refit`` is ``'ok'``, ``'failed'`` or
        ``'underdetermined: dof N'``; for each free parameter ``p``, its
        refitted value ``p`` and ``p_shift_sigma``, the shift from the full
        fit in units of the full fit's HESSE error. A failed refit is kept,
        with NaN values.
    comparison : pandas.DataFrame
        The stated model and each ``compare=`` model fitted to the same
        points with the same errors: ``model``, ``role``, ``success``,
        ``n_free``, ``dof``, ``chi_squared``, ``reduced_chi_squared``,
        ``aic``, ``bic``, ``delta_aic``, ``delta_bic`` (compared minus
        stated: **positive favours the stated model**), ``criterion``.
    chi_squared_quantile : float
        ``P(chi2_dof <= chi_squared)`` for this fit's dof: where the fit's
        chi-squared sits in the distribution it would follow if the form and
        the errors were right. NaN when the fit failed.
    scatter_below_errors : bool
        ``chi_squared_quantile < LOW_CHI2_TAIL``: the points scatter about
        the curve far less than their errors allow. The errors are then
        overstated -- typically a location error floored at the scan's grid
        spacing -- and the HESSE errors, the band and the information
        criteria all inherit that (see :func:`fit_boundary`).
    warnings : tuple of str
        The messages of the warnings raised for this fit, in order.
    params : dict
        The recipe: ``model``, ``model_description``, ``criterion``,
        ``signal`` (a TransitionResult's), ``swept_axis``, ``location``,
        ``held``, ``source``, ``select`` (the result's record for that axis),
        ``other_axis_sweep_ids``, ``exclude``, ``errors`` (``'stat'`` or
        ``'stat+sys'``), ``yerr_sys`` (whether one was given, its source and
        its per-sweep values), ``compare``, ``initial``, ``limits``,
        ``fixed``, ``band_points``, ``low_chi2_tail``,
        ``information_criteria``, ``weighting`` and ``calamari_version``.
    total : BoundaryFit or None
        The same fit weighted by the statistical and systematic errors in
        quadrature, with its own diagnostics, when ``yerr_sys`` (or a
        systematic column) was given; ``None`` otherwise. Its own ``total``
        is ``None``.
    model : ModelDefinition
        The fitted model's definition, for :meth:`band_at`.
    """

    fit: FitResult = field(repr=False)
    estimates: pd.DataFrame = field(repr=False)
    sweeps: pd.DataFrame = field(repr=False)
    band: pd.DataFrame = field(repr=False)
    jackknife: pd.DataFrame = field(repr=False)
    comparison: pd.DataFrame = field(repr=False)
    chi_squared_quantile: float
    scatter_below_errors: bool
    warnings: tuple[str, ...]
    params: dict = field(repr=False)
    total: BoundaryFit | None = field(default=None, repr=False)
    model: ModelDefinition | None = field(default=None, repr=False)

    def __repr__(self) -> str:  # DataFrames make the default repr unusable
        p = self.params
        roles = _ROLES[p["swept_axis"]]
        counts = self.sweeps["status"].value_counts()
        used = int(counts.get(_USED, 0))
        excluded = int(counts.get(_EXCLUDED, 0))
        without = len(self.sweeps) - used - excluded
        if self.fit.success:
            quality = f"chi2/dof {self.fit.reduced_chi_squared:.3g} (dof {self.fit.dof})"
            if self.scatter_below_errors:
                quality += ", scatter below errors"
        else:
            quality = f"fit FAILED (dof {self.fit.dof})"
        errors = "stat" if p["errors"] == "stat" else "stat+sys"
        total = "; stat+sys fit in .total" if self.total is not None else ""
        return (
            f"BoundaryFit({p['model']!r} model of {roles['location']} against "
            f"{roles['held']}, criterion {p['criterion']}, {errors} errors: {used} used, "
            f"{excluded} excluded, {without} without a point; {quality}{total})"
        )

    def band_at(self, values: ArrayLike) -> pd.DataFrame:
        """The model and its 1-sigma band at chosen values of the held coordinate.

        :attr:`band` stops at the used points' range, deliberately. Beyond it
        nothing in the points constrains the curve, and the value and band
        there are the assumed form speaking: two forms that agree across the
        points can part company quickly outside them. Ask for those values
        here, where the call says so.

        Parameters
        ----------
        values : array_like
            Values of the held coordinate, in its own unit: ``H_T`` (Tesla)
            for a T-swept fit, ``T_K`` (Kelvin) for an H-swept one.

        Returns
        -------
        pandas.DataFrame
            Columns as :attr:`band`. All NaN when the fit failed.
        """
        return _band(self.fit, self.model, np.asarray(values, dtype=float), self.params)


def _is_real(value: Any) -> bool:
    """True for a real number (numpy scalars included), False for a bool."""
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_))


def _check_axis(swept_axis: Any) -> str:
    if swept_axis not in _ROLES:
        raise FitInputError(f"fit_boundary(): swept_axis must be 'T' or 'H', got {swept_axis!r}.")
    return swept_axis


def _rows_from_result(
    result: TransitionResult, swept_axis: str | None
) -> tuple[pd.DataFrame, dict]:
    """The family table of a ``select()``-ed result: one row per sweep, with its status.

    Every sweep the batch knows of is accounted for: searched sweeps from
    ``params['per_sweep']``, refused ones from ``skipped`` and
    ``params['skipped_sweep_ids']``. A searched sweep whose candidates were
    all swept along the other axis belongs to the other family, and is
    recorded in ``other_axis_sweep_ids`` rather than listed.
    """
    selects = result.params.get("select") or {}
    if swept_axis is None:
        if not selects:
            raise FitInputError(
                "fit_boundary(): this TransitionResult has not been select()-ed, so a sweep "
                "can hold several candidates -- other features, or the same feature seen "
                "by several criteria -- and counting them all would count one sweep "
                "several times. Choose one point per sweep first, e.g. "
                "result.select(swept_axis='T', method='peak', sign='max', "
                "t_window_k=(lo, hi))."
            )
        if len(selects) > 1:
            raise FitInputError(
                f"fit_boundary(): this result is selected on {sorted(selects)}, and a "
                "T-swept point (T_K at a held H_T) and an H-swept one (H_T at a held T_K) "
                "have their error on different axes, so they belong to different fits. "
                "Pass swept_axis='T' or swept_axis='H'."
            )
        swept_axis = next(iter(selects))
    swept_axis = _check_axis(swept_axis)
    if swept_axis not in selects:
        raise FitInputError(
            f"fit_boundary(): this result has no selection on swept_axis={swept_axis!r} "
            f"(selected on: {sorted(selects) or 'none'}), so a sweep can hold several "
            "candidates. Choose one point per sweep first with "
            f"result.select(swept_axis={swept_axis!r}, method=..., sign=..., "
            f"{'t_window_k' if swept_axis == 'T' else 'h_window_t'}=...)."
        )
    spec = selects[swept_axis]
    roles = _ROLES[swept_axis]
    loc_col, held_col, err_col = roles["location"], roles["held"], roles["err"]
    full, _ = result._full()
    view = result.candidates
    chosen = view[(view["swept_axis"] == swept_axis).to_numpy(dtype=bool)]
    full_on_axis = full[(full["swept_axis"] == swept_axis).to_numpy(dtype=bool)]
    chosen_by_sweep = {int(row.sweep_id): row for row in chosen.itertuples(index=False)}
    held_by_sweep = {
        int(sid): float(held)
        for sid, held in zip(full_on_axis["sweep_id"], full_on_axis[held_col], strict=True)
    }

    files: dict[int, str] = {}
    if "per_sweep" in result.params:
        searched = [int(entry["sweep_id"]) for entry in result.params["per_sweep"]]
        files.update(
            {int(e["sweep_id"]): str(e["measurement_file"]) for e in result.params["per_sweep"]}
        )
    else:
        searched = sorted({int(s) for s in full["sweep_id"]}) if len(full) else [0]
    for sid, path in zip(full["sweep_id"], full["measurement_file"], strict=True):
        files.setdefault(int(sid), str(path))
    skipped = {
        int(sid): (str(path), str(reason))
        for sid, (path, reason) in zip(
            result.params.get("skipped_sweep_ids", []), result.skipped, strict=True
        )
    }
    with_candidates = {int(s) for s in full["sweep_id"]} if len(full) else set()
    empty = {int(s) for s in spec.get("empty_sweeps", [])}
    criterion = f"{spec['method']}/{spec['sign']}"

    rows, other_axis = [], []
    for sid in sorted(set(searched) | set(skipped)):
        row = {"sweep_id": sid, held_col: np.nan, loc_col: np.nan, err_col: np.nan,
               "significance": np.nan, "criterion": criterion,
               "measurement_file": files.get(sid, np.nan)}
        if sid in skipped:
            path, reason = skipped[sid]
            row["status"] = f"skipped: {reason}"
            row["measurement_file"] = path
        elif sid in chosen_by_sweep:
            cand = chosen_by_sweep[sid]
            row.update({
                "status": _USED,
                held_col: float(getattr(cand, held_col)),
                loc_col: float(getattr(cand, loc_col)),
                err_col: float(getattr(cand, err_col)),
                "significance": float(cand.significance),
            })
        elif sid in empty:
            row["status"] = _EMPTY
            row[held_col] = held_by_sweep.get(sid, np.nan)
        elif sid in with_candidates:
            other_axis.append(sid)
            continue
        else:
            row["status"] = _NONE_DETECTED
        rows.append(row)
    meta = {
        "swept_axis": swept_axis,
        "criterion": criterion,
        "signal": spec["signal"],
        "source": "TransitionResult",
        "select": dict(spec),
        "other_axis_sweep_ids": other_axis,
        "sys_column": False,
    }
    return pd.DataFrame(rows), meta


def _rows_from_frame(frame: pd.DataFrame, swept_axis: str | None) -> tuple[pd.DataFrame, dict]:
    """The family table of a DataFrame of points, validated against :data:`POINT_COLUMNS`."""
    if swept_axis is None:
        raise FitInputError(
            "fit_boundary(): a DataFrame of points needs swept_axis='T' (the location is "
            "T_K at a held H_T) or swept_axis='H' (H_T at a held T_K); the frame alone "
            "does not say which coordinate carries the error."
        )
    swept_axis = _check_axis(swept_axis)
    roles = _ROLES[swept_axis]
    required = POINT_COLUMNS[swept_axis]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        hint = ""
        if "criterion" in missing and {"method", "sign"} <= set(frame.columns):
            hint = (
                " A candidates table carries method and sign rather than criterion: pass "
                "the select()-ed TransitionResult itself, or add "
                "criterion = method + '/' + sign."
            )
        raise FitInputError(
            f"fit_boundary(): the points frame is missing column(s) {missing}. For "
            f"swept_axis={swept_axis!r} it needs {list(required)} (one row per sweep), and "
            f"may carry {list(_OPTIONAL_POINT_COLUMNS[swept_axis])}.{hint}"
        )
    loc_col, held_col, err_col, sys_col = (
        roles["location"], roles["held"], roles["err"], roles["sys"]
    )
    table = frame.copy()  # DM-1: the caller's frame is never touched
    if table["sweep_id"].isna().any():
        raise FitInputError(
            "fit_boundary(): the points frame has rows with no sweep_id. Every row is one "
            "sweep, and exclude= and yerr_sys= address rows by it."
        )
    duplicated = _sorted_ids(table.loc[table["sweep_id"].duplicated(), "sweep_id"].tolist())
    if duplicated:
        raise FitInputError(
            f"fit_boundary(): sweep_id(s) {duplicated} appear more than once. One sweep is "
            "one point: several smoothing windows or fitting ranges see the same data, so "
            "they are not separate rows. Keep one row per sweep and pass the spread across "
            "windows or ranges as its systematic error (yerr_sys=)."
        )
    for column in (loc_col, held_col, err_col):
        table[column] = pd.to_numeric(table[column], errors="raise").astype(float)
    has_point = np.isfinite(table[loc_col].to_numpy(dtype=float))
    criteria = table.loc[has_point, "criterion"]
    if criteria.isna().any():
        bad = table.loc[has_point & table["criterion"].isna().to_numpy(), "sweep_id"].tolist()
        raise FitInputError(
            f"fit_boundary(): sweep(s) {bad} give a location but no criterion. Every point "
            "must say which criterion placed it."
        )
    _refuse_mixed(sorted({str(c) for c in criteria}), "criterion")
    for column in ("method", "sign"):
        if column in table.columns:
            _refuse_mixed(sorted({str(v) for v in table.loc[has_point, column].dropna()}), column)
    criterion = str(criteria.iloc[0]) if len(criteria) else "none"

    rows = pd.DataFrame({
        "sweep_id": table["sweep_id"].to_numpy(),
        "status": np.where(has_point, _USED, _NO_POINT),
        held_col: table[held_col].to_numpy(dtype=float),
        loc_col: table[loc_col].to_numpy(dtype=float),
        err_col: np.where(has_point, table[err_col].to_numpy(dtype=float), np.nan),
        "significance": (
            pd.to_numeric(table["significance"]).to_numpy(dtype=float)
            if "significance" in table.columns else np.nan
        ),
        "criterion": criterion,
        "measurement_file": (
            table["measurement_file"].to_numpy() if "measurement_file" in table.columns
            else np.nan
        ),
    })
    sys_column = sys_col in table.columns
    if sys_column:
        rows[sys_col] = pd.to_numeric(table[sys_col]).to_numpy(dtype=float)
    meta = {
        "swept_axis": swept_axis,
        "criterion": criterion,
        "signal": None,
        "source": "DataFrame",
        "select": None,
        "other_axis_sweep_ids": [],
        "sys_column": sys_column,
    }
    return rows, meta


def _sorted_ids(ids: Iterable) -> list:
    """Distinct sweep ids in order: numerically for numbers, as text for a mixture."""
    distinct = list(dict.fromkeys(ids))
    try:
        return sorted(distinct)
    except TypeError:  # ids of mixed types from a DataFrame
        return sorted(distinct, key=str)


def _refuse_mixed(values: list[str], what: str) -> None:
    if len(values) > 1:
        raise FitInputError(
            f"fit_boundary(): the points mix {what} values {values}. Different criteria mark "
            "systematically different places on one feature, so one curve through them "
            "would draw the change of criterion as structure. Fit each criterion "
            "separately (one fit_boundary call per criterion) and compare the curves."
        )


def _apply_exclude(rows: pd.DataFrame, exclude: Iterable) -> list:
    """Mark the caller's exclusions, refusing ids that are unknown or have no point."""
    if isinstance(exclude, (str, bytes)) or not isinstance(exclude, Iterable):
        raise FitInputError(
            f"fit_boundary(): exclude must be a collection of sweep ids, got {exclude!r}. "
            "Pass e.g. exclude=[3]."
        )
    wanted = list(exclude)
    known = {sid: status for sid, status in zip(rows["sweep_id"], rows["status"], strict=True)}
    unknown = [s for s in wanted if s not in known]
    if unknown:
        raise FitInputError(
            f"fit_boundary(): exclude names sweep id(s) {unknown} that are not in this "
            f"family (sweep ids: {list(known)})."
        )
    pointless = [s for s in wanted if known[s] != _USED]
    if pointless:
        raise FitInputError(
            f"fit_boundary(): exclude names sweep id(s) {pointless}, which give no point "
            f"(status {[known[s] for s in pointless]}), so there is nothing to leave out."
        )
    mask = rows["sweep_id"].isin(wanted).to_numpy()
    rows.loc[mask, "status"] = _EXCLUDED
    return _sorted_ids(wanted)


def _resolve_sys(yerr_sys: Any, rows: pd.DataFrame, sys_col: str, column: bool) -> dict:
    """Fill ``rows[sys_col]`` from ``yerr_sys`` or the frame's column; return the record."""
    has_point = rows["status"].isin([_USED, _EXCLUDED]).to_numpy()
    if yerr_sys is None and not column:
        rows[sys_col] = np.nan
        return {"given": False, "source": None, "values": {}}
    if yerr_sys is not None and column:
        raise FitInputError(
            f"fit_boundary(): the points frame has a {sys_col!r} column and yerr_sys= was "
            "also given. Pass one of them, so the record says which systematic was used."
        )
    if column:
        source = f"column {sys_col}"
    elif _is_real(yerr_sys):
        source = "scalar"
        rows[sys_col] = np.where(has_point, float(yerr_sys), np.nan)
    elif isinstance(yerr_sys, (Mapping, pd.Series)):
        source = "per sweep"
        given = dict(yerr_sys.items())
        known = set(rows["sweep_id"])
        unknown = [k for k in given if k not in known]
        if unknown:
            raise FitInputError(
                f"fit_boundary(): yerr_sys names sweep id(s) {unknown} that are not in this "
                f"family (sweep ids: {sorted(known, key=str)})."
            )
        values = []
        for sid, point in zip(rows["sweep_id"], has_point, strict=True):
            value = given.get(sid, np.nan)
            values.append(float(value) if _is_real(value) else np.nan)
        rows[sys_col] = np.where(has_point, values, np.nan)
    else:
        raise FitInputError(
            f"fit_boundary(): yerr_sys must be a number, or a mapping (dict or Series) from "
            f"sweep_id to a systematic error; got {type(yerr_sys).__name__}."
        )
    sys_values = rows[sys_col].to_numpy(dtype=float)
    bad = [
        sid for sid, value, point in zip(rows["sweep_id"], sys_values, has_point, strict=True)
        if point and not (np.isfinite(value) and value >= 0)
    ]
    if bad:
        raise FitInputError(
            f"fit_boundary(): sweep(s) {bad} give a point but no finite, non-negative "
            f"systematic error in {'yerr_sys' if not column else sys_col}. Supply one for "
            "every sweep with a point (0 where there is none), or exclude= the sweep."
        )
    return {
        "given": True,
        "source": source,
        "values": {
            sid: float(v)
            for sid, v, point in zip(rows["sweep_id"], sys_values, has_point, strict=True)
            if point
        },
    }


def _band(result: FitResult, defn: ModelDefinition, grid: np.ndarray, params: dict) -> pd.DataFrame:
    """The model and its delta-method 1-sigma error at each held value, via ``derive``."""
    roles = _ROLES[params["swept_axis"]]
    unit = roles["unit"]
    names = defn.param_names
    values, errors = [], []
    for g in grid:
        point = np.array([g])
        value, err = result.derive(
            lambda p, point=point: float(defn.function(point, *[p[n] for n in names])[0])
        )
        values.append(value)
        errors.append(err)
    values_arr, errors_arr = np.asarray(values, dtype=float), np.asarray(errors, dtype=float)
    return pd.DataFrame({
        roles["held"]: grid,
        f"model_{unit}": values_arr,
        f"model_err_{unit}": errors_arr,
        f"lower_{unit}": values_arr - errors_arr,
        f"upper_{unit}": values_arr + errors_arr,
        "criterion": params["criterion"],
    })


def _free_names(defn: ModelDefinition, fixed: dict | None) -> list[str]:
    return [n for n in defn.param_names if n not in (fixed or {})]


def _jackknife(
    defn: ModelDefinition,
    full: FitResult,
    sweep_ids: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    e: np.ndarray,
    fit_kwargs: dict,
    params: dict,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Refit without each used sweep in turn; shifts in HESSE sigma and the jackknife SE.

    Each refit is seeded at the full fit, so it answers "where does the
    minimum move?" rather than "which minimum does a cold start find?".
    """
    held_col = _ROLES[params["swept_axis"]]["held"]
    free = _free_names(defn, fit_kwargs.get("fixed"))
    columns = ["sweep_id", held_col, "refit", "dof"]
    for name in free:
        columns += [name, f"{name}_shift_sigma"]
    columns.append("criterion")
    if not full.success:
        return pd.DataFrame(columns=columns), {name: math.nan for name in free}
    seed = {**(fit_kwargs.get("initial") or {}), **full.parameters}
    rows = []
    n = len(x)
    for i in range(n):
        keep = np.arange(n) != i
        refit = fit(defn, x[keep], y[keep], yerr=e[keep], initial=seed,
                    limits=fit_kwargs.get("limits"), fixed=fit_kwargs.get("fixed"))
        if refit.success:
            state = "ok"
        elif refit.dof <= 0:
            state = f"underdetermined: dof {refit.dof}"
        else:
            state = "failed"
        row = {"sweep_id": sweep_ids[i], held_col: x[i], "refit": state, "dof": refit.dof,
               "criterion": params["criterion"]}
        for name in free:
            value = refit.parameters[name] if refit.success else math.nan
            sigma = full.uncertainties.get(name, math.nan)
            row[name] = value
            row[f"{name}_shift_sigma"] = (
                (value - full.parameters[name]) / sigma if sigma and sigma > 0 else math.nan
            )
        rows.append(row)
    table = pd.DataFrame(rows, columns=columns)
    se: dict[str, float] = {}
    all_ok = bool(len(table)) and bool((table["refit"] == "ok").all())
    for name in free:
        if not all_ok or n < 2:
            se[name] = math.nan
            continue
        leave_out = table[name].to_numpy(dtype=float)
        se[name] = float(np.sqrt((n - 1) / n * np.sum((leave_out - leave_out.mean()) ** 2)))
    return table, se


def _information(result: FitResult, n: int) -> tuple[float, float, int]:
    """AIC and BIC from chi-squared with the errors taken as known, and the free count."""
    k = result.n_points - result.dof  # free parameters: dof is n_points - n_free (FT-1)
    if not result.success:
        return math.nan, math.nan, k
    return result.chi_squared + 2.0 * k, result.chi_squared + k * math.log(n), k


def _comparison(
    stated: FitResult,
    defn: ModelDefinition,
    others: list[ModelDefinition],
    x: np.ndarray,
    y: np.ndarray,
    e: np.ndarray,
    params: dict,
) -> tuple[pd.DataFrame, list[str]]:
    n = len(x)
    aic0, bic0, k0 = _information(stated, n)
    rows = [{
        "model": defn.name, "role": "stated", "success": stated.success, "n_free": k0,
        "dof": stated.dof, "chi_squared": stated.chi_squared,
        "reduced_chi_squared": stated.reduced_chi_squared, "aic": aic0, "bic": bic0,
        "delta_aic": 0.0 if stated.success else math.nan,
        "delta_bic": 0.0 if stated.success else math.nan,
    }]
    failed = []
    for other in others:
        result = fit(other, x, y, yerr=e)
        aic, bic, k = _information(result, n)
        if not result.success:
            failed.append(other.name)
        rows.append({
            "model": other.name, "role": "compared", "success": result.success, "n_free": k,
            "dof": result.dof, "chi_squared": result.chi_squared,
            "reduced_chi_squared": result.reduced_chi_squared, "aic": aic, "bic": bic,
            "delta_aic": aic - aic0, "delta_bic": bic - bic0,
        })
    table = pd.DataFrame(rows)
    table["criterion"] = params["criterion"]
    return table, failed


def _fit_once(
    defn: ModelDefinition,
    rows: pd.DataFrame,
    weight: np.ndarray,
    others: list[ModelDefinition],
    fit_kwargs: dict,
    params: dict,
) -> BoundaryFit:
    """One weighted fit with every diagnostic; ``weight`` is each row's error for this fit."""
    axis = params["swept_axis"]
    roles = _ROLES[axis]
    which = "statistical errors" if params["errors"] == "stat" else "stat+sys errors"
    used = (rows["status"] == _USED).to_numpy()
    x = rows[roles["held"]].to_numpy(dtype=float)[used]
    y = rows[roles["location"]].to_numpy(dtype=float)[used]
    e = weight[used]
    sweep_ids = rows["sweep_id"].to_numpy()[used]
    result = fit(defn, x, y, yerr=e, **fit_kwargs)
    messages: list[str] = []
    if not result.success:
        messages.append(
            f"fit_boundary: the {defn.name!r} fit with {which} did not converge "
            f"(success=False; {result.n_points} points, dof {result.dof}). There is no curve "
            "to report. Check the sweeps table, the starting values (initial=) and the "
            "form itself."
        )

    table = rows.copy()
    table[roles["fit_err"]] = np.where(
        table["status"].isin([_USED, _EXCLUDED]).to_numpy(), weight, np.nan
    )
    held = table[roles["held"]].to_numpy(dtype=float)
    predicted = np.full(held.shape, np.nan)
    known = np.isfinite(held)
    if result.success and known.any():
        with np.errstate(all="ignore"):
            predicted[known] = result.curve(held[known])
    residual = table[roles["location"]].to_numpy(dtype=float) - predicted
    table[roles["predicted"]] = predicted
    table[roles["residual"]] = residual
    table["pull"] = residual / table[roles["fit_err"]].to_numpy(dtype=float)
    table = table.reindex(columns=SWEEP_COLUMNS[axis])
    try:
        table = table.sort_values("sweep_id", kind="stable")
    except TypeError:  # sweep ids of mixed types from a DataFrame: order them as text
        table = table.sort_values("sweep_id", key=lambda s: s.map(str), kind="stable")
    table = table.reset_index(drop=True)

    grid = (
        np.linspace(float(x.min()), float(x.max()), BAND_POINTS) if x.size else np.array([])
    )
    band = _band(result, defn, grid, params)

    quantile = math.nan
    low = False
    if result.success and result.dof >= 1:
        quantile = float(stats.chi2.cdf(result.chi_squared, result.dof))
        low = quantile < LOW_CHI2_TAIL
    if low:
        messages.append(
            f"fit_boundary: with {which}, chi2 = {result.chi_squared:.3g} on {result.dof} dof "
            f"(chi2/dof = {result.reduced_chi_squared:.3g}); P(chi2 <= that) = {quantile:.2g} "
            f"< {LOW_CHI2_TAIL:.2g}, so the points scatter far less than their errors "
            "allow. The errors are overstated (a location error floored at the grid "
            "spacing does this), and the HESSE errors, the band and the information "
            "criteria inherit it: the parameter errors are too large, and every delta_aic "
            "and delta_bic is squeezed towards its parameter penalty, which favours the "
            "simpler model. Compare jackknife_se with hesse_err in estimates."
        )

    jackknife, se = _jackknife(defn, result, sweep_ids, x, y, e, fit_kwargs, params)
    not_ok = jackknife[jackknife["refit"] != "ok"] if len(jackknife) else jackknife
    if len(not_ok):
        messages.append(
            f"fit_boundary: with {which}, the leave-one-sweep-out refit without sweep(s) "
            f"{not_ok['sweep_id'].tolist()} did not succeed ({not_ok['refit'].tolist()}), so "
            "jackknife_se is NaN; those rows stay in the jackknife table."
        )

    comparison, failed = _comparison(result, defn, others, x, y, e, params)
    if failed:
        messages.append(
            f"fit_boundary: with {which}, the compared model(s) {failed} did not converge "
            "(success=False); their rows in comparison are NaN. Check their starting "
            "values (param_defaults or guess of the ModelDefinition)."
        )

    estimates = []
    for name in defn.param_names:
        is_fixed = name in (fit_kwargs.get("fixed") or {})
        shift_col = f"{name}_shift_sigma"
        shifts = (
            jackknife[shift_col].to_numpy(dtype=float)
            if shift_col in jackknife.columns and len(jackknife) else np.array([])
        )
        finite = np.isfinite(shifts)
        if finite.any():
            where = int(np.nanargmax(np.abs(np.where(finite, shifts, np.nan))))
            max_shift, influential = float(abs(shifts[where])), jackknife["sweep_id"].iloc[where]
        else:
            max_shift, influential = math.nan, None
        estimates.append({
            "parameter": name,
            "value": result.parameters.get(name, math.nan) if result.success else math.nan,
            "hesse_err": result.uncertainties.get(name, math.nan),
            "fixed": is_fixed,
            "jackknife_se": se.get(name, math.nan),
            "max_abs_shift_sigma": max_shift,
            "most_influential_sweep_id": influential,
            "criterion": params["criterion"],
        })

    for message in messages:
        warnings.warn(message, UserWarning, stacklevel=3)
    return BoundaryFit(
        fit=result,
        estimates=pd.DataFrame(estimates),
        sweeps=table,
        band=band,
        jackknife=jackknife,
        comparison=comparison,
        chi_squared_quantile=quantile,
        scatter_below_errors=low,
        warnings=tuple(messages),
        params=params,
        model=defn,
    )


def fit_boundary(
    points: TransitionResult | pd.DataFrame,
    model: str | ModelDefinition,
    *,
    swept_axis: str | None = None,
    compare: Sequence[str | ModelDefinition] = (),
    yerr_sys: float | Mapping | pd.Series | None = None,
    exclude: Iterable = (),
    initial: dict[str, float] | None = None,
    limits: dict[str, tuple[float, float]] | None = None,
    fixed: dict[str, float] | None = None,
) -> BoundaryFit:
    """Fit a stated form to one criterion's points, one per sweep, and say how far to trust it.

    The second stage of the two-stage route: detection and
    :meth:`~calamari.transitions.TransitionResult.select` give each sweep at
    most one point, and this fits a curve through them -- the location
    against the held coordinate, weighted by each point's error -- with
    :func:`~calamari.fitting.engine.fit` (iminuit MIGRAD + HESSE, FT-1). The form is
    the caller's assumption. The points cannot choose it: two forms can pass
    through the same points inside their errors and disagree everywhere
    else, which is why ``compare=`` and the band's restriction to the data
    range exist.

    Parameters
    ----------
    points : TransitionResult or pandas.DataFrame
        One point per sweep, in one of two forms.

        - A :class:`~calamari.transitions.TransitionResult` that has been
          ``select()``-ed on the swept axis. Its selection sets the
          criterion (``method/sign``). An unselected result is refused,
          because a sweep then holds several candidates and they would be
          counted as separate data. The family is every sweep the result
          knows of: searched (``params['per_sweep']``) and skipped
          (``skipped``); sweeps swept along the other axis belong to the
          other family, and their ids are recorded in
          ``params['other_axis_sweep_ids']``.
        - A DataFrame with one row per sweep and the columns
          :data:`POINT_COLUMNS` names for ``swept_axis``: ``sweep_id``, the
          held coordinate, the location, its statistical error and
          ``criterion``. Optional columns: ``significance``, the systematic
          error (``T_err_sys_K`` or ``H_err_sys_T``, instead of
          ``yerr_sys=``) and ``measurement_file``. A row with a NaN location
          is a sweep with no point, and is listed as such. This is the form
          for per-sweep :func:`~calamari.fitting.breakpoint.fit_breakpoint` results:
          ``T_K=x_break``, ``T_err_K=x_break_err``, and a criterion such as
          ``'breakpoint:segmented_linear'``. Duplicate ``sweep_id`` values
          are refused: several smoothing windows or fitting ranges of one
          sweep are one point with a systematic spread, never several rows.

    model : str or ModelDefinition
        The form: a registered model name or a
        :class:`~calamari.fitting.models.ModelDefinition`, called as ``f(held,
        *params)`` and returning the location. A monotonic boundary means a
        monotonic parametric form; nonparametric isotonic regression is not
        offered in this version (it gives no parameters and no band).
    swept_axis : {'T', 'H'}, optional
        Which coordinate the criterion located. ``'T'``: ``T_K +/- T_err_K``
        against the held ``H_T``. ``'H'``: ``H_T +/- H_err_T`` against the
        held ``T_K``. Required for a DataFrame, and for a result selected on
        both axes; otherwise taken from the one selection.
    compare : sequence of str or ModelDefinition, optional
        Simpler forms to fit to the same points with the same errors, for
        :attr:`BoundaryFit.comparison`. Each starts from its own
        ``param_defaults`` and ``guess``; to seed one differently, define it
        with different defaults.
    yerr_sys : float or mapping, optional
        A systematic error per sweep, supplied by the caller: the half-spread
        of a feature's location across the smoothing windows of a
        :func:`~calamari.scales.scan_smoothing` scan, say, or across the
        ranges of a :func:`~calamari.fitting.windows.fit_windows` run. A mapping (or
        Series) from ``sweep_id`` to the error, in the location's unit, with a
        finite, non-negative value for every sweep with a point, or one
        number for all. Nothing is derived by default. When given, the fit is
        made twice and both are reported: with the statistical errors alone
        (the returned object) and with statistical and systematic in
        quadrature (``.total``).
    exclude : collection of sweep ids, optional
        Points to leave out of the fit. They stay in ``sweeps``, marked
        ``'excluded by caller'``, with their residual from the fit made
        without them, and in ``params['exclude']``.
    initial, limits, fixed : dict, optional
        As for :func:`~calamari.fitting.engine.fit`, for the stated model. The
        jackknife refits use the same limits and fixed values, seeded at the
        full fit.

    Returns
    -------
    BoundaryFit
        The fit with statistical errors, its diagnostics and the recipe; the
        stat+sys fit, when there is one, in ``.total``.

    Raises
    ------
    ValueError
        For an unselected result, or one selected on another axis; a result
        selected on both axes without ``swept_axis``; a DataFrame without
        ``swept_axis``, missing a required column (named), or with a missing
        or duplicate ``sweep_id``; points of more than one criterion; a used
        point whose held coordinate is not finite or whose error is not
        finite and positive; ``exclude`` or ``yerr_sys`` naming a sweep that
        is unknown or has no point; a malformed ``yerr_sys``, or one given
        beside a systematic column.
    FitConstraintError
        A forwarded fixed value conflicts with its limits; propagated from
        :func:`~calamari.fitting.engine.fit` before minimisation.
    KeyError
        For an unknown model name, here or in ``compare``, or an unknown
        parameter in ``initial``, ``limits`` or ``fixed``.

    Warns
    -----
    UserWarning
        When a fit does not converge, when the scatter is below the errors
        (:data:`LOW_CHI2_TAIL`), when a jackknife refit fails, and when a
        compared model does not converge. Each is also kept in
        :attr:`BoundaryFit.warnings`, and the result is returned regardless.

    Notes
    -----
    **Weights.** Each point is weighted by its location error alone
    (absolute errors, the ``fit(..., yerr=)`` convention). ``significance``
    is shown in ``sweeps`` and never used as a weight. It is a
    normalised strength diagnostic used in filtering, not a calibrated
    probability that a feature is real. How well the point is
    *located* is its error's job, and a weak feature that is sharply placed
    is sharply placed. Weighting by significance as well would count the
    strength twice, and would have no likelihood behind it.

    **Grid-floored errors.** A candidate's location error is floored at the
    scan's grid spacing (``docs/DETECTION.md``). On clean sweeps the floor
    is all of it, the points sit far closer to the curve than their errors,
    and chi2/dof comes out near zero. :attr:`BoundaryFit.scatter_below_errors`
    flags that, from the chi-squared distribution for the fit's dof (see
    :data:`LOW_CHI2_TAIL`). The HESSE errors and the band are then the floor
    propagated, not the scatter, and the jackknife standard error, which is
    driven by the scatter itself, comes out far smaller. Above the floor the
    error comes from a local quadratic fit to smoothed data, whose residuals
    the smoothing correlates, so it is somewhat optimistic: chi2/dof well
    above 1 means that, or a form that does not fit.

    **The band** is ``value +/- sqrt(J C J^T)`` at each held value, the
    first-order propagation of the fit covariance ``C``
    (:meth:`~calamari.fitting.engine.FitResult.derive`). It is exact for a
    form linear in its parameters and good when the relative errors are
    small. It describes the model under its assumptions and says nothing
    about whether the form is right.

    **The jackknife.** Each used sweep is left out and the model refitted.
    ``*_shift_sigma`` is the shift of each parameter, in units of its HESSE
    error, and the sweep with the largest is the one the curve leans on
    most. The jackknife standard error, ``sqrt((n - 1)/n * sum((p_i -
    p_mean)^2))``, estimates the parameter's spread from the points' own
    scatter rather than from their stated errors, so set beside HESSE it is
    a second check on those errors. With few sweeps it is itself noisy. It
    is NaN unless every refit succeeded, because the formula needs all n.
    Leaving one observation out at a time is Quenouille's jackknife (M. H.
    Quenouille, "Notes on bias in estimation", *Biometrika* **43**, 353
    (1956)); the variance estimate is the one B. Efron and C. Stein analyse
    in "The jackknife estimate of variance", *Ann. Statist.* **9**, issue 3
    (1981), doi:10.1214/aos/1176345462.

    **Information criteria.** With the errors taken as known, ``-2 ln L`` is
    chi-squared up to a constant, so ``AIC = chi2 + 2k`` and ``BIC = chi2 +
    k ln n`` for ``k`` free parameters and ``n`` used points. ``delta_aic``
    and ``delta_bic`` are the compared model's minus the stated model's, so
    **positive favours the stated model**. They inherit the errors' scale:
    overstated errors (``scatter_below_errors``) shrink every chi-squared
    difference and leave the parameter penalty, which favours the simpler
    form; understated ones do the opposite. They describe; they do not test.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from calamari import fit_boundary
    >>> rng = np.random.default_rng(5)
    >>> h = np.linspace(0.0, 1.8, 10)
    >>> t = 14.0 - 1.5 * h**2 + rng.normal(scale=0.05, size=h.size)
    >>> points = pd.DataFrame({"sweep_id": range(10), "H_T": h, "T_K": t,
    ...                        "T_err_K": 0.05, "criterion": "peak/max"})
    >>> from calamari.fitting import ModelDefinition
    >>> quadratic = ModelDefinition(
    ...     name="quadratic", function=lambda h, t0, b: t0 - b * np.asarray(h) ** 2,
    ...     param_names=("t0", "b"), param_defaults={"t0": 10.0, "b": 1.0},
    ...     description="t0 - b H^2")
    >>> bf = fit_boundary(points, quadratic, swept_axis="T", compare=["linear"])
    >>> print(bf.estimates[["parameter", "value", "hesse_err"]].round(3).to_string(index=False))
    parameter  value  hesse_err
           t0 13.975      0.023
            b  1.477      0.015
    >>> bool(bf.comparison.loc[1, "delta_bic"] > 0)   # the straight line fits far worse
    True
    """
    defn = _resolve_model(model)
    if isinstance(compare, (str, ModelDefinition)):
        compare = (compare,)
    others = [_resolve_model(m) for m in compare]

    if isinstance(points, TransitionResult):
        rows, meta = _rows_from_result(points, swept_axis)
    elif isinstance(points, pd.DataFrame):
        rows, meta = _rows_from_frame(points, swept_axis)
    else:
        raise FitInputError(
            "fit_boundary(): points must be a select()-ed TransitionResult or a DataFrame of "
            f"one point per sweep (columns {dict(POINT_COLUMNS)}); got {type(points).__name__}."
        )
    axis = meta["swept_axis"]
    roles = _ROLES[axis]
    excluded = _apply_exclude(rows, exclude)
    sys_record = _resolve_sys(yerr_sys, rows, roles["sys"], meta["sys_column"])

    used = (rows["status"] == _USED).to_numpy()
    with_point = rows["status"].isin([_USED, _EXCLUDED]).to_numpy()
    held = rows[roles["held"]].to_numpy(dtype=float)
    err = rows[roles["err"]].to_numpy(dtype=float)
    bad_held = rows.loc[with_point & ~np.isfinite(held), "sweep_id"].tolist()
    if bad_held:
        raise FitInputError(
            f"fit_boundary(): sweep(s) {bad_held} give a location but no finite held "
            f"coordinate ({roles['held']}), so they cannot be placed on the curve."
        )
    bad_err = rows.loc[with_point & ~(np.isfinite(err) & (err > 0)), "sweep_id"].tolist()
    if bad_err:
        raise FitInputError(
            f"fit_boundary(): sweep(s) {bad_err} give a location whose statistical error "
            f"({roles['err']}) is not finite and positive, so they cannot be weighted. "
            "Supply an error, or exclude= those sweeps."
        )

    fit_kwargs = {"initial": initial, "limits": limits, "fixed": fixed}
    params = {
        "model": defn.name,
        "model_description": defn.description,
        "criterion": meta["criterion"],
        "signal": meta["signal"],
        "swept_axis": axis,
        "location": roles["location"],
        "held": roles["held"],
        "source": meta["source"],
        "select": meta["select"],
        "other_axis_sweep_ids": meta["other_axis_sweep_ids"],
        "exclude": excluded,
        "n_used": int(used.sum()),
        "errors": "stat",
        "yerr_sys": sys_record,
        "compare": [m.name for m in others],
        "initial": dict(initial) if initial else None,
        "limits": dict(limits) if limits else None,
        "fixed": dict(fixed) if fixed else None,
        "band_points": BAND_POINTS,
        "low_chi2_tail": LOW_CHI2_TAIL,
        "information_criteria": _IC_TEXT,
        "weighting": "location error only; significance shown, never a weight",
        "calamari_version": _calamari_version(),
    }
    stat = _fit_once(defn, rows, err, others, fit_kwargs, params)
    if not sys_record["given"]:
        return stat
    total_err = np.hypot(err, rows[roles["sys"]].to_numpy(dtype=float))
    total = _fit_once(defn, rows, total_err, others, fit_kwargs, {**params, "errors": "stat+sys"})
    return replace(stat, total=total)
