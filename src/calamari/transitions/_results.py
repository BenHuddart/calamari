"""Candidate linking, result views, selection and reconciliation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from calamari._chains import _chain_walk
from calamari._provenance import describe_callable as _describe_callable
from calamari.constants import OE_PER_TESLA
from calamari.exceptions import CurveInputError, DetectionInputError, SmoothingSensitivityError
from calamari.measurement import SCAN_H_TOL_OE, SCAN_T_TOL_K
from calamari.quality import _plain_label
from calamari.transitions._metadata import (
    CHAIN_COLUMNS,
    DEFAULT_LINK_BY,
    DEFAULT_MIN_SIGNIFICANCE,
    METHODS,
)


def _stouffer(significances: np.ndarray) -> float:
    """Stouffer-like diagnostic combination: ``sum(scores) / sqrt(n)``.

    The non-negative prominence ratios are not centred standard normal
    scores, and selected chain members need not be independent. Even a
    constant positive score grows as sqrt(n) under this combination. Chain
    length can therefore increase the value without establishing a real
    boundary; it is not a calibrated significance or false-alarm probability.
    """
    values = np.asarray(significances, dtype=float)
    if values.size == 0:
        return float("nan")
    return float(values.sum() / math.sqrt(values.size))


#: Per swept axis: which candidate column indexes the family of sweeps,
#: and which one locates the feature within a sweep. A T-scan is one of a
#: family taken at different fixed fields, so H_T orders the family and
#: T_K is where the boundary sits; an H-scan is the mirror image.
_LINK_GEOMETRY = {"T": ("H_T", "T_K"), "H": ("T_K", "H_T")}


def _unique_or_none(values):
    """The single value a chain's members agree on, else ``None``.

    Only ``by`` columns are constant within a chain by construction; a
    column the caller left out of ``by`` may genuinely vary, and inventing
    a winner for it would put a value in the chain table that no member
    has.
    """
    seen = {v for v in values if not (isinstance(v, float) and math.isnan(v))}
    return next(iter(seen)) if len(seen) == 1 else None


def _link_candidates(
    cands: pd.DataFrame, by: tuple[str, ...], t_tol_k: float | None, h_tol_t: float | None
) -> tuple[np.ndarray, pd.DataFrame]:
    """Assign every candidate to exactly one cross-sweep chain.

    The deterministic walk documented on :meth:`TransitionResult.link`,
    run by :func:`_chain_walk` over :data:`_LINK_GEOMETRY` (the held
    coordinate orders the family, the swept one places the feature).
    Returns the per-row ``chain_id`` array and the chain table
    (:data:`CHAIN_COLUMNS`), ids ordered by descending
    ``chain_significance`` then by where the chain starts.
    """
    n_rows = len(cands)
    if n_rows == 0:
        return np.array([], dtype=int), pd.DataFrame(columns=CHAIN_COLUMNS)

    chains, fam, loc = _chain_walk(
        cands, by, geometry=_LINK_GEOMETRY, tolerances={"T": t_tol_k, "H": h_tol_t}
    )
    axis_col = cands["swept_axis"].to_numpy()
    sig = cands["significance"].to_numpy(dtype=float)
    t_vals = cands["T_K"].to_numpy(dtype=float)
    h_vals = cands["H_T"].to_numpy(dtype=float)

    def finite_or_last(value: float) -> float:
        """NaN sorts last rather than comparing false against everything."""
        return math.inf if math.isnan(value) else value

    def chain_key(members: list[int]) -> tuple:
        combined = _stouffer(sig[members])
        start = members[0]
        return (
            math.inf if math.isnan(combined) else -combined,
            finite_or_last(fam[start]),
            finite_or_last(loc[start]),
            start,
        )

    def summarise(values: np.ndarray, how) -> float:
        """``how`` over the present entries; NaN for an all-missing column
        (numpy's nan-aware reductions warn and return NaN there, and a
        RuntimeWarning is not what an absent coordinate deserves). NaN, not
        non-finite: an infinite significance is a real value here (a
        noiseless sweep reports one) and must survive to the chain table."""
        present = values[~np.isnan(values)]
        return float(how(present)) if present.size else float("nan")

    chains.sort(key=chain_key)
    chain_ids = np.full(n_rows, -1, dtype=int)
    rows = []
    for cid, members in enumerate(chains):
        chain_ids[members] = cid
        member_sig = sig[members]
        rows.append(
            {
                "chain_id": cid,
                "n": len(members),
                "signal": _unique_or_none(cands["signal"].to_numpy()[members])
                if "signal" in cands.columns
                else None,
                "method": _unique_or_none(cands["method"].to_numpy()[members]),
                "sign": _unique_or_none(cands["sign"].to_numpy()[members]),
                "swept_axis": _unique_or_none(axis_col[members]),
                "T_K_min": summarise(t_vals[members], np.min),
                "T_K_max": summarise(t_vals[members], np.max),
                "H_T_min": summarise(h_vals[members], np.min),
                "H_T_max": summarise(h_vals[members], np.max),
                "significance_median": summarise(member_sig, np.median),
                "significance_max": summarise(member_sig, np.max),
                "chain_significance": _stouffer(member_sig),
            }
        )
    table = pd.DataFrame(rows, columns=CHAIN_COLUMNS)
    return chain_ids, table


#: Each method's ``sign`` vocabulary, as the detectors write it (see
#: :func:`find_transitions`). An onset inherits the sign of the derivative
#: extremum it was built from. :meth:`TransitionResult.select` checks a
#: requested sign against this, so ``sign='max'`` on a derivative -- which
#: can never match a row -- is refused by name instead of emptying every
#: sweep for a reason the result cannot show.
_SIGN_VOCABULARY = {
    "peak": ("max", "min"),
    "derivative": ("rise", "fall"),
    "kink": ("up", "down"),
    "onset": ("rise", "fall"),
    "fisher": ("max", "min"),
}

#: Per swept axis, for :meth:`TransitionResult.select`: the window keyword
#: and its unit, then the fixed coordinate a callable window is given and
#: its unit.
_SELECT_WINDOW = {
    "T": ("t_window_k", "K", "H_T", "T"),
    "H": ("h_window_t", "T", "T_K", "K"),
}


# Optional sensitivity gates are independent of significance and cross-sweep recurrence.
_SENSITIVITY_GATES = {
    "min_smoothing_n": ("n_present", "min", None),
    "min_smoothing_persistence": ("persistence", "min", None),
    "min_smoothing_sign_consistency": ("sign_consistency", "min", None),
    "max_smoothing_spread_k": ("T_K_spread", "max", "T"),
    "max_smoothing_spread_t": ("H_T_spread", "max", "H"),
    "require_smoothing_interval_overlap": ("interval_overlap", "bool", None),
}


def _sensitivity_reasons(cands: pd.DataFrame, thresholds: dict) -> pd.Series:
    """Reasons sensitivity gates reject each full candidate, including unassessed rows."""
    requested = {k: v for k, v in thresholds.items()
                 if k in _SENSITIVITY_GATES and v is not None}
    if not requested:
        return pd.Series("", index=cands.index, dtype=str)
    if requested and "smoothing_status" not in cands:
        raise SmoothingSensitivityError(
            "robust(): sensitivity thresholds need scan.reference(window, "
            "min_member_significance=...) first."
        )
    for name, value in requested.items():
        if name == "require_smoothing_interval_overlap":
            valid = isinstance(value, (bool, np.bool_))
        else:
            valid = (not isinstance(value, (bool, np.bool_))
                     and isinstance(value, (int, float, np.integer, np.floating))
                     and np.isfinite(value) and value >= 0)
            if name in ("min_smoothing_persistence", "min_smoothing_sign_consistency"):
                valid = valid and value <= 1
            if name == "min_smoothing_n":
                valid = valid and value == int(value)
        if not valid:
            raise SmoothingSensitivityError(
                f"robust(): invalid {name}={value!r}; use a non-negative finite "
                "threshold (fractions in [0, 1], counts integral, overlap boolean)."
            )
    reasons = []
    for _, row in cands.iterrows():
        failed = []
        for name, threshold in requested.items():
            stat, direction, axis = _SENSITIVITY_GATES[name]
            if direction == "bool" and not threshold:
                continue
            if axis is not None and row.swept_axis != axis:
                continue
            if row.smoothing_status != "assessed":
                failed.append(str(row.smoothing_status))
                continue
            value = row[f"smoothing_{stat}"]
            passed = False if pd.isna(value) else (
                bool(value) if direction == "bool" else
                value >= threshold if direction == "min" else value <= threshold
            )
            if not passed:
                failed.append(name)
        reasons.append(";".join(dict.fromkeys(failed)))
    return pd.Series(reasons, index=cands.index, dtype=str)


def _robust_mask(cands: pd.DataFrame, thresholds: dict) -> pd.Series:
    """Which rows of ``cands`` pass the :meth:`TransitionResult.robust` gates.

    Shared by ``robust()`` itself and by the methods that re-apply recorded
    thresholds after their own work (``link()``, ``select()``), so a gate is
    validated and applied identically whichever call triggered it.
    """
    min_chain_significance = thresholds.get("min_chain_significance")
    min_chain_n = thresholds.get("min_chain_n")
    wants_chain = min_chain_significance is not None or min_chain_n is not None
    if wants_chain and "chain_significance" not in cands.columns:
        raise DetectionInputError(
            "robust(): a chain threshold was requested but these candidates "
            "carry no chains -- call link(t_tol_k=...) or link(h_tol_t=...) "
            "first (or detect_transitions_all(..., link={...})). Gating on a "
            "statistic that was never computed would keep every row."
        )
    mask = pd.Series(True, index=cands.index)

    def gate(column: str, threshold: float) -> pd.Series:
        values = cands[column]
        if len(values) and values.isna().all():
            raise DetectionInputError(
                f"robust(): every candidate has NaN {column!r}, so a threshold "
                "on it would drop all rows for a reason the table cannot show. "
                "Remove the threshold or use a statistic these methods report."
            )
        return values >= threshold

    mask &= gate("significance", thresholds["min_significance"])
    for key, column in (
        ("min_n_methods", "n_methods"),
        ("min_sweep_range_over_noise", "sweep_range_over_noise"),
        ("min_chain_significance", "chain_significance"),
        ("min_chain_n", "chain_n"),
    ):
        if thresholds.get(key) is not None:
            mask &= gate(column, thresholds[key])
    if thresholds.get("exclude_edge", True) and len(cands):
        mask &= ~cands["edge"].astype(bool)
    mask &= _sensitivity_reasons(cands, thresholds).eq("")
    return mask


def _check_select_window(value, what: str) -> tuple[float, float]:
    """``value`` as a validated ``(lo, hi)`` pair of plain floats.

    A malformed window is the caller's error and raises, naming ``what``:
    treating a reversed or non-finite window as merely empty would record
    every sweep as having no boundary, for a reason nobody could see.
    """
    try:
        lo, hi = value
    except (TypeError, ValueError):
        raise DetectionInputError(f"select(): {what} {value!r}, which is not a (lo, hi) pair.") from None
    for bound in (lo, hi):
        if isinstance(bound, bool) or not isinstance(bound, (int, float, np.integer, np.floating)):
            raise DetectionInputError(f"select(): {what} {value!r}, whose bounds are not numbers.")
    lo, hi = float(lo), float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
        raise DetectionInputError(
            f"select(): {what} ({lo:g}, {hi:g}), which is not a pair of finite "
            "bounds with lo < hi."
        )
    return lo, hi


def _resolve_select_window(window, axis: str, sweep_id: int, fixed_value: float):
    """One sweep's window for :meth:`TransitionResult.select`.

    ``None`` stays ``None`` (search the whole sweep). A pair is validated.
    A callable is invoked with the sweep's fixed coordinate -- ``H_T`` in
    Tesla for a T-swept sweep, ``T_K`` in Kelvin for an H-swept one -- and
    its answer validated here, in the style of
    :func:`_resolve_batch_smooth_window`, so a broken callable is reported as
    *its* fault, naming the sweep it failed on, rather than surfacing as an
    opaque comparison error later.
    """
    keyword, _, fixed_col, fixed_unit = _SELECT_WINDOW[axis]
    if window is None:
        return None
    if not callable(window):
        return _check_select_window(window, f"{keyword} is")
    name = _describe_callable(window)
    at = f"sweep {sweep_id} ({fixed_col} = {fixed_value:g} {fixed_unit})"
    try:
        resolved = window(fixed_value)
    except Exception as exc:
        raise DetectionInputError(f"select(): {keyword} callable {name!r} raised {exc!r} for {at}.") from exc
    return _check_select_window(resolved, f"{keyword} callable {name!r} returned, for {at},")


def _apply_select(view: pd.DataFrame, axis: str, spec: dict) -> tuple[pd.DataFrame, dict]:
    """Apply one recorded :meth:`TransitionResult.select` to ``view``.

    Rows swept along the other axis pass through untouched. On ``axis``, at
    most one row per ``sweep_id`` survives: among the rows matching the
    recorded signal, method and sign whose location lies inside that
    sweep's recorded window (inclusive), the highest ``significance`` --
    ties broken by location ascending, then by ``candidate_id``, so the
    choice never depends on row order (TR-1). A NaN significance ranks last.

    Returns the filtered view and the record for ``params['select'][axis]``,
    with ``empty_sweeps`` recomputed for this view: the same selection
    re-applied after a looser ``robust()`` can fill a sweep that was empty.
    """
    _, loc_col = _LINK_GEOMETRY[axis]
    windows = spec["windows"]
    on_axis = (view["swept_axis"] == axis).to_numpy(dtype=bool)
    match = (
        on_axis
        & (view["signal"] == spec["signal"]).to_numpy(dtype=bool)
        & (view["method"] == spec["method"]).to_numpy(dtype=bool)
        & (view["sign"] == spec["sign"]).to_numpy(dtype=bool)
    )
    loc = view[loc_col].to_numpy(dtype=float)
    sig = view["significance"].to_numpy(dtype=float)
    sweep = view["sweep_id"].to_numpy()
    cid = view["candidate_id"].to_numpy()

    def rank(i: int) -> tuple:
        s = sig[i]
        return (math.inf if math.isnan(s) else -s, loc[i], int(cid[i]))

    best: dict[int, int] = {}
    for i in np.flatnonzero(match):
        if not math.isfinite(loc[i]):
            continue
        key = int(sweep[i])
        window = windows[key]
        if window is not None and not (window[0] <= loc[i] <= window[1]):
            continue
        if key not in best or rank(i) < rank(best[key]):
            best[key] = i

    keep = ~on_axis
    keep[list(best.values())] = True
    record = {
        "method": spec["method"],
        "sign": spec["sign"],
        "signal": spec["signal"],
        "window": spec["window"],
        "windows": dict(windows),
        "empty_sweeps": sorted(k for k in windows if k not in best),
    }
    return view[keep], record


#: Columns :meth:`TransitionResult.reconcile` adds to the candidates table.
#: The two residuals carry their unit in their name (DM-2) and each is NaN
#: on the other axis's rows; the rest apply to every row.
_RECONCILE_COLUMNS = [
    "cross_residual_K",
    "cross_residual_T",
    "cross_residual_sigma",
    "cross_covered",
    "cross_concordant",
]

#: Per swept axis, for :meth:`TransitionResult.reconcile`: the column
#: holding a point's own error on its swept axis; the residual column its
#: comparison with the *other* family writes and the tolerance keyword
#: that judges it; and the tolerance within which two
#: held values of this family count as one, with its unit and the constant
#: it comes from. The held-value tolerances are the scan-classification
#: ones of :mod:`calamari.measurement` -- the same numbers that decide a
#: coordinate is "held" at all -- with the field one converted from Oe to
#: the candidates' Tesla, as :func:`detect_transitions` converts the field.
_RECONCILE_AXES = {
    "T": {
        "own_err": "T_err_K",
        "residual": "cross_residual_T",
        "tol": "h_tol_t",
        "held_tol": SCAN_H_TOL_OE / OE_PER_TESLA,
        "held_unit": "T",
        "held_const": "SCAN_H_TOL_OE",
    },
    "H": {
        "own_err": "H_err_T",
        "residual": "cross_residual_K",
        "tol": "t_tol_k",
        "held_tol": SCAN_T_TOL_K,
        "held_unit": "K",
        "held_const": "SCAN_T_TOL_K",
    },
}


def _check_reconcile_tolerance(name: str, value) -> float:
    """``value`` as a positive finite plain float, or a ValueError naming it.

    ``None`` is refused with the reason there is no default: how far apart
    two families may sit and still be called one boundary is physics only
    the caller can supply, exactly as for :meth:`TransitionResult.link`.
    """
    if value is None:
        raise DetectionInputError(
            f"reconcile(): {name} is None -- how far the two families may disagree "
            "and still trace one boundary is physics only you can supply, so there "
            "is no default."
        )
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise DetectionInputError(f"reconcile(): {name} must be a number, got {value!r}.")
    value = float(value)
    if not (math.isfinite(value) and value > 0):
        raise DetectionInputError(f"reconcile(): {name} must be positive and finite, got {value!r}.")
    return value


def _check_reconcile_selects(selects: dict | None) -> None:
    """Refuse unless ``params['select']`` holds both ``'T'`` and ``'H'``.

    Checked by :meth:`TransitionResult.reconcile` before recording, and
    again on every rebuild that re-applies it.
    """
    missing = [axis for axis in ("T", "H") if axis not in (selects or {})]
    if missing:
        have = sorted(selects or {})
        raise DetectionInputError(
            f"reconcile(): needs a selection on both swept axes, but params['select'] "
            f"holds {have!r}. Reconciling sweeps that still carry several candidates "
            "is meaningless: each boundary would interpolate between arbitrary "
            "features. Call select(swept_axis=...) for "
            f"{' and '.join(repr(a) for a in missing)} first."
        )


def _boundary_nodes(view: pd.DataFrame, axis: str) -> tuple[np.ndarray, ...]:
    """One family's selected points as sorted interpolation nodes.

    Returns ``(held, location, error, sweep_id)`` for the rows swept along
    ``axis``, sorted by held value. For the T family the held value is
    ``H_T`` and the location ``T_K``, so the nodes describe ``T_b(H)``; the H
    family is the mirror image. A row whose held value or location is not
    finite cannot be a node and is left out.

    Raises if two nodes share a held value within the family's scan
    tolerance: a piecewise-linear boundary through two locations at one
    abscissa is not defined, and any choice between them would be a guess.
    """
    held_col, loc_col = _LINK_GEOMETRY[axis]
    spec = _RECONCILE_AXES[axis]
    rows = view[(view["swept_axis"] == axis).to_numpy(dtype=bool)]
    held = rows[held_col].to_numpy(dtype=float)
    loc = rows[loc_col].to_numpy(dtype=float)
    err = rows[spec["own_err"]].to_numpy(dtype=float)
    sid = rows["sweep_id"].to_numpy()
    usable = np.isfinite(held) & np.isfinite(loc)
    held, loc, err, sid = held[usable], loc[usable], err[usable], sid[usable]
    order = np.argsort(held, kind="stable")
    held, loc, err, sid = held[order], loc[order], err[order], sid[order]

    close = np.diff(held) <= spec["held_tol"]
    if close.any():
        clashes = []
        i = 0
        while i < len(close):
            if not close[i]:
                i += 1
                continue
            j = i
            while j < len(close) and close[j]:
                j += 1
            members = range(i, j + 1)
            ids = [_plain_label(sid[k]) for k in members]
            values = ", ".join(f"{float(held[k]):g}" for k in members)
            clashes.append(f"sweep_ids {ids} at {held_col} = {values} {spec['held_unit']}")
            i = j + 1
        raise DetectionInputError(
            f"reconcile(): {len(clashes)} group(s) of {axis}-swept selections share a "
            f"held {held_col} within {spec['held_tol']:g} {spec['held_unit']} "
            f"({spec['held_const']}): {'; '.join(clashes)}. A boundary interpolated "
            "through two locations at one held value would be a guess between them. "
            "Keep one sweep per held value -- thin repeats before detection "
            "(Catalog.select_representative) or combine them (calamari.repeats) -- "
            "and detect again."
        )
    return held, loc, err, sid


def _interpolate_boundary(
    xq: np.ndarray, held: np.ndarray, loc: np.ndarray, err: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Piecewise-linear boundary, its local slope and its error at ``xq``.

    Returns ``(covered, value, slope, error)``. ``covered`` is False where
    the family has fewer than two nodes or ``xq`` lies outside
    ``[held[0], held[-1]]`` (or is not finite), and the other three are NaN
    there: the boundary is never extrapolated. Inside, the segment is the
    one containing ``xq``; a query exactly on an interior node takes the
    segment above it, which is deterministic and leaves the value
    unchanged (the interpolant is continuous) -- only the slope term of the
    uncertainty depends on the choice. The error is the node errors
    interpolated linearly along the same segment (see
    :meth:`TransitionResult.reconcile` for why that and not a quadrature
    sum).
    """
    n = len(held)
    shape = xq.shape
    value = np.full(shape, np.nan)
    slope = np.full(shape, np.nan)
    error = np.full(shape, np.nan)
    if n < 2:
        return np.zeros(shape, dtype=bool), value, slope, error
    covered = np.isfinite(xq) & (xq >= held[0]) & (xq <= held[-1])
    q = xq[covered]
    k = np.clip(np.searchsorted(held, q, side="right") - 1, 0, n - 2)
    width = held[k + 1] - held[k]
    w = (q - held[k]) / width
    value[covered] = loc[k] + w * (loc[k + 1] - loc[k])
    slope[covered] = (loc[k + 1] - loc[k]) / width
    error[covered] = (1.0 - w) * err[k] + w * err[k + 1]
    return covered, value, slope, error


def _apply_reconcile(view: pd.DataFrame, spec: dict, selects: dict) -> tuple[pd.DataFrame, dict]:
    """Compute the :meth:`TransitionResult.reconcile` columns on ``view``.

    ``view`` is the already-filtered view (thresholds, then selections), and
    nothing is dropped from it. Returns the view with
    :data:`_RECONCILE_COLUMNS` appended and the record for
    ``params['reconcile']``, whose counts are recomputed for this view.
    """
    _check_reconcile_selects(selects)
    tolerances = {"t_tol_k": spec["t_tol_k"], "h_tol_t": spec["h_tol_t"]}
    nodes = {axis: _boundary_nodes(view, axis) for axis in ("T", "H")}

    out = view.copy()
    n_rows = len(out)
    swept = out["swept_axis"].to_numpy()
    residual_k = np.full(n_rows, np.nan)
    residual_t = np.full(n_rows, np.nan)
    sigma = np.full(n_rows, np.nan)
    covered_all = np.zeros(n_rows, dtype=bool)
    concordant = pd.array([pd.NA] * n_rows, dtype="boolean")
    record = dict(tolerances)
    for axis, other in (("T", "H"), ("H", "T")):
        axis_spec = _RECONCILE_AXES[axis]
        held_col, loc_col = _LINK_GEOMETRY[axis]
        rows = np.flatnonzero(swept == axis)
        # A point on this axis sits at its own location along the other
        # family's held coordinate: an H-swept point (T0, Hc) is read off
        # T_b at H = Hc, and compared with its held T0.
        xq = out[loc_col].to_numpy(dtype=float)[rows]
        own = out[held_col].to_numpy(dtype=float)[rows]
        own_err = out[axis_spec["own_err"]].to_numpy(dtype=float)[rows]
        covered, boundary, slope, boundary_err = _interpolate_boundary(xq, *nodes[other][:3])
        resid = own - boundary
        combined = np.hypot(boundary_err, slope * own_err)
        ok = covered & np.isfinite(combined) & (combined > 0)
        z = np.full(len(rows), np.nan)
        np.divide(resid, combined, out=z, where=ok)
        tol = tolerances[axis_spec["tol"]]
        agree = np.abs(resid) <= tol
        (residual_t if axis == "T" else residual_k)[rows] = resid
        sigma[rows] = z
        covered_all[rows] = covered
        for i, row in enumerate(rows):
            if covered[i]:
                concordant[row] = bool(agree[i])
        record[axis] = {
            "residual": axis_spec["residual"],
            "n_points": int(len(rows)),
            "n_covered": int(covered.sum()),
            "n_concordant": int((covered & agree).sum()),
        }
    out["cross_residual_K"] = residual_k
    out["cross_residual_T"] = residual_t
    out["cross_residual_sigma"] = sigma
    out["cross_covered"] = covered_all
    out["cross_concordant"] = concordant
    return out, record


# ---------------------------------------------------------------------------
# Result object.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class TransitionResult:
    """Detected transition candidates plus the evidence and the recipe.

    Attributes
    ----------
    candidates : pandas.DataFrame
        One row per grouped feature (:data:`CANDIDATE_COLUMNS`). The
        current -- possibly :meth:`robust`-filtered and :meth:`select`-ed
        -- view, carrying the ``cross_*`` columns once :meth:`reconcile`
        has run.
    detections : pandas.DataFrame
        One row per raw detector hit (:data:`DETECTION_COLUMNS`), linked
        by ``candidate_id``; this is where "the cusp and the steepest
        slope were 0.3 K apart" stays visible after grouping. Filtered
        alongside ``candidates``: a detection survives iff its candidate
        does.
    params : dict
        Every parameter *as resolved* (methods, criterion priority,
        windows in physical units, noise source, package version, and --
        on a filtered result -- the ``'robust'`` thresholds and the
        ``'select'`` choices applied, and any ``'reconcile'`` tolerances
        and counts). Measurement-level moment searches also record the exact
        resolved moment and paired error columns and their native/canonical
        units in ``per_signal['moment_emu']['moment_source']`` (inside each
        ``per_sweep`` entry for batches). This records detection choices,
        not a complete replay manifest: retain the input files and external
        preprocessing, row selections, repeat aggregation and sample facts.
    skipped : tuple
        Batch-level ``(path, reason)`` pairs for sweeps that were refused
        or failed; empty for single-sweep results. An empty candidates
        table with a populated ``skipped`` is "everything failed", not
        "no transitions". Their ``sweep_id`` values are in
        ``params['skipped_sweep_ids']``, in the same order.
    curves : dict
        One ``{'x', 'y_smooth', 'dydx', 'd2ydx2'}`` mapping per searched
        sweep and signal -- the smoothed representations that produced the
        hits, for auditing a candidate against the curve behind it. Read
        one with :meth:`curve`, which works the same whatever produced the
        result; the keys do not. A batch from
        :func:`detect_transitions_all` keys by ``(sweep_id, signal)``,
        because a path cannot name a sweep: the isotherms
        :func:`~calamari.sweeps.split_isotherms` cuts from one file, and
        ``restrict()`` slices of one file, all share its path, and keyed by
        path each sweep overwrote the one before. A single-sweep
        :func:`detect_transitions` result keys by ``(measurement_file,
        signal)``, and :func:`find_transitions` by ``("", "")``. The
        curvature is always present, whether or not ``'kink'`` was run:
        it is how a caller checks whether a feature reported as a
        derivative extremum is really a slope change. When ``'fisher'``
        ran, the entry also holds ``'yT'`` (the smoothed y*T, in the
        signal's unit times K) and ``'dyT_dT'`` (its derivative, in the
        signal's own unit once the Kelvins cancel).
    chains : pandas.DataFrame or None
        One row per cross-sweep chain (:data:`CHAIN_COLUMNS`) once
        :meth:`link` has run; ``None`` before that. On a filtered result
        it lists only chains with a surviving candidate, keeping the
        statistics as computed at link time -- the chain statistic is what
        justified keeping those rows, so recomputing it from the survivors
        would erase the evidence.
    """

    candidates: pd.DataFrame
    detections: pd.DataFrame
    params: dict
    skipped: tuple = ()
    curves: dict = field(default_factory=dict, repr=False)
    chains: pd.DataFrame | None = None
    # Full (unfiltered) frames, carried by a robust()-filtered or select()-ed
    # result so thresholds can be *loosened*, and a selection changed,
    # without re-running detection. None on an unfiltered result
    # (candidates/detections ARE the full frames there).
    full_candidates: pd.DataFrame | None = field(default=None, repr=False)
    full_detections: pd.DataFrame | None = field(default=None, repr=False)
    # The unrestricted chain table behind ``chains`` on a filtered result,
    # so all() restores every chain and not just the surviving ones.
    full_chains: pd.DataFrame | None = field(default=None, repr=False)

    def __repr__(self) -> str:  # DataFrames make the default repr unusable
        n_c, n_d = len(self.candidates), len(self.detections)
        states = []
        if "robust" in self.params:
            states.append("robust-filtered")
        if self.params.get("select"):
            states.append("selected on " + " and ".join(self.params["select"]))
        if "reconcile" in self.params:
            record = self.params["reconcile"]
            points = sum(record[a]["n_points"] for a in ("T", "H"))
            covered = sum(record[a]["n_covered"] for a in ("T", "H"))
            concordant = sum(record[a]["n_concordant"] for a in ("T", "H"))
            states.append(
                f"reconciled: {covered} of {points} covered, {concordant} concordant"
            )
        filtered = f" ({', '.join(states)})" if states else ""
        skip = f", {len(self.skipped)} skipped" if self.skipped else ""
        return f"TransitionResult({n_c} candidates, {n_d} detections{filtered}{skip})"

    def _full(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.full_candidates is not None:
            return self.full_candidates, self.full_detections
        return self.candidates, self.detections

    def _full_chains(self) -> pd.DataFrame | None:
        return self.full_chains if self.full_chains is not None else self.chains

    def all(self) -> TransitionResult:
        """The unfiltered result this one was filtered from (self if unfiltered).

        Undoes both filters, :meth:`robust` and :meth:`select`, and drops
        their records from ``params``. Chain columns and the chain table
        survive: :meth:`link` is not a filter, so undoing the filters must
        not undo it. :meth:`reconcile` is dropped with them, columns and
        record: it is not a filter either, but it describes the selected
        view, and without the selection there is nothing it could mean.
        """
        if self.full_candidates is None:
            return self
        params = {
            k: v for k, v in self.params.items() if k not in ("robust", "select", "reconcile")
        }
        return TransitionResult(
            candidates=self.full_candidates,
            detections=self.full_detections,
            params=params,
            skipped=self.skipped,
            curves=self.curves,
            chains=self._full_chains(),
        )

    def curve(self, sweep_id: int = 0, signal: str | None = None) -> dict:
        """The smoothed curves one sweep's candidates were found in.

        Why this exists: auditing a candidate means looking at the curve
        behind it -- is a derivative extremum really a slope change, where
        does the smoothing put the peak -- and ``curves`` is keyed
        differently by each producer (see the class docstring). A batch
        cannot key by path, because sweeps cut from one file share it, so
        a lookup that works on one sweep breaks on a family. This accessor
        is the one spelling that works on either, and on every filtered or
        linked view of them: a candidate row's own ``sweep_id`` and
        ``signal`` always address its curve, ``result.curve(row.sweep_id,
        row.signal)``.

        Parameters
        ----------
        sweep_id : int, optional
            The sweep, as in the ``sweep_id`` column. In a batch it is the
            input position given to :func:`detect_transitions_all`. A
            single-sweep result holds one sweep, ``0`` (the default).
        signal : str, optional
            The searched signal (``'chi_real'``, ``'moment_emu'``, ...).
            ``None`` (the default) takes the sweep's only signal. It is
            refused when the sweep has several (``signal='all'``), because
            picking one would be picking χ′ or χ″ for you.

        Returns
        -------
        dict
            A new dict holding ``'x'``, ``'y_smooth'``, ``'dydx'`` and
            ``'d2ydx2'``, plus ``'yT'`` and ``'dyT_dT'`` when ``'fisher'``
            ran. The arrays are the stored ones, not copies.

        Raises
        ------
        CurveInputError
            If ``sweep_id`` is not an integer; if no sweep has that id --
            naming the ids that do, and saying so when the batch skipped
            that sweep, with its reason; if ``signal`` is not among the
            sweep's signals; or if ``signal=None`` and the sweep has more
            than one.
        """
        if isinstance(sweep_id, bool) or not isinstance(sweep_id, (int, np.integer)):
            raise CurveInputError(
                f"curve(): sweep_id must be an integer, got {sweep_id!r}. Pass the "
                "sweep's sweep_id, as in the candidates table's sweep_id column."
            )
        sweep_id = int(sweep_id)
        # A batch is marked by its per-sweep record, not by its keys: a batch
        # whose every sweep was skipped has no curves at all, and must still
        # say "skipped" rather than "single-sweep".
        if "per_sweep" in self.params:
            searched = [entry["sweep_id"] for entry in self.params["per_sweep"]]
            skipped_ids = list(self.params.get("skipped_sweep_ids", []))
            if sweep_id in skipped_ids:
                path, reason = self.skipped[skipped_ids.index(sweep_id)]
                # Most refusals already name the file; say it once.
                why = reason if path in reason else f"{path}: {reason}"
                raise CurveInputError(
                    f"curve(): sweep_id={sweep_id} was skipped by detect_transitions_all, "
                    f"so it has no curves. Searched sweep ids: {searched}. Why it was "
                    f"skipped: {why}"
                )
            if sweep_id not in searched:
                raise CurveInputError(
                    f"curve(): no sweep_id={sweep_id} in this result. It holds curves "
                    f"for sweep ids {searched} (skipped: {skipped_ids}); an id is the "
                    "sweep's position in the input to detect_transitions_all. Pass one "
                    "of the searched ids."
                )
            entries = {sig: c for (sid, sig), c in self.curves.items() if sid == sweep_id}
        else:
            if sweep_id != 0:
                raise CurveInputError(
                    f"curve(): sweep_id={sweep_id}, but this is a single-sweep result, "
                    "whose one sweep has sweep_id 0. Call curve() or curve(0, signal); "
                    "other ids number the sweeps of a detect_transitions_all batch."
                )
            entries = {sig: c for (_, sig), c in self.curves.items()}
        if not entries:
            raise CurveInputError(
                f"curve(): sweep_id={sweep_id} was searched but recorded no curves: "
                "none of its requested signals had data to smooth. Check the "
                "signal= passed to detection against the file's columns."
            )
        available = list(entries)
        if signal is None:
            if len(entries) > 1:
                raise CurveInputError(
                    f"curve(): sweep_id={sweep_id} has curves for several signals "
                    f"{available}, so signal=None is ambiguous. Pass signal= one of "
                    f"them, e.g. curve({sweep_id}, signal={available[0]!r})."
                )
            (signal,) = available
        elif signal not in entries:
            raise CurveInputError(
                f"curve(): sweep_id={sweep_id} has no curve for signal={signal!r}. "
                f"Available: {available}. Pass signal= one of those."
            )
        return dict(entries[signal])

    def robust(
        self,
        min_significance: float = DEFAULT_MIN_SIGNIFICANCE,
        *,
        min_n_methods: int | None = None,
        min_sweep_range_over_noise: float | None = None,
        min_chain_significance: float | None = None,
        min_chain_n: int | None = None,
        exclude_edge: bool = True,
        min_smoothing_n: int | None = None,
        min_smoothing_persistence: float | None = None,
        min_smoothing_sign_consistency: float | None = None,
        max_smoothing_spread_k: float | None = None,
        max_smoothing_spread_t: float | None = None,
        require_smoothing_interval_overlap: bool | None = None,
    ) -> TransitionResult:
        """Filter to the candidates worth publishing; thresholds are recorded.

        Always filters from the FULL underlying detection (not from a
        previous ``robust()`` view), so loosening after tightening works.
        The thresholds land in ``params['robust']`` -- they are provenance
        that belongs in a figure legend or a methods section, and
        :func:`calamari.plot.overlay_transitions` prints them.

        ``min_significance`` defaults to :data:`DEFAULT_MIN_SIGNIFICANCE`
        -- a diagnostic screening convention, see there for what it does and
        does not mean. The other thresholds have no defaults: they are
        physics-interest judgements. ``exclude_edge`` drops candidates
        supported only by a terminal smoothing window, where the
        Savitzky-Golay derivative artefact is largest.

        ``min_chain_significance`` and ``min_chain_n`` gate on the
        cross-sweep chain a candidate belongs to (see :meth:`link`), and
        are the way to keep a boundary that is weak in every sweep but
        present in all of them: threshold the combined statistic, not the
        per-sweep one. They require a linked result -- gating on a chain
        that was never computed would silently keep everything.

        Sensitivity gates require ``scan.reference(window)``. Each is opt-in:
        ``min_smoothing_n`` counts qualified distinct representations;
        ``min_smoothing_persistence`` is their fraction of resolved windows;
        ``min_smoothing_sign_consistency`` is their modal polarity fraction.
        ``max_smoothing_spread_k`` / ``max_smoothing_spread_t`` bound the
        qualified location range on T / H sweeps respectively, in K / Tesla.
        ``require_smoothing_interval_overlap=True`` requires a common intersection
        of all qualified coordinate ± conditional-error intervals. This is a
        sensitivity description, not a joint confidence test. Empty qualified
        sets and unassessed tracks never pass an active sensitivity gate.
        No sensitivity threshold has a universal default. Consult
        :attr:`sensitivity_audit` for rejected candidates and reasons.

        A recorded :meth:`select` is re-applied after the thresholds, to
        the candidates that pass them, so ``robust()`` and ``select()``
        commute: whichever order they are called in, the thresholds decide
        which candidates exist and the selection then picks among those. A
        recorded :meth:`reconcile` is recomputed last, on the new view.

        Raises
        ------
        ValueError
            If a requested threshold names a statistic that is NaN on
            every row -- silently dropping everything (NaN fails any
            comparison) would be indistinguishable from "nothing robust".
            Or if a chain threshold is requested on a result that
            :meth:`link` has not been run on.
        """
        cands, dets = self._full()
        thresholds = {
            "min_significance": min_significance,
            "min_n_methods": min_n_methods,
            "min_sweep_range_over_noise": min_sweep_range_over_noise,
            "min_chain_significance": min_chain_significance,
            "min_chain_n": min_chain_n,
            "exclude_edge": exclude_edge,
            **{key: value for key, value in {
                "min_smoothing_n": min_smoothing_n,
                "min_smoothing_persistence": min_smoothing_persistence,
                "min_smoothing_sign_consistency": min_smoothing_sign_consistency,
                "max_smoothing_spread_k": max_smoothing_spread_k,
                "max_smoothing_spread_t": max_smoothing_spread_t,
                "require_smoothing_interval_overlap": require_smoothing_interval_overlap,
            }.items() if value is not None},
        }
        params = {k: v for k, v in self.params.items() if k != "robust"}
        params["robust"] = thresholds
        return _filtered_result(
            cands, dets, self._full_chains(), params, self.skipped, self.curves
        )

    def link(
        self,
        *,
        t_tol_k: float | None = None,
        h_tol_t: float | None = None,
        by: Sequence[str] = DEFAULT_LINK_BY,
    ) -> TransitionResult:
        """Link candidates into chains across a family of sweeps.

        Why this exists: a weak phase boundary can recur in every sweep of
        a family, at a position that varies smoothly with the fixed
        coordinate, while its per-sweep ``significance`` sits below any
        threshold worth publishing. Noise candidates land at unrelated
        positions instead. Recurrence is the physically meaningful
        robustness signal, and nothing in a per-sweep result records it.

        Each chain carries ``chain_significance`` = ``sum(significance) / sqrt(n)``.
        This Stouffer-like combination is a diagnostic: its non-negative input
        ratios are not centred z-scores. Constant positive inputs grow as sqrt(n),
        including chance chains; neither this score nor chain length establishes
        a real boundary or a false-alarm probability. Record the linking
        tolerances and gates; see :func:`_stouffer`.

        The linking itself is deterministic (TR-1): within each group of
        ``by`` columns plus ``swept_axis``, seeds are taken in descending
        ``significance`` (ties broken by family value then location), and
        each seed walks to the adjacent sweep upward and then downward,
        taking the nearest unclaimed candidate within tolerance and
        stopping at the first sweep that offers none. The ordered set of
        sweeps is the sorted distinct family values among *all* candidates
        on that swept axis -- so a sweep that fired detections but none in
        this group counts as a gap and ends the chain, rather than being
        stepped over silently. Every candidate belongs to exactly one
        chain; a candidate that links to nothing is a chain of one.

        This is a *description*, not a filter: nothing is dropped, and
        gating on the result is :meth:`robust`'s job
        (``min_chain_significance=``, ``min_chain_n=``). Linking always
        works from the full detection and re-applies any ``robust()``
        thresholds and then any :meth:`select` afterwards, so linking
        commutes with both: a selected candidate keeps the chain it has
        when the whole detection is linked. A recorded :meth:`reconcile`
        is recomputed after those.

        Parameters
        ----------
        t_tol_k : float, optional
            How far a T-swept candidate's temperature may move between
            adjacent sweeps and still be the same boundary, in K. Required
            when any candidate was swept along T.
        h_tol_t : float, optional
            The same for H-swept candidates, in Tesla.
        by : sequence of str, optional
            Candidate columns that must match within a chain, on top of
            ``swept_axis``. Default :data:`DEFAULT_LINK_BY`.

        Returns
        -------
        TransitionResult
            A new result (DM-1) whose candidates carry ``chain_id``,
            ``chain_n`` and ``chain_significance``, with the per-chain
            table on ``chains`` (:data:`CHAIN_COLUMNS`) and the settings
            in ``params['link']``.

        Raises
        ------
        ValueError
            If the candidates are not (T, H)-coordinated; if the tolerance
            for a swept axis that is present was not given, or is not
            positive; or if ``by`` names a column the table does not have.
        """
        cands, dets = self._full()
        by = tuple(by)

        # Shape first, then the keywords: an array-level result is missing
        # the coordinates AND most of the default ``by`` columns, and
        # "no signal column" would be a confusing way to be told so.
        required = ("T_K", "H_T", "swept_axis")
        if any(c not in cands.columns for c in required):
            raise DetectionInputError(
                "link(): linking needs (T, H)-coordinated candidates with a "
                f"swept_axis column ({', '.join(required)}), which an array-level "
                "find_transitions result does not carry. Detect with "
                "detect_transitions / detect_transitions_all instead."
            )
        missing = [c for c in by if c not in cands.columns]
        if missing:
            raise DetectionInputError(
                f"link(): by names column(s) {missing!r} that the candidates table "
                f"does not have. Available: {list(cands.columns)!r}."
            )
        for name, value in (("t_tol_k", t_tol_k), ("h_tol_t", h_tol_t)):
            if value is not None and not (float(value) > 0):
                raise DetectionInputError(f"link(): {name} must be positive, got {value!r}.")
        if len(cands) and (cands["T_K"].isna().all() or cands["H_T"].isna().all()):
            raise DetectionInputError(
                "link(): every candidate is missing its T_K or its H_T, so no "
                "family of sweeps can be ordered. Chains are placed in the (T, H) "
                "plane; detect with detect_transitions / detect_transitions_all."
            )
        axes = set(cands["swept_axis"].dropna()) if len(cands) else set()
        for axis, name, value in (("T", "t_tol_k", t_tol_k), ("H", "h_tol_t", h_tol_t)):
            if axis in axes and value is None:
                raise DetectionInputError(
                    f"link(): candidates swept along {axis} are present but {name} "
                    f"is None -- how far the feature may move between adjacent "
                    "sweeps is physics only you can supply, so there is no default."
                )

        chain_ids, chains = _link_candidates(cands, by, t_tol_k, h_tol_t)
        linked = cands.copy()
        linked["chain_id"] = chain_ids
        sizes = chains.set_index("chain_id")["n"] if len(chains) else None
        stats = chains.set_index("chain_id")["chain_significance"] if len(chains) else None
        linked["chain_n"] = (
            linked["chain_id"].map(sizes).astype(int)
            if sizes is not None
            else pd.Series(dtype=int, index=linked.index)
        )
        linked["chain_significance"] = (
            linked["chain_id"].map(stats).astype(float)
            if stats is not None
            else pd.Series(dtype=float, index=linked.index)
        )

        params = {k: v for k, v in self.params.items() if k not in ("robust", "select")}
        params["link"] = {
            "t_tol_k": t_tol_k,
            "h_tol_t": h_tol_t,
            "by": by,
            "statistic": "stouffer",
        }
        # Re-apply the recorded filters, rather than carry their output, so
        # link() commutes with robust() and select(). Re-inserted after
        # 'link' so params reads in the order the steps take effect.
        for key in ("robust", "select"):
            if key in self.params:
                params[key] = self.params[key]
        return _filtered_result(linked, dets, chains, params, self.skipped, self.curves)

    def select(
        self,
        *,
        swept_axis: str,
        method: str,
        sign: str,
        signal: str | None = None,
        t_window_k=None,
        h_window_t=None,
    ) -> TransitionResult:
        """Keep one candidate per sweep: one criterion, inside a physical window.

        Why this exists: a phase boundary is drawn from one point per
        sweep, but a sweep usually yields several candidates -- a stronger
        feature elsewhere in the sweep, a derivative flank, a hump that is
        significant and still not the transition. Which one marks *this*
        boundary is a physics judgement: the criterion you cite (``method``
        and ``sign``) and where you expect the boundary to be (the window).
        ``select`` makes that judgement a recorded call instead of a pandas
        filter in a script, so a figure or a methods section can state it.
        Like :meth:`robust` it is a filter, never applied inside detection
        (TR-1).

        Only candidates swept along ``swept_axis`` are touched; the other
        axis passes through untouched, so a mixed campaign chains::

            r.select(swept_axis="T", method="peak", sign="max",
                     t_window_k=(5, 12)).select(
                swept_axis="H", method="derivative", sign="rise",
                h_window_t=(1, 4))

        On that axis, each sweep (``sweep_id``) keeps at most one
        candidate: among those with the given signal, ``method`` and
        ``sign`` whose location lies inside the window (inclusive), the one
        with the highest ``significance``, ties broken by location
        ascending and then by ``candidate_id`` -- deterministic, whatever
        the row order. Every other candidate on that axis is dropped.
        ``method`` matches the candidate's own ``method`` column, the
        criterion that supplied its coordinate: a cusp whose derivative
        flanks were grouped into it is a ``'peak'`` row, and is not
        offered to ``method='derivative'``.

        What it does not do: judge whether the chosen candidate is real.
        The best candidate inside a window can be noise, and ``select``
        will keep it. Compose with :meth:`robust` for that.

        Composition. ``select`` works on the current view, so after
        :meth:`robust` it picks among the candidates that pass. The choice
        is recorded and re-applied, not carried: :meth:`robust` and
        :meth:`link` both re-apply it after their own work, so the order of
        the three calls does not matter -- thresholds decide which
        candidates exist, then the selection picks among them. Selecting
        the same axis again replaces the earlier choice for that axis
        (working, like ``robust()``, from the unselected view, so a window
        can be widened). :meth:`all` undoes it along with ``robust()``.
        A recorded :meth:`reconcile`, which needs a selection on both axes,
        is recomputed on the new selection.
        ``detections`` keep the rows of surviving candidates, and ``chains``
        the chains with a surviving member, statistics as at link time.

        Parameters
        ----------
        swept_axis : {'T', 'H'}
            Which candidates to select among: those from temperature sweeps
            or those from field sweeps.
        method : str
            The criterion, a :data:`METHODS` key. Required, with no default:
            letting it vary from sweep to sweep would interleave
            systematically different criteria along one boundary -- the
            failure :data:`CRITERION_PRIORITY` exists to prevent.
        sign : str
            The extremum, in that method's vocabulary (peak ``'max'`` /
            ``'min'``, derivative and onset ``'rise'`` / ``'fall'``, kink
            ``'up'`` / ``'down'``, fisher ``'max'`` / ``'min'`` of
            d(yT)/dT). Required for the same reason.
        signal : str, optional
            The channel (``'chi_real'``, ``'chi_imag'``, ``'moment_emu'``).
            ``None`` requires the candidates on this axis to come from
            exactly one signal, and raises naming them otherwise.
        t_window_k : (float, float) or callable, optional
            The window for ``swept_axis='T'``, on ``T_K``, in Kelvin.
        h_window_t : (float, float) or callable, optional
            The window for ``swept_axis='H'``, on ``H_T``, in Tesla -- the
            candidates' unit, not the Oersted of
            :func:`detect_transitions`' ``h_window_oe``, which bounds the
            measurement's own field column. Only the swept axis's window is
            accepted; the other raises (DM-2: a keyword means one quantity).
            A window is an inclusive ``(lo, hi)`` with finite ``lo < hi``,
            or a callable ``f(fixed_value) -> (lo, hi)`` so it can follow a
            boundary that moves across the family: it is called once per
            sweep with the sweep's fixed coordinate, ``H_T`` (Tesla) for a
            T-swept sweep and ``T_K`` (Kelvin) for an H-swept one.
            ``None`` (default) searches the whole sweep.

        Returns
        -------
        TransitionResult
            A new result (DM-1). ``params['select'][swept_axis]`` records
            ``method``, ``sign``, ``signal`` (as resolved), ``window`` (the
            pair, a callable's name, or ``None``), ``windows`` (each sweep's
            resolved window, by ``sweep_id``) and ``empty_sweeps``: the
            ``sweep_id`` of every sweep with candidates on this axis in the
            full detection but none kept by this selection -- nothing in
            its window, or everything there removed by ``robust()``. It is
            never dropped silently. A sweep that produced no candidate on
            this axis at all is not in the record, since there is nothing
            to select from; ``params['per_sweep']`` and
            ``params['skipped_sweep_ids']`` of a batch account for those.
            ``params['select']`` is keyed by axis, so a T selection and an
            H selection are both kept.

        Raises
        ------
        TypeError
            If ``method`` or ``sign`` is not given.
        ValueError
            If the candidates carry no ``swept_axis`` / ``sweep_id`` (an
            array-level result); if ``swept_axis`` is not ``'T'`` or
            ``'H'``, or no candidate was swept along it; if ``method`` or
            ``sign`` is ``None`` or not in the vocabulary; if ``signal`` is
            ambiguous or absent; if the other axis's window is given; or if
            a window, or a callable's answer for some sweep, is malformed.
        """
        cands, dets = self._full()
        required = ("T_K", "H_T", "swept_axis", "sweep_id", "signal", "method", "sign")
        if any(c not in cands.columns for c in required):
            raise DetectionInputError(
                "select(): selecting one candidate per sweep needs (T, H)-coordinated "
                f"candidates with sweep ids ({', '.join(required)}), which an "
                "array-level find_transitions result does not carry. Detect with "
                "detect_transitions / detect_transitions_all instead."
            )
        if swept_axis not in _SELECT_WINDOW:
            raise DetectionInputError(f"select(): swept_axis must be 'T' or 'H', got {swept_axis!r}.")
        if method is None or sign is None:
            raise DetectionInputError(
                "select(): method and sign are required. The criterion that marks "
                "a boundary is your choice to state, and letting it vary from sweep "
                "to sweep would interleave different criteria along one line."
            )
        if method not in METHODS:
            raise DetectionInputError(
                f"select(): unknown method {method!r}; choose from {list(METHODS)!r}."
            )
        vocabulary = _SIGN_VOCABULARY.get(method, ())
        if sign not in vocabulary:
            raise DetectionInputError(
                f"select(): sign {sign!r} is not in the {method!r} vocabulary "
                f"{list(vocabulary)!r} -- each method writes its own signs, so this "
                "selection could never match a row."
            )
        keyword = _SELECT_WINDOW[swept_axis][0]
        window, other = (
            (t_window_k, h_window_t) if swept_axis == "T" else (h_window_t, t_window_k)
        )
        if other is not None:
            other_keyword = _SELECT_WINDOW["H" if swept_axis == "T" else "T"][0]
            raise DetectionInputError(
                f"select(): {other_keyword} was given for swept_axis={swept_axis!r}. "
                f"A {swept_axis}-swept candidate is located along "
                f"{_LINK_GEOMETRY[swept_axis][1]}, so its window is {keyword}; the "
                "other coordinate is the sweep's fixed value, not a place to search."
            )

        on_axis = cands[cands["swept_axis"] == swept_axis]
        if not len(on_axis):
            present = sorted({str(a) for a in cands["swept_axis"].dropna()})
            raise DetectionInputError(
                f"select(): no candidate in this result was swept along {swept_axis} "
                f"(swept axes present: {present!r}), so there is nothing to select. "
                "An empty result with entries in skipped means every sweep failed."
            )
        # Resolved from the FULL detection, never the robust() view, so the
        # resolution cannot change with the thresholds -- that is what lets
        # robust() and select() commute.
        signals = sorted({str(v) for v in on_axis["signal"].dropna()})
        if signal is None:
            if len(signals) != 1:
                raise DetectionInputError(
                    f"select(): the {swept_axis}-swept candidates come from "
                    f"{len(signals)} signals {signals!r}; pass signal= to choose one "
                    "-- a chi' cusp and a chi'' shoulder are different measurements."
                )
            signal = signals[0]
        elif signal not in signals:
            raise DetectionInputError(
                f"select(): no {swept_axis}-swept candidate has signal {signal!r}; "
                f"the signals present are {signals!r}."
            )

        # One window per sweep, resolved over every sweep with candidates on
        # this axis in the full detection (all of a sweep's rows share its
        # fixed coordinate, so the first row's value is the sweep's).
        # A callable is recorded by name (the smooth_window precedent): the
        # object itself is not a printable recipe, and ``windows`` holds what
        # it answered for every sweep.
        if window is None:
            window_record = None
        elif callable(window):
            window_record = _describe_callable(window)
        else:
            window = window_record = _check_select_window(window, f"{keyword} is")
        fixed_col = _SELECT_WINDOW[swept_axis][2]
        firsts = on_axis.sort_values(["sweep_id", "candidate_id"]).drop_duplicates("sweep_id")
        windows = {
            int(sid): _resolve_select_window(window, swept_axis, int(sid), float(fixed))
            for sid, fixed in zip(firsts["sweep_id"], firsts[fixed_col], strict=True)
        }

        params = dict(self.params)
        selects = dict(self.params.get("select", {}))
        selects[swept_axis] = {
            "method": method,
            "sign": sign,
            "signal": signal,
            "window": window_record,
            "windows": windows,
        }
        params["select"] = selects
        return _filtered_result(
            cands, dets, self._full_chains(), params, self.skipped, self.curves
        )

    def reconcile(self, *, t_tol_k: float, h_tol_t: float) -> TransitionResult:
        """Say how far each family's points sit from the other family's boundary.

        Why this exists: a campaign usually traces one phase boundary two
        ways -- temperature scans at held fields give ``T_b(H)``, isotherms
        at held temperatures give ``H_b(T)`` -- and nothing per sweep says
        whether the two agree. They need not: the criteria differ (a cusp in
        M(T) against a step in M(H)), demagnetisation and thermal lag act
        along different axes, and a family can pick up a feature that is not
        the boundary at all. ``reconcile`` measures the disagreement, point
        by point, in both units and in units of the combined uncertainty.

        This is a *description*, not a filter, in the same sense as
        :meth:`link`: it adds columns and a record and drops nothing.
        Whether a disagreement matters is your call, stated as the two
        tolerances; ``cross_concordant`` applies them, and nothing acts on
        it.

        Both families must already be down to one point per sweep: it
        requires a :meth:`select` on both axes. The boundaries are then
        drawn from the current view -- after :meth:`robust` and both
        selections -- as piecewise-linear interpolants:

        * the T-swept points ``(T_N, H_0)``, sorted by held field, give
          ``T_b(H)``;
        * the H-swept points ``(T_0, H_c)``, sorted by held temperature,
          give ``H_b(T)``.

        Each H-swept point gets ``cross_residual_K = T_0 - T_b(H_c)``, and
        each T-swept point ``cross_residual_T = H_0 - H_b(T_N)``: positive
        means the point lies at higher temperature, or higher field, than
        the other family's boundary there. Each residual is NaN on the
        other axis's rows.

        **Uncertainty.** Take an H-swept point (the T-swept case is the
        mirror image, with T and H exchanged). To first order, with the
        sweeps independent, the residual ``r = T_0 - T_b(H_c)`` has variance::

            sigma_r**2 = sigma_Tb(H_c)**2 + (dT_b/dH)**2 * sigma_Hc**2 + sigma_T0**2

        * ``sigma_Tb(H_c)`` is the T family's error at the point: the two
          bracketing nodes' ``T_err_K`` interpolated linearly, with the same
          weights as the boundary itself. That is exact at a node and, in
          between, larger than the quadrature sum
          ``sqrt((1-w)^2 s_k^2 + w^2 s_{k+1}^2)`` that independent node
          errors would give (by up to sqrt(2) at mid-segment). The larger
          one is used on purpose: the chord between two sweeps also carries
          the boundary's curvature, which no node error describes, so the
          quadrature sum would be the optimistic choice.
        * ``dT_b/dH`` is the slope of the segment containing ``H_c`` (on an
          interior node, the segment above it) and ``sigma_Hc`` is the
          point's own ``H_err_T``. Its location error moves it along the
          boundary, which shifts ``T_b`` by the slope times that error.
        * ``sigma_T0``, the error of the held coordinate, is taken as zero.
          The row does carry ``T_span_K``, the range of the held
          temperature over the sweep, but a range is not a standard error.
          Turning it into one needs an assumption about where in the sweep
          the crossing happened (uniformly anywhere gives ``span/sqrt(12)``)
          that the detection has no evidence for, and every other consumer
          of the held coordinate -- :meth:`link`, :meth:`select`, the maps
          -- treats it as exact. The span stays on the row to compare with
          the residual by eye; it is bounded by the scan-classification
          tolerances (``SCAN_T_TOL_K``, ``SCAN_H_TOL_OE``) whenever
          :func:`detect_transitions` accepted the sweep as held.

        ``cross_residual_sigma`` is the residual divided by ``sigma_r``.
        Where ``sigma_r`` is zero or not finite it is NaN, never infinite.
        The location errors it rests on are floored at the grid spacing and
        somewhat optimistic (see :func:`find_transitions`), so read it as a
        scale, not a calibrated z-score.

        **Coverage.** A point is covered only if the other family has at
        least two points and spans it: an H-swept point's ``H_c`` must lie
        within the T family's range of held fields (inclusive), and a
        T-swept point's ``T_N`` within the H family's range of held
        temperatures. An uncovered point has ``cross_covered = False`` and
        NaN residuals and sigma. The boundary is never extrapolated: beyond
        the last sweep there is no measurement of it to disagree with.

        **Composition.** The columns depend on the selected view, so the
        call is recorded and recomputed, not carried. :meth:`robust`,
        :meth:`select` and :meth:`link` all rebuild from the full detection
        and re-apply a recorded reconcile last, after their own work. A
        looser threshold that restores a point therefore also restores its
        residual, and moves the boundaries it takes part in. :meth:`all`
        drops it, because it undoes the selections it depends on.

        Parameters
        ----------
        t_tol_k : float
            Largest ``|cross_residual_K|`` still called concordant, in K.
            Required, positive and finite, with no default: how far apart
            two criteria may sit and still mark one boundary is physics only
            you can supply, as for :meth:`link`.
        h_tol_t : float
            The same for ``|cross_residual_T|``, in Tesla.

        Returns
        -------
        TransitionResult
            A new result (DM-1), with no candidate dropped, whose candidates
            carry five more columns:

            * ``cross_residual_K`` (float, K) on H-swept rows, NaN elsewhere;
            * ``cross_residual_T`` (float, T) on T-swept rows, NaN elsewhere;
            * ``cross_residual_sigma`` (float), the row's residual over
              ``sigma_r``;
            * ``cross_covered`` (bool);
            * ``cross_concordant`` (pandas nullable ``boolean``):
              ``|residual| <= tolerance`` on covered rows and ``<NA>`` on
              uncovered ones, where agreement is undefined rather than
              false.

            ``params['reconcile']`` records ``t_tol_k`` and ``h_tol_t``
            and, per swept axis (``'T'``, ``'H'``), the residual column
            and ``n_points``, ``n_covered`` and ``n_concordant`` in this
            view.

        Raises
        ------
        ValueError
            If ``params['select']`` does not hold both ``'T'`` and ``'H'``;
            if a tolerance is ``None``, not a number, or not positive and
            finite; or if two selections on one axis share a held value
            within that axis's scan tolerance (``SCAN_H_TOL_OE`` for the
            field of T-scans, ``SCAN_T_TOL_K`` for the temperature of
            isotherms), listing their ``sweep_id`` values -- a boundary
            through two locations at one held value would be a guess.
        """
        tolerances = {
            "t_tol_k": _check_reconcile_tolerance("t_tol_k", t_tol_k),
            "h_tol_t": _check_reconcile_tolerance("h_tol_t", h_tol_t),
        }
        _check_reconcile_selects(self.params.get("select"))
        cands, dets = self._full()
        params = dict(self.params)
        params["reconcile"] = tolerances
        return _filtered_result(
            cands, dets, self._full_chains(), params, self.skipped, self.curves
        )

    @property
    def sensitivity_audit(self) -> pd.DataFrame:
        """Full reference candidates, assessment, sensitivity verdict and current inclusion.

        Recomputed from the full detection, so rejected and unassessed rows remain
        exportable with ``to_csv`` or ``to_json`` after any filtering or selection.
        ``sensitivity_pass`` is nullable when no gate applies or assessment is
        unavailable; ``retained`` describes the final view including other gates
        and selection. The complete recipe is ``params['smoothing']`` plus
        ``params['robust']``.
        """
        cands, _ = self._full()
        if "smoothing" not in self.params:
            raise SmoothingSensitivityError(
                "sensitivity_audit needs scan.reference(window) first."
            )
        audit = cands.copy()
        thresholds = self.params.get("robust", {})
        reasons = _sensitivity_reasons(cands, thresholds)
        audit["sensitivity_reasons"] = reasons
        verdicts = []
        for i, row in cands.iterrows():
            active = any(
                thresholds.get(k) is not None
                and not (direction == "bool" and not thresholds[k])
                and (axis is None or row.swept_axis == axis)
                for k, (_, direction, axis) in _SENSITIVITY_GATES.items()
            )
            verdicts.append(
                reasons.loc[i] == "" if active and row.smoothing_status == "assessed" else None
            )
        audit["sensitivity_pass"] = pd.array(verdicts, dtype="boolean")
        audit["retained"] = audit.candidate_id.isin(self.candidates.candidate_id)
        return audit

    def provenance(self) -> pd.DataFrame:
        """Candidates joined with each method's name and citations.

        The bibliography lives once, in :data:`METHODS`; this is the
        convenience join for a report or a paper's methods section.
        """
        cands = self.candidates.copy()
        cands["method_name"] = cands["method"].map(
            lambda k: METHODS[k].name if k in METHODS else k
        )
        cands["citations"] = cands["method"].map(
            lambda k: "; ".join(c.text for c in METHODS[k].citations) if k in METHODS else ""
        )
        return cands


def _filtered_result(
    cands: pd.DataFrame,
    dets: pd.DataFrame,
    chains: pd.DataFrame | None,
    params: dict,
    skipped: tuple,
    curves: dict,
) -> TransitionResult:
    """Build the view ``params`` records from the full frames.

    The one place the filters compose, in one fixed order whichever method
    asked: the ``robust()`` thresholds first, then each recorded
    ``select()``. Every filtering method records its settings and calls
    this, rather than filtering the view it was handed, which is what makes
    ``robust()``, ``select()`` and ``link()`` commute. Detections follow
    their candidates; chains keep only those with a surviving member, with
    their link-time statistics. A recorded ``reconcile()`` comes after
    both filters: it drops nothing, but its columns describe the selected
    view, so it is recomputed on that view every time.
    """
    if "robust" not in params and not params.get("select"):
        return TransitionResult(
            candidates=cands,
            detections=dets,
            params=params,
            skipped=skipped,
            curves=curves,
            chains=chains,
        )
    view = cands
    if "robust" in params:
        view = view[_robust_mask(view, params["robust"])]
    if params.get("select"):
        records = {}
        for axis, spec in params["select"].items():
            view, records[axis] = _apply_select(view, axis, spec)
        params = {**params, "select": records}
    if "reconcile" in params:
        # Last, because its columns describe the selected view: recomputed
        # here on every rebuild, never carried, and moved to the end of
        # params so the record reads in the order the steps take effect.
        view, record = _apply_reconcile(view, params["reconcile"], params.get("select"))
        params = {k: v for k, v in params.items() if k != "reconcile"}
        params["reconcile"] = record

    kept = view.reset_index(drop=True)
    kept_ids = set(kept["candidate_id"]) if len(kept) else set()
    kept_dets = dets[dets["candidate_id"].isin(kept_ids)].reset_index(drop=True)
    kept_chains = chains
    if chains is not None:
        surviving = set(kept["chain_id"]) if "chain_id" in kept.columns else set()
        kept_chains = chains[chains["chain_id"].isin(surviving)].reset_index(drop=True)
    return TransitionResult(
        candidates=kept,
        detections=kept_dets,
        params=params,
        skipped=skipped,
        curves=curves,
        chains=kept_chains,
        full_candidates=cands,
        full_detections=dets,
        full_chains=chains,
    )
