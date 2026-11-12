"""Repeat-measurement grouping and aggregation.

:func:`combine` and :func:`group_consecutive` operate on arrays;
:func:`repeat_groups` and :func:`aggregate_repeats` operate on measurements.
Measurements report repeat counts, but aggregation requires an explicit call because
it changes observations and their uncertainties.

Grouping
--------
Group consecutive observations within each measurement family, preserving separate
sweep branches. Families come from :func:`calamari.measurement.row_families`, which
uses populated channels when transport metadata is absent. Ambiguous rows are refused
when several families are present.

Each family independently resolves its grouping mechanism (instrument counter or
setpoint tolerance), tolerance scales and separation check. Every proposed group must
agree in setpoint coordinates or raise
:class:`~calamari.exceptions.RepeatGroupError`. Tolerance grouping also requires
separated within-group and between-group steps; see :func:`_check_separation`.

Uncertainties
-------------
The default combines repeats by inverse-variance weighting with Birge-scaled error,
``sigma_int * max(1, sqrt(chi2/nu))``. ``error='internal'`` and ``error='external'``
are explicit alternatives. The default has upward bias for small groups; see
:func:`combine` for its assumptions and the two-repeat case.

For repeats at identical x with fixed standard errors, the original weighted
least-squares cost and that of the combined mean with internal propagated error differ
only by a parameter-independent term. Parameter estimates and Hessians therefore agree.
Setpoint jitter, averaging x or changing the error model can break this equivalence.
Record aggregation and weighting choices; they also affect detection and presentation.

Grouping and combination are deterministic. Recover constituent rows by rerunning
:func:`repeat_groups` on :attr:`AggregatedMeasurement.source` with its recorded settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Not used by this module's own code: AggregatedMeasurement inherits
# ``path: Path`` from Measurement, and the docs build resolves that string
# annotation in *this* module's namespace, so Path must exist here at
# runtime.
from pathlib import Path  # noqa: F401
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from calamari.exceptions import (
    InputTypeError,
    MeasurementFamilyError,
    RepeatGroupError,
    RepeatWeightError,
)
from calamari.measurement import (
    _AC_DRIVE_BASES,
    _AC_FREQUENCY_BASES,
    _CHI_IMAG_ERROR_BASES,
    _CHI_REAL_ERROR_BASES,
    _FIELD_BASES,
    _MOMENT_ERROR_BASES,
    _TEMPERATURE_BASES,
    FAMILY_ORDER,
    HOUSEKEEPING,
    Measurement,
    _split_unit_suffix,
    row_families,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

__all__ = [
    "CombinedPoint",
    "AggregatedMeasurement",
    "combine",
    "group_consecutive",
    "repeat_groups",
    "aggregate_repeats",
    "REPEAT_T_TOL_K",
    "REPEAT_H_TOL_OE",
    "REPEAT_DRIVE_RTOL",
    "REPEAT_FREQUENCY_RTOL",
    "MAX_REPEATS",
]

# ---------------------------------------------------------------------------
# Tolerances.
#
# These are NOT calamari.measurement's SCAN_T_TOL_K / SCAN_H_TOL_OE, and must
# not be unified with them. Those answer "is this a sweep?" and are
# deliberately loose (0.5 K, 10 Oe); distinct repeated setpoints can be closer
# than these thresholds, so reusing them could merge the sweep into one group.
# These answer the much tighter question "is this the same setpoint measured
# again?".
# ---------------------------------------------------------------------------

#: Absolute temperature span (K) allowed within a proposed repeat group.
#: This empirical working tolerance accommodates settling jitter; it is not
#: an instrument accuracy specification. Counter groups still require setpoint
#: validation, and tolerance-based groups require inter-group separation.
REPEAT_T_TOL_K = 0.025

#: Absolute field span (Oe) allowed within a proposed repeat group.
#: At high fields the same absolute tolerance is a smaller relative allowance.
#: Inspect ungrouped repeats and supply an explicit override when justified;
#: this working default is not a universal stability or accuracy guarantee.
REPEAT_H_TOL_OE = 1.0

#: AC drive amplitude tolerance, **relative**. It gets its own tolerance rather
#: than sharing REPEAT_H_TOL_OE because a drive amplitude is routinely ~1 Oe,
#: where 1 Oe absolute is a 100% tolerance.
#:
#: Different drives can probe different physical responses; group only when
#: drive magnitudes agree within this recorded relative tolerance.
REPEAT_DRIVE_RTOL = 1e-3

#: AC drive frequency tolerance, relative. Same reasoning as the drive: a
#: frequency is a magnitude, not an offset, so a relative tolerance is the
#: meaningful one. Grouping across frequencies would average two different
#: measurements -- this is the failure a `1,1,2,2` counter produces.
REPEAT_FREQUENCY_RTOL = 1e-3

#: Sanity bound on how many rows one repeat group may hold. This is not a
#: statement about how many repeats an experiment may take; it is a guard on
#: the tolerance being wrong. `sweeps.segment_by_constant` documents that
#: consecutive-difference chaining *deliberately* tolerates unbounded
#: cumulative drift -- right for finding plateaus, wrong here, because a slow
#: measure-while-sweeping ramp whose consecutive steps fall under tolerance
#: would otherwise chain into one group and average an entire sweep to a
#: point. Raise it explicitly if a run really does repeat more than this.
MAX_REPEATS = 32

#: Minimum ratio of between-group spacing to within-group span.
#: A uniform ramp has ratio 1. The empirical guard 10 requires a clear
#: separation of proposed repeat clusters; it is not a calibrated confidence.
_MIN_SEPARATION_RATIO = 10.0

#: MPMS3's per-row repetition counter; absent or unusable in some exports.
_MEASUREMENT_NUMBER_BASES = ("Measurement Number",)

#: `calamari.measurement.row_families`' label for a row it cannot place: one
#: populating two families' columns at once, or none at all. Spelled out here
#: because that module states it as a literal rather than a named constant; if it
#: ever gains one, import it instead of restating it.
_UNPLACED = "unknown"

#: Housekeeping rows -- touchdowns and locate scans -- are carried through
#: untouched as singleton groups rather than dropped. They are recognised through
#: `calamari.measurement.row_families`, which labels them `HOUSEKEEPING`, rather
#: than through a Transport Action code table restated here: the module that
#: decides which rows are measurements and the module that decides which rows may
#: be grouped then cannot drift apart, and the one classifier answers both
#: questions for every dialect at once.
#:
#: Summary exports can contain trailing locate scans. Keep them out of repeat
#: groups; DC measurement rows, including transport code 6, remain eligible.

# ---------------------------------------------------------------------------
# Which columns hold a quantity, and which column holds its error.
#
# The first four tables are imported from calamari.measurement so the
# vocabulary has one home. The fifth covers quantities real files carry that
# no Measurement accessor returns -- leaving `AC Susceptibility` at "first
# value of the group" while combining `AC X'` beside it would look exactly
# like a bug. The aliases below accept alternate punctuation for the same
# error quantities.
# ---------------------------------------------------------------------------

_EXTRA_ERROR_BASES = {
    "AC Moment": "AC M. Std Err.",
    "AC Susceptibility": "AC Suscept. Std Err.",
}

#: Alternate error-column spellings, supported by documentary rather than
#: observed-export evidence.
_EXTRA_ERROR_ALIASES = {
    "AC Moment": "AC M. Std. Err.",
    "AC Susceptibility": "AC Suscept. Std. Err.",
}

#: Columns holding a setpoint: averaged unweighted across a group.
_SETPOINT_BASES = (
    *_TEMPERATURE_BASES,
    *_FIELD_BASES,
    *_AC_FREQUENCY_BASES,
    *_AC_DRIVE_BASES,
)

#: Extremes over the instrument's own measurement of one point.
#: Combining a group means widening them, not averaging them.
_MIN_BASES = ("Min. Temperature", "Min. Field")
_MAX_BASES = ("Max. Temperature", "Max. Field")

#: Suffixes for the provenance columns aggregation adds. Square brackets are
#: not part of any Quantum Design column spelling, so a `Map NN` column
#: renamed by an operator is very unlikely to collide -- and if one does, the
#: collision raises rather than silently overwriting the operator's data.
_N_SUFFIX = "[n]"
_BIRGE_SUFFIX = "[Birge]"


@dataclass(frozen=True)
class CombinedPoint:
    """One repeat group reduced to a single value, with its diagnostics.

    Attributes
    ----------
    value : float
        The combined value: an inverse-variance weighted mean where usable
        errors were supplied, an unweighted mean otherwise. NaN for an empty
        group.
    error : float
        The uncertainty to report, chosen by :func:`combine`'s ``error``
        argument.
    n : int
        How many points were actually combined -- the number of **finite**
        values in the group, which is not the group's row count when a column
        is blank on some rows. In an interleaved AC/DC file that is the normal
        case, not an error.
    internal : float
        ``1 / sqrt(sum(1/sigma_i**2))`` -- what the instrument's own per-point
        errors imply. NaN when no usable errors were supplied.
    external : float
        The uncertainty implied by the observed scatter. Equal to
        ``sigma_int * S``, which for equal input errors reduces exactly to the
        familiar ``s / sqrt(n)``. NaN for ``n < 2``.
    birge : float
        The scale factor ``S = sqrt(chi2/nu)``, ``nu = n - 1``: how far the
        repeats disagree relative to how far their error bars say they should.
        1 means "exactly as expected". NaN for ``n < 2`` or when unweighted.
    weighted : bool
        Whether inverse-variance weighting was actually used.
    """

    value: float
    error: float
    n: int
    internal: float
    external: float
    birge: float
    weighted: bool


def combine(
    values: ArrayLike,
    errors: ArrayLike | None = None,
    *,
    error: str = "birge",
) -> CombinedPoint:
    """Combine repeat measurements of one quantity into a single point.

    Array-first and unit-agnostic: ``values`` and ``errors`` are whatever unit
    you supplied, and the result is in that same unit, so this is usable well
    outside magnetometry. The parameters therefore carry no unit suffix, which
    would be an active lie for every other caller.

    Parameters
    ----------
    values : array_like
        The repeated measurements. Non-finite entries are dropped, and the
        count that survives is reported as :attr:`CombinedPoint.n` -- a blank
        row is not a measurement, and in an interleaved AC/DC file most
        quantities are blank on most rows.
    errors : array_like, optional
        Per-point standard errors aligned to ``values``. When omitted, or when
        every error in the group is zero or missing, the mean is unweighted
        and the uncertainty comes from the observed scatter.
    error : {'birge', 'internal', 'external'}, optional
        Which uncertainty to report on :attr:`CombinedPoint.error`:

        ``'birge'`` (default)
            ``sigma_int * max(1, S)``, the Particle Data Group convention.
            When the repeats agree no better than chance you get the
            propagated error; when they disagree by more than their error bars
            allow, the error is inflated to match the disagreement rather than
            reporting a precision the data does not support.
        ``'internal'``
            The propagated error alone, ``sigma_int``. This is the choice that
            leaves a downstream weighted fit numerically identical to fitting
            the unaggregated rows.
        ``'external'``
            The observed scatter alone. Honest about disagreement but noisy,
            and exactly 0 when two repeats happen to land on the same value.
            This is also the escape hatch for a group whose errors cannot be
            turned into weights: asking for the scatter is asking for no
            weighting, so it combines such a group unweighted instead of
            raising :class:`~calamari.exceptions.RepeatWeightError`.

        An unweighted group reports the external error whatever is asked,
        since there is no propagated error to report.

    Returns
    -------
    CombinedPoint

    Raises
    ------
    RepeatWeightError
        If the group mixes zero (or missing) errors with real ones, **and**
        ``error`` is not ``'external'``. These instruments write a **zero**
        error to mean "only one scan was averaged", so the two are different
        conventions rather than different numbers, and weighting cannot span
        them -- see the exception. ``error='external'`` does not need weights
        and so is answerable rather than refused.
    ValueError
        If ``errors`` has a different shape from ``values``, or ``error`` is
        not one of the three names above.

    Notes
    -----
    **The n = 2 caveat.** Two repeats give ``nu = 1``, so ``S = |z|`` and the
    scale factor is itself as uncertain as the measurement. Because
    ``max(1, S)`` is one-sided it inflates on average even when nothing is
    wrong: ``E[max(1, |z|)] = 1.167`` for independent Gaussian errors with
    correctly specified standard deviations and a common underlying value.
    This is a deliberate upward scaling bias; ``error='internal'`` omits it
    under those assumptions. Two observations alone cannot identify a bad
    member or distinguish drift from random scatter.

    Examples
    --------
    >>> from calamari.repeats import combine
    >>> point = combine([1.00, 1.02], [0.01, 0.01])
    >>> round(point.value, 4)
    1.01
    >>> round(point.birge, 4)   # 2 sigma apart, so the error is inflated
    1.4142
    >>> round(point.internal, 6), round(point.error, 6)
    (0.007071, 0.01)
    """
    if error not in ("birge", "internal", "external"):
        raise RepeatWeightError(
            f"error must be 'birge', 'internal' or 'external', got {error!r}"
        )

    values = np.asarray(values, dtype=float)
    if errors is None:
        errs = None
    else:
        errs = np.asarray(errors, dtype=float)
        if errs.shape != values.shape:
            raise RepeatWeightError(
                f"values and errors must have the same shape, got "
                f"{values.shape} vs {errs.shape}"
            )

    keep = np.isfinite(values)
    v = values[keep]
    e = None if errs is None else errs[keep]
    n = int(v.shape[0])

    if n == 0:
        return CombinedPoint(np.nan, np.nan, 0, np.nan, np.nan, np.nan, False)
    if n == 1:
        single = float(e[0]) if e is not None else np.nan
        return CombinedPoint(float(v[0]), single, 1, single, np.nan, np.nan, False)

    weighted = e is not None
    if weighted:
        # A zero or missing error cannot become a weight. All of them means
        # the file simply records no scatter estimate for this group; some of
        # them means two conventions in one column, which is not ours to
        # reconcile.
        unusable = ~np.isfinite(e) | (e == 0.0)
        if unusable.all():
            weighted = False
        elif unusable.any() and error == "external":
            # The documented escape hatch, and it has to be honoured HERE
            # rather than below: `error='external'` asks for the observed
            # scatter and no weighting at all, which is exactly what a group
            # with unusable weights can still deliver. Raising anyway would
            # make the message on the exception below name a resolution that
            # does not work.
            weighted = False
        elif unusable.any():
            raise RepeatWeightError(
                f"a repeat group of {n} points mixes unusable standard errors "
                f"({int(unusable.sum())} that are zero or missing) with "
                f"{int((~unusable).sum())} real ones. A zero error means 'one "
                "scan averaged', not 'exact', so inverse-variance weighting "
                "cannot span the two -- a zero is an infinite weight and would "
                "silently discard every other point in the group. Either drop "
                "the offending rows with Measurement.restrict(mask=...), or "
                "pass error='external' to weight nothing and use the observed "
                f"scatter. Values: {v.tolist()}; errors: {e.tolist()}."
            )

    if weighted:
        w = 1.0 / e**2
        w_sum = w.sum()
        mean = float((v * w).sum() / w_sum)
        internal = float(1.0 / np.sqrt(w_sum))
        chi2 = float((((v - mean) / e) ** 2).sum())
        birge = float(np.sqrt(chi2 / (n - 1)))
        external = internal * birge
        if error == "internal":
            reported = internal
        elif error == "external":
            reported = external
        else:
            reported = internal * max(1.0, birge)
    else:
        mean = float(v.mean())
        internal = np.nan
        birge = np.nan
        external = float(v.std(ddof=1) / np.sqrt(n))
        reported = external

    return CombinedPoint(mean, reported, n, internal, external, birge, weighted)


def _span_ok(
    keys: Sequence[np.ndarray], tols: Sequence[float], rows: Sequence[int]
) -> tuple[bool, str | None]:
    """Whether ``rows`` form one setpoint; if not, which key says otherwise.

    A key that is non-finite for *every* row in the candidate group is
    ignored: an all-blank column carries no information about whether these
    rows are the same setpoint, and in an interleaved file the AC columns are
    blank on every VSM row. A key that is finite for only *some* of them fails
    -- a row whose setpoint is unknown is never merged with one whose is not.
    """
    for i, (key, tol) in enumerate(zip(keys, tols, strict=True)):
        vals = key[list(rows)]
        finite = np.isfinite(vals)
        if not finite.any():
            continue
        if not finite.all():
            return False, f"key {i} (blank on some rows)"
        span = float(vals.max() - vals.min())
        if span > tol:
            return False, f"key {i} (span {span:.6g} > tol {tol:.6g})"
    return True, None


def group_consecutive(
    keys: Sequence[ArrayLike],
    tols: Sequence[float],
    *,
    partition: ArrayLike | None = None,
    max_repeats: int = MAX_REPEATS,
) -> list[np.ndarray]:
    """Group consecutive rows that share every key to within tolerance.

    The generic primitive under :func:`repeat_groups`, operating on plain
    arrays. Unlike :func:`calamari.sweeps.segment_by_constant`, which starts a
    new segment on a large consecutive *difference*, this bounds the **total
    span** of each group. The difference matters: consecutive-difference
    chaining tolerates unbounded cumulative drift by design -- correct for
    finding a plateau, and catastrophic here, where it would fold a slow
    measure-while-sweeping ramp into a single "repeat group" and average a
    whole sweep to a point.

    Parameters
    ----------
    keys : sequence of array_like
        The setpoint columns, all the same length. A group is extended only
        while *every* key stays within its tolerance.
    tols : sequence of float
        One absolute tolerance per key, in that key's own units.
    partition : array_like, optional
        Row labels that may never be grouped together. Rows of one label need
        not be adjacent in the input: "consecutive" means adjacent *within a
        partition*, because in a real interleaved VSM/AC file no two rows of
        the same kind are ever file-adjacent, and requiring file-adjacency
        would make grouping such a file impossible.
    max_repeats : int, optional
        Raise rather than build a group larger than this. Default
        :data:`MAX_REPEATS`.

    Returns
    -------
    list of numpy.ndarray
        One integer index array per group, in file order (sorted by first
        row), together covering every row exactly once. Index arrays -- not
        slices -- so a caller can build a row mask directly for
        :meth:`~calamari.measurement.Measurement.restrict`.

    Raises
    ------
    RepeatGroupError
        If a group would exceed ``max_repeats``, which means the tolerances do
        not match the file.
    ValueError
        If ``keys`` and ``tols`` differ in length, or the arrays differ in
        length from one another or from ``partition``.
    """
    if len(keys) != len(tols):
        raise RepeatGroupError(
            f"keys and tols must have the same length, got "
            f"{len(keys)} vs {len(tols)}"
        )
    arrays = [np.asarray(k, dtype=float) for k in keys]
    if not arrays:
        raise RepeatGroupError("group_consecutive needs at least one key")
    n_rows = arrays[0].shape[0]
    for i, arr in enumerate(arrays):
        if arr.shape[0] != n_rows:
            raise RepeatGroupError(
                f"every key must have the same length; key 0 has {n_rows} "
                f"rows but key {i} has {arr.shape[0]}"
            )
    if max_repeats < 1:
        raise RepeatGroupError(f"max_repeats must be >= 1, got {max_repeats}")

    if partition is None:
        order_sets = [np.arange(n_rows)]
    else:
        labels = np.asarray(partition)
        if labels.shape[0] != n_rows:
            raise RepeatGroupError(
                f"partition has {labels.shape[0]} rows but the keys have "
                f"{n_rows}"
            )
        # NaN never equals itself, so a blank label would scatter its rows
        # into singleton partitions; treat all blanks as one label instead.
        blank = pd.isna(labels)
        uniques = pd.unique(labels[~blank])
        order_sets = [np.flatnonzero(~blank & (labels == u)) for u in uniques]
        if blank.any():
            order_sets.append(np.flatnonzero(blank))

    groups: list[np.ndarray] = []
    for indices in order_sets:
        current: list[int] = []
        for row in indices:
            if not current:
                current = [int(row)]
                continue
            ok, _ = _span_ok(arrays, tols, [*current, int(row)])
            if ok:
                if len(current) + 1 > max_repeats:
                    raise RepeatGroupError(
                        f"rows {current[0]}-{row} would form a repeat group of "
                        f"{len(current) + 1} rows, over max_repeats="
                        f"{max_repeats}. A group that large almost always "
                        "means the tolerances are too loose for this file "
                        "rather than that the run repeated that many times -- "
                        "consecutive rows chain, so a slow ramp whose steps "
                        "each fall under tolerance collapses into one group. "
                        "Tighten the tolerances, or raise max_repeats "
                        "deliberately if the run really does repeat this often."
                    )
                current.append(int(row))
            else:
                groups.append(np.array(current, dtype=int))
                current = [int(row)]
        if current:
            groups.append(np.array(current, dtype=int))

    groups.sort(key=lambda g: int(g[0]))
    return groups


def _check_separation(
    groups: Sequence[np.ndarray],
    keys: Sequence[np.ndarray],
    tols: Sequence[float],
    names: Sequence[str],
    measurement: Measurement,
    family: str | None = None,
) -> None:
    """Refuse a tolerance grouping with no separation of scales.

    Bounding each group's span stops a slow ramp chaining into *one* group,
    but not from being chopped into fixed-width chunks and averaged: at a
    0.002 K step under a 0.025 K tolerance, a 100-point ramp still becomes
    a handful of tolerance-width "repeat groups" of ~13 rows each, silently
    turning a sweep into a coarser sweep. Nothing about a group's span alone
    can tell the two cases apart.

    What tells them apart is separation of scales, and the honest way to
    measure it is on the *consecutive differences* themselves. A file of
    repeats has two clearly separated populations of them -- tiny inside a
    repeat, large between setpoints -- while a ramp has exactly one. So:

    - ``W`` -- the largest setpoint step *inside* any group,
    - ``B`` -- the smallest setpoint step *between* adjacent groups,

    both in units of each key's own tolerance, taking the largest such change
    over the keys. The grouping is believable only when ``B >= R * W``.

    Use one scalar comparison across keys: a held axis cannot veto separation,
    and a single jump on another axis cannot validate a ramp. A uniform ramp
    has ``B == W`` and fails the required factor of ten.

    Compare adjacent groups within one family, including interleaved rows.
    ``groups`` must contain that family's groups only, so another family's
    ramp cannot veto its repeats.

    Applied only to tolerance grouping. When the instrument's own counter
    proposed the groups it has told us directly that those rows are repeats,
    which is evidence this test does not have and must not overrule.
    """
    if len(groups) < 2:
        return

    def step(i: int, j: int) -> tuple[float, str | None]:
        """Largest change between two rows, in units of each key's tolerance.

        Normalising by the tolerance is what lets one number stand for every
        setpoint at once; the cost is that the verdict depends on the ratio of
        the tolerances to each other, so a file whose swept axis steps only a
        few tolerance-units while another wanders can raise. That case is
        already marginal, and it fails loudly with both numbers named.
        """
        worst, which = 0.0, None
        for key, tol, name in zip(keys, tols, names, strict=True):
            if tol <= 0:
                continue
            a, b = key[i], key[j]
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            change = abs(float(b) - float(a)) / tol
            if change > worst:
                worst, which = change, name
        return worst, which

    within, within_axis = 0.0, None
    for group in groups:
        for a, b in zip(group[:-1], group[1:]):
            change, axis = step(int(a), int(b))
            if change > within:
                within, within_axis = change, axis

    between, between_axis = np.inf, None
    for x, y in zip(range(len(groups) - 1), range(1, len(groups))):
        change, axis = step(int(groups[x][-1]), int(groups[y][0]))
        if change < between:
            between, between_axis = change, axis

    if not np.isfinite(between):
        # No two groups to compare -- nothing to judge.
        return
    if between >= _MIN_SEPARATION_RATIO * within:
        # Note this passes when `within` is 0, i.e. every group is recorded
        # bit-identically. That is the tidiest file there is, not a suspicious
        # one: infinitely good separation.
        return

    where = f" among the {family} rows" if family else ""
    raise RepeatGroupError(
        f"{measurement.path}: grouping on setpoint tolerance produced repeat "
        f"groups{where} no better separated than the rows inside them. The largest "
        f"setpoint step within a group is {within:.4g} tolerance-units"
        f"{f' (in {within_axis})' if within_axis else ''}, while the smallest "
        f"step between adjacent groups is {between:.4g}"
        f"{f' (in {between_axis})' if between_axis else ''} -- a ratio of "
        f"{between / within if within else np.inf:.2f}, against the "
        f"{_MIN_SEPARATION_RATIO:g} required. Repeats of one setpoint sit far "
        "closer to each other than to the next setpoint; when they do not, "
        "the data is a continuum and any grouping of it is arbitrary. This is "
        "what a measure-while-sweeping ramp looks like, and averaging it "
        "would quietly turn a sweep into a coarser sweep. Tighten the "
        "tolerance for the swept key if the file really does hold repeats, or "
        "do not aggregate this file."
        + (
            f" Only the {family} rows are judged here, so if another family in "
            "this file does hold repeats, m.of_type(...) will aggregate it."
            if family
            else ""
        )
    )


def _counter_groups(counter: np.ndarray, indices: np.ndarray) -> list[list[int]]:
    """Propose groups from a repetition counter over one partition.

    A new group starts wherever the counter fails to increase. This is only
    ever a *proposal*: the caller verifies every group against the setpoints.
    """
    proposed: list[list[int]] = []
    current: list[int] = []
    previous = None
    for row in indices:
        value = counter[row]
        if current and not (np.isfinite(value) and previous is not None
                            and value > previous):
            proposed.append(current)
            current = []
        current.append(int(row))
        previous = value if np.isfinite(value) else None
    if current:
        proposed.append(current)
    return proposed


def _resolve_plain(measurement: Measurement, bases: Sequence[str]) -> str | None:
    """Name of the first column matching ``bases`` exactly, or None.

    For bookkeeping columns that carry no unit suffix and need no conversion,
    so `_resolve_column`'s unit machinery would be beside the point.
    """
    wanted = [b.lower() for b in bases]
    for column in measurement.data.columns:
        if _split_unit_suffix(str(column))[0].lower() in wanted:
            return str(column)
    return None


def _setpoint_keys(
    measurement: Measurement,
    rows: np.ndarray,
    *,
    t_tol_K: float,
    h_tol_Oe: float,
    drive_rtol: float,
    frequency_rtol: float,
) -> tuple[list[np.ndarray], list[float], list[str]]:
    """Build the setpoint arrays, their absolute tolerances, and their names.

    Values come from the unit-bearing accessors, so a file recorded in Tesla
    or A m^2 is compared on the same footing as one in Oe or emu -- the
    tolerances above are stated in canonical units and would otherwise mean
    different things for different files.

    The frequency and drive tolerances are *relative* and are turned into
    absolute ones here, against the median magnitude of the column.

    Parameters
    ----------
    rows : numpy.ndarray
        Which rows the *tolerances* are scaled to -- one measurement family's,
        since a relative tolerance taken over a whole combined file would be a
        median of two populations at once. The returned keys are still
        **full-length**, indexed by the file's own row numbers, because that is
        what :func:`_verify` and :func:`_span_ok` are handed. A column that is
        blank on every one of ``rows`` drops out entirely, which is how the AC
        frequency stops being a key for the VSM family of a combined run.
    """
    keys: list[np.ndarray] = []
    tols: list[float] = []
    names: list[str] = []

    def add_absolute(series, tol, name):
        if series is None:
            return
        keys.append(np.asarray(series, dtype=float))
        tols.append(tol)
        names.append(name)

    def add_relative(series, rtol, name):
        if series is None:
            return
        values = np.asarray(series, dtype=float)
        here = values[rows]
        finite = here[np.isfinite(here)]
        if finite.size == 0:
            return
        scale = float(np.median(np.abs(finite)))
        keys.append(values)
        tols.append(rtol * scale)
        names.append(name)

    add_absolute(measurement.temperature, t_tol_K, "temperature (K)")
    add_absolute(measurement.field_oe, h_tol_Oe, "field (Oe)")
    add_relative(measurement.ac_frequency, frequency_rtol, "AC frequency (Hz)")
    add_relative(measurement.ac_drive_oe, drive_rtol, "AC drive (Oe)")
    return keys, tols, names


def _families(measurement: Measurement) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    """Label every row with its family, and split the measured rows by it.

    Returns the label array (aligned to ``data``) and one ``(family, rows)``
    pair per population to plan independently, in :data:`FAMILY_ORDER`.

    Two rules earn their place here:

    **A row that cannot be placed is refused only when there is something to
    confuse it with.** ``row_families`` says ``'unknown'`` both for a row that
    populates two families' columns at once and for one that populates none --
    and the second is ordinary. A single-family file whose moment happens to be
    blank on a row would otherwise stop aggregating over a row that says
    nothing, so where only one family is present the unplaceable rows join it,
    which is exactly the behaviour that file had before families existed. Where
    two or more are present, such a row is the one that could be fused with a
    measurement of the other kind, and calamari cannot tell which it belongs
    to, so it says so.

    **Housekeeping comes from the same classifier** as the families rather than
    from a second reading of ``Transport Action``, so "which rows are
    measurements" and "which rows may be grouped together" cannot drift apart.
    """
    try:
        labels = row_families(measurement.data, path=measurement.path).to_numpy()
    except MeasurementFamilyError as exc:
        raise MeasurementFamilyError(
            f"{exc}\n\nRepeat grouping asks this of every file, because rows of "
            "different measurement families must never share a group -- so a "
            "file whose families cannot be told apart cannot be aggregated as "
            "it stands. Isolate the rows you mean with "
            "Measurement.restrict(mask=...) and aggregate that."
        ) from exc

    live = np.flatnonzero(labels != HOUSEKEEPING)
    present = [f for f in FAMILY_ORDER if np.any(labels[live] == f)]
    unplaceable = np.flatnonzero(labels[live] == _UNPLACED)

    if len(present) <= 1:
        # One population, so nothing can be fused with anything: plan the file
        # as a whole exactly as it was planned before families were considered.
        if not live.size:
            return labels, []
        return labels, [(present[0] if present else _UNPLACED, live)]

    if unplaceable.size:
        row = int(live[unplaceable[0]])
        raise MeasurementFamilyError(
            f"{measurement.path}: row {row} cannot be assigned to a measurement "
            f"family, and this file holds {', '.join(present)}, so the question "
            f"cannot be left open ({unplaceable.size} row(s) are like this). "
            "Such a row populates two families' columns at once, or none of "
            "them; either way calamari cannot tell whose repeats it belongs "
            "with, and grouping it by setpoint alone would combine two "
            "different measurements into one point. Take the family you want "
            "with m.of_type(...) and aggregate that, or select the rows "
            "yourself with m.restrict(mask=...)."
        )

    return labels, [(f, live[labels[live] == f]) for f in present]


def _plan(
    measurement: Measurement,
    *,
    t_tol_K: float = REPEAT_T_TOL_K,
    h_tol_Oe: float = REPEAT_H_TOL_OE,
    drive_rtol: float = REPEAT_DRIVE_RTOL,
    frequency_rtol: float = REPEAT_FREQUENCY_RTOL,
    max_repeats: int = MAX_REPEATS,
    use_counter: bool = True,
) -> tuple[list[np.ndarray], str, dict[str, str]]:
    """Group the rows and report which mechanism did it, per family.

    Each measurement family is planned on its own -- its own tolerance scales,
    its own choice between the counter and the setpoints, its own separation
    check -- and the groups are merged back into one file-ordered list at the
    end. See this module's docstring for why every one of those has to be per
    family rather than per file.
    """
    if len(measurement.data) == 0:
        return [], "empty", {}

    labels, populations = _families(measurement)

    # Housekeeping rows are not measurements, so they are never grouped -- but
    # they are carried through as singletons so aggregation preserves them.
    groups = [
        np.array([int(i)], dtype=int) for i in np.flatnonzero(labels == HOUSEKEEPING)
    ]

    counter_column = _resolve_plain(measurement, _MEASUREMENT_NUMBER_BASES)
    counter_values = (
        None
        if counter_column is None
        else measurement.data[counter_column].to_numpy(dtype=float)
    )

    mechanisms: dict[str, str] = {}
    for family, rows in populations:
        keys, tols, names = _setpoint_keys(
            measurement,
            rows,
            t_tol_K=t_tol_K,
            h_tol_Oe=h_tol_Oe,
            drive_rtol=drive_rtol,
            frequency_rtol=frequency_rtol,
        )
        if not keys:
            raise RepeatGroupError(
                f"{measurement.path}: no setpoint column to group by. Repeat "
                "aggregation needs at least one of temperature, field, AC "
                "frequency or AC drive amplitude to decide which rows are "
                "measurements of the same thing; without one, any grouping would "
                "be a guess. Columns present: "
                f"{', '.join(repr(str(c)) for c in measurement.data.columns)}."
            )

        # Usable *for this family*. A counter populated on the AC rows and blank
        # on the VSM ones is evidence about AC repeats and no evidence at all
        # about VSM ones -- and on the counter path a blank breaks the group at
        # every row, so judging it file-wide returned the VSM repeats as
        # singletons.
        counter = None
        if use_counter and counter_values is not None:
            here = counter_values[rows]
            if np.unique(here[np.isfinite(here)]).size > 1:
                counter = counter_values

        if counter is not None:
            for proposed in _counter_groups(counter, rows):
                _verify(
                    proposed, keys, tols, names, measurement, max_repeats,
                    counter_column,
                )
                groups.append(np.array(proposed, dtype=int))
            mechanisms[family] = "measurement-number"
        else:
            found = [
                rows[g]
                for g in group_consecutive(
                    [k[rows] for k in keys], tols, max_repeats=max_repeats
                )
            ]
            _check_separation(
                found, keys, tols, names, measurement,
                family if len(populations) > 1 else None,
            )
            groups.extend(found)
            mechanisms[family] = "setpoint-tolerance"

    groups.sort(key=lambda g: int(g[0]))
    return groups, _compose_mechanism(mechanisms), mechanisms


def _compose_mechanism(mechanisms: dict[str, str]) -> str:
    """Join the per-family mechanisms into one label.

    Follows :attr:`~calamari.measurement.Measurement.measurement_type`: a single
    name while every family agrees, and a composite in
    :data:`~calamari.measurement.FAMILY_ORDER` when they differ. The structured
    form is on :attr:`AggregatedMeasurement.grouped_by_family`, and is what to
    branch on; parsing this string is not intended.
    """
    if not mechanisms:
        return "empty"
    distinct = set(mechanisms.values())
    if len(distinct) == 1:
        return next(iter(distinct))
    ordered = [f for f in FAMILY_ORDER if f in mechanisms]
    ordered += [f for f in mechanisms if f not in ordered]
    return "+".join(f"{f}:{mechanisms[f]}" for f in ordered)


def _verify(
    rows: Sequence[int],
    keys: Sequence[np.ndarray],
    tols: Sequence[float],
    names: Sequence[str],
    measurement: Measurement,
    max_repeats: int,
    counter_column: str,
) -> None:
    """Check a counter-proposed group really is one setpoint.

    The counter is a proposal, never a verdict. This is what stops a counter
    that runs `1,1,2,2` from pairing repeat 1 of one setpoint with repeat 2 of
    the previous, and a counter that is really a global measurement index from
    proposing a single group containing the whole file.
    """
    if len(rows) > max_repeats:
        raise RepeatGroupError(
            f"{measurement.path}: {counter_column!r} proposes a repeat group "
            f"of {len(rows)} rows ({rows[0]}-{rows[-1]}), over max_repeats="
            f"{max_repeats}. A counter that never resets -- a global "
            "measurement index rather than a per-setpoint repetition number -- "
            "produces exactly this. Pass use_counter=False to group on the "
            "setpoints instead."
        )
    ok, why = _span_ok(keys, tols, rows)
    if ok:
        return
    index = int(why.split()[1]) if why and why.startswith("key") else 0
    name = names[index] if index < len(names) else "a setpoint"
    detail = why.split("(", 1)[1].rstrip(")") if "(" in (why or "") else why
    raise RepeatGroupError(
        f"{measurement.path}: {counter_column!r} proposes rows "
        f"{list(rows)} as one repeat group, but they are not one setpoint -- "
        f"{name} {detail}. The counter is a proposal, not a verdict: one that "
        "runs 1,1,2,2 rather than 1,2,1,2 pairs repeat 1 of one setpoint with "
        "repeat 2 of the previous, which would average two different "
        "measurements (in a multi-frequency AC file, two different "
        "frequencies). Pass use_counter=False to group on the setpoints "
        "instead, or widen the tolerance if this span really is one setpoint."
    )


def repeat_groups(
    measurement: Measurement,
    *,
    t_tol_K: float = REPEAT_T_TOL_K,
    h_tol_Oe: float = REPEAT_H_TOL_OE,
    drive_rtol: float = REPEAT_DRIVE_RTOL,
    frequency_rtol: float = REPEAT_FREQUENCY_RTOL,
    max_repeats: int = MAX_REPEATS,
    use_counter: bool = True,
) -> list[np.ndarray]:
    """Find the repeat groups in a measurement, without combining anything.

    The "look before you leap" entry point, and the way to recover an
    :class:`AggregatedMeasurement`'s constituent rows: grouping is
    deterministic, so re-running this on ``.source`` with ``.params``
    reproduces exactly the groups that were combined.

    Parameters
    ----------
    measurement : Measurement
        The file to inspect.
    t_tol_K, h_tol_Oe : float, optional
        Absolute setpoint tolerances, defaulting to :data:`REPEAT_T_TOL_K` and
        :data:`REPEAT_H_TOL_OE`. Deliberately much tighter than
        ``SCAN_T_TOL_K`` / ``SCAN_H_TOL_OE``, which answer a different
        question -- see this module's constants.
    drive_rtol, frequency_rtol : float, optional
        *Relative* tolerances for the AC drive amplitude and frequency, which
        are magnitudes rather than offsets. The drive one is load-bearing for
        response consistency -- see :data:`REPEAT_DRIVE_RTOL`.
    max_repeats : int, optional
        Sanity bound on group size; see :data:`MAX_REPEATS`.
    use_counter : bool, optional
        Whether to let the instrument's ``Measurement Number`` column propose
        the groups (default True). Set False to group purely on setpoints --
        the escape hatch when a file's counter turns out to mean something
        else.

    Returns
    -------
    list of numpy.ndarray
        One integer index array per group, in file order, covering every row
        exactly once. A file with no repeats returns one singleton group per
        row, which is the honest answer rather than an empty list.

    Raises
    ------
    RepeatGroupError
        If a proposed group is not one setpoint, if a group exceeds
        ``max_repeats``, or if the file has no setpoint column at all.

    Examples
    --------
    >>> import numpy as np
    >>> from calamari import load, simulate
    >>> from calamari.repeats import repeat_groups
    >>> path = simulate.write_mpms3_file(          # doctest: +SKIP
    ...     "repeats.dat", app="SQUID AC",
    ...     temperature_K=np.repeat([2.0, 3.0], 2),
    ...     field_oe=np.zeros(4), chi_real=[1.0, 1.1, 2.0, 2.2],
    ...     extra_columns={"Measurement Number": [1, 2, 1, 2]},
    ... )
    >>> [g.tolist() for g in repeat_groups(load(path))]   # doctest: +SKIP
    [[0, 1], [2, 3]]
    """
    groups, _, _ = _plan(
        measurement,
        t_tol_K=t_tol_K,
        h_tol_Oe=h_tol_Oe,
        drive_rtol=drive_rtol,
        frequency_rtol=frequency_rtol,
        max_repeats=max_repeats,
        use_counter=use_counter,
    )
    return groups


def _error_pairs(columns: Sequence[str]) -> dict[str, str | None]:
    """Map each value column present to its error column, if the file has one.

    Pairing follows the same explicit quantity/error tables used by the
    Measurement accessors, preserving moment and susceptibility units.
    """
    tables = [
        _MOMENT_ERROR_BASES,
        _CHI_REAL_ERROR_BASES,
        _CHI_IMAG_ERROR_BASES,
        _EXTRA_ERROR_BASES,
        _EXTRA_ERROR_ALIASES,
    ]
    by_base: dict[str, str] = {}
    for name in columns:
        by_base.setdefault(_split_unit_suffix(str(name))[0].lower(), str(name))

    pairs: dict[str, str | None] = {}
    for table in tables:
        for value_base, error_base in table.items():
            value_column = by_base.get(value_base.lower())
            if value_column is None:
                continue
            error_column = by_base.get(error_base.lower())
            # An alias table must not overwrite a pairing the observed table
            # already made, but it may fill one in.
            if pairs.get(value_column) is None:
                pairs[value_column] = error_column
    return pairs


def _reduce_present(values: ArrayLike, reduce) -> float:
    """``reduce`` over what a group actually recorded, or NaN if it recorded
    nothing.

    The reductions this wraps are the NaN-aware ones, and every one of them
    both warns and returns NaN when handed a block that is blank throughout --
    ``RuntimeWarning: Mean of empty slice`` from :func:`numpy.nanmean`,
    ``All-NaN slice encountered`` from :func:`numpy.nanmin` and
    :func:`numpy.nanmax`.

    In a combined run that block is not an edge case, it is the normal case for
    half the file. A real MPMS3 DC+AC export leaves ``AC Drive`` and
    ``AC Frequency`` empty on every DC row, so each of its DC groups reduced two
    all-blank setpoint columns and a single aggregation emitted one warning per
    column per group -- 274 of them, on a file whose every value came out right.
    Warnings that fire in the ordinary case are worse than useless: they train
    the reader to ignore the ones that mean something.

    NaN is the correct answer here and is returned directly rather than
    computed and warned about. This is not the silent NaN ``AGENTS.md`` argues
    against: nothing failed and nothing was guessed at. The instrument recorded
    no AC drive for a DC measurement, the source rows are blank, and the
    aggregate is blank in exactly the same way -- inventing a number would be
    the defect. Which rows a value came from is on the ``[n]`` column beside it.
    """
    values = np.asarray(values, dtype=float)
    if np.isnan(values).all():
        return float("nan")
    return float(reduce(values))


def aggregate_repeats(
    measurement: Measurement,
    *,
    error: str = "birge",
    t_tol_K: float = REPEAT_T_TOL_K,
    h_tol_Oe: float = REPEAT_H_TOL_OE,
    drive_rtol: float = REPEAT_DRIVE_RTOL,
    frequency_rtol: float = REPEAT_FREQUENCY_RTOL,
    max_repeats: int = MAX_REPEATS,
    use_counter: bool = True,
) -> AggregatedMeasurement:
    """Collapse each repeat group into a single row.

    Returns an :class:`AggregatedMeasurement` -- a real
    :class:`~calamari.measurement.Measurement` subclass, so every accessor,
    plot helper, fit and transition detector takes it unchanged.

    Parameters
    ----------
    measurement : Measurement
        The file to aggregate. Aggregating an already-aggregated measurement
        raises: after one pass the ``Measurement Number`` column holds
        first-values, so a second pass would silently fall through to the
        tolerance path.
    error : {'birge', 'internal', 'external'}, optional
        Which uncertainty to write into each error column; see :func:`combine`,
        including the ``n = 2`` caveat on the default.
    t_tol_K, h_tol_Oe, drive_rtol, frequency_rtol, max_repeats, use_counter
        Passed to :func:`repeat_groups`.

    Returns
    -------
    AggregatedMeasurement

    Raises
    ------
    RepeatGroupError
        From :func:`repeat_groups`, or if a provenance column name collides
        with a column the file already has.
    RepeatWeightError
        From :func:`combine`.
    TypeError
        If ``measurement`` is already an :class:`AggregatedMeasurement`.

    Notes
    -----
    What happens to each column:

    - A **quantity with a paired error column** (DC moment, chi'/chi'', the AC
      moment components, AC moment/susceptibility amplitude) is combined by
      :func:`combine`, and both its value and its error column are rewritten.
      A quantity with no error column in the file is averaged unweighted.
    - **Setpoint columns** (temperature, field, AC frequency, AC drive) get the
      unweighted group mean.
    - ``Min.``/``Max.`` **columns** get the min of mins and the max of maxes,
      which is what those columns mean.
    - **Everything else** -- ``Time Stamp``, ``Range``, the ``* Status (code)``
      columns, ``Measure Count``, ``Transport Action``, ``Measurement Number``,
      ``Map 01``..``Map 16`` -- keeps the **first value** of its group. Notably
      that includes ``AC Phase``, which is an angle on a -90..270 range where a
      naive mean is wrong near the wrap; averaging it would be worse than not.

    Two provenance columns are added per combined quantity, ``X [n]`` and
    ``X [Birge]``. They are per quantity rather than per group on purpose: a
    group's row count is not the number of points that went into a given
    column, and in an interleaved AC/DC file that is the normal case rather
    than an anomaly.
    """
    if isinstance(measurement, AggregatedMeasurement):
        raise InputTypeError(
            "this Measurement is already aggregated. Aggregating twice is "
            "refused rather than silently permitted: after the first pass the "
            "'Measurement Number' column holds each group's first value, so a "
            "second pass would quietly fall through to setpoint-tolerance "
            "grouping and combine already-combined points. Aggregate the "
            "original instead -- it is available as .source."
        )

    groups, mechanism, by_family = _plan(
        measurement,
        t_tol_K=t_tol_K,
        h_tol_Oe=h_tol_Oe,
        drive_rtol=drive_rtol,
        frequency_rtol=frequency_rtol,
        max_repeats=max_repeats,
        use_counter=use_counter,
    )

    data = measurement.data
    pairs = _error_pairs([str(c) for c in data.columns])
    error_columns = {c for c in pairs.values() if c is not None}

    setpoint = {b.lower() for b in _SETPOINT_BASES}
    minima = {b.lower() for b in _MIN_BASES}
    maxima = {b.lower() for b in _MAX_BASES}

    out: dict[str, list] = {str(c): [] for c in data.columns}
    added: dict[str, list[float]] = {}
    for value_column in pairs:
        base = _split_unit_suffix(value_column)[0]
        for suffix in (_N_SUFFIX, _BIRGE_SUFFIX):
            name = f"{base} {suffix}"
            if name in out:
                raise RepeatGroupError(
                    f"{measurement.path}: cannot add the provenance column "
                    f"{name!r} because the file already has a column of that "
                    "name. Aggregation would overwrite it. Rename the file's "
                    "column (it is most likely an operator-titled 'Map NN') "
                    "before aggregating."
                )
            added[name] = []

    for rows in groups:
        for column in data.columns:
            name = str(column)
            base = _split_unit_suffix(name)[0].lower()
            values = data[column].to_numpy()
            block = values[rows]

            if name in error_columns:
                continue  # written alongside its value column
            if name in pairs:
                errs = None
                error_column = pairs[name]
                if error_column is not None:
                    errs = data[error_column].to_numpy(dtype=float)[rows]
                point = combine(
                    np.asarray(block, dtype=float), errs, error=error
                )
                out[name].append(point.value)
                if error_column is not None:
                    out[error_column].append(point.error)
                stem = _split_unit_suffix(name)[0]
                added[f"{stem} {_N_SUFFIX}"].append(point.n)
                added[f"{stem} {_BIRGE_SUFFIX}"].append(point.birge)
            elif base in setpoint:
                out[name].append(_reduce_present(block, np.nanmean))
            elif base in minima:
                out[name].append(_reduce_present(block, np.nanmin))
            elif base in maxima:
                out[name].append(_reduce_present(block, np.nanmax))
            else:
                out[name].append(block[0])

    frame = pd.DataFrame({**out, **added})
    params = {
        "t_tol_K": t_tol_K,
        "h_tol_Oe": h_tol_Oe,
        "drive_rtol": drive_rtol,
        "frequency_rtol": frequency_rtol,
        "max_repeats": max_repeats,
        "use_counter": use_counter,
    }
    return AggregatedMeasurement(
        path=measurement.path,
        data_format=measurement.data_format,
        measurement_type=measurement.measurement_type,
        file_role=measurement.file_role,
        title=measurement.title,
        header=dict(measurement.header),
        data=frame,
        moment_base=measurement.moment_base,
        source=measurement,
        grouped_by=mechanism,
        grouped_by_family=by_family,
        error_rule=error,
        params=params,
    )


@dataclass(repr=False, eq=False)
class AggregatedMeasurement(Measurement):
    """A :class:`~calamari.measurement.Measurement` whose repeats are combined.

    Produced by :func:`aggregate_repeats`, never constructed directly. It is a
    real ``Measurement`` subclass, and deliberately so: because aggregation
    writes the combined values back under the file's **own column spellings**,
    every accessor -- ``moment_emu``, ``moment_err_emu``, ``chi_real``,
    ``field_oe``, the whole unit-resolution machinery -- is inherited and
    simply works, and every consumer in :mod:`calamari.plot`,
    :mod:`calamari.fitting`, :mod:`calamari.transitions` and
    :mod:`calamari.sweeps` takes one unchanged. Not one accessor is
    overridden.

    ``restrict`` and ``with_moment`` return an ``AggregatedMeasurement`` too,
    with the provenance intact, because they derive via
    :func:`dataclasses.replace`, which constructs ``type(self)``. The
    per-quantity ``[n]`` and ``[Birge]`` diagnostics are ordinary columns
    rather than parallel arrays for the same reason: ``restrict`` subsets them
    correctly with no code of its own.

    Attributes
    ----------
    data : pandas.DataFrame
        **Not** the raw table the base class documents. This is the one
        derived case in calamari: the value and error columns hold *combined*
        numbers under the file's original column names, so
        ``data['Moment (emu)']`` is no longer what the instrument wrote. The
        unaggregated table is on :attr:`source`. Two provenance columns per
        combined quantity, ``X [n]`` and ``X [Birge]``, are added.
    source : Measurement
        The measurement this was aggregated from. Constituent rows are
        recoverable by re-running
        :func:`~calamari.repeats.repeat_groups` on it with :attr:`params`,
        since grouping is deterministic -- which is why no row-membership
        bookkeeping is stored here.

        Note this is the *original*, always: after ``restrict()`` it still
        refers to the unfiltered measurement.
    grouped_by : str
        Which mechanism grouped the rows: ``'measurement-number'``,
        ``'setpoint-tolerance'``, or ``'empty'`` when there was nothing to group
        -- a file with no rows, or one holding only housekeeping rows, which are
        carried through as themselves either way.

        Mechanisms are chosen **per family**, so a combined run whose AC rows
        carry a usable ``Measurement Number`` and whose VSM rows do not reports
        the composite ``'VSM:setpoint-tolerance+AC:measurement-number'``, in
        :data:`~calamari.measurement.FAMILY_ORDER`. This attribute follows
        :attr:`~calamari.measurement.Measurement.measurement_type` exactly:
        one plain name while the families agree -- which is every single-family
        file -- and a composite only when they genuinely differ. Branch on
        :attr:`grouped_by_family` rather than parsing it.
    grouped_by_family : dict
        The same answer keyed by measurement family,
        ``{'VSM': 'setpoint-tolerance', 'AC': 'measurement-number'}``. This is
        the structured form of :attr:`grouped_by`, and the one to read: which
        mechanism grouped a given family decides how much its ``[n]`` column
        can be trusted, and a file can now answer differently for each.

        Empty when :attr:`grouped_by` is ``'empty'``. A file calamari could
        place in no family at all -- one with no moment or susceptibility column
        populated anywhere -- keys its single population as ``'unknown'``.
    error_rule : str
        Which uncertainty was written: ``'birge'``, ``'internal'`` or
        ``'external'``.
    params : dict
        The tolerances and options actually used, not merely the mechanism's
        name -- without them, "re-running the grouping reproduces the groups"
        would only hold for default calls.

    Notes
    -----
    The error columns carry **two conventions at once** after aggregation. A
    group of one keeps whatever the instrument wrote, including its
    "0 means a single scan, not exact" zero (see
    :attr:`~calamari.measurement.Measurement.moment_err_emu`); a group of two
    or more holds a combined, possibly Birge-inflated error. The ``[n]``
    column is what tells the two apart, and is the thing to check before using
    an error column as a fit weight.
    """

    source: Measurement | None = None
    grouped_by: str = ""
    grouped_by_family: dict = field(default_factory=dict)
    error_rule: str = ""
    params: dict = field(default_factory=dict)

    def _repeat_counts(self) -> np.ndarray:
        """Every ``[n]`` column's values, flattened; empty if there are none."""
        columns = [c for c in self.data.columns if str(c).endswith(_N_SUFFIX)]
        if not columns:
            return np.empty(0, dtype=float)
        return self.data[columns].to_numpy(dtype=float).ravel()

    def _repr_bits(self) -> list[str]:
        """Append the reduction to the base class's own repr fragments.

        The median is computed from the ``[n]`` columns rather than stored,
        so it stays true after ``restrict()`` rather than reporting a stored
        reduction that the surviving rows no longer show.
        """
        bits = super()._repr_bits()
        counts = self._repeat_counts()
        counts = counts[np.isfinite(counts) & (counts > 0)]
        if counts.size:
            bits.append(f"combined n={int(np.median(counts))}")
        bits.append(self.error_rule or "?")
        return bits

    def summary(self) -> dict:
        """The base summary plus how the aggregation was done.

        Adds ``grouped_by``, ``grouped_by_family``, ``error_rule``, ``median_n``
        and ``max_birge``.
        ``n_points`` means what it says -- rows in *this* object, i.e. groups
        -- so it is no longer the file's row count; that is ``len(source.data)``.
        """
        summary = super().summary()
        counts = self._repeat_counts()
        counts = counts[np.isfinite(counts) & (counts > 0)]
        birge_columns = [
            c for c in self.data.columns if str(c).endswith(_BIRGE_SUFFIX)
        ]
        birge = (
            self.data[birge_columns].to_numpy(dtype=float).ravel()
            if birge_columns
            else np.empty(0)
        )
        birge = birge[np.isfinite(birge)]
        summary.update(
            {
                "grouped_by": self.grouped_by,
                "grouped_by_family": dict(self.grouped_by_family),
                "error_rule": self.error_rule,
                "median_n": float(np.median(counts)) if counts.size else None,
                "max_birge": float(birge.max()) if birge.size else None,
            }
        )
        return summary
