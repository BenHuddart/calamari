"""Per-point quality screening: which points of a file deserve a second look.

:meth:`Measurement.quality() <calamari.measurement.Measurement.quality>`
returns a :class:`QualityReport` -- one row per measured point, stating what
the file offers about that point's trustworthiness, and flagging the points
whose value breaks the continuity of its neighbours.

**One gate, and evidence beside it.** A point is flagged on its continuity
score alone (:func:`continuity_scores`): its departure from a straight line
through its neighbours, scaled by how far those neighbours stray from that
line themselves.
The threshold is an empirical diagnostic choice, not a probability or a
calibrated test of instrument quality. The remaining fields provide context:
relative uncertainty, within-point temperature/field spread, fit quality,
status departures and range changes. None alone proves a point defective;
nonzero statuses and range changes can accompany ordinary measurements or
physical features.

**Nothing is dropped.** The report describes; a caller decides. Three
consumers take that decision through an ``outliers=`` keyword, all warning
by default and all using the same score: the map helpers
(:func:`calamari.plot.chi_map_grid`), which can exclude flagged points from
a map; the family plots (:func:`calamari.plot.plot_family` and its
aliases), which can exclude them from each curve -- where a waterfall's
``baseline=`` would otherwise let one glitch move every point of its curve;
and :func:`calamari.transitions.detect_transitions`, which can exclude them
from the search -- a glitch on a smooth sweep is otherwise a sharp extremum,
exactly what a peak finder reports. The family plots and detection share
one contract, whose checks live at the end of this module: the same modes,
the same threshold, and a supplied report refused unless it describes the
measurement it is applied to. These deliberate shared policy/report refusals use
:class:`~calamari.exceptions.QualityInputError` (also ``ValueError`` and
:class:`~calamari.exceptions.CalamariError`).

This module imports no plotting library (PK-1): :mod:`calamari.plot` calls
:func:`continuity_scores` for its map glitch check, and the shared
``outliers=`` checks for its family plots, not the other way round.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from calamari._sweep_rules import frequency_clusters
from calamari.exceptions import QualityInputError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from calamari.measurement import Measurement

__all__ = [
    "CONTINUITY_THRESHOLD",
    "CONTINUITY_WINDOW",
    "QUALITY_COLUMNS",
    "QualityReport",
    "assess",
    "continuity_scores",
]

#: Neighbours on each side a point is compared against (so up to four
#: neighbours, two on each side). Two is the smallest window whose
#: leave-one-out median survives one bad neighbour; wider windows start to
#: smooth over the sharp cusps the check must not flag.
CONTINUITY_WINDOW = 2

#: A point is flagged when its departure from a straight line through its
#: neighbours (fitted against the swept quantity) exceeds this multiple of
#: the larger of (a) the neighbours' own disorder about that line and (b)
#: the series-wide robust noise (1.4826 x MAD of every departure). A line
#: absorbs a smooth slope. Extended steps or cusps can also raise neighbour
#: disorder, whereas an isolated departure on a smooth response can score
#: strongly. Scores depend on sampling, noise and feature shape; no fixed
#: range separates physical features from defective readings.
#:
#: Empirical working threshold: emphasise departures from a locally smooth
#: response without interpreting the score as a probability. It is not a
#: calibrated false-alarm rate or a guarantee that every flagged point is bad.
#: Inspect scores and response shape; record any override.
CONTINUITY_THRESHOLD = 20.0

#: Median absolute deviation to standard deviation for a normal
#: distribution (1 / Phi^-1(3/4)).
_MAD_TO_SIGMA = 1.4826

#: Relative size below which a departure is floating-point rounding, not a
#: departure: least-squares arithmetic on float64 errs by a few 1e-16 of the
#: values' magnitude, and no instrument resolves one part in 1e12.
_ROUNDING = 1e-12

#: Columns of :attr:`QualityReport.rows`, in order. Fixed, so a report on a
#: file that offers no evidence of some kind still has the column (NaN, or
#: empty) and code reading reports never branches on the schema.
QUALITY_COLUMNS = (
    "family",
    "frequency_hz",
    "signal",
    "value",
    "error",
    "continuity_score",
    "flagged",
    "err_ratio",
    "T_spread_K",
    "T_spread_ratio",
    "H_spread_Oe",
    "fit_quality",
    "status_departure",
    "range_change",
)

#: Instrument fit-quality column paired with each moment column: the fit
#: that produced *that* moment. DC Free/Fixed Fit pair with their respective
#: moments; VSM and AC moments have no such column.
_FIT_QUALITY_BASES = {
    "dc moment free ctr": "DC Free Fit",
    "dc moment fixed ctr": "DC Fixed Fit",
}

#: SQUID range column per family. A range change is evidence of a possible
#: step, not a verdict -- a steep real feature drives one too.
_RANGE_BASES = {"VSM": "Range", "DC": "Range", "AC": "AC Range"}

#: Report departures in waveform count as diagnostic evidence. AC Cycles is
#: excluded by default; this empirical choice is not an instrument-wide quality rule.
_STATUS_EXTRA_BASES = ("Measure Count",)


def _window(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Each point's window neighbours as an ``n x 2k`` array.

    Neighbours past either end of the series are NaN, so every row has the
    same width and the edges need no special indexing.
    """
    k = CONTINUITY_WINDOW
    n = len(values)
    padded = np.concatenate([np.full(k, np.nan), values, np.full(k, np.nan)])
    offsets = [o for o in range(-k, k + 1) if o != 0]
    return np.stack([padded[k + o : k + o + n] for o in offsets], axis=1)


def _departures(
    y: NDArray[np.float64], x: NDArray[np.float64] | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Each point's departure from its neighbours' trend, and their disorder.

    With three or more finite neighbours, a straight line is fitted through
    them (least squares, against ``x`` -- the swept quantity -- or against
    position along the series when ``x`` is None): the departure is the
    point's distance from that line at its own ``x``, and the disorder is the
    range of the neighbours' own residuals about it. A line absorbs a smooth slope --
    which is the point: measured against the raw spread of neighbours on a
    steep flank, a sign-flipped point once scored below the threshold,
    because the flank's own rise counted as disorder. A step or a cusp
    cannot be fitted by a line, so its residuals stay large and the score
    stays small.

    The line is fitted against the swept quantity, not row position, because
    real sweeps are often sampled unevenly -- denser near a feature, or with
    a setpoint jittered by the magnet -- and against position a signal
    linear in field would read as curved there. Neighbours sharing one ``x``
    (repeats at a setpoint) give no slope; the line is then flat, at their
    mean.

    With exactly two neighbours (a series endpoint, or a gap) a line would
    fit them exactly and leave no disorder to measure, so the median and
    raw range are used instead. Fewer than two, or a NaN point: NaN.
    """
    n = len(y)
    if x is None:
        x = np.arange(n, dtype=float)
    neighbours = _window(y)
    finite = np.isfinite(neighbours) & np.isfinite(_window(x))
    count = finite.sum(axis=1)
    # Positions relative to the point itself, so the fitted line is read off
    # at 0 and large absolute fields cost no precision.
    rel = np.where(finite, _window(x) - x[:, None], np.nan)
    neighbours = np.where(finite, neighbours, np.nan)
    departure = np.full(n, np.nan)
    disorder = np.full(n, np.nan)

    line = (count >= 3) & np.isfinite(y) & np.isfinite(x)
    if line.any():
        xs, vs = rel[line], neighbours[line]
        x_mean = np.nanmean(xs, axis=1, keepdims=True)
        v_mean = np.nanmean(vs, axis=1, keepdims=True)
        sxx = np.nansum((xs - x_mean) ** 2, axis=1, keepdims=True)
        sxy = np.nansum((xs - x_mean) * (vs - v_mean), axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(sxx > 0, sxy / sxx, 0.0)
        at_point = (v_mean - slope * x_mean)[:, 0]
        residual = vs - (v_mean + slope * (xs - x_mean))
        departure[line] = y[line] - at_point
        disorder[line] = np.nanmax(residual, axis=1) - np.nanmin(residual, axis=1)

    pair = (count == 2) & np.isfinite(y)
    if pair.any():
        vs = neighbours[pair]
        departure[pair] = y[pair] - np.nanmedian(vs, axis=1)
        disorder[pair] = np.nanmax(vs, axis=1) - np.nanmin(vs, axis=1)

    # A departure within floating-point rounding of the values is none: an
    # exactly linear series fits its line to ~1e-16, and dividing that by
    # an exactly zero disorder would score inf where nothing departs.
    magnitude = np.fmax(np.abs(y), np.nanmax(np.abs(np.where(finite, neighbours, 0.0)), axis=1))
    departure[np.abs(departure) <= _ROUNDING * magnitude] = 0.0
    disorder[disorder <= _ROUNDING * magnitude] = 0.0
    return departure, disorder


def continuity_scores(
    ys: Sequence[NDArray[np.float64]],
    xs: Sequence[NDArray[np.float64] | None] | None = None,
) -> list[NDArray[np.float64]]:
    """Continuity score for every point of every series.

    Each point is compared with a straight line through its
    :data:`CONTINUITY_WINDOW` neighbours on either side, itself excluded (so
    the point under test cannot drag its own reference), fitted against the
    matching entry of ``xs`` -- the swept quantity -- or against position
    along the series where ``xs`` (or its entry) is None. The departure from
    that line is normalised by the larger of the neighbours' own disorder
    about it and a robust noise level (:data:`_MAD_TO_SIGMA` x MAD of every
    departure, pooled across all the series passed together) -- see
    :func:`_departures` for why a line and not a median, and
    :data:`CONTINUITY_THRESHOLD` for the threshold rationale. Points with fewer than
    two finite neighbours score NaN, and so does a NaN point.

    Pass the series of one family together (the map glitch check passes
    every sweep of a map) so the noise floor is estimated from all of them;
    pass one series alone to score it on its own terms.

    A point that does not depart scores 0, even where the disorder and the
    pooled noise are both zero ("no departure, no score" rather than NaN
    from 0/0); one that departs from a noiseless, perfectly regular series
    scores inf.
    """
    departures = []
    disorders = []
    xs = [None] * len(ys) if xs is None else list(xs)
    if len(xs) != len(ys):
        raise QualityInputError(f"xs has {len(xs)} entries for {len(ys)} series")
    for y, x in zip(ys, xs, strict=True):
        dep, dis = _departures(
            np.asarray(y, dtype=float), None if x is None else np.asarray(x, dtype=float)
        )
        departures.append(dep)
        disorders.append(dis)

    pooled = np.concatenate(departures) if departures else np.empty(0)
    pooled = pooled[np.isfinite(pooled)]
    noise = (
        _MAD_TO_SIGMA * float(np.median(np.abs(pooled - np.median(pooled))))
        if len(pooled)
        else 0.0
    )
    scores = []
    for dep, dis in zip(departures, disorders, strict=True):
        scale = np.maximum(dis, noise)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.abs(dep) / scale
        s[dep == 0] = 0.0
        scores.append(s)
    return scores


def _neighbour_ratio(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Each value over the median of its window neighbours (NaN if none).

    The same window as the continuity score, so "larger than its
    neighbours'" means the same neighbours in every column of a report.
    """
    k = CONTINUITY_WINDOW
    n = len(values)
    padded = np.concatenate([np.full(k, np.nan), values, np.full(k, np.nan)])
    offsets = [o for o in range(-k, k + 1) if o != 0]
    neighbours = np.stack([padded[k + o : k + o + n] for o in offsets], axis=1)
    out = np.full(n, np.nan)
    has = np.isfinite(neighbours).any(axis=1)
    if has.any():
        reference = np.nanmedian(neighbours[has], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[has] = values[has] / reference
    return out


@dataclass(frozen=True)
class QualityReport:
    """What a file says about the trustworthiness of each of its points.

    Returned by :meth:`Measurement.quality()
    <calamari.measurement.Measurement.quality>`; see :mod:`calamari.quality`
    for what is a gate and what is evidence, and why.

    Attributes
    ----------
    path : Path or None
        The file the report describes.
    threshold : float
        The continuity score above which a point is flagged.
    rows : pandas.DataFrame
        One row per scored point, indexed by the point's row label in
        ``Measurement.data`` (so ``m.data.loc[report.flagged.index]`` is the
        raw rows). Columns are :data:`QUALITY_COLUMNS`:

        ``family``, ``frequency_hz``
            The series the point was scored in: a measurement family
            (``'VSM'``, ``'DC'``, ``'AC'``) and, for AC, one frequency. Each
            series is scored on its own, in file order. Scored together, a
            combined run's interleaved rows would put the other family's (or
            frequency's) values among every point's neighbours, and their
            offset would set the neighbours' spread -- hiding real glitches
            in either.
        ``signal``, ``value``, ``error``
            The accessor scored (``'moment_emu'`` or ``'chi_real'``), its
            value and paired standard error at this point.
        ``continuity_score``, ``flagged``
            The gate: :func:`continuity_scores`, and whether it exceeds
            ``threshold``.
        ``err_ratio``
            ``error`` over the median of its neighbours' errors. Large on a
            glitch -- and on the first point of every field scan.
        ``T_spread_K``, ``T_spread_ratio``, ``H_spread_Oe``
            Max minus min temperature and field over the point (the MPMS3
            ``Min./Max.`` columns), and the temperature spread relative to
            the neighbours'. In settled measurements the temperature spread
            is a stability figure, not ramp information.
        ``fit_quality``
            The instrument's own fit quality (0-1) for the moment column in
            use: ``DC Free Fit`` for the free-centre moment, ``DC Fixed Fit``
            for the fixed-centre one. Follows
            :meth:`~calamari.measurement.Measurement.with_moment`. NaN for
            VSM and AC, which have none.
        ``status_departure``
            Status columns whose value at this point differs from the
            file's usual value for this family, as ``"<column>=<value>
            (usual <value>)"`` joined by ``"; "``, or ``""``. Never read
            against a fixed "0 = OK": on real files ``1`` is the normal value
            of several codes.
        ``range_change``
            Whether the SQUID range differs from the previous point of the
            series.
    """

    path: object
    threshold: float
    rows: pd.DataFrame

    @property
    def flagged(self) -> pd.DataFrame:
        """The flagged points only -- the rows worth a second look."""
        return self.rows[self.rows["flagged"]]

    def summary(self) -> dict:
        """Counts for a catalog row: how much of this file deserves a look.

        Keys: ``n_scored``, ``n_flagged``, ``max_continuity_score``,
        ``n_status_departures``, ``n_range_changes``, ``fit_quality_min``,
        ``T_spread_max_K``. A value the file cannot supply (no fit-quality
        column, no ``Min./Max.`` columns) is None, never 0 -- "none
        recorded" and "none found" are different answers.
        """
        rows = self.rows

        def finite_max(column):
            values = rows[column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            return float(values.max()) if values.size else None

        def finite_min(column):
            values = rows[column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            return float(values.min()) if values.size else None

        return {
            "n_scored": int(rows["continuity_score"].notna().sum()),
            "n_flagged": int(rows["flagged"].sum()),
            "max_continuity_score": finite_max("continuity_score"),
            "n_status_departures": int((rows["status_departure"] != "").sum()),
            "n_range_changes": int(rows["range_change"].sum()),
            "fit_quality_min": finite_min("fit_quality"),
            "T_spread_max_K": finite_max("T_spread_K"),
        }

    def __repr__(self) -> str:
        """One line, in the style of the Measurement repr: the file's name,
        how many points are flagged, and the evidence counts."""
        s = self.summary()
        score = (
            "no points scored"
            if s["max_continuity_score"] is None
            else f"max score {s['max_continuity_score']:.3g}"
        )
        name = getattr(self.path, "name", None) or str(self.path)
        return (
            f"<QualityReport {name} | {s['n_flagged']} of {len(self.rows)} "
            f"points flagged ({score}, threshold {self.threshold:g}) | "
            f"{s['n_status_departures']} status departures | "
            f"{s['n_range_changes']} range changes>"
        )


def _series(m: Measurement, name: str, bases: Sequence[str], units: dict) -> pd.Series | None:
    """One optional column resolved and unit-converted, or None."""
    from calamari.measurement import _columns_with_base

    if not _columns_with_base(m.data, bases):
        return None
    return m._resolve_column(tuple(bases), units, quantity=name)


def _frequency_clusters(freq: NDArray[np.float64]) -> list[float]:
    """Distinct AC frequencies, clustered as the map helpers cluster them."""
    from calamari.repeats import REPEAT_FREQUENCY_RTOL

    return frequency_clusters(freq[np.isfinite(freq)], rtol=REPEAT_FREQUENCY_RTOL)


def _status_departures(data: pd.DataFrame, rows: pd.Index) -> pd.Series:
    """Per-row description of status values departing from their usual one.

    "Usual" is the modal value over this family's rows, per column. A code
    column that is blank throughout the family says nothing and is skipped.
    """
    from calamari.measurement import _columns_with_base

    columns = [c for c in data.columns if str(c).endswith("(code)")]
    columns += _columns_with_base(data, _STATUS_EXTRA_BASES)
    out = pd.Series("", index=rows, dtype=object)
    for column in columns:
        values = pd.to_numeric(data.loc[rows, column], errors="coerce")
        present = values.dropna()
        if present.empty:
            continue
        usual = present.mode().iloc[0]
        departs = values.notna() & (values != usual)
        for label in departs[departs].index:
            text = f"{column}={values[label]:g} (usual {usual:g})"
            out[label] = f"{out[label]}; {text}" if out[label] else text
    return out


def assess(m: Measurement, *, threshold: float = CONTINUITY_THRESHOLD) -> QualityReport:
    """Build the :class:`QualityReport` for one measurement.

    The implementation behind :meth:`Measurement.quality()
    <calamari.measurement.Measurement.quality>`, which is the spelling to
    use; see there.
    """
    from calamari.measurement import (
        _FIELD_UNITS,
        _MOMENT_BASES,
        _TEMPERATURE_UNITS,
        FAMILY_ORDER,
        _split_unit_suffix,
    )
    from calamari.repeats import REPEAT_FREQUENCY_RTOL

    if not (threshold > 0):
        raise QualityInputError(f"threshold must be positive, got {threshold!r}")

    frames = []
    measured = m._measurement_mask()
    t_min = _series(m, "minimum temperature", ("Min. Temperature",), _TEMPERATURE_UNITS)
    t_max = _series(m, "maximum temperature", ("Max. Temperature",), _TEMPERATURE_UNITS)
    h_min = _series(m, "minimum field", ("Min. Field",), _FIELD_UNITS)
    h_max = _series(m, "maximum field", ("Max. Field",), _FIELD_UNITS)

    # Families are computed once: of_type() would recompute them for every
    # family (and restrict() again inside it), which was most of the cost
    # of a catalog build. The per-family view is the same object of_type
    # returns -- the family's rows, re-indexed from 0, restamped -- built
    # directly, and its labels map back to the file's one to one.
    families = m.row_families
    present = [f for f in FAMILY_ORDER if f in set(families)]
    for family in present:
        labels = m.data.index[(families == family).to_numpy()]
        part = replace(
            m, data=m.data.loc[labels].reset_index(drop=True), measurement_type=family
        )

        def relabel(series, labels=labels):
            return None if series is None else pd.Series(series.to_numpy(), index=labels)

        if family == "AC":
            signal = "chi_real"
            value, error = relabel(part.chi_real), relabel(part.chi_real_err)
            fit = None
        else:
            signal = "moment_emu"
            value, error = relabel(part.moment_emu), relabel(part.moment_err_emu)
            in_use = part.moment_base or part._column_in_use(_MOMENT_BASES)
            fit_base = _FIT_QUALITY_BASES.get((in_use or "").lower())
            fit = (
                None
                if fit_base is None
                else relabel(_series(part, "fit quality", (fit_base,), {}))
            )
        if value is None:
            continue
        # The swept quantity the line is fitted against: field for a field
        # scan, temperature for a temperature scan, row order otherwise
        # (a time series at fixed T and H, or a sweep of both).
        kind = part._safe(lambda part=part: part.scan_kind, "unknown")
        axis = {"H-scan": part.field_oe, "T-scan": part.temperature}.get(kind)
        axis = relabel(axis)

        keep = value.notna() & measured.reindex(value.index, fill_value=True)
        index = value.index[keep]
        frequency_series = relabel(part.ac_frequency) if family == "AC" else None
        if frequency_series is not None:
            freq = frequency_series.reindex(index).to_numpy(dtype=float)
            clusters = _frequency_clusters(freq)
        else:
            freq, clusters = None, []
        groups = (
            [(None, np.ones(len(index), dtype=bool))]
            if not clusters
            else [(c, np.isclose(freq, c, rtol=REPEAT_FREQUENCY_RTOL)) for c in clusters]
        )

        range_base = _RANGE_BASES[family]
        range_cols = [
            c for c in m.data.columns if _split_unit_suffix(str(c))[0] == range_base
        ]
        departures = _status_departures(m.data, index)

        for frequency, select in groups:
            idx = index[select]
            y = value.reindex(idx).to_numpy(dtype=float)
            err = (
                np.full(len(idx), np.nan)
                if error is None
                else error.reindex(idx).to_numpy(dtype=float)
            )
            x = None if axis is None else axis.reindex(idx).to_numpy(dtype=float)
            score = continuity_scores([y], [x])[0]

            def spread(lo, hi, idx=idx):
                if lo is None or hi is None:
                    return np.full(len(idx), np.nan)
                return (hi - lo).reindex(idx).to_numpy(dtype=float)

            t_spread = spread(t_min, t_max)
            if range_cols:
                rng = pd.to_numeric(m.data.loc[idx, range_cols[0]], errors="coerce").to_numpy(
                    dtype=float
                )
                change = np.r_[False, (rng[1:] != rng[:-1]) & np.isfinite(rng[1:])
                               & np.isfinite(rng[:-1])]
            else:
                change = np.zeros(len(idx), dtype=bool)

            frames.append(
                pd.DataFrame(
                    {
                        "family": family,
                        "frequency_hz": np.nan if frequency is None else frequency,
                        "signal": signal,
                        "value": y,
                        "error": err,
                        "continuity_score": score,
                        "flagged": np.nan_to_num(score, nan=0.0) > threshold,
                        "err_ratio": _neighbour_ratio(err),
                        "T_spread_K": t_spread,
                        "T_spread_ratio": _neighbour_ratio(t_spread),
                        "H_spread_Oe": spread(h_min, h_max),
                        "fit_quality": (
                            np.full(len(idx), np.nan)
                            if fit is None
                            else fit.reindex(idx).to_numpy(dtype=float)
                        ),
                        "status_departure": departures.reindex(idx).to_numpy(),
                        "range_change": change,
                    },
                    index=idx,
                )
            )

    rows = (
        pd.concat(frames).sort_index()
        if frames
        else pd.DataFrame({c: pd.Series(dtype=object) for c in QUALITY_COLUMNS})
    )
    rows = rows[list(QUALITY_COLUMNS)]
    rows["flagged"] = rows["flagged"].astype(bool)
    rows["range_change"] = rows["range_change"].astype(bool)
    return QualityReport(path=m.path, threshold=float(threshold), rows=rows)


# ---------------------------------------------------------------------------
# The outliers= contract shared by the consumers of a report. One home, so
# the vocabulary, the validation and the report-matching checks cannot drift
# apart between detect_transitions and the family plots.
# ---------------------------------------------------------------------------

#: Accepted string values of an ``outliers=`` keyword (a QualityReport -- or,
#: for a family, one per measurement -- is also accepted: an explicit drop of
#: its flagged rows). The same vocabulary everywhere it appears
#: (:func:`calamari.plot.chi_map_grid`, :func:`calamari.plot.plot_family`,
#: :func:`calamari.transitions.detect_transitions`), so one word means one
#: thing everywhere.
_VALID_OUTLIER_MODES = ("warn", "drop", "ignore")

#: Flagged points listed in full in an outlier warning before the rest are
#: summarised as a count: enough to see the pattern, short enough that a
#: warning is still a warning and not a data dump. Shared by every consumer
#: (the maps, the family plots, detection), so their warnings read alike.
_OUTLIER_WARN_LIST = 10


def _plain_label(label):
    """A row label as the caller would write it: ``0``, not ``np.int64(0)``.

    Depending on the pandas version, integer row labels come back as numpy
    scalars, whose repr under NumPy 2 is ``np.int64(0)``. That is noise in an
    error message and in ``params``, and it varied across the CI matrix.
    """
    return int(label) if isinstance(label, (int, np.integer)) else label


def _check_outlier_threshold(outlier_threshold) -> None:
    """Refuse an ``outlier_threshold=`` that is not None or a positive, finite score."""
    if outlier_threshold is None:
        return
    if (
        isinstance(outlier_threshold, bool)
        or not isinstance(outlier_threshold, (int, float, np.integer, np.floating))
        or not math.isfinite(outlier_threshold)
        or outlier_threshold <= 0
    ):
        raise QualityInputError(
            "outlier_threshold must be a positive, finite continuity score (or None "
            f"for calamari.quality.CONTINUITY_THRESHOLD = {CONTINUITY_THRESHOLD:g}), "
            f"got {outlier_threshold!r}"
        )


def _check_outlier_args(
    outliers, outlier_threshold, *, caller: str, per_measurement: bool = False
) -> None:
    """Refuse a malformed ``outliers=`` / ``outlier_threshold=`` up front.

    Shared by every consumer, so a typo raises before any work is done --
    before a batch lands it, reworded, in the ``skipped`` list of every
    sweep, and before a family plot has drawn half its curves. ``caller``
    names the function in the one message that must say whose scoring is
    meant.

    ``per_measurement`` selects the report form. A consumer of one
    measurement (:func:`~calamari.transitions.detect_transitions`) takes one
    ``QualityReport``; a consumer of several
    (:func:`~calamari.plot.plot_family`) takes a sequence of them, one per
    measurement -- a single report there is refused by name, because it
    could describe only one of the curves. The sequence's *length* is the
    caller's to check: only the caller knows how many measurements it has.
    """
    if per_measurement:
        if isinstance(outliers, QualityReport):
            raise QualityInputError(
                f"{caller} takes one QualityReport per measurement, as a sequence "
                "aligned with the measurements in the order given -- e.g. "
                "outliers=[m.quality() for m in measurements] -- not a single report, "
                "which can describe only one of the curves."
            )
        if isinstance(outliers, Sequence) and not isinstance(outliers, str):
            for i, report in enumerate(outliers):
                if not isinstance(report, QualityReport):
                    raise QualityInputError(
                        f"outliers[{i}] is a {type(report).__name__}, not a "
                        "QualityReport: a sequence passed as outliers= must hold one "
                        "report per measurement, from each one's quality()."
                    )
            if outlier_threshold is not None:
                raise QualityInputError(
                    f"outlier_threshold applies when {caller} scores the measurements "
                    "itself; a QualityReport was scored already, at its own threshold. "
                    "Pass one or the other -- e.g. "
                    "outliers=[m.quality(threshold=<score>) for m in measurements]."
                )
            return
        accepted = "a sequence of QualityReport, one per measurement from its quality()"
    else:
        if isinstance(outliers, QualityReport):
            if outlier_threshold is not None:
                raise QualityInputError(
                    f"outlier_threshold applies when {caller} scores the "
                    "measurement itself; a QualityReport was scored already, at its own "
                    f"threshold ({outliers.threshold:g}). Pass one or the other -- e.g. "
                    "outliers=m.quality(threshold=<score>)."
                )
            return
        accepted = "a QualityReport from this measurement's quality()"
    if not isinstance(outliers, str) or outliers not in _VALID_OUTLIER_MODES:
        raise QualityInputError(
            f"outliers must be 'warn', 'drop', 'ignore', or {accepted}, got {outliers!r}"
        )
    _check_outlier_threshold(outlier_threshold)


def _check_report_matches(report: QualityReport, measurement: Measurement) -> None:
    """Refuse a QualityReport that does not describe ``measurement``.

    A report addresses points by row label, and a label is only meaningful
    in the view it was taken on. Three checks, each catching a way a report
    can silently point at the wrong rows: another file (``path``); a larger
    view of this file (a label this measurement lacks); and a *reset*
    view -- :meth:`~calamari.measurement.Measurement.restrict` renumbers
    from 0, so a report taken before it names labels that still exist but
    now hold other points. Only the values can catch the last, so the
    report's ``value`` column is compared with the accessor it scored.
    """
    if str(report.path) != str(measurement.path):
        raise QualityInputError(
            f"outliers=QualityReport describes {report.path}, but this measurement "
            f"is {measurement.path} -- a report's row labels mean nothing in another "
            "file. Pass this measurement's own quality(), or outliers='drop' to "
            "score it here."
        )
    rows = report.rows
    missing = rows.index.difference(measurement.data.index)
    if len(missing):
        raise QualityInputError(
            f"{measurement.path}: the QualityReport names {len(missing)} row label(s) "
            f"this measurement does not have (first: {_plain_label(missing[0])!r}), so it was taken "
            "on a different view of the file -- typically before restrict(), which "
            "renumbers the rows. Score the measurement you pass: "
            "outliers=m.quality() on this view, or outliers='drop'."
        )
    for signal, group in rows.groupby("signal", sort=False):
        series = getattr(measurement, signal)
        expected = group["value"].to_numpy(dtype=float)
        actual = (
            np.full(len(group), np.nan)
            if series is None
            else series.reindex(group.index).to_numpy(dtype=float)
        )
        same = np.isclose(actual, expected, rtol=1e-12, atol=0.0, equal_nan=True)
        if not same.all():
            i = int(np.flatnonzero(~same)[0])
            raise QualityInputError(
                f"{measurement.path}: the QualityReport's {signal} at row "
                f"{_plain_label(group.index[i])!r} is {expected[i]:.6g}, but this measurement holds "
                f"{actual[i]:.6g} there -- the report was taken on a different view of "
                "the file (restrict() renumbers the rows, so its labels now name other "
                "points). Score the measurement you pass: outliers=m.quality() on this "
                "view, or outliers='drop'."
            )
