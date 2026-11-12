"""Measurement selection, screening and single/batch detection orchestration."""

from __future__ import annotations

import inspect
import math
import warnings
from collections.abc import Callable, Iterable, Sequence
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from calamari._provenance import calamari_version as _calamari_version
from calamari._provenance import describe_callable as _describe_callable
from calamari._sweep_rules import frequency_clusters as _frequency_clusters
from calamari._sweep_rules import select_frequency as _select_recorded_frequency
from calamari._sweep_rules import validate_frequency as _validate_frequency
from calamari.constants import OE_PER_TESLA
from calamari.exceptions import CalamariError, DetectionInputError
from calamari.measurement import SCAN_H_TOL_OE, SCAN_T_TOL_K, _split_unit_suffix
from calamari.quality import (
    _OUTLIER_WARN_LIST,
    CONTINUITY_THRESHOLD,
    QualityReport,
    _check_outlier_args,
    _check_report_matches,
    _plain_label,
)
from calamari.repeats import REPEAT_FREQUENCY_RTOL
from calamari.transitions._metadata import (
    _SIGNALS,
    CANDIDATE_COLUMNS,
    CRITERION_PRIORITY,
    DEFAULT_PROMINENCE_FRACTION,
    DETECTION_COLUMNS,
)
from calamari.transitions._numerics import _check_detection_options, find_transitions
from calamari.transitions._results import TransitionResult

if TYPE_CHECKING:
    from calamari.measurement import Measurement


def _measured_array(measurement: Measurement, series) -> np.ndarray | None:
    """Measurement-rows-only float array for a Measurement series (the
    housekeeping mask every aggregate description goes through -- a
    locate-scan row must not shift a published coordinate)."""
    series = measurement._measured(series)
    if series is None:
        return None
    return series.to_numpy(dtype=float)


def _fixed_value_and_span(
    measurement: Measurement, param: str
) -> tuple[float, float]:
    """Masked mean and span of the sweep's nominally fixed parameter,
    refusing a drifting one -- mirrors the plot layer's family labelling
    (same tolerances), so a detected point and its map row cannot
    disagree by construction. Returns values in the accessor's own unit
    (K, or Oe for field)."""
    accessor, tol = (
        ("temperature", SCAN_T_TOL_K) if param == "temperature" else ("field_oe", SCAN_H_TOL_OE)
    )
    arr = _measured_array(measurement, getattr(measurement, accessor))
    if arr is not None:
        arr = arr[np.isfinite(arr)]
    if arr is None or len(arr) == 0:
        raise DetectionInputError(
            f"{measurement.path}: no {param} values to place this sweep's "
            "transitions at -- the fixed coordinate of every candidate would be "
            "unknown."
        )
    value, span = float(arr.mean()), float(arr.max() - arr.min())
    if span > tol:
        raise DetectionInputError(
            f"{measurement.path}: {param} is not fixed in this sweep -- it spans "
            f"{span:.4g} against a tolerance of {tol:.4g}. A transition placed at "
            "its mean would be a mislabelled point on a phase diagram. Restrict "
            "the sweep (t_window_k=/h_window_oe=) or split it upstream."
        )
    return value, span


def _select_frequency(
    measurement: Measurement, frequency: float | None
) -> tuple[np.ndarray | None, float | None]:
    """Row mask selecting one AC frequency, refusing silent mixing.

    A multi-frequency file interleaves rows at each setpoint; searching
    them together would dedupe to whichever frequency came first -- and
    the frequency dependence of a chi' peak is itself the measurement for
    glassy systems. Same refusal semantics as the map layer.
    """
    freq = _measured_array(measurement, measurement.ac_frequency)
    if freq is None:
        if frequency is not None:
            _select_recorded_frequency([], frequency, rtol=REPEAT_FREQUENCY_RTOL,
                                       error_type=DetectionInputError)
        return None, None
    finite = freq[np.isfinite(freq)]
    if finite.size == 0:
        if frequency is not None:
            _select_recorded_frequency([], frequency, rtol=REPEAT_FREQUENCY_RTOL,
                                       error_type=DetectionInputError)
        return None, None
    clusters = _frequency_clusters(np.round(finite, 12), rtol=REPEAT_FREQUENCY_RTOL)
    if frequency is None:
        if len(clusters) > 1:
            listed = ", ".join(f"{c:g}" for c in clusters)
            raise DetectionInputError(
                f"{measurement.path}: {len(clusters)} distinct AC frequencies "
                f"({listed} Hz) -- transitions are one frequency at a time (their "
                "frequency dependence is physics, not noise). Pass "
                "frequency=<value> to select one."
            )
        selected = clusters[0]
    else:
        selected = _select_recorded_frequency(
            clusters, frequency, rtol=REPEAT_FREQUENCY_RTOL, error_type=DetectionInputError,
        )
    mask = np.abs(freq - selected) <= selected * REPEAT_FREQUENCY_RTOL
    return mask, selected


def _resolve_signals(measurement: Measurement, signal) -> list[str]:
    """Which channels to search. ``None`` -> ONE primary channel per probe
    (chi' for AC data, the moment for DC) -- searching every channel by
    default would double-report every transition from chi' and chi''
    without any physics to merge them. ``'all'`` opts into every
    populated channel; an explicit name or sequence restricts.
    """

    def populated(name: str) -> bool:
        arr = _measured_array(measurement, getattr(measurement, name))
        return arr is not None and np.isfinite(arr).any()

    if signal is None:
        for name in ("chi_real", "moment_emu"):
            if populated(name):
                return [name]
        raise DetectionInputError(
            f"{measurement.path}: neither chi_real nor moment_emu carries data -- "
            "nothing to search. Pass signal= to name a channel explicitly."
        )
    if isinstance(signal, str) and signal == "all":
        names = [name for name in _SIGNALS if populated(name)]
        if not names:
            raise DetectionInputError(f"{measurement.path}: no searchable channel carries data.")
        return names
    if isinstance(signal, str):
        signal = [signal]
    for name in signal:
        if name not in _SIGNALS:
            raise DetectionInputError(
                f"signal must be one of {tuple(_SIGNALS)}, 'all', or a sequence "
                f"of those, got {name!r}"
            )
    return list(signal)


def _default_methods(scan_kind: str, signal: str) -> tuple[str, ...]:
    """Scan-kind defaults (module docstring): H-scans use the single
    differential-response criterion, one derivative apart for chi vs M;
    T-scans run both coexisting conventions."""
    if scan_kind == "T-scan":
        return ("peak", "derivative")
    return ("derivative",) if signal == "moment_emu" else ("peak",)


def _increasing_branch_indices(field: np.ndarray) -> np.ndarray:
    """Indices of the leading increasing-``|field|`` run (the logic of
    calamari.sweeps.increasing_field_branch, returning indices so aligned
    arrays -- y_err, frequency mask -- can be sliced consistently)."""
    from calamari.sweeps import increasing_field_branch

    _, idx = increasing_field_branch(field, np.arange(len(field), dtype=float))
    return idx.astype(int)


def _outlier_record(
    source: Measurement,
    report: QualityReport,
    mode: str,
    *,
    swept_t: bool,
    signals: Sequence[str],
    selected_freq: float | None,
    t_window_k: tuple[float, float] | None,
    h_window_oe: tuple[float, float] | None,
    dropping: bool,
) -> dict:
    """Which flagged points detection would search, as ``params['outliers']``.

    ``source`` is the measurement as passed (the report's labels are its
    labels). A flagged point is *in the search* when it lies inside the
    ``t_window_k`` / ``h_window_oe`` bounds, belongs to a measurement family
    that one of ``signals`` reads (AC rows for chi, DC/VSM rows for the
    moment), and -- for AC -- sits at the selected frequency, by the same
    test :func:`_select_frequency` uses. A point failing any of these is
    never searched, and warning about it would send the caller to look at
    data the candidates do not depend on.
    """
    flagged = report.flagged
    labels = flagged.index
    inside = np.ones(len(labels), dtype=bool)
    if t_window_k is not None:
        temperature = source.temperature.reindex(labels).to_numpy(dtype=float)
        inside &= (temperature >= t_window_k[0]) & (temperature <= t_window_k[1])
    if h_window_oe is not None:
        field_oe = source.field_oe.reindex(labels).to_numpy(dtype=float)
        inside &= (field_oe >= h_window_oe[0]) & (field_oe <= h_window_oe[1])
    n_in_window = int(inside.sum())

    probes = {_SIGNALS[s][1] for s in signals}
    family = flagged["family"].to_numpy()
    relevant = np.where(family == "AC", "AC" in probes, "DC" in probes)
    if selected_freq is not None and source.ac_frequency is not None:
        freq = source.ac_frequency.reindex(labels).to_numpy(dtype=float)
        at_selected = np.abs(freq - selected_freq) <= selected_freq * REPEAT_FREQUENCY_RTOL
        relevant &= (family != "AC") | at_selected
    searched = flagged[inside & relevant]

    loc_key = "T_K" if swept_t else "H_T"
    coordinate = source.temperature if swept_t else source.field_tesla
    coords = (
        np.full(len(searched), np.nan)
        if coordinate is None
        else coordinate.reindex(searched.index).to_numpy(dtype=float)
    )
    points = [
        {
            "row": _plain_label(label),
            loc_key: float(coord),
            "value": float(value),
            "score": float(score),
        }
        for label, coord, value, score in zip(
            searched.index, coords, searched["value"], searched["continuity_score"],
            strict=True,
        )
    ]
    return {
        "mode": mode,
        "threshold": float(report.threshold),
        "screened": True,
        "error": None,
        "n_flagged_in_search": len(points),
        # Rows the drop removed from the searched window -- every flagged row
        # inside it, including any at an unsearched frequency or family
        # (removing those cannot change a candidate), hence possibly more
        # than n_flagged_in_search.
        "n_dropped": n_in_window if dropping else 0,
        "flagged": points,
    }


def _unscreened_record(mode: str, outlier_threshold: float | None, error: Exception) -> dict:
    """``params['outliers']`` for a sweep ``quality()`` could not score.

    Same keys as the screened record, so a batch reads one schema:
    ``screened=False`` with the error, and zero counts -- zero here means
    "nothing was looked at", which ``screened`` says, not "nothing found".
    """
    return {
        "mode": mode,
        "threshold": float(
            CONTINUITY_THRESHOLD if outlier_threshold is None else outlier_threshold
        ),
        "screened": False,
        "error": f"{type(error).__name__}: {error}",
        "n_flagged_in_search": 0,
        "n_dropped": 0,
        "flagged": [],
    }


def _warn_unscreened(measurement: Measurement, error: Exception) -> None:
    """The ``outliers='warn'`` message when screening itself failed.

    Silence would read as "screened, nothing flagged"; the warning says the
    sweep went through unscreened, why, and how to accept that knowingly.
    """
    name = getattr(measurement.path, "name", None) or str(measurement.path)
    warnings.warn(
        f"detect_transitions: {name} was not screened for bad points -- "
        f"Measurement.quality() raised {type(error).__name__}: {error} "
        "Detection went ahead on every point as measured. Pass outliers='ignore' to "
        "accept that without this warning; params['outliers'] records "
        "screened=False and the error.",
        stacklevel=3,
    )


def _warn_outliers(
    record: dict, report: QualityReport, measurement: Measurement, swept_t: bool
) -> None:
    """The ``outliers='warn'`` message, in the map helpers' style.

    Names the file, each point by its swept coordinate, the quantity scored
    and its value, and its score, then the three ways out -- so the warning
    is actionable where it is read, without a trip to the docs.
    """
    points = record["flagged"]
    loc_key, unit = ("T_K", "K") if swept_t else ("H_T", "T")
    symbol = "T" if swept_t else "H"
    scored = report.rows["signal"]
    lines = [
        f"  {scored.loc[p['row']]} {p['value']:.4g} at {symbol} = {p[loc_key]:g} {unit} "
        f"(row {p['row']}), score {p['score']:.0f}"
        for p in points[:_OUTLIER_WARN_LIST]
    ]
    if len(points) > _OUTLIER_WARN_LIST:
        lines.append(f"  ... and {len(points) - _OUTLIER_WARN_LIST} more")
    n = len(points)
    name = getattr(measurement.path, "name", None) or str(measurement.path)
    warnings.warn(
        f"detect_transitions: {n} point{'s' if n > 1 else ''} of {name} in the searched "
        f"rows depart{'' if n > 1 else 's'} from the trend of the neighbouring points by "
        "far more than those neighbours stray from it themselves (continuity score > "
        f"{record['threshold']:g}, Measurement.quality()):\n" + "\n".join(lines) + "\n"
        "Kept as measured, so it is searched like any other point and can create or "
        "displace a candidate. Pass outliers='drop' to exclude it, outliers='ignore' "
        "to keep it without this warning, or outlier_threshold=<score> to move the "
        "line; params['outliers'] records the flagged points in every mode.",
        stacklevel=3,
    )


def _check_fisher(
    measurement: Measurement, swept_t: bool, signals: list[str], field_oe: float | None
) -> None:
    """Refuse ``'fisher'`` wherever d(yT)/dT is not Fisher's quantity.

    Fisher's relation ties the magnetic specific heat to d(chi T)/dT, with
    chi the static susceptibility and the derivative along temperature.
    Each refusal below is a case where the searched ``y*T`` is not that
    quantity, so a candidate would carry Fisher's citation for something
    else. Checked once the channels are known and before any warning about
    the sweep, so a refused sweep produces no noise first.
    """
    path = measurement.path
    if not swept_t:
        raise DetectionInputError(
            f"{path}: 'fisher' needs a temperature sweep. Fisher's criterion is "
            "d(chi T)/dT, a derivative along temperature, and this file is an "
            "H-scan. Use 'peak' or 'derivative' along the field instead."
        )
    if "chi_imag" in signals:
        raise DetectionInputError(
            f"{path}: 'fisher' was asked for on chi_imag. chi'' is the dissipative "
            "response, not a static susceptibility, and Fisher's relation to the "
            "specific heat is about the static chi. Search signal='chi_real' (or "
            "the moment) with 'fisher', and chi_imag without it."
        )
    if "moment_emu" in signals and field_oe is not None and abs(field_oe) <= SCAN_H_TOL_OE:
        raise DetectionInputError(
            f"{path}: 'fisher' on moment_emu needs a held, non-zero field. M*T is "
            "proportional to chi*T only at fixed H != 0, and this sweep's field is "
            f"{field_oe:g} Oe, within {SCAN_H_TOL_OE:g} Oe (SCAN_H_TOL_OE) of zero, "
            "where the moment is remanence rather than a response to the field."
        )


def _moment_column_record(series: pd.Series) -> dict:
    """Describe a resolved moment/error Series without repeating selection."""
    return {
        "column": str(series.name),
        "native_unit": _split_unit_suffix(str(series.name))[1],
        "canonical_unit": "emu",
    }


def _check_measurement_options(signal, frequency, branch, top, t_window_k, h_window_oe):
    """Validate caller selections independently of any sweep's available data."""
    if signal is not None and not (isinstance(signal, str) and signal == "all"):
        names = (signal,) if isinstance(signal, str) else signal
        try:
            valid = bool(len(names)) and all(name in _SIGNALS for name in names)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise DetectionInputError(f"signal must name channels from {tuple(_SIGNALS)}, got {signal!r}")
    if branch not in (None, "increasing"):
        raise DetectionInputError(f"branch must be None or 'increasing', got {branch!r}")
    if frequency is not None:
        _validate_frequency(frequency, DetectionInputError)
    if top is not None and (
        isinstance(top, bool) or not isinstance(top, (int, np.integer)) or top < 0
    ):
        raise DetectionInputError(f"top must be a non-negative integer or None, got {top!r}")
    for name, window in (("t_window_k", t_window_k), ("h_window_oe", h_window_oe)):
        if window is None:
            continue
        try:
            lo, hi = window
            valid = (
                isinstance(lo, Real) and isinstance(hi, Real)
                and not np.isnan(lo) and not np.isnan(hi) and lo <= hi
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise DetectionInputError(f"{name} must be ordered bounds (lo, hi) without NaN, got {window!r}")


def detect_transitions(
    measurement: Measurement,
    *,
    signal=None,
    methods: str | Sequence[str] | None = None,
    criterion: str | None = None,
    frequency: float | None = None,
    branch: str | None = None,
    smooth_window: float | None = None,
    polyorder: int = 3,
    prominence_fraction: float = DEFAULT_PROMINENCE_FRACTION,
    top: int | None = None,
    t_window_k: tuple[float, float] | None = None,
    h_window_oe: tuple[float, float] | None = None,
    noise: str | float = "auto",
    outliers: str | QualityReport = "warn",
    outlier_threshold: float | None = None,
) -> TransitionResult:
    """Detect transition candidates in one Measurement's sweep.

    Resolves the swept axis from ``scan_kind``, the fixed coordinate from
    the masked column (refusing drift), the channels to search, and the
    scan-kind default methods -- then runs :func:`find_transitions` per
    channel and assembles the (T, H)-coordinated tables
    (:data:`CANDIDATE_COLUMNS` / :data:`DETECTION_COLUMNS`).

    Refused by name rather than guessed around: a ``'mixed'`` or
    ``'static/other'`` file (split stacked isotherms with
    :func:`calamari.sweeps.split_isotherms` first); a non-monotonic sweep
    without ``branch=``; a multi-frequency AC file without ``frequency=``;
    a drifting fixed parameter; systematically repeated setpoints, i.e.
    more than 5% of points duplicating a swept-axis value (aggregate with
    :mod:`calamari.repeats` first). Isolated duplicates below that line --
    a settle re-read, a segment join -- are dropped keep-first, with the
    count recorded as ``params['n_duplicates_dropped']`` (never silently).

    Parameters
    ----------
    measurement : Measurement
        One T-scan or H-scan.
    signal : optional
        ``None`` (default) searches ONE primary channel -- chi' for AC
        data, the moment for DC -- so a transition is reported once, not
        once per channel. ``'all'`` searches every populated channel
        (rows tagged by ``signal``; nothing merges across channels --
        the chi'' peak sitting below the chi' peak is physics, not
        duplication). A name or sequence restricts. A file with several
        populated moment columns raises ``AmbiguousColumnError`` exactly
        as the accessor does -- choose with ``with_moment()``.
    methods : optional
        ``None`` -> scan-kind defaults (module docstring). Or any subset
        of ``('peak', 'derivative', 'kink', 'onset', 'fisher')``. Three are
        always opt-in and none is ever a default (``'fisher'`` is described
        below): ``'onset'`` presupposes a
        baseline-plus-rise shape and is the method for
        superconducting-type steps, and ``'kink'`` is for a boundary that
        shows as a change of slope rather than a cusp -- add it as
        ``methods=("peak", "derivative", "kink")`` when a chi'(T) family's
        cusp flattens into a kink partway along the boundary, so that the
        slope change is reported once at its centre instead of twice, on
        alternating flanks, by the derivative detector. It does not cost
        the cusps their single row, nor move a rounded step's coordinate
        onto a shoulder -- see :func:`_group_detections` for the geometry
        that keeps each shape to one candidate. ``'onset'`` without
        ``'derivative'`` reports onsets without their parent derivative
        rows. The ``sign`` vocabulary is per method: peak
        ``'max'``/``'min'``, derivative ``'rise'``/``'fall'``, kink
        ``'up'``/``'down'``, fisher ``'max'``/``'min'``.

        ``'fisher'`` (Fisher's d(chi T)/dT criterion, see
        :func:`find_transitions`) is a third opt-in, never a default, and
        its candidates are never grouped with the others: it marks a
        systematically different temperature from the cusp, so
        ``methods=("peak", "fisher")`` reports both, side by side. It is
        refused by name where d(yT)/dT is not Fisher's quantity: on an
        H-scan (the derivative is along T); on ``chi_imag`` (chi'' is not
        a static susceptibility); and on ``moment_emu`` when the held
        field is within :data:`~calamari.measurement.SCAN_H_TOL_OE` of
        zero. Elsewhere on the moment it is allowed, because at a fixed
        non-zero field M*T = H*(chi*T). No conversion to chi or to molar
        units is made, and none is needed for detection: multiplying ``y``
        by a constant positive factor scales d(yT)/dT, its prominence and
        its noise alike, so every position and significance is unchanged.
        :func:`calamari.plot.fisher_series` draws the same quantity in
        physical units.
    criterion : optional
        Method promoted to the front of :data:`CRITERION_PRIORITY` to
        report each grouped feature. ``criterion='kink'`` reports only a
        group's own slope change -- a lone kink, or a cusp's central
        curvature extremum -- never a cusp's flank or a step's shoulder;
        a group without one reports as under the default order (see
        :func:`find_transitions`).
    frequency : float, optional
        Selects one AC frequency from a multi-frequency file (required
        there); recorded in ``params`` and the ``frequency_hz`` column.
        Must be finite and positive and match exactly one recorded cluster
        within 0.1% of its representative (no absolute tolerance). Missing or
        ambiguous matches raise. Requires an AC signal; omit for DC analysis.
    branch : {'increasing'}, optional
        Fold a hysteresis loop to its leading increasing-``|field|`` branch
        before detection. Without it a non-monotonic sweep is refused.
    smooth_window : float, optional
        Savitzky-Golay window in the swept axis's units (K for a T-scan,
        T for an H-scan). Resolved per signal in
        ``params['per_signal'][<signal>]``: ``smooth_points``, the odd
        point count on the resampled grid (at least 5, at most the sweep),
        and ``smooth_window``, that count times the grid step. Two
        requested windows can resolve to the same count on a short or
        coarse sweep; :func:`calamari.scales.scan_smoothing` warns when
        they do.
    polyorder, prominence_fraction : optional
        Forwarded to :func:`find_transitions`.
    top : int, optional
        Keep only the ``top`` highest-significance candidates per
        (signal, method) -- a visible, recorded cap for quick looks, not
        a silent gate. Default: no cap.
    t_window_k, h_window_oe : (float, float), optional
        Inclusive bounds forwarded to
        :meth:`Measurement.restrict <calamari.measurement.Measurement.restrict>`
        (temperature in Kelvin, field in Oersted -- the restrict units).
    noise : {'auto', 'instrument', 'scatter'} or float, optional
        Which noise estimate ``significance`` is scored against --
        forwarded to :func:`find_transitions` for each searched channel;
        see there for the error/scatter comparison and its assumptions. ``'auto'`` (default) prefers the instrument
        column when usable and warns if it looks inflated relative to the
        scatter (see :data:`NOISE_RATIO_WARN`); ``'instrument'`` and
        ``'scatter'`` pin the source (the former raising if the file
        carries no usable per-point errors); a positive float supplies
        sigma directly, in the signal's own units. Recorded per signal in
        ``params['per_signal'][<signal>]`` (``noise``, ``noise_source``,
        ``noise_sigma``, ``noise_sigma_instrument``,
        ``noise_sigma_scatter``, ``noise_ratio``). With ``'fisher'``,
        the same entry also holds ``fisher_noise``: the separate record
        resolved for y*T (see :func:`find_transitions`).
    outliers : {'warn', 'drop', 'ignore'} or QualityReport, optional
        What to do about points :meth:`Measurement.quality()
        <calamari.measurement.Measurement.quality>` flags as glitches.
        One bad reading on a smooth chi'(T) is a sharp local extremum,
        which is exactly what the peak and derivative detectors look for:
        a single planted glitch has produced several spurious candidates,
        while a transition's own cusp or step is not flagged (its
        neighbours disagree too, so it scores near 1). ``'warn'``
        (default) keeps the flagged points and warns once, naming the file
        and each point's swept coordinate, value and score; ``'drop'``
        excludes them before the search; ``'ignore'`` keeps them without
        the warning. A ``QualityReport`` is an explicit drop of *its*
        flagged rows -- a report you have read, perhaps at your own
        threshold -- and is refused unless it describes this measurement:
        the same ``path``, row labels this measurement has, and the same
        values at them (the check that catches a report taken before
        :meth:`~calamari.measurement.Measurement.restrict`, which renumbers
        the rows). Dropping is never the default: discarding a measured
        point changes what the result says, and that is the caller's call
        with the point in front of them.

        Scoring happens on the measurement *as passed*, before
        ``t_window_k`` / ``h_window_oe``, so each point is judged against
        the file's own neighbours; the drop then rides on the same
        ``restrict`` call as the windows. Only flagged points in the rows
        searched warn or count as ``n_flagged_in_search``: inside the
        windows, in a family a searched signal reads (AC rows for chi,
        DC/VSM rows for the moment) and, for AC, at the selected frequency.
        Two consequences of the report's design to keep in mind. It scores
        chi' for AC data and the moment for DC/VSM, so a ``chi_imag``
        search is screened by the chi'-based flag -- a glitch visible in
        chi'' alone is not caught. And dropping is row-level: a dropped
        row leaves every channel of that point, not only the one scored.
        With ``branch='increasing'``, a flagged point on the discarded
        branch still counts as searched.

        Screening can fail where detection would not: ``quality()`` reads
        columns the search never touches (the ``Min./Max.`` spreads, fit
        quality, status codes), so an unrecognised unit on one of them
        makes it raise on a sweep whose searched channel is fine. Screening
        is then skipped, not allowed to block detection: under ``'warn'``
        the sweep is searched unscreened with one warning naming the file
        and the error; under ``'ignore'`` the same, without the warning.
        ``'drop'`` raises instead (``ValueError``, chaining the original):
        removal was asked for and cannot be honoured, and proceeding would
        pass an unscreened sweep off as a cleaned one. A passed
        ``QualityReport`` needs no scoring, so it is unaffected.
    outlier_threshold : float, optional
        Continuity score above which a point is flagged. ``None``
        (default) means :data:`calamari.quality.CONTINUITY_THRESHOLD`, the
        same empirical diagnostic threshold used by map helpers, so a
        point a map warns about is the point detection warns about. Not
        accepted with a ``QualityReport``, which carries its own.

    Notes
    -----
    Moment searches record ``params['per_signal']['moment_emu']['moment_source']``:
    ``column`` is the exact resolved header, ``native_unit`` its unit token
    (``None`` if absent), and ``canonical_unit`` is ``'emu'``. ``error`` holds
    the same three fields for the populated paired error column, or ``None``.
    Its presence identifies the available error input; ``noise_source`` records
    whether it supplied the scoring noise. AC-only and array-level searches
    do not invent a moment source. Retain external preprocessing separately;
    this recipe is not a complete replay manifest.

    Returns
    -------
    TransitionResult
        Every row has ``sweep_id = 0``: one call searches one sweep, and
        :func:`detect_transitions_all` renumbers by input position.
        ``curves`` is keyed by ``(measurement_file, signal)``;
        :meth:`TransitionResult.curve` reads it the same way on a batch.
        ``params['outliers']`` records the screening in every mode:
        ``mode`` (``'warn'``, ``'drop'``, ``'ignore'`` or ``'report'``),
        ``threshold``, ``screened`` (False when ``quality()`` raised, with
        ``error`` as ``"<Type>: <message>"``; True and None otherwise --
        the counts of an unscreened sweep are 0 because nothing was
        looked at), ``n_flagged_in_search``, ``n_dropped`` (rows the
        drop removed from the window -- every flagged row inside it,
        including any at an unsearched frequency or family, so it can
        exceed ``n_flagged_in_search``; 0 unless dropping) and ``flagged``,
        one ``{'row', 'T_K' or 'H_T', 'value', 'score'}`` per point in the
        search, where ``row`` is the label in the passed measurement's
        ``data`` and ``value`` is the scored quantity (chi' or moment).

    Warns
    -----
    UserWarning
        With ``outliers='warn'``, when a flagged point lies in the searched
        rows, or when the sweep could not be screened (see ``outliers``).

    Raises
    ------
    ValueError
        With ``outliers='drop'``, when ``quality()`` cannot score the sweep;
        and for ``'fisher'`` on an H-scan, on ``chi_imag``, or on
        ``moment_emu`` at zero held field.
    """
    methods = _check_detection_options(
        methods, criterion, smooth_window, polyorder, prominence_fraction, noise
    )
    _check_measurement_options(signal, frequency, branch, top, t_window_k, h_window_oe)
    _check_outlier_args(outliers, outlier_threshold, caller="detect_transitions")
    # Score before the window: a point's neighbours must be the file's own
    # neighbours, not whatever the window leaves -- a glitch at a window
    # edge would otherwise be scored against one side only, and the score
    # would change with the window.
    source = measurement
    screen_error: Exception | None = None
    if isinstance(outliers, QualityReport):
        _check_report_matches(outliers, source)
        report, mode = outliers, "report"
    else:
        mode = outliers
        try:
            report = source.quality(threshold=outlier_threshold)
        except (CalamariError, ValueError) as exc:
            # Screening reads columns detection never touches (the Min./Max.
            # spreads, fit quality, status codes), so it can refuse a file --
            # an unrecognised unit, say -- whose searched channel is fine. The
            # catalog already tolerates exactly this (Measurement._safe). Only
            # a caller who asked for removal is refused: a drop that cannot be
            # honoured must not pass silently as if nothing were flagged.
            if mode == "drop":
                raise DetectionInputError(
                    f"{source.path}: outliers='drop' asked for flagged points to be "
                    "removed, but the sweep could not be screened -- "
                    f"Measurement.quality() raised {type(exc).__name__}: {exc} "
                    "Pass outliers='ignore' to detect without screening, or screen "
                    "the sweep yourself and pass the rows to keep with "
                    "measurement.restrict(mask=...)."
                ) from exc
            report, screen_error = None, exc
    dropping = mode in ("drop", "report")
    # The drop is folded into the one restrict() call, so the windowed view
    # is built once, from the labels the report addresses (restrict resets
    # the index; a second call would need the labels translated).
    keep = (
        ~source.data.index.isin(report.flagged.index)
        if dropping and len(report.flagged)
        else None
    )
    if t_window_k is not None or h_window_oe is not None or keep is not None:
        measurement = measurement.restrict(t_range=t_window_k, h_range=h_window_oe, mask=keep)

    scan_kind = measurement.scan_kind
    if scan_kind not in ("T-scan", "H-scan"):
        raise DetectionInputError(
            f"{measurement.path}: scan_kind is {scan_kind!r}, so this file is not "
            "one sweep along one axis. A file holding stacked isotherms splits "
            "with calamari.sweeps.split_isotherms; pass the resulting pairs to "
            "detect_transitions_all."
        )
    swept_t = scan_kind == "T-scan"

    signals = _resolve_signals(measurement, signal)
    freq_mask, selected_freq = (None, None)
    if any(_SIGNALS[s][1] == "AC" for s in signals):
        freq_mask, selected_freq = _select_frequency(measurement, frequency)
    elif frequency is not None:
        raise DetectionInputError(
            "frequency selection requires an AC signal; choose chi_real/chi_imag "
            "from an AC measurement, or omit frequency for DC moment analysis."
        )

    x_all = _measured_array(
        measurement, measurement.temperature if swept_t else measurement.field_tesla
    )
    if x_all is None:
        axis = "temperature" if swept_t else "field"
        raise DetectionInputError(f"{measurement.path}: no {axis} column -- nothing to sweep along.")

    fixed_param = "field" if swept_t else "temperature"
    fixed_value, fixed_span = _fixed_value_and_span(measurement, fixed_param)
    held_field_oe = fixed_value if fixed_param == "field" else None
    if fixed_param == "field":
        fixed_value, fixed_span = fixed_value / OE_PER_TESLA, fixed_span / OE_PER_TESLA

    requested_methods = methods
    if requested_methods is not None and "fisher" in requested_methods:
        _check_fisher(measurement, swept_t, signals, held_field_oe)

    # Resolved only now, once the sweep is known to be searchable: a
    # warning about a point in a sweep that is then refused would be noise.
    if screen_error is not None:
        outlier_record = _unscreened_record(mode, outlier_threshold, screen_error)
        if mode == "warn":
            _warn_unscreened(source, screen_error)
    else:
        outlier_record = _outlier_record(
            source,
            report,
            mode,
            swept_t=swept_t,
            signals=signals,
            selected_freq=selected_freq,
            t_window_k=t_window_k,
            h_window_oe=h_window_oe,
            dropping=dropping,
        )
        if mode == "warn" and outlier_record["flagged"]:
            _warn_outliers(outlier_record, report, source, swept_t)

    cand_frames, det_frames, curves = [], [], {}
    per_signal_params = {}
    for name in signals:
        y_series = getattr(measurement, name)
        y_all = _measured_array(measurement, y_series)
        if y_all is None:
            continue
        err_series = getattr(measurement, _SIGNALS[name][0])
        err_all = _measured_array(measurement, err_series)

        x_arr, y_arr = x_all, y_all
        err_arr = err_all
        if freq_mask is not None and _SIGNALS[name][1] == "AC":
            x_arr, y_arr = x_arr[freq_mask], y_arr[freq_mask]
            if err_arr is not None:
                err_arr = err_arr[freq_mask]
        if branch == "increasing" and not swept_t:
            idx = _increasing_branch_indices(x_arr)
            x_arr, y_arr = np.abs(x_arr)[idx], y_arr[idx]
            if err_arr is not None:
                err_arr = err_arr[idx]

        resolved_methods = (
            _default_methods(scan_kind, name) if requested_methods is None else requested_methods
        )
        result = find_transitions(
            x_arr,
            y_arr,
            methods=resolved_methods,
            x_name="T" if swept_t else "H",
            x_unit="K" if swept_t else "T",
            y_err=err_arr,
            smooth_window=smooth_window,
            polyorder=polyorder,
            prominence_fraction=prominence_fraction,
            criterion=criterion,
            label=f"{measurement.path} [{name}]",
            noise=noise,
        )
        loc = "T_K" if swept_t else "H_T"
        # Returned frames avoid pandas' deprecated inplace keyword and leave
        # the array-level detection result untouched.
        rename_columns = {f"{loc}_err": "loc_err"}
        cands = result.candidates.rename(columns=rename_columns)
        dets = result.detections.rename(columns=rename_columns)
        for frame in (cands, dets):
            frame["signal"] = name
            frame["measurement_file"] = str(measurement.path)
            # One call, one sweep; detect_transitions_all renumbers by input
            # position.
            frame["sweep_id"] = 0
        fixed_loc = "H_T" if swept_t else "T_K"
        cands[fixed_loc] = fixed_value
        dets[fixed_loc] = fixed_value
        cands["T_err_K"] = cands["loc_err"] if swept_t else np.nan
        cands["H_err_T"] = np.nan if swept_t else cands["loc_err"]
        cands["T_span_K"] = np.nan if swept_t else fixed_span
        cands["H_span_T"] = fixed_span if swept_t else np.nan
        cands["swept_axis"] = "T" if swept_t else "H"
        cands["frequency_hz"] = selected_freq if _SIGNALS[name][1] == "AC" else np.nan
        cand_frames.append(cands)
        det_frames.append(dets)
        curves[(str(measurement.path), name)] = result.curves[("", "")]
        per_signal_params[name] = {
            k: result.params[k]
            for k in (
                "methods",
                "smooth_window",
                # The point count the window resolved to on this sweep's grid.
                # Two requested windows can clamp to one count (see
                # _window_points), and only the count shows it.
                "smooth_points",
                "noise_sigma",
                "noise_source",
                "noise_sigma_instrument",
                "noise_sigma_scatter",
                "noise_ratio",
                "noise",
                "n_duplicates_dropped",
            )
        }
        if "fisher_noise" in result.params:
            per_signal_params[name]["fisher_noise"] = dict(result.params["fisher_noise"])
        if name == "moment_emu":
            # Series names survive the accessor's unit rescaling. Read the
            # actual consumed columns rather than resolving them a second time.
            per_signal_params[name]["moment_source"] = {
                **_moment_column_record(y_series),
                "error": None if err_series is None else _moment_column_record(err_series),
            }

    if cand_frames:
        candidates = pd.concat(cand_frames, ignore_index=True)
        detections = pd.concat(det_frames, ignore_index=True)
        # Re-key candidate ids uniquely across signals.
        candidates["candidate_id"] = range(len(candidates))
        id_map = {}
        offset = 0
        for frame in cand_frames:
            for old in frame["candidate_id"]:
                id_map[(frame["signal"].iloc[0], old)] = offset
                offset += 1
        detections["candidate_id"] = [
            id_map[(s, c)] for s, c in zip(detections["signal"], detections["candidate_id"])
        ]
        if top is not None:
            keep_ids: list[int] = []
            for _, grp in candidates.groupby(["signal", "method"], sort=False):
                keep_ids += list(
                    grp.sort_values("significance", ascending=False)["candidate_id"][:top]
                )
            candidates = candidates[candidates["candidate_id"].isin(keep_ids)]
            detections = detections[detections["candidate_id"].isin(keep_ids)]
        candidates = candidates.reindex(columns=CANDIDATE_COLUMNS).reset_index(drop=True)
        detections = detections.reindex(columns=DETECTION_COLUMNS).reset_index(drop=True)
    else:
        candidates = pd.DataFrame(columns=CANDIDATE_COLUMNS)
        detections = pd.DataFrame(columns=DETECTION_COLUMNS)

    params = {
        "scan_kind": scan_kind,
        "signals": signals,
        "criterion_priority": (
            (criterion, *[m for m in CRITERION_PRIORITY if m != criterion])
            if criterion
            else CRITERION_PRIORITY
        ),
        "frequency_hz": selected_freq,
        "branch": branch,
        "polyorder": polyorder,
        "prominence_fraction": prominence_fraction,
        "top": top,
        "t_window_k": t_window_k,
        "h_window_oe": h_window_oe,
        "per_signal": per_signal_params,
        "outliers": outlier_record,
        "calamari_version": _calamari_version(),
    }
    return TransitionResult(
        candidates=candidates, detections=detections, params=params, curves=curves
    )


def _resolve_batch_smooth_window(
    smooth_window: float | Callable[[Measurement], float | None] | None,
    measurement: Measurement,
) -> float | None:
    """Per-sweep ``smooth_window`` for one batch entry.

    A plain value (``None`` or a float) passes straight through -- that is
    the pre-existing, whole-batch behaviour. A callable is invoked with
    the sweep's own ``Measurement`` so a campaign whose sweeps do not all
    share one point density can choose a window per sweep when sampling,
    noise or physical feature widths justify it. Validated here rather than left to :func:`_window_points` so a
    caller's broken callable is reported as *its* fault, not smuggled in
    as an opaque "smooth_window must be positive" from three calls away.
    """
    if not callable(smooth_window):
        return smooth_window
    name = _describe_callable(smooth_window)
    try:
        resolved = smooth_window(measurement)
    except Exception as exc:
        raise DetectionInputError(f"smooth_window callable {name!r} raised {exc!r}") from exc
    if resolved is None:
        return None
    if isinstance(resolved, bool) or not isinstance(resolved, (int, float)):
        raise DetectionInputError(
            f"smooth_window callable {name!r} returned {resolved!r}, which is not a "
            "float or None"
        )
    resolved = float(resolved)
    if not math.isfinite(resolved) or resolved <= 0:
        raise DetectionInputError(
            f"smooth_window callable {name!r} returned {resolved!r}, which is not a "
            "positive, finite window"
        )
    return resolved


def detect_transitions_all(
    measurements: Iterable,
    *,
    link: dict | None = None,
    robust: dict | None = None,
    smooth_window: float | Callable[[Measurement], float | None] | None = None,
    **kwargs,
) -> TransitionResult:
    """Detect over many sweeps, collecting failures instead of hiding them.

    Accepts Measurements and/or the ``(T_K, Measurement)`` pairs that
    :func:`calamari.sweeps.split_isotherms` returns (the pair's
    measurement is used; its own columns carry the coordinates). A sweep
    that is refused (mixed scan kind, drifting fixed parameter,
    multi-frequency without ``frequency=``, ...) or fails lands in
    ``skipped`` as ``(path, reason)`` -- an empty table with a populated
    ``skipped`` is "everything failed", not "no transitions".

    Every row carries ``sweep_id``, the sweep's position in
    ``measurements``, counted over every entry including the skipped ones:
    an id therefore names the same sweep however many others fail, and the
    gaps are listed in ``params['skipped_sweep_ids']`` (parallel to
    ``skipped``). ``measurement_file`` cannot do this job, because the
    isotherms :func:`~calamari.sweeps.split_isotherms` cuts from one file
    all carry that file's path. Each ``params['per_sweep']`` entry records
    its ``sweep_id`` too, and ``curves`` is keyed by ``(sweep_id, signal)``
    for the same reason; read a sweep's curves with
    :meth:`TransitionResult.curve`.

    Adds a ``probe`` column (``'AC'``/``'DC'``, from each row's signal).
    ``link=`` and ``robust=`` (mappings of :meth:`TransitionResult.link`
    and :meth:`TransitionResult.robust` keyword arguments) apply those
    steps in the same call for pipeline ergonomics; the settings are
    recorded identically either way. The order is link *then* robust,
    which is the only useful one: gating on a chain statistic requires the
    chains to exist, and this is the entry point where a whole family of
    sweeps is in hand -- the only place cross-sweep recurrence is visible
    at all. ``**kwargs`` forward to :func:`detect_transitions`.

    Parameters
    ----------
    smooth_window : None, float, or callable, optional
        A single physical-units window applied to every sweep -- the
        historical behaviour -- may need adjustment for sweeps with different
        sampling densities, noise or physical feature widths. Pass a callable ``f(measurement) -> float
        | None`` instead and it is invoked per sweep (the pair's
        ``Measurement`` for a ``split_isotherms`` entry), so a caller can
        key the window off e.g. the sweep's own point count or spacing.
        An exception from the callable, or a return value that is not a
        positive finite number or ``None``, is treated as that sweep's
        failure and recorded in ``skipped`` (never raised out of the
        batch, and never silently substituted). What was actually passed
        to :func:`detect_transitions` for every searched sweep is
        recorded in ``params['per_sweep']``, in ``results`` order, so the
        resolved windows remain available without retaining the callable.
        Each entry's ``per_signal`` records include any resolved moment source;
        original inputs and external preprocessing must still be retained.
    **kwargs
        Forwarded to :func:`detect_transitions`. Shared configuration is
        checked before consuming any sweeps, including an empty batch.
        Invalid methods, noise, smoothing or selection settings raise rather
        than skipping every sweep. Data-dependent refusals remain in ``skipped``;
        each sweep is then screened on its own (so ``'warn'`` warns once per sweep
        with a flagged point in its search), and each sweep's
        ``params['outliers']`` is copied into its ``per_sweep`` entry --
        where ``screened`` shows which sweeps went through unscreened
        because ``quality()`` could not score them. Under ``'drop'`` such a
        sweep is refused into ``skipped`` instead. A
        ``QualityReport`` describes one file, so in a batch it is refused
        -- into ``skipped`` -- by every sweep it does not describe; pass a
        mode instead.
    """
    # Binding also rejects misspelt keywords for an empty batch, before consuming inputs.
    config = inspect.signature(detect_transitions).bind(None, **kwargs)
    config.apply_defaults()
    settings = config.arguments
    methods = _check_detection_options(
        settings["methods"], settings["criterion"],
        None if callable(smooth_window) else smooth_window,
        settings["polyorder"], settings["prominence_fraction"], settings["noise"],
    )
    if "methods" in kwargs:
        kwargs = {**kwargs, "methods": methods}
    _check_measurement_options(
        settings["signal"], settings["frequency"], settings["branch"], settings["top"],
        settings["t_window_k"], settings["h_window_oe"],
    )
    _check_outlier_args(
        kwargs.get("outliers", "warn"), kwargs.get("outlier_threshold"),
        caller="detect_transitions",
    )
    results: list[TransitionResult] = []
    sweep_ids: list[int] = []
    skipped: list[tuple[str, str]] = []
    skipped_sweep_ids: list[int] = []
    per_sweep: list[dict] = []
    # sweep_id is the input position, counted over EVERY entry, skipped ones
    # included: numbering only the survivors would renumber every later sweep
    # whenever an earlier one fails, so an id recorded yesterday (in a
    # select() window, a notebook, a figure caption) would name a different
    # sweep today.
    for sweep_id, entry in enumerate(measurements):
        m = entry[1] if isinstance(entry, tuple) and len(entry) == 2 else entry
        try:
            resolved_window = _resolve_batch_smooth_window(smooth_window, m)
            result = detect_transitions(m, smooth_window=resolved_window, **kwargs)
        except (CalamariError, ValueError) as exc:
            skipped.append((str(getattr(m, "path", m)), str(exc)))
            skipped_sweep_ids.append(sweep_id)
            continue
        results.append(result)
        sweep_ids.append(sweep_id)
        per_sweep.append(
            {
                "sweep_id": sweep_id,
                "measurement_file": str(getattr(m, "path", m)),
                "smooth_window": resolved_window,
                "per_signal": {k: dict(v) for k, v in result.params["per_signal"].items()},
                "outliers": {
                    **result.params["outliers"],
                    "flagged": [dict(p) for p in result.params["outliers"]["flagged"]],
                },
            }
        )

    # Re-key ids uniquely by offsetting each sweep's block positionally.
    # Never key the mapping on measurement_file: split_isotherms pairs are
    # sub-Measurements of ONE file, so several sweeps legitimately share a
    # path while each restarts candidate_id at 0 -- a path-keyed map would
    # silently attach their detections to the wrong candidate. The same
    # fact is why every row also carries sweep_id: it is the only column
    # that tells those sweeps apart.
    cand_frames, det_frames = [], []
    offset = 0
    for sweep_id, r in zip(sweep_ids, results, strict=True):
        if not len(r.candidates):
            continue
        cands, dets = r.candidates.copy(), r.detections.copy()
        cands["candidate_id"] += offset
        dets["candidate_id"] += offset
        cands["sweep_id"] = sweep_id
        dets["sweep_id"] = sweep_id
        # Advance past the block's LARGEST id, not its length: a top=-capped
        # result keeps its original (now sparse) ids, so len() could collide.
        offset = int(cands["candidate_id"].max()) + 1
        cand_frames.append(cands)
        det_frames.append(dets)
    # sweep_id distinguishes curves from sweeps that share a file path.
    curves: dict = {}
    for sweep_id, r in zip(sweep_ids, results, strict=True):
        for (_, signal), curve in r.curves.items():
            curves[(sweep_id, signal)] = curve

    if cand_frames:
        candidates = pd.concat(cand_frames, ignore_index=True)
        detections = pd.concat(det_frames, ignore_index=True)
    else:
        candidates = pd.DataFrame(columns=CANDIDATE_COLUMNS)
        detections = pd.DataFrame(columns=DETECTION_COLUMNS)
    candidates["probe"] = candidates["signal"].map(
        lambda s: _SIGNALS.get(s, (None, ""))[1]
    )

    # Record the callable itself as its name (or repr, for a lambda / partial
    # / other unnamed callable): the object is not printable/reproducible in
    # params, but its identity is worth keeping for a human reading the batch
    # recipe back.
    smooth_window_repr = (
        _describe_callable(smooth_window)
        if callable(smooth_window)
        else smooth_window
    )
    params = {
        "n_sweeps": len(results) + len(skipped),
        "n_searched": len(results),
        "n_with_candidates": sum(1 for r in results if len(r.candidates)),
        "detect_kwargs": {**kwargs, "smooth_window": smooth_window_repr},
        "per_sweep": per_sweep,
        # Parallel to ``skipped`` (same order), kept out of its (path, reason)
        # pairs so code unpacking those two-tuples keeps working.
        "skipped_sweep_ids": skipped_sweep_ids,
        "calamari_version": _calamari_version(),
    }
    result = TransitionResult(
        candidates=candidates.reset_index(drop=True),
        detections=detections.reset_index(drop=True),
        params=params,
        skipped=tuple(skipped),
        curves=curves,
    )
    if link is not None:
        result = result.link(**link)
    if robust is not None:
        result = result.robust(**robust)
    return result
