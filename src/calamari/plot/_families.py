"""Family composition, presentation shifts and screening."""

from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from calamari._plot_data import (
    _FAMILY_AUTO,
    _FAMILY_PARAMS,
    _FIELD_AXIS_UNITS,
    PlotSeries,
    _branch_slices,
    _family_value,
    _resolve_chi_source,
)
from calamari.exceptions import CalamariError, PlotInputError, QualityInputError
from calamari.plot._curves import _Y_TRANSFORM_KW, plot_chi_vs_h, plot_chi_vs_t, plot_mh, plot_mt
from calamari.quality import (
    _OUTLIER_WARN_LIST,
    CONTINUITY_THRESHOLD,
    _check_outlier_args,
    _check_report_matches,
    _plain_label,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from calamari.measurement import Measurement
    from calamari.quality import QualityReport

# ---------------------------------------------------------------------------
# Waterfalls: plot_family's offset= / baseline=. A display-only y-transform
# applied to the resolved series of each measurement, between resolution and
# drawing, by the module's own helpers.
# ---------------------------------------------------------------------------

#: The helpers that honour the private y-transform (they pop
#: :data:`_Y_TRANSFORM_KW` from ``**style`` and apply it per baseline
#: group). ``plot_family`` refuses ``offset=``/``baseline=`` for any other
#: ``plot`` callable: a foreign function would forward the keyword to
#: matplotlib, or silently draw the curves unshifted.
_Y_TRANSFORM_HELPERS = frozenset({plot_chi_vs_t, plot_chi_vs_h, plot_mh, plot_mt})

#: ``baseline=`` modes spelled by name. The other two forms are a float
#: ``x0`` (value interpolated there) and an ``(x_lo, x_hi)`` window (mean).
_BASELINE_MODES = ("start", "min")


def _is_real_number(value: Any) -> bool:
    """True for an int/float (numpy scalars included), False for a bool.

    ``bool`` subclasses ``int``, so ``baseline=True`` would otherwise pass
    as ``x0 = 1`` -- a typo turned into a silently wrong figure.
    """
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _check_family_shift(offset: Any, baseline: Any) -> None:
    """Validate the shape of ``offset=``/``baseline=`` before anything is drawn.

    Only the form is checked here; whether an ``x0`` lies on a curve, or a
    window holds any points, depends on each measurement and is checked
    when that measurement's series are resolved.
    """
    if offset is not None and not (_is_real_number(offset) and np.isfinite(offset)):
        raise PlotInputError(
            f"offset must be None or a finite number (a step in the plotted y "
            f"unit), got {offset!r}."
        )
    if baseline is None:
        return
    if isinstance(baseline, str):
        if baseline in _BASELINE_MODES:
            return
        raise PlotInputError(
            f"baseline={baseline!r} is not a known mode; expected 'start', 'min', "
            "a number x0 (the value interpolated at x = x0), or an (x_lo, x_hi) "
            "window (the mean over it)."
        )
    if _is_real_number(baseline):
        if not np.isfinite(baseline):
            raise PlotInputError(f"baseline x0 must be finite, got {baseline!r}.")
        return
    if (
        isinstance(baseline, (tuple, list))
        and len(baseline) == 2
        and all(_is_real_number(v) for v in baseline)
    ):
        lo, hi = baseline
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
            raise PlotInputError(
                f"baseline window must be (x_lo, x_hi) with finite x_lo < x_hi, "
                f"got {tuple(baseline)!r}."
            )
        return
    raise PlotInputError(
        f"baseline must be None, 'start', 'min', a number x0, or an (x_lo, x_hi) "
        f"window, got {baseline!r}."
    )


def _pooled_points(
    group: list[PlotSeries],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Every plotted point of one baseline group, each point once.

    :func:`moment_series` repeats each turning point at the end of one
    branch and the start of the next, so split branches draw as a connected
    loop. Pooling them naively would count every turning point twice and
    make a window-mean baseline depend on ``split_branches`` -- a display
    choice. The leading point of a series is dropped when it is exactly the
    previous series' last point, which is that duplication and nothing else.
    """
    xs: list[NDArray[np.float64]] = []
    ys: list[NDArray[np.float64]] = []
    previous = None
    for series in group:
        x, y = series.x, series.y
        if (
            previous is not None
            and len(x)
            and x[0] == previous.x[-1]
            and y[0] == previous.y[-1]
        ):
            x, y = x[1:], y[1:]
        xs.append(x)
        ys.append(y)
        if len(series.x):
            previous = series
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys)


def _baseline_value(group: list[PlotSeries], baseline: Any, path: Any) -> float:
    """The one baseline for a group of series, in the plotted y unit.

    Computed from the *resolved* values -- after ``units=``, ``sample=``,
    ``inverse=`` and ``imag_scale`` -- because the baseline is subtracted
    from what is drawn and must be in the same unit.

    ``x0`` interpolation is done along each monotonic segment that spans
    ``x0`` and the results averaged. A hysteresis loop visits most fields
    twice; interpolating the pooled, x-sorted points would mix the branches
    arbitrarily. On a plain sweep there is one segment and this is ordinary
    linear interpolation.
    """
    x, y = _pooled_points(group)
    xlabel = group[0].xlabel if group else "x"
    if len(x) == 0:
        raise PlotInputError(f"{path}: no plotted points to take a baseline from.")
    span = f"this curve spans {x.min():g} to {x.max():g} on the x-axis, {xlabel}"
    if baseline == "start":
        return float(y[x == x.min()].mean())
    if baseline == "min":
        return float(y.min())
    if isinstance(baseline, (tuple, list)):
        lo, hi = baseline
        inside = (x >= lo) & (x <= hi)
        if not inside.any():
            raise PlotInputError(
                f"{path}: the baseline window ({lo:g}, {hi:g}) contains no plotted "
                f"point -- {span}. Choose a window inside the curve."
            )
        return float(y[inside].mean())
    x0 = float(baseline)
    if not x.min() <= x0 <= x.max():
        raise PlotInputError(
            f"{path}: baseline x0 = {x0:g} lies outside the curve -- {span}. "
            "Interpolation cannot supply a value there, and extrapolation would "
            "invent one."
        )
    values = []
    for series in group:
        for sl in _branch_slices(series.x):
            xs, ys = series.x[sl], series.y[sl]
            if len(xs) and xs.min() <= x0 <= xs.max():
                order = np.argsort(xs, kind="stable")
                values.append(float(np.interp(x0, xs[order], ys[order])))
    if not values:
        raise PlotInputError(
            f"{path}: baseline x0 = {x0:g} falls in a gap between this curve's "
            f"series -- no single series spans it ({span})."
        )
    return float(np.mean(values))


def _shift_group(
    group: list[PlotSeries], *, baseline: Any, shift: float, path: Any
) -> list[PlotSeries]:
    """Subtract one pooled baseline from a group and raise it by ``shift``.

    ``yerr`` is left alone: the baseline is treated as exact, and a
    constant shift does not change a point's own uncertainty.
    """
    b = 0.0 if baseline is None else _baseline_value(group, baseline, path)
    return [replace(series, y=series.y - b + shift) for series in group]


# ---------------------------------------------------------------------------
# Family screening: plot_family's outliers= / outlier_threshold=. The same
# contract as detect_transitions (calamari.quality holds the shared checks),
# applied to each measurement before its curve is drawn.
# ---------------------------------------------------------------------------


def _external_stacklevel() -> int:
    """``stacklevel`` that points a warning at the first frame outside this package.

    Called by the function that warns. The family aliases add a frame
    between the caller and :func:`plot_family`, so a fixed ``stacklevel``
    would blame the alias on one route in and the caller on the other.
    Counting frames until the stack leaves ``calamari.plot`` names the
    caller's own line either way.
    """
    frame = sys._getframe(1)  # the function about to warn
    level = 1
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if module != "calamari.plot" and not module.startswith("calamari.plot."):
            break
        frame = frame.f_back
        level += 1
    return level


def _family_drawn_signals(
    measurement: Measurement, plot: Callable[..., Axes], kwargs: dict
) -> frozenset[str] | None:
    """The report signals one family member's curve draws; None means "any".

    A :class:`~calamari.quality.QualityReport` row names the accessor it
    scored: ``'chi_real'`` on AC rows, ``'moment_emu'`` on DC and VSM rows.
    A chi' plot of a combined VSM+AC file never draws the VSM rows (chi' is
    NaN there), so a glitch in the moment is no reason to warn about -- or
    drop anything from -- that figure. For this module's helpers the drawn
    accessor follows from the helper and its keywords, resolved by the rule
    :func:`chi_series` itself uses (:func:`_resolve_chi_source`), so the two
    cannot disagree. chi'' is not scored, so a chi'' curve is screened by the
    chi' flag of the same row. A foreign ``plot`` callable could draw
    anything: None, and every flagged point counts.
    """
    if plot in (plot_mh, plot_mt):
        return frozenset({"moment_emu"})
    if plot not in (plot_chi_vs_t, plot_chi_vs_h):
        return None
    component = kwargs.get("component", "real")
    source = kwargs.get("source", "auto")
    if component not in ("real", "imag", "both") or source not in ("auto", "ac", "dc"):
        return None  # the plot call refuses it by name, so nothing is drawn
    if component == "both":  # always AC -- see _plot_chi
        return frozenset({"chi_real"})
    resolved = _resolve_chi_source(measurement, component, source)
    return frozenset({"chi_real" if resolved == "ac" else "moment_emu"})


def _family_x_axis(measurement: Measurement, plot: Callable[..., Axes]) -> str | None:
    """The swept coordinate a member's curve is drawn against, for the warning.

    Known for this module's helpers; for a foreign callable, read from the
    sweep's ``scan_kind``, and None when that names no single axis (the
    warning then gives both T and H).
    """
    if plot in (plot_chi_vs_t, plot_mt):
        return "temperature"
    if plot in (plot_chi_vs_h, plot_mh):
        return "field"
    kind = measurement._safe(lambda: measurement.scan_kind)
    return {"T-scan": "temperature", "H-scan": "field"}.get(kind)


def _family_point_locations(
    measurement: Measurement, labels: Any, x: str | None, field_unit: str
) -> list[str]:
    """``'T = 5 K'`` / ``'H = 2.5 T'`` for each row label, in the plotted unit."""

    def values(series: Any) -> NDArray[np.float64]:
        if series is None:
            return np.full(len(labels), np.nan)
        return series.reindex(labels).to_numpy(dtype=float)

    columns = []
    if x != "field":
        columns.append(("T", values(measurement.temperature), "K"))
    if x != "temperature":
        accessor, _ = _FIELD_AXIS_UNITS[field_unit]
        unit = "Oe" if field_unit == "oe" else "T"
        columns.append(("H", values(getattr(measurement, accessor)), unit))
    return [
        ", ".join(f"{symbol} = {v[i]:g} {unit}" for symbol, v, unit in columns)
        for i in range(len(labels))
    ]


def _screen_family(
    loaded: list[Measurement],
    plot: Callable[..., Axes],
    kwargs: dict,
    outliers: Any,
    outlier_threshold: float | None,
) -> tuple[list[Measurement], list[list[dict]], list[Exception | None]]:
    """Screen every member of a family before any curve is drawn.

    Returns ``(views, flagged, errors)``, each aligned with ``loaded``: the
    measurement to draw (the input itself, or a ``restrict``-ed copy without
    its dropped rows); the flagged points a ``'warn'`` lists for it; and the
    error ``quality()`` raised on it, or None. Aligned with the input, not
    the draw order, so that sorting cannot pair a curve with another's.

    Each member is scored as passed -- the whole view, before the plot
    resolves anything -- so a point is judged against its own neighbours in
    the file, as :func:`~calamari.transitions.detect_transitions` does. A
    member that is itself a view (``restrict()``, or an isotherm cut by
    :func:`~calamari.sweeps.split_isotherms`, which shares its file's path)
    is scored as that view, and a supplied report must describe that same
    view: :func:`calamari.quality._check_report_matches` refuses one that
    does not.
    """
    views: list[Measurement] = list(loaded)
    flagged: list[list[dict]] = [[] for _ in loaded]
    errors: list[Exception | None] = [None] * len(loaded)
    if isinstance(outliers, str) and outliers == "ignore":
        return views, flagged, errors  # no quality() pass: ignoring costs nothing
    given = not isinstance(outliers, str)
    dropping = given or outliers == "drop"
    field_unit = kwargs.get("field_unit", "oe")
    if field_unit not in _FIELD_AXIS_UNITS:
        field_unit = "oe"  # the plot call refuses it by name before drawing
    for i, m in enumerate(loaded):
        if given:
            report = outliers[i]
            try:
                _check_report_matches(report, m)
            except QualityInputError as exc:
                raise PlotInputError(f"plot_family, outliers[{i}]: {exc}") from exc
        else:
            try:
                report = m.quality(threshold=outlier_threshold)
            except (CalamariError, ValueError) as exc:
                # quality() reads columns no plot touches (the Min./Max.
                # spreads, fit quality, status codes), so it can refuse a file
                # whose drawn channel is fine. Only a caller who asked for
                # removal is refused: a drop that cannot be honoured must not
                # pass silently as a cleaned curve.
                if dropping:
                    raise PlotInputError(
                        f"{m.path}: outliers='drop' asked for flagged points to be "
                        "removed, but the curve could not be screened -- "
                        f"Measurement.quality() raised {type(exc).__name__}: {exc} "
                        "Pass outliers='ignore' to draw without screening, or screen "
                        "the measurement yourself and pass the rows to keep with "
                        "measurement.restrict(mask=...)."
                    ) from exc
                errors[i] = exc
                continue
        points = report.flagged
        if len(points):
            signals = _family_drawn_signals(m, plot, kwargs)
            if signals is not None:
                points = points[points["signal"].isin(sorted(signals))]
        if not len(points):
            continue
        if dropping:
            views[i] = m.restrict(mask=~m.data.index.isin(points.index))
            continue
        name = getattr(m.path, "name", None) or str(m.path)
        where = _family_point_locations(m, points.index, _family_x_axis(m, plot), field_unit)
        flagged[i] = [
            {
                "file": name,
                "row": _plain_label(label),
                "at": at,
                "signal": signal,
                "value": float(value),
                "score": float(score),
            }
            for label, at, signal, value, score in zip(
                points.index, where, points["signal"], points["value"],
                points["continuity_score"], strict=True,
            )
        ]
    return views, flagged, errors


def _warn_family_outliers(points: list[dict], threshold: float, baseline: Any) -> None:
    """The ``outliers='warn'`` message: one per family, in the maps' style.

    Names each point by file, swept coordinate (in the plotted unit), the
    accessor scored and its value, and its score, then the three ways out,
    so the warning is actionable where it is read. With ``baseline=`` it
    says why a flagged point matters more than its own position: it can
    *set* the baseline, and so move every point of its curve.
    """
    lines = [
        f"  {p['file']}: {p['signal']} {p['value']:.4g} at {p['at']} (row {p['row']}), "
        f"score {p['score']:.0f}"
        for p in points[:_OUTLIER_WARN_LIST]
    ]
    if len(points) > _OUTLIER_WARN_LIST:
        lines.append(f"  ... and {len(points) - _OUTLIER_WARN_LIST} more")
    n = len(points)
    consequence = "Kept as measured, so it is drawn like any other point"
    if baseline is not None:
        consequence += (
            f", and a flagged point can set the baseline (baseline={baseline!r}), "
            "shifting every point of its curve"
        )
    warnings.warn(
        f"plot_family: {n} point{'s' if n > 1 else ''} drawn depart"
        f"{'' if n > 1 else 's'} from the trend of the neighbouring points by far "
        "more than those neighbours stray from it themselves (continuity score > "
        f"{threshold:g}, Measurement.quality()):\n" + "\n".join(lines) + "\n"
        f"{consequence}. Pass outliers='drop' to exclude it, outliers='ignore' to keep "
        "it without this warning, or outlier_threshold=<score> to move the line; "
        "m.quality() lists the flagged points with their evidence.",
        stacklevel=_external_stacklevel(),
    )


def _warn_family_unscreened(unscreened: list[tuple[Measurement, Exception]]) -> None:
    """The ``outliers='warn'`` message for members ``quality()`` could not score.

    Silence would read as "screened, nothing flagged"; the warning says which
    curves were drawn unscreened, why, and how to accept that knowingly.
    """
    lines = [
        f"  {getattr(m.path, 'name', None) or m.path}: {type(exc).__name__}: {exc}"
        for m, exc in unscreened[:_OUTLIER_WARN_LIST]
    ]
    if len(unscreened) > _OUTLIER_WARN_LIST:
        lines.append(f"  ... and {len(unscreened) - _OUTLIER_WARN_LIST} more")
    n = len(unscreened)
    warnings.warn(
        f"plot_family: {n} curve{'s' if n > 1 else ''} not screened for bad points -- "
        f"Measurement.quality() could not score {'them' if n > 1 else 'it'}:\n"
        + "\n".join(lines) + "\n"
        "Drawn as measured. Pass outliers='ignore' to accept that without this warning.",
        stacklevel=_external_stacklevel(),
    )


def plot_family(
    measurements: Iterable[Any],
    plot: Callable[..., Axes],
    *,
    ax: Axes | None = None,
    label_by: str = "auto",
    labels: Iterable[str] | None = None,
    sort: bool = True,
    cmap: Any | None = None,
    offset: float | None = None,
    baseline: str | float | tuple[float, float] | None = None,
    outliers: str | Sequence[QualityReport] = "warn",
    outlier_threshold: float | None = None,
    **kwargs: Any,
) -> Axes:
    """Overlay one curve per measurement, labelled by each sweep's fixed parameter.

    The engine behind :func:`plot_chi_vs_h_family` and friends -- use those
    aliases unless you are composing with your own plot function. Each
    measurement is drawn by one call to ``plot`` (any of the
    measurement-first helpers in this module) onto the same axes, with a
    legend label derived from the parameter that sweep held fixed:
    ``T = 5 K`` for a family of field sweeps, ``H = 3 T`` for a family of
    temperature sweeps, ``f = 997 Hz`` for ``label_by='frequency'``.

    The label is never a guess: the fixed value is the measured-row mean,
    and a sweep whose nominally-fixed parameter varies beyond calamari's
    held-quantity tolerance is refused by name rather than mislabelled
    (pass ``labels=`` to name such curves yourself). Curves draw in
    ascending order of the fixed value, so legend, draw order and colours
    are monotonic in the physics.

    **Waterfalls.** A family whose curves sit on top of one another --
    chi'(H) isotherms whose background drifts with temperature, say --
    hides the feature it was measured to show. ``baseline=`` subtracts a
    per-measurement baseline so the curves share a zero, and ``offset=``
    then stacks them, curve *i* in draw order raised by ``i * offset``.
    Both are **display-only**: no ``Measurement`` changes, and
    :func:`chi_series`/:func:`moment_series` still return the measured
    values. The y-label says so, gaining ``" (baseline-subtracted)"``
    and/or ``" (offset for clarity)"``, so a shifted figure cannot pass for
    raw data. The tick values then describe the shifted curves; hiding them
    (``ax.set_yticklabels([])``) is the caller's decision, per PL-1.

    The shift is applied where each helper resolves its series, before
    anything is drawn, not by moving artists afterwards. Autoscaling
    (``relim``) does not account for artist transforms, so offset curves
    would fall outside the view limits. An error-bar container is several
    artists (line, caps, bar collection), and moving them consistently
    after the fact is fragile. Shifting the arrays first means the axes,
    the limits and ``ax.lines`` all agree on what is drawn. The price is
    that only this module's helpers can take part: a foreign ``plot``
    callable has no resolution step to shift, and is refused.

    The baseline is computed per measurement from the *resolved* series,
    in the plotted unit (after ``units=``, ``sample=``, ``inverse=``,
    ``imag_scale=``). All the branches of one M(H) loop share one pooled
    baseline, and so do the frequency curves of a multi-frequency sweep, so
    their relative positions survive. With ``component='both'``, chi' and
    chi'' each get their own baseline. The offset is per measurement, so
    every curve of one sweep moves together. Error bars are the points'
    own and are not changed: the baseline is treated as exact, and its
    own uncertainty is not propagated.

    **Bad points.** One glitched reading is one wrong point on an
    ordinary plot, but a waterfall magnifies it: ``baseline='min'`` on a
    curve with one reading far below its neighbours subtracts that reading,
    and every clean point of the curve moves by the glitch's depth -- which
    can be several times the feature the figure exists to show. So each
    measurement is screened with :meth:`Measurement.quality()
    <calamari.measurement.Measurement.quality>` before its curve is drawn,
    under the same ``outliers=`` contract as
    :func:`~calamari.transitions.detect_transitions` and the maps: by
    default a flagged point is kept and named in one warning for the whole
    family. The warning, and the drop, cover only the points a curve draws:
    moment points for :func:`plot_mh`/:func:`plot_mt` and for a chi plot
    whose source resolves to DC ``M/H``, chi' points for an AC chi plot --
    so a glitch in the VSM rows of a combined VSM+AC file does not touch a
    chi' figure. A foreign ``plot`` callable could draw any column, so
    there every flagged point counts. Screening happens on the
    ``Measurement`` before ``plot`` is called, so -- unlike ``offset=`` and
    ``baseline=`` -- it works with a foreign callable too. The return value
    stays the axes: the record of what was flagged is ``m.quality()``
    itself, which lists every flagged point with its evidence.

    Parameters
    ----------
    measurements : iterable of Measurement or path
        One sweep per entry; paths are loaded with
        :func:`calamari.load <calamari.io.load>`.
    plot : callable
        A measurement-first plot function from this module
        (:func:`plot_chi_vs_h`, :func:`plot_mh`, ...). Called as
        ``plot(m, ax=ax, label=..., **kwargs)``.
    ax : matplotlib.axes.Axes, optional
        Draw into these axes, or omit to create a figure.
    label_by : {'auto', 'temperature', 'field', 'frequency'}, optional
        Which fixed parameter labels the curves. ``'auto'`` reads each
        sweep's ``scan_kind`` (an H-scan is labelled by its temperature, a
        T-scan by its field) and requires every sweep to agree; it never
        picks ``'frequency'``, which must be asked for.
    labels : iterable of str, optional
        Explicit legend labels, one per measurement, used verbatim in the
        given order (``label_by`` and ``sort`` are ignored) -- the escape
        hatch for families distinguished by something other than T/H/f.
    sort : bool, optional
        Draw in ascending order of the fixed value. Default True.
    cmap : str or matplotlib colormap, optional
        Colour the curves sequentially by their fixed value (by position,
        when ``labels`` is given) instead of the property cycle -- the
        readable choice for a family of many curves. Incompatible with an
        explicit ``color=`` in ``**kwargs``.
    offset : float, optional
        Waterfall step in the plotted y unit. Curve *i* in draw order
        (ascending fixed value; the given order with ``labels=`` or
        ``sort=False``) is raised by ``i * offset``. A negative step stacks
        downwards. Default None (no offset).
    baseline : {'start', 'min'} or float or (float, float), optional
        Subtract one baseline per measurement before offsetting.
        ``'start'`` is the value at the curve's smallest x (the mean, if
        several points share it). ``'min'`` is the curve's minimum. A
        number ``x0`` is the value linearly interpolated at ``x = x0``, in
        the plotted x unit (K, Oe or T -- a field family drawn with
        ``field_unit='tesla'`` takes ``x0`` and windows in Tesla); it must
        lie on the curve. An ``(x_lo, x_hi)`` window is the mean of the
        points inside it (inclusive); it must contain at least one.
        Default None.
    outliers : {'warn', 'drop', 'ignore'} or sequence of QualityReport, optional
        What to do about points :meth:`Measurement.quality()
        <calamari.measurement.Measurement.quality>` flags as glitches.
        ``'warn'`` (default) draws them and warns once for the whole family,
        naming each point's file, swept coordinate (in the plotted unit:
        ``field_unit='tesla'`` gives H in T), scored value and score --
        and, with ``baseline=``, that a flagged point can set the baseline.
        ``'drop'`` removes them before drawing, ``'ignore'`` draws them
        without the warning and without screening at all (no ``quality()``
        pass). A sequence of reports, one per measurement and aligned with
        the measurements **in the order given** (before any sorting), is an
        explicit drop of *their* flagged points -- reports you have read,
        perhaps at your own threshold -- and each is refused unless it
        describes its measurement as passed (same ``path``, row labels it
        has, the same values at them), which catches a report taken before
        :meth:`~calamari.measurement.Measurement.restrict` or on another
        isotherm of the same file. Dropping is never the default: removing
        a measured point changes the figure, and that is the caller's call
        with the point in front of them.

        Each measurement is scored as passed -- the whole view -- so a
        point is judged against its neighbours in the file. Only flagged
        points the curve draws warn or are dropped (see *Bad points*
        above), and the drop is by row: a dropped chi' row takes its chi''
        with it, and chi'' -- which ``quality()`` does not score -- is
        screened only through chi'. When ``quality()`` cannot score a
        measurement (it reads columns no plot touches, such as the
        ``Min./Max.`` spreads, so an unrecognised unit there makes it
        raise), ``'warn'`` draws the curve as measured and says it was not
        screened, while ``'drop'`` raises, since the removal asked for
        cannot be honoured.
    outlier_threshold : float, optional
        Continuity score above which a point is flagged. ``None`` (default)
        means :data:`calamari.quality.CONTINUITY_THRESHOLD`, the line
        detection and the maps use, so a point one warns about is the point
        the others warn about. Not accepted with reports, which carry their
        own.
    **kwargs
        Forwarded to every ``plot`` call (``units=``, ``sample=``,
        ``source=``, ``inverse=``, ``field_unit=``, style kwargs, ...).

    Returns
    -------
    matplotlib.axes.Axes
        The shared axes. Call ``ax.legend()`` to show the labels.

    Warns
    -----
    UserWarning
        With ``outliers='warn'``: once, when a drawn point is flagged; and
        once, when some measurement could not be screened.

    Raises
    ------
    PlotInputError
        For an empty family; a sweep whose fixed parameter is not actually
        fixed (named, with its span); sweeps whose ``scan_kind`` disagree
        under ``'auto'``; a family whose labels all come out identical
        (a mislabelled family -- pick another ``label_by``); a ``labels``
        length mismatch; or ``cmap`` combined with ``color=``. For
        ``offset=``/``baseline=``: a malformed value, a ``plot`` callable
        that is not one of this module's helpers, an ``x0`` outside a curve,
        or a window containing none of a curve's points. The last two are
        found as each curve is resolved, so curves drawn before it stay on
        the axes; the failing curve itself draws nothing. Also for a report
        count mismatch, a report that does not describe its measurement, or
        ``'drop'`` on a measurement ``quality()`` cannot score; these are
        found before any curve is drawn.
    calamari.exceptions.QualityInputError
        Malformed ``outliers=``/``outlier_threshold=`` values, a single report
        instead of a sequence, invalid sequence entries, or reports combined
        with a threshold. Raised before drawing; also a ``ValueError``.
    """
    # Imported here, not at module top: calamari.io itself imports at
    # package import time, and plot must stay importable independent of it.
    from calamari.io import load

    loaded = [load(m) if isinstance(m, (str, Path)) else m for m in measurements]
    if not loaded:
        raise PlotInputError("plot_family requires at least one measurement.")
    if cmap is not None and "color" in kwargs:
        raise PlotInputError("Pass cmap= or color=, not both: they both set the colour.")
    _check_family_shift(offset, baseline)
    shifting = offset is not None or baseline is not None
    if shifting and plot not in _Y_TRANSFORM_HELPERS:
        name = getattr(plot, "__name__", repr(plot))
        raise PlotInputError(
            f"offset= and baseline= need one of calamari's own plot helpers "
            f"(plot_chi_vs_t, plot_chi_vs_h, plot_mh, plot_mt), not {name}: the "
            "shift is applied to the resolved series before drawing, and a "
            "foreign plot function has no such step. Shift your own arrays "
            "(chi_series/moment_series return them) before drawing instead."
        )
    _check_outlier_args(outliers, outlier_threshold, caller="plot_family", per_measurement=True)
    if not isinstance(outliers, str) and len(outliers) != len(loaded):
        raise PlotInputError(
            f"outliers has {len(outliers)} QualityReports for {len(loaded)} "
            "measurements; they must pair one-to-one, in the order the "
            "measurements are given."
        )

    # Entries pair each curve's input position with its label; sorting then
    # reorders the pairs, so anything aligned with the input (the reports,
    # the screened views) stays with its own measurement.
    if labels is not None:
        label_list = [str(lab) for lab in labels]
        if len(label_list) != len(loaded):
            raise PlotInputError(
                f"labels has {len(label_list)} entries for {len(loaded)} "
                "measurements; they must pair one-to-one."
            )
        entries = list(enumerate(label_list))
        positions = np.linspace(0.0, 1.0, len(entries)) if len(entries) > 1 else [0.5]
    else:
        if label_by == "auto":
            params = []
            for m in loaded:
                kind = m.scan_kind
                if kind not in _FAMILY_AUTO:
                    raise PlotInputError(
                        f"{m.path}: scan_kind is {kind!r}, which names no single "
                        "fixed parameter to label this curve by. Pass "
                        "label_by='temperature'/'field'/'frequency', or labels=[...]."
                    )
                params.append(_FAMILY_AUTO[kind])
            if len(set(params)) > 1:
                raise PlotInputError(
                    "The sweeps disagree about what is fixed (mixed T-scans and "
                    "H-scans); label_by='auto' cannot label them consistently. "
                    "Pass label_by=... or labels=[...]."
                )
            param = params[0]
        elif label_by in _FAMILY_PARAMS:
            param = label_by
        else:
            raise PlotInputError(
                f"label_by must be 'auto', 'temperature', 'field', or "
                f"'frequency', got {label_by!r}"
            )
        formatter = _FAMILY_PARAMS[param][3]
        values = [_family_value(m, param) for m in loaded]
        entries = [(i, formatter(v)) for i, v in enumerate(values)]
        if len(entries) > 1 and len({lab for _, lab in entries}) == 1:
            raise PlotInputError(
                f"Every curve in this family would be labelled {entries[0][1]!r} -- "
                f"the sweeps do not differ in {param}. Label by the parameter that "
                "actually distinguishes them (label_by=...), or pass labels=[...]."
            )
        if sort:
            order = np.argsort(values, kind="stable")
            entries = [entries[i] for i in order]
            values = [values[i] for i in order]
        lo, hi = min(values), max(values)
        positions = (
            [0.5] * len(values)
            if hi == lo
            else [(v - lo) / (hi - lo) for v in values]
        )

    colours = [None] * len(entries)
    if cmap is not None:
        import matplotlib as mpl

        colormap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
        colours = [colormap(p) for p in positions]

    # Screened after the labels (a refused label costs no quality() pass)
    # and before any curve is drawn (a refused report strands nothing).
    views, flagged, errors = _screen_family(loaded, plot, kwargs, outliers, outlier_threshold)

    step = 0.0 if offset is None else float(offset)
    for i, ((index, curve_label), colour) in enumerate(zip(entries, colours)):
        call_kwargs = dict(kwargs)
        if colour is not None:
            call_kwargs["color"] = colour
        if shifting:
            call_kwargs[_Y_TRANSFORM_KW] = partial(
                _shift_group, baseline=baseline, shift=i * step, path=loaded[index].path
            )
        ax = plot(views[index], ax=ax, label=curve_label, **call_kwargs)

    # Warned once the family is drawn -- a family refused part-way has an
    # error to report, and a warning about its points would be noise -- and
    # listed in draw order, the order of the legend.
    points = [p for index, _ in entries for p in flagged[index]]
    if points:
        _warn_family_outliers(
            points,
            CONTINUITY_THRESHOLD if outlier_threshold is None else outlier_threshold,
            baseline,
        )
    unscreened = [
        (loaded[index], errors[index]) for index, _ in entries if errors[index] is not None
    ]
    if unscreened:
        _warn_family_unscreened(unscreened)

    # A zero step, or a family of one, offsets nothing; saying otherwise on
    # the axis would be false.
    suffix = " (baseline-subtracted)" if baseline is not None else ""
    if step != 0.0 and len(entries) > 1:
        suffix += " (offset for clarity)"
    if suffix:
        ax.set_ylabel(ax.get_ylabel() + suffix)
    return ax


def plot_chi_vs_h_family(measurements: Iterable[Any], **kwargs: Any) -> Axes:
    """A family of chi(H) sweeps on one axes, each curve labelled by its temperature.

    ``plot_chi_vs_h_family(sweeps, units="molar", sample=sample)`` is
    :func:`plot_family` with :func:`plot_chi_vs_h`; every keyword of
    either is accepted, including ``field_unit='tesla'``, the waterfall's
    ``offset=`` and ``baseline=``, and the glitch screen's ``outliers=`` and
    ``outlier_threshold=`` (a flagged point warns by default). Call
    ``ax.legend()`` on the result.
    """
    return plot_family(measurements, plot_chi_vs_h, **kwargs)


def plot_chi_vs_t_family(measurements: Iterable[Any], **kwargs: Any) -> Axes:
    """A family of chi(T) sweeps on one axes, each curve labelled by its field.

    :func:`plot_family` with :func:`plot_chi_vs_t`; pass
    ``label_by='frequency'`` for a family distinguished by drive
    frequency instead of field.
    """
    return plot_family(measurements, plot_chi_vs_t, **kwargs)


def plot_mh_family(measurements: Iterable[Any], **kwargs: Any) -> Axes:
    """A family of M(H) sweeps on one axes, each curve labelled by its temperature.

    :func:`plot_family` with :func:`plot_mh` (so ``split_branches=``,
    ``field_unit=`` and the moment ``units=`` all pass through).
    """
    return plot_family(measurements, plot_mh, **kwargs)


def plot_mt_family(measurements: Iterable[Any], **kwargs: Any) -> Axes:
    """A family of M(T) sweeps on one axes, each curve labelled by its field.

    :func:`plot_family` with :func:`plot_mt`.
    """
    return plot_family(measurements, plot_mt, **kwargs)
