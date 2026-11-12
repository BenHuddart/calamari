"""Rendering of transition, smoothing and boundary results."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from calamari._plot_data import (
    _AXIS_LABELS,
    _CHI_SYMBOLS,
    _CHI_UNITS,
    _FIELD_AXIS_UNITS,
    _FISHER_SYMBOLS,
    PlotGrid,
    _check_field_unit,
    axis_labels,
)
from calamari.constants import OE_PER_TESLA
from calamari.exceptions import PlotInputError
from calamari.plot._curves import _make_ax

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from calamari.boundary import BoundaryFit
    from calamari.scales import SmoothingScanResult
    from calamari.transitions import TransitionResult


# ---------------------------------------------------------------------------
# Transition overlays: candidates from calamari.transitions drawn onto a 1-D
# sweep plot or a 2-D T-H map. The module boundary is the tidy candidates
# table -- this code consumes DataFrame columns and never imports detection
# machinery beyond the METHODS registry (marker strings and display names).
# ---------------------------------------------------------------------------

#: Display vocabulary for the ``signal`` column on figure legends.
_SIGNAL_LABELS = {
    "chi_real": r"$\chi'$",
    "chi_imag": r"$\chi''$",
    "moment_emu": r"$M$",
}

#: Fallback marker cycle for encoding columns with no registry backing
#: (``method`` takes its canonical markers from calamari.transitions.METHODS
#: so the same criterion has the same shape on every calamari figure).
_OVERLAY_MARKERS = ("o", "^", "s", "D", "v", "P", "X")

#: Colour cycle for ``color_by`` groups: matplotlib's default categorical
#: palette, referenced by name so a user style sheet can restyle it.
_OVERLAY_COLORS = tuple(f"C{i}" for i in range(10))

#: Above this many distinct ``color_by`` values the per-value legend
#: entries are suppressed. Tied to the palette length: past it the cycle
#: repeats, so a legend would be both unreadably long AND wrong (two
#: values sharing a colour). ``color_by='chain_id'`` on a real campaign
#: routinely produces hundreds.
_MAX_LEGEND_COLOR_VALUES = len(_OVERLAY_COLORS)

#: Line width, colour and transparency of the ``connect=`` chain lines.
#: Deliberately unobtrusive: the line asserts "these markers are the same
#: feature", and must never compete with the markers, which are the data.
_CONNECT_STYLE = {"linewidth": 0.8, "alpha": 0.6, "zorder": 1.5}

#: Neutral connector colour when there is no ``color_by`` to inherit.
_CONNECT_COLOR = "0.4"


def _transitions_frame(transitions) -> tuple[Any, dict | None]:
    """Normalise overlay input to (candidates DataFrame, robust thresholds).

    Accepts a TransitionResult (or anything with ``.candidates`` and
    ``.params``), a bare DataFrame, or a sequence of either. The robust
    thresholds -- printed on the legend so the figure states its own
    inclusion criterion -- survive only when every input carries the SAME
    thresholds; a bare DataFrame carries none, which is why filtering via
    ``.robust()`` beats a hand-rolled pandas filter for publication figures.
    """
    import pandas as pd

    if hasattr(transitions, "candidates"):
        params = getattr(transitions, "params", {}) or {}
        robust = params.get("robust")
        if "smoothing" in params:
            recipe = params["smoothing"]
            robust = {**(robust or {}), "smoothing_reference": {
                key: recipe[key] for key in (
                    "reference_window", "windows", "min_member_significance",
                    "track_tol_k", "track_tol_t", "smooth_points",
                )
            }}
        return transitions.candidates, robust
    if isinstance(transitions, pd.DataFrame):
        return transitions, None
    frames, thresholds = [], []
    for entry in transitions:
        frame, robust = _transitions_frame(entry)
        frames.append(frame)
        thresholds.append(robust)
    if not frames:
        raise PlotInputError("overlay_transitions: no transitions supplied.")
    same = thresholds[0] if all(t == thresholds[0] for t in thresholds) else None
    return pd.concat(frames, ignore_index=True), same


def _overlay_group_key(value):
    """Use one stable style key for NaN, None and pandas nullable values."""
    import pandas as pd

    return None if pd.isna(value) else value


def _overlay_group_mask(series, value):
    """Keep missing encodings drawable: equality alone never selects NaN."""
    selected = series.isna() if value is None else series.eq(value).fillna(False)
    return selected.to_numpy(dtype=bool)


def _overlay_style_maps(frame, marker_by, color_by, markers, colors):
    """Resolve the (marker value, colour value) -> style mappings.

    Deterministic ordering: METHODS registry order for the ``method``
    column, sorted order otherwise, so the same data always produces the
    same legend.
    """
    from calamari.transitions import METHODS

    def ordered(column):
        if column is None:
            return [None]
        values = list(dict.fromkeys(_overlay_group_key(v) for v in frame[column].unique()))
        if column == "method":
            keys = list(METHODS)
            return sorted(values, key=lambda v: (keys.index(v) if v in keys else len(keys), str(v)))
        return sorted(values, key=str)

    marker_values, color_values = ordered(marker_by), ordered(color_by)
    marker_map = {_overlay_group_key(k): v for k, v in (markers or {}).items()}
    for i, v in enumerate(marker_values):
        if v not in marker_map:
            if marker_by == "method" and v in METHODS:
                marker_map[v] = METHODS[v].marker
            else:
                marker_map[v] = _OVERLAY_MARKERS[i % len(_OVERLAY_MARKERS)]
    color_map = {_overlay_group_key(k): v for k, v in (colors or {}).items()}
    for i, v in enumerate(color_values):
        if v not in color_map:
            color_map[v] = _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)]
    return marker_values, color_values, marker_map, color_map


def _overlay_label(column: str | None, value) -> str:
    """Legend text for one encoding value, through the display vocabularies."""
    from calamari.transitions import METHODS

    if value is None:
        return "" if column is None else "(missing)"
    if column == "method" and value in METHODS:
        return METHODS[value].name.split(" (")[0]
    if column == "signal":
        return _SIGNAL_LABELS.get(value, str(value))
    return str(value)


def _overlay_group_label(marker_by, mv, color_by, cv, suppress_color, used) -> str | None:
    """Legend text for one (marker, colour) group, or None for no entry.

    Two suppressions, both so the legend stays readable: a ``color_by``
    column with more values than the palette has colours contributes no
    text at all (see :data:`_MAX_LEGEND_COLOR_VALUES`), and a label that
    has already been used is not repeated -- otherwise suppressing the
    colour half would emit one identical entry per colour group.
    """
    parts = [_overlay_label(marker_by, mv)]
    if not suppress_color:
        parts.append(_overlay_label(color_by, cv))
    label = " · ".join(t for t in parts if t)
    if not label or label in used:
        return None
    used.add(label)
    return label


def _overlay_legend_title(marker_by, color_by, robust) -> str:
    parts = []
    if marker_by is not None:
        parts.append(f"shape: {marker_by}")
    if color_by is not None:
        parts.append(f"colour: {color_by}")
    title = " · ".join(parts)
    if robust:
        gates = []
        if robust.get("min_significance") is not None:
            gates.append(f"significance ≥ {robust['min_significance']:g}")
        if robust.get("min_n_methods") is not None:
            gates.append(f"n_methods ≥ {robust['min_n_methods']:g}")
        if robust.get("min_sweep_range_over_noise") is not None:
            gates.append(f"range/noise ≥ {robust['min_sweep_range_over_noise']:g}")
        if robust.get("min_chain_significance") is not None:
            gates.append(f"chain significance ≥ {robust['min_chain_significance']:g}")
        if robust.get("min_chain_n") is not None:
            gates.append(f"chain length ≥ {robust['min_chain_n']:g}")
        if robust.get("exclude_edge"):
            gates.append("edges excluded")
        for key, label, unit in (
            ("min_smoothing_n", "smoothing support ≥", ""),
            ("min_smoothing_persistence", "smoothing fraction ≥", ""),
            ("min_smoothing_sign_consistency", "smoothing polarity ≥", ""),
            ("max_smoothing_spread_k", "smoothing spread ≤", " K"),
            ("max_smoothing_spread_t", "smoothing spread ≤", " T"),
        ):
            if robust.get(key) is not None:
                gates.append(f"{label} {robust[key]:g}{unit}")
        if robust.get("require_smoothing_interval_overlap"):
            gates.append("smoothing intervals overlap")
        if robust.get("smoothing_reference"):
            recipe = robust["smoothing_reference"]
            gates.append(f"reference window {recipe['reference_window']:g}")
            gates.append(f"declared windows {recipe['windows']}")
            resolved = sorted({
                tuple(entry["smooth_points"][w] for w in sorted(entry["smooth_points"]))
                for entry in recipe["smooth_points"]
            }, key=str)
            gates.append(f"resolved point counts {resolved}")
            if recipe["min_member_significance"] is not None:
                gates.append(f"member significance ≥ {recipe['min_member_significance']:g}")
            for key, unit in (("track_tol_k", "K"), ("track_tol_t", "T")):
                if recipe[key] is not None:
                    gates.append(f"track tolerance {recipe[key]:g} {unit}")
        if gates:
            title = title + "\n" + ", ".join(gates) if title else ", ".join(gates)
    return title


def plot_transitions(
    transitions,
    *,
    ax: Axes | None = None,
    field_unit: str = "tesla",
    marker_by: str | None = "method",
    color_by: str | None = None,
    markers: dict | None = None,
    colors: dict | None = None,
    connect: str | None = None,
    show_errors: bool = True,
    legend: bool = True,
    **style: Any,
) -> Axes:
    """Draw candidate coordinates without manufacturing a background map.

    This is a physical temperature--field diagram, with temperature on x
    in Kelvin and applied field on y. Every supplied candidate is retained;
    no boundary is fitted or interpolated and no phase is assigned.
    Use :func:`overlay_transitions` for rules on a sweep or for a map whose
    family coordinates may be indexed.

    Parameters
    ----------
    transitions : object
        A :class:`~calamari.transitions.TransitionResult`, a pandas DataFrame,
        or a sequence of either, as in :func:`overlay_transitions`. Coordinates
        are ``T_K`` (Kelvin) and ``H_T`` (Tesla), regardless of display units.
        Array-level ``find_transitions`` results lack these physical coordinates;
        use ``detect_transitions`` or ``detect_transitions_all`` instead.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; created when omitted. Existing axes must use physical
        coordinates, never indexed map rows.
    field_unit : {'tesla', 'oe'}, default 'tesla'
        Display unit for field coordinates and their errors. Tesla denotes
        applied mu0 H, without a demagnetisation correction.
    marker_by, color_by : str or None, optional
        Candidate columns encoding shape and colour; defaults to method shapes
        and white faces. Registry method markers agree with map overlays.
    markers, colors : dict, optional
        Override mappings from column values to marker shapes or colours.
        Missing encodings form a drawable ``(missing)`` group; use ``None``
        as its override key (NaN and pandas nullable values share this key).
    connect : str or None, optional
        Only equal values of this explicitly named column are connected, in
        increasing temperature order (e.g. ``chain_id`` from ``link()``). No
        connections are drawn by default; the column must identify one feature.
    show_errors : bool, default True
        Draw ``T_err_K`` and ``H_err_T`` where supplied. Missing columns or NaN
        entries give no bar in that direction; the point remains. These are
        detected-axis location standard errors, not fixed-axis spans or a
        complete systematic uncertainty budget (calibration, criterion and
        smoothing choices are not included).
    legend : bool, default True
        Show encoding groups and any recorded ``robust()`` thresholds.
    **style
        Passed to ``Axes.errorbar``, overriding marker and error-bar defaults.
        Marker lines are disabled; use ``connect`` to state their grouping.

    Returns
    -------
    matplotlib.axes.Axes
        The supplied or created axes. Group artists are available through
        ``ax.containers`` as matplotlib ``ErrorbarContainer`` objects.
    """
    from matplotlib.cbook import normalize_kwargs
    from matplotlib.lines import Line2D

    _check_field_unit(field_unit, "field")
    frame, robust = _transitions_frame(transitions)
    missing = [c for c in ("T_K", "H_T") if c not in frame.columns]
    if missing:
        raise PlotInputError(
            f"plot_transitions: candidates lack physical coordinate columns {missing}. "
            "Use detect_transitions or detect_transitions_all; array-level "
            "find_transitions does not supply physical temperature and field coordinates."
        )
    style = normalize_kwargs(style, Line2D)
    style["linestyle"] = "none"
    ax = _make_ax(ax)
    if len(frame):
        _draw_transition_points(
            frame.reset_index(drop=True), robust, ax=ax, field_unit=field_unit,
            marker_by=marker_by, color_by=color_by, markers=markers, colors=colors,
            connect=connect, show_errors=show_errors, legend=legend, scatter_kwargs=style,
        )
    ax.set_xlabel(axis_labels("temperature", "k"))
    ax.set_ylabel(axis_labels("field", field_unit))
    return ax


def overlay_transitions(
    transitions,
    *,
    ax: Axes,
    grid: PlotGrid | None = None,
    family_scale: str = "linear",
    x: str | None = None,
    field_unit: str | None = None,
    marker_by: str | None = "method",
    color_by: str | None = None,
    markers: dict | None = None,
    colors: dict | None = None,
    connect: str | None = None,
    show_errors: bool = True,
    legend: bool = True,
    scatter_kwargs: dict | None = None,
) -> dict:
    """Draw transition candidates onto existing axes.

    For a standalone physical T-H diagram use :func:`plot_transitions`.

    Two modes, chosen by ``grid``:

    - **1-D sweep overlay** (``grid=None``): vertical rules at each
      candidate's swept coordinate on a sweep plot (:func:`plot_mt`,
      :func:`plot_chi_vs_t`, ...) -- the audit view: the marks on the
      curve that produced them. ``x='temperature'``/``'field'``
      restricts to rows swept along that axis (default: every row, each
      on its own axis value). A field rule sits at ``H_T``, which is in
      Tesla, so it has to be scaled into the unit the sweep was plotted
      in (the ``field_unit=`` given to :func:`plot_chi_vs_h`/
      :func:`plot_mh`) -- and the two choices put it a factor of 10^4
      apart. ``field_unit=None`` (the default) reads that unit from the
      axes' x-label when it is calamari's own (``Field (Oe)`` or the
      mu0 H (T) label): the plotting call wrote it from the same
      ``field_unit``, so it is a record, not a guess. Any other label --
      one you have replaced, or none -- names no unit calamari can
      trust, so the default then refuses, naming the fix, rather than
      assume Oe and put a Tesla sweep's rule at 10^4 times its field.
      That refusal comes only when a field rule would actually be drawn:
      T-swept rules need no unit, so a chi(T) or M(T) overlay works on
      any label. An explicit ``'oe'`` or ``'tesla'`` is taken at its word
      on a label you have replaced, but must agree with calamari's own
      label where there is one; a contradiction is refused with the fix.
      Every refusal comes before anything is drawn.
    - **T-H map overlay** (``grid=`` the map's :class:`PlotGrid`): points
      at (T, H) over :func:`draw_map` output. The grid is REQUIRED here
      because the axes alone cannot reveal a ``family_scale='index'``
      rendering, where physical coordinates would land on wrong rows --
      the silently-wrong-points failure this parameter exists to prevent.
      The map's field axis is always mu0 H in Tesla, so ``field_unit``
      has nothing to choose here: the default and ``'tesla'`` (simply
      true -- one ``field_unit='tesla'`` can serve every overlay in a
      script) are accepted and change nothing, and ``'oe'`` is refused,
      since no map axis is in Oe.
      Pass the same ``family_scale`` the map was drawn with; in
      ``'index'`` mode points snap to family rows by the same
      insertion-index rule (``searchsorted``) as the map's own sample
      overlay, so a candidate and the sample it came from land on the
      same row. The one-call forms
      (``plot_chi_map(..., transitions=...)``) wire both through
      automatically.

    Encoding is generic column mapping: ``marker_by``/``color_by`` name
    candidate columns (``method``, ``signal``, ``swept_axis``, ``probe``,
    ...); ``method`` markers come from the :data:`calamari.transitions.METHODS`
    registry so the same criterion has the same shape on every figure.
    The combined legend shows one entry per observed (marker, colour)
    pair, and its title carries the encoding rule plus -- when the input
    is a ``.robust()``-filtered result -- the thresholds applied, so the
    figure states its own inclusion criterion. A bare DataFrame (e.g.
    ``result.candidates`` after a pandas filter) carries no thresholds;
    prefer filtering with ``.robust()``.

    When the ``color_by`` column has more distinct values than the palette
    has colours (:data:`_MAX_LEGEND_COLOR_VALUES`), the colours still
    cycle but the per-value legend entries are dropped: past that point
    the cycle repeats, so the legend would be both unreadably long and
    actively wrong. ``color_by='chain_id'`` is the case this exists for.

    ``connect=`` names a column -- ``'chain_id'`` from
    :meth:`~calamari.transitions.TransitionResult.link` is the intended
    one -- whose equal values are joined by a thin line in family order,
    one line per value with two or more members. That is the visible form
    of the cross-sweep argument: a boundary that recurs traces a curve
    across the map, while unlinked noise stays a scatter of dots. The
    line takes its colour from ``color_by`` when that is set (the first
    member's, since a chain's members normally share the value being
    encoded) and a neutral grey otherwise, sits below the markers, and
    never enters the legend. Drawing the line is map-only: on a 1-D
    overlay the candidates are vertical rules on ONE sweep, where a line
    joining other sweeps' candidates would have nothing to run along, so
    no line is drawn there. The column name is still checked in both
    modes -- naming a column the frame does not carry is a caller error
    either way, and its usual cause is a result that was never linked,
    which accepting the argument in silence would hide.

    Error bars (``show_errors=True``) draw the detected-axis standard
    errors (``T_err_K``/``H_err_T``); the fixed-axis spans are provenance,
    not uncertainties, and are never drawn as bars.

    Returns
    -------
    dict
        ``(marker value, colour value) -> list of matplotlib artists``
        -- the handles for a custom legend or removal. Error-bar
        containers are included (they are not ``PathCollection``\\ s).
        ``connect=`` adds ``('connect', <column value>)`` keys for the
        chain lines, so they can be restyled or removed independently of
        the markers.
    """
    # Before the empty-frame return: a wrong unit is a caller error whether
    # or not this particular filter left anything to draw, and it would
    # surface only on the next, non-empty call.
    if field_unit is not None and field_unit not in _FIELD_AXIS_UNITS:
        raise PlotInputError(
            f"field_unit must be one of {sorted(_FIELD_AXIS_UNITS)}, or None to read "
            f"the unit from calamari's own axis label; got {field_unit!r}."
        )
    if grid is not None and field_unit == "oe":
        raise PlotInputError(
            "field_unit='oe' cannot apply to a map overlay -- a map's field axis "
            "is always mu0 H in Tesla (draw_map's fixed orientation), and map "
            "points are placed there from H_T. Leave field_unit unset, or pass "
            "field_unit='tesla'."
        )
    # The unit calamari itself wrote on this axis, or None for a label it did
    # not write. Only an exact match counts: a label the caller composed is
    # theirs, and reading a unit out of free text would be the guess this
    # check exists to avoid.
    label_unit = None
    if grid is None:
        xlabel = ax.get_xlabel()
        label_unit = next(
            (u for u in _FIELD_AXIS_UNITS if xlabel == axis_labels("field", u)), None
        )
        if field_unit is not None and label_unit not in (None, field_unit):
            raise PlotInputError(
                f"overlay_transitions: this axis is in {_FIELD_AXIS_UNITS[label_unit][1]} "
                f'(it carries calamari\'s own label "{xlabel}") but '
                f"field_unit={field_unit!r}, which would place every field rule a "
                f"factor of 10^4 off. Pass field_unit={label_unit!r}, or leave it "
                "unset to follow the label."
            )

    frame, robust = _transitions_frame(transitions)
    if len(frame) == 0:
        return {}
    frame = frame.reset_index(drop=True)

    if grid is None and family_scale != "linear":
        raise PlotInputError(
            "family_scale applies to map overlays only -- pass grid=<the map's "
            "PlotGrid> (the axes alone cannot reveal how the map was drawn)."
        )

    if grid is not None:
        return _draw_transition_points(
            frame, robust, ax=ax, grid=grid, family_scale=family_scale,
            marker_by=marker_by, color_by=color_by, markers=markers, colors=colors,
            connect=connect, show_errors=show_errors, legend=legend,
            scatter_kwargs=scatter_kwargs,
        )

    marker_values, color_values, marker_map, color_map = _overlay_style_maps(
        frame, marker_by, color_by, markers, colors
    )
    suppress_color = (
        color_by is not None and len(color_values) > _MAX_LEGEND_COLOR_VALUES
    )
    used_labels: set = set()

    # Deliberately before the 1-D/map branch. Only the line is map-only;
    # the name is a caller error in either mode, and the usual cause is a
    # result that was never linked. Validating only where the line gets
    # drawn would let that mistake pass silently on a 1-D overlay.
    if connect is not None and connect not in frame.columns:
        raise PlotInputError(
            f"connect={connect!r} is not a candidates column. Chains come from "
            "TransitionResult.link(), which adds 'chain_id'; available columns: "
            f"{list(frame.columns)!r}."
        )

    artists: dict = {}
    extra = dict(scatter_kwargs or {})

    if grid is None:
        # 1-D mode: vertical rules at the swept coordinate, field rules in
        # the unit the sweep was drawn in (H_T is always Tesla).
        rows = frame
        if x is not None:
            if x not in ("temperature", "field"):
                raise PlotInputError(f"x must be 'temperature' or 'field', got {x!r}")
            rows = frame[frame["swept_axis"] == ("T" if x == "temperature" else "H")]
        unit = field_unit if field_unit is not None else label_unit
        # A row is field-swept unless it says "T" -- the same test the
        # drawing loop applies -- and a frame without the column is all T.
        has_field_rows = "swept_axis" in rows.columns and bool((rows["swept_axis"] != "T").any())
        if unit is None and has_field_rows:
            # Only now, with a field rule actually to place: T-swept rules
            # need no unit, so a chi(T) overlay must work on any label.
            shown = f'its x-label "{xlabel}" is not one calamari wrote' if xlabel else (
                "it has no x-label"
            )
            raise PlotInputError(
                "overlay_transitions: cannot read the field axis unit from this axis "
                f"-- {shown} -- and a field rule at H_T lands a factor of 10^4 apart "
                "in Oe and Tesla. Pass field_unit='oe' or field_unit='tesla', "
                "whichever the curve was plotted in."
            )
        # Unused when unit is None: then every rule drawn is a T-swept one.
        h_scale = OE_PER_TESLA if unit == "oe" else 1.0
        for mv in marker_values:
            for cv in color_values:
                sel = rows
                if marker_by is not None:
                    sel = sel[_overlay_group_mask(sel[marker_by], mv)]
                if color_by is not None:
                    sel = sel[_overlay_group_mask(sel[color_by], cv)]
                if len(sel) == 0:
                    continue
                colour = color_map[cv] if color_by is not None else "black"
                label = _overlay_group_label(
                    marker_by, mv, color_by, cv, suppress_color, used_labels
                )
                group = []
                for _, row in sel.iterrows():
                    swept_t = row.get("swept_axis", "T") == "T"
                    coord = row["T_K"] if swept_t else row["H_T"] * h_scale
                    group.append(
                        ax.axvline(coord, color=colour, linestyle="--", alpha=0.8,
                                   label=label if not group else None, **extra)
                    )
                artists[(mv, cv)] = group
        if legend and used_labels:
            ax.legend(title=_overlay_legend_title(marker_by, color_by, robust) or None)
        return artists


def _draw_transition_points(
    frame, robust, *, ax, grid=None, family_scale="linear", field_unit="tesla",
    marker_by="method", color_by=None, markers=None, colors=None, connect=None,
    show_errors=True, legend=True, scatter_kwargs=None,
) -> dict:
    """Share candidate encoding between physical diagrams and map overlays."""
    import numpy as _np
    from matplotlib.cbook import normalize_kwargs
    from matplotlib.lines import Line2D

    marker_values, color_values, marker_map, color_map = _overlay_style_maps(
        frame, marker_by, color_by, markers, colors
    )
    suppress_color = (
        color_by is not None and len(color_values) > _MAX_LEGEND_COLOR_VALUES
    )
    used_labels: set = set()

    # Reject an unlinked or misspelt grouping column before drawing any points.
    if connect is not None and connect not in frame.columns:
        raise PlotInputError(
            f"connect={connect!r} is not a candidates column. Chains come from "
            "TransitionResult.link(), which adds 'chain_id'; available columns: "
            f"{list(frame.columns)!r}."
        )

    artists: dict = {}
    extra = normalize_kwargs(scatter_kwargs or {}, Line2D)

    # Physical T-H coordinates; indexed maps replace only the family coordinate.
    if family_scale not in ("linear", "log", "index"):
        raise PlotInputError(
            f"family_scale must be 'linear', 'log', or 'index', got {family_scale!r}"
        )
    family_on_x = grid is None or grid.family_axis == "temperature"
    fam_centres = None if grid is None else (
        grid.t_k if family_on_x else grid.h_oe / OE_PER_TESLA
    )
    h_scale = OE_PER_TESLA if field_unit == "oe" else 1.0

    t_vals = frame["T_K"].to_numpy(dtype=float)
    h_vals = frame["H_T"].to_numpy(dtype=float) * h_scale
    t_err = (
        frame["T_err_K"].to_numpy(dtype=float)
        if "T_err_K" in frame.columns
        else _np.full(len(frame), _np.nan)
    )
    h_err = (
        frame["H_err_T"].to_numpy(dtype=float) * h_scale
        if "H_err_T" in frame.columns
        else _np.full(len(frame), _np.nan)
    )

    fam_vals = t_vals if family_on_x else h_vals
    keep = _np.ones(len(frame), dtype=bool)
    if family_scale == "log":
        keep &= fam_vals > 0
        if (~keep).any():
            warnings.warn(
                f"overlay_transitions: dropped {int((~keep).sum())} candidate(s) at "
                "non-positive family-axis values -- a log axis cannot place them.",
                stacklevel=2,
            )
    if family_scale == "index":
        # Snap by searchsorted (the insertion index, not nearest-by-
        # distance) DELIBERATELY: it is the exact rule draw_map's sample
        # overlay uses, and a candidate's fixed coordinate is computed by
        # the same masked mean as the map's row value, so the match is
        # exact in practice -- agreeing with the samples matters more than
        # the tie-break for a hypothetical between-rows point.
        snapped = _np.searchsorted(fam_centres, fam_vals).clip(0, len(fam_centres) - 1)
        fam_plot = snapped.astype(float)
    else:
        fam_plot = fam_vals

    x_all = fam_plot if family_on_x else t_vals
    y_all = h_vals if family_on_x else fam_plot
    if grid is None:
        # Matplotlib omits NaN bars while retaining their point; a missing
        # uncertainty must not acquire a zero-length bar with visible caps.
        xerr_all, yerr_all = t_err, h_err
    elif family_on_x and family_scale == "index":
        xerr_all, yerr_all = _np.zeros(len(frame)), _np.nan_to_num(h_err)
    elif family_on_x:
        xerr_all, yerr_all = _np.nan_to_num(t_err), _np.nan_to_num(h_err)
    elif family_scale == "index":
        xerr_all, yerr_all = _np.nan_to_num(t_err), _np.zeros(len(frame))
    else:
        xerr_all, yerr_all = _np.nan_to_num(t_err), _np.nan_to_num(h_err)

    for mv in marker_values:
        for cv in color_values:
            sel = keep.copy()
            if marker_by is not None:
                sel &= _overlay_group_mask(frame[marker_by], mv)
            if color_by is not None:
                sel &= _overlay_group_mask(frame[color_by], cv)
            if not sel.any():
                continue
            # Stroked markers: a "high-contrast colour" does not exist
            # against an arbitrary colormap, but a light face with a dark
            # edge (or a colour_by face with a dark edge) survives any
            # background.
            face = color_map[cv] if color_by is not None else "white"
            label = _overlay_group_label(
                marker_by, mv, color_by, cv, suppress_color, used_labels
            )
            style = dict(
                fmt=marker_map[mv] if marker_by is not None else "o",
                markerfacecolor=face, markeredgecolor="black", ecolor="black",
                elinewidth=1.0, capsize=2.0, linestyle="none", label=label,
            )
            style.update(extra)
            container = ax.errorbar(
                x_all[sel], y_all[sel],
                xerr=xerr_all[sel] if show_errors else None,
                yerr=yerr_all[sel] if show_errors else None,
                **style,
            )
            artists[(mv, cv)] = [container]

    if connect is not None:
        # Drawn after the markers but at a lower zorder, so the line runs
        # UNDER them: the markers are the measurement, the line is only the
        # claim that they are one feature.
        values = frame[connect].to_numpy()
        for value in sorted({v for v in values[keep]}, key=str):
            members = _np.flatnonzero(keep & (values == value))
            if len(members) < 2:
                continue
            # Family order, not detection order: the line must trace the
            # boundary across the family of sweeps, not zigzag.
            members = members[_np.argsort(fam_vals[members], kind="stable")]
            colour = _CONNECT_COLOR
            if color_by is not None:
                colour = color_map[_overlay_group_key(frame[color_by].iloc[members[0]])]
            artists.setdefault(("connect", value), []).extend(
                ax.plot(
                    x_all[members], y_all[members], color=colour, label=None,
                    **_CONNECT_STYLE,
                )
            )

    if legend and used_labels:
        ax.legend(title=_overlay_legend_title(marker_by, color_by, robust) or None)
    return artists


# ---------------------------------------------------------------------------
# Detection diagnostics: the smoothed curves a detection searched, how they
# change across a declared set of smoothing windows, and where each feature
# sits at each window. Single-axes helpers (PL-1); the standard multi-panel
# figure is composed by the caller.
# ---------------------------------------------------------------------------

DETECTION_REPRESENTATIONS = ("y_smooth", "dydx", "d2ydx2", "dyT_dT")

#: The unit of each searched signal as detection reads it: the file's own
#: extensive quantity (``Measurement.chi_real`` and ``chi_imag`` are emu/Oe on
#: every dialect, ``moment_emu`` is emu). Detection never converts, so the
#: curves are in these units whatever a neighbouring plot was drawn in.
_DETECTION_SIGNAL_UNITS = {"chi_real": "emu/Oe", "chi_imag": "emu/Oe", "moment_emu": "emu"}

#: Per swept axis: the variable as a mathtext fragment, its unit, and the
#: (quantity, units) of its x-label. Detection sweeps a field as mu0 H in
#: Tesla (``field_tesla``), so a field curve's x is in Tesla.
_DETECTION_AXES = {
    "T": ("T", "K", ("temperature", "k")),
    "H": (r"(\mu_0 H)", "T", ("field", "tesla")),
}

#: How a candidate is marked on a detection curve: a dashed black rule, as
#: :func:`overlay_transitions` draws on a sweep plot, so the two read alike,
#: and the method's registry marker on the curve at that location, with a
#: white face so the curve stays visible under it. Fixed, like the
#: failed-window lines of :func:`plot_fit_windows`: ``**style`` describes the
#: curve, not the marks.
_CANDIDATE_RULE_STYLE = {"color": "black", "linestyle": "--", "alpha": 0.8, "linewidth": 0.8}
_CANDIDATE_MARKER_STYLE = {
    "linestyle": "none",
    "color": "black",
    "markerfacecolor": "white",
    "markeredgecolor": "black",
    "zorder": 3,
}

#: Line width of the track lines in :func:`plot_scale_tracks`. Thin, for the
#: reason :data:`_CONNECT_STYLE` gives: the line only claims that the
#: markers are one feature; the markers are the detections.
_TRACK_LINE_WIDTH = 0.8

#: Legend text of the first sign-flipped (hollow) marker in
#: :func:`plot_scale_tracks`.
_FLIPPED_LABEL = "sign differs from its track"


def _check_representation(representation: str, who: str) -> None:
    if representation not in DETECTION_REPRESENTATIONS:
        raise PlotInputError(
            f"{who}: unknown representation={representation!r}; expected one of "
            f"{list(DETECTION_REPRESENTATIONS)} (the smoothed signal, its first and "
            "second derivatives along the swept axis, and Fisher's d(yT)/dT)."
        )


def _detection_labels(
    signal: str, representation: str, axis: str | None
) -> tuple[str | None, str]:
    """(x-label or None, y-label) for one representation of one signal.

    The smoothed signal is labelled exactly as the raw helpers label the raw
    quantity in the same unit (``plot_chi_vs_t``'s default, ``plot_mt``), so
    drawing it over them keeps their label. A derivative's unit is the
    signal's divided by the swept unit once or twice; Kelvin cancels in
    Fisher's d(yT)/dT, which is in the signal's own unit.
    """
    unit = _DETECTION_SIGNAL_UNITS.get(signal)
    symbol = _SIGNAL_LABELS.get(signal, signal).strip("$")
    xlabel = None if axis is None else axis_labels(*_DETECTION_AXES[axis][2])
    if representation == "y_smooth":
        if signal == "moment_emu":
            return xlabel, axis_labels("moment", "emu")
        return xlabel, f"${symbol}$" + (f" ({unit})" if unit else "")
    if representation == "dyT_dT":
        fisher = rf"$\mathrm{{d}}({symbol} T)/\mathrm{{d}}T$"
        return xlabel, fisher + (f" ({unit})" if unit else "")
    variable, x_unit = _DETECTION_AXES[axis][:2] if axis else ("x", None)
    power = "" if representation == "dydx" else "^2"
    ylabel = rf"$\mathrm{{d}}{power}{symbol}/\mathrm{{d}}{variable}{power}$"
    if unit and x_unit:
        per = x_unit if representation == "dydx" else f"{x_unit}$^2$"
        numerator, _, denominator = unit.partition("/")
        ylabel += (
            f" ({numerator}/({denominator} {per}))" if denominator else f" ({numerator}/{per})"
        )
    return xlabel, ylabel


def _calamari_y_labels() -> set[str]:
    """Every y-label a calamari sweep helper writes, for recognising one.

    A label in this set is a record of the quantity and unit a helper drew,
    so it can be trusted the way :func:`overlay_transitions` trusts
    calamari's own field labels; any other label is the caller's and is
    left alone.
    """
    labels = {
        text for (quantity, _), text in _AXIS_LABELS.items() if quantity in ("moment", "chi")
    }
    for _, unit_text, inverse_unit_text in _CHI_UNITS.values():
        for symbol, inverse_symbol in (*_CHI_SYMBOLS.values(), (r"$\chi$", r"$1/\chi$")):
            labels.add(f"{symbol} ({unit_text})")
            labels.add(f"{inverse_symbol} ({inverse_unit_text})")
        for symbol in _FISHER_SYMBOLS.values():
            labels.add(f"{symbol} ({unit_text})")
    for signal in _DETECTION_SIGNAL_UNITS:
        for representation in DETECTION_REPRESENTATIONS:
            for axis in _DETECTION_AXES:
                labels.add(_detection_labels(signal, representation, axis)[1])
    return labels


def _check_axes_agree(ax: Axes | None, xlabel: str | None, ylabel: str | None, who: str) -> None:
    """Refuse to draw a detection curve over axes calamari labelled otherwise.

    The curves are in the detected signal's file units, against the swept
    axis in K or Tesla. A raw sweep drawn in another unit -- field in Oe, chi
    per mole or per gram -- shares the axes but not the scale, and the two
    curves would sit orders of magnitude apart under one label. Only
    calamari's own labels are read, and only an exact match counts, as in
    :func:`overlay_transitions`: a label the caller wrote is theirs.
    """
    if ax is None:
        return
    shown_x = ax.get_xlabel()
    own_x = {axis_labels(*spec[2]) for spec in _DETECTION_AXES.values()}
    own_x.add(axis_labels("field", "oe"))
    if xlabel is not None and shown_x in own_x and shown_x != xlabel:
        fix = (
            " Draw the raw field sweep in Tesla, e.g. plot_chi_vs_h(m, field_unit='tesla'): "
            "detection sweeps mu0 H in Tesla, and in Oe every point sits 10^4 times further "
            "along."
            if shown_x == axis_labels("field", "oe")
            else " Draw it on axes for the same swept axis."
        )
        raise PlotInputError(
            f'{who}: these axes carry calamari\'s x-label "{shown_x}", but the curve is '
            f'against "{xlabel}", so it would land on the wrong scale.{fix}'
        )
    shown_y = ax.get_ylabel()
    if ylabel is not None and shown_y and shown_y != ylabel and shown_y in _calamari_y_labels():
        raise PlotInputError(
            f'{who}: these axes carry calamari\'s y-label "{shown_y}", but this curve is '
            f'"{ylabel}". Detection smooths the signal in the file\'s own unit and never '
            "converts it, so the two would share an axis but not a scale. Draw the raw "
            "sweep in the file's unit (plot_chi_vs_t / plot_chi_vs_h with the default "
            "units='emu_per_oe', or plot_mt / plot_mh for a moment), or put this curve on "
            "its own axes."
        )


def _curve_signal(result: TransitionResult, sweep_id: int, signal: str | None) -> str:
    """The signal ``result.curve(sweep_id, signal)`` resolved to.

    Call only after ``curve()`` has succeeded, which guarantees that a
    ``signal=None`` sweep has exactly one signal.
    """
    if signal is not None:
        return signal
    if "per_sweep" in result.params:
        return next(s for (sid, s) in result.curves if sid == int(sweep_id))
    return next(s for (_, s) in result.curves)


def _curve_axis(result: TransitionResult, sweep_id: int, signal: str) -> str | None:
    """'T' or 'H' for one sweep's curve, or None when nothing records it.

    Read from that sweep's rows in the unfiltered candidates, or from a
    single-sweep result's ``scan_kind``. A batch sweep with no candidate at
    all records its swept axis nowhere, so the answer is then None.
    """
    full = result.all().candidates
    rows = full[(full["sweep_id"] == int(sweep_id)) & (full["signal"] == signal)]
    axes = set(rows["swept_axis"].dropna())
    if len(axes) == 1:
        return axes.pop()
    return {"T-scan": "T", "H-scan": "H"}.get(result.params.get("scan_kind"))


def _require_measurement_level(result: TransitionResult, who: str) -> None:
    missing = [c for c in ("sweep_id", "signal", "swept_axis") if c not in result.candidates]
    if missing:
        raise PlotInputError(
            f"{who}: this result's candidates have no {missing} column(s): it is an "
            "array-level find_transitions result, which records neither the swept axis "
            "nor the signal needed to place and label its curves. Detect with "
            "detect_transitions or detect_transitions_all."
        )


def _representation_xy(
    curve: dict, representation: str, sweep_id: int, signal: str, who: str, rerun: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if representation not in curve:
        raise PlotInputError(
            f"{who}: representation='dyT_dT' is Fisher's d(yT)/dT, and sweep_id={sweep_id} "
            f"({signal}) has no Fisher curve: 'fisher' was not among the detection's "
            f"methods. {rerun} with methods=(..., 'fisher'), e.g. methods=('peak', 'fisher'), "
            "on a temperature sweep."
        )
    return np.asarray(curve["x"], dtype=float), np.asarray(curve[representation], dtype=float)


def plot_detection_curve(
    result: TransitionResult,
    *,
    sweep_id: int = 0,
    signal: str | None = None,
    representation: str = "y_smooth",
    ax: Axes | None = None,
    candidates: bool = True,
    **style: Any,
) -> Axes:
    """Draw one smoothed curve a detection searched, with that sweep's candidates on it.

    Why this exists: every candidate was found in a smoothed representation
    -- a cusp in the smoothed signal, a derivative extremum in dy/dx, a kink
    in d²y/dx² -- and auditing one means looking at that curve, not at the
    raw points. The result keeps those curves
    (:meth:`~calamari.transitions.TransitionResult.curve`); this draws one
    of them against the swept axis, with a dashed rule at each candidate of
    that sweep and signal and the method's
    :data:`~calamari.transitions.METHODS` marker where the rule meets the
    curve. On the ``'d2ydx2'`` curve a kink's marker should sit on a
    curvature extremum; if it does not, the figure shows it.

    **The raw data are not in the result**, so this does not draw them.
    Draw the raw sweep first and this curve on the same axes::

        r = detect_transitions(m, methods=("peak", "derivative", "kink"))
        ax = plot_chi_vs_t(m, linestyle="none", marker=".")
        plot_detection_curve(r, ax=ax)  # 'y_smooth', the default

    **Units.** The curves are in the searched signal's file units, never
    converted: chi' and chi'' in emu/Oe, a moment in emu; against
    temperature in K, or against mu0 H in Tesla for a field sweep. That is
    what :func:`plot_chi_vs_t` and :func:`plot_mt` draw by default, so that
    composition shares one scale. Two others do not:

    - a field sweep drawn by :func:`plot_chi_vs_h` or :func:`plot_mh` with
      their default ``field_unit='oe'`` puts the raw points 10^4 times
      further along x than the curve. Draw them with ``field_unit='tesla'``;
    - chi drawn with ``units='molar'``, ``'mass'`` or ``'molar_per_ion'``
      is the curve times the sample's conversion factor, so the two would
      sit orders of magnitude apart under one label.

    Both are refused, naming the fix, whenever the axes carry the label
    calamari wrote for the other unit. A label you wrote yourself is
    yours, and is not checked. Two more mismatches are not units, and are
    not refused: a multi-frequency file is searched at one frequency
    (``params['frequency_hz']``) while :func:`plot_chi_vs_t` draws every
    frequency, and a field sweep detected with ``branch='increasing'`` is a
    curve against abs(mu0 H) on that branch only. Say so in the caption
    when either applies.

    The curve is on the uniform grid detection resampled the searched range
    to (after any ``t_window_k`` / ``h_window_oe`` and outlier drop), so it
    can be shorter than the raw sweep.

    Parameters
    ----------
    result : calamari.transitions.TransitionResult
        From :func:`~calamari.transitions.detect_transitions` or
        :func:`~calamari.transitions.detect_transitions_all`, any filtered
        or linked view of one (``robust()``, ``select()``, ``link()``), or
        one window of a scan (``scan.at(window)``). The candidates drawn are
        the view's own, so a ``robust()`` view draws only what survived. An
        array-level :func:`~calamari.transitions.find_transitions` result is
        refused: it records no swept axis or signal.
    sweep_id : int, optional
        The sweep, as in the ``sweep_id`` column; ``0`` (the default) for a
        single-sweep result.
    signal : str, optional
        The searched signal; ``None`` takes the sweep's only one. Resolved
        and refused exactly as by
        :meth:`~calamari.transitions.TransitionResult.curve`.
    representation : {'y_smooth', 'dydx', 'd2ydx2', 'dyT_dT'}, optional
        The smoothed signal (default), its first or second derivative along
        the swept axis, or Fisher's d(yT)/dT, which exists only when
        ``'fisher'`` was among the detection's methods.
    ax : matplotlib.axes.Axes, optional
        Draw into these axes -- normally those the raw sweep is on; omitted,
        a new single-axes figure is created.
    candidates : bool, optional
        Mark the sweep's candidates (default ``True``). Each gets a rule and
        a marker; the first marker of each (method, sign) pair carries a
        legend label naming both (``'curvature extremum (down)'``), so
        ``ax.legend()`` explains them. Calling the legend is left to you.
    **style
        Forwarded to ``ax.plot`` for the curve (``color=``, ``label=``,
        ...). The candidate marks have a fixed style and ignore it.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into. The x-label names the swept axis and the
        y-label the representation, in mathtext with its unit
        (``$\\mathrm{d}\\chi'/\\mathrm{d}T$ (emu/(Oe K))``). A batch sweep
        with no candidates at all records its swept axis nowhere, so its
        x-label is left as it was.

    Raises
    ------
    PlotInputError
        For an unknown ``representation`` (naming the valid ones);
        ``'dyT_dT'`` when Fisher did not run (naming ``methods=(...,
        'fisher')``); an array-level result; and axes whose calamari label
        names another unit (see above). Raised before drawing.
    calamari.exceptions.CurveInputError
        An unknown ``sweep_id`` or ``signal`` (from
        :meth:`~calamari.transitions.TransitionResult.curve`, naming the
        choices). Raised before drawing; also a ``ValueError``.
    """
    who = "plot_detection_curve"
    _check_representation(representation, who)
    _require_measurement_level(result, who)
    curve = result.curve(sweep_id, signal)
    signal = _curve_signal(result, sweep_id, signal)
    x, y = _representation_xy(curve, representation, sweep_id, signal, who, "Re-run detection")
    axis = _curve_axis(result, sweep_id, signal)
    xlabel, ylabel = _detection_labels(signal, representation, axis)
    _check_axes_agree(ax, xlabel, ylabel, who)

    rows = result.candidates
    rows = rows[(rows["sweep_id"] == int(sweep_id)) & (rows["signal"] == signal)]
    ax = _make_ax(ax)  # after every refusal: a refused call must not leak a figure
    ax.plot(x, y, **style)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if candidates:
        _draw_candidate_marks(ax, rows, x, y)
    return ax


def _draw_candidate_marks(ax: Axes, rows: Any, x: NDArray, y: NDArray) -> None:
    """A rule and a registry marker per candidate, labelled once per (method, sign).

    Each row is placed on its own swept axis (``T_K`` or ``H_T``), which is
    the axis the curve's x is on: detection sweeps temperature in K and
    field as mu0 H in Tesla, and stores both the curve and the candidate
    location that way.
    """
    from calamari.transitions import METHODS

    used: set = set()
    for _, row in rows.iterrows():
        location = float(row["T_K"] if row["swept_axis"] == "T" else row["H_T"])
        if not np.isfinite(location):
            continue
        method = row["method"]
        text = f"{_overlay_label('method', method)} ({row['sign']})"
        label = None if text in used else text
        used.add(text)
        ax.axvline(location, **_CANDIDATE_RULE_STYLE)
        marker = METHODS[method].marker if method in METHODS else _OVERLAY_MARKERS[0]
        ax.plot(
            [location], [np.interp(location, x, y)], marker=marker, label=label,
            **_CANDIDATE_MARKER_STYLE,
        )


def _scan_signal(scan: SmoothingScanResult, sweep_id: int, signal: str | None, who: str) -> str:
    """Check a scan searched (sweep_id, signal), and return the signal.

    Read from ``params['smooth_points']``, the scan's own record of every
    (sweep, signal) it searched, so a sweep with no tracks at all is still
    recognised, and refusals name what the scan does hold, as
    :meth:`~calamari.transitions.TransitionResult.curve` does.
    """
    if isinstance(sweep_id, bool) or not isinstance(sweep_id, (int, np.integer)):
        raise PlotInputError(
            f"{who}: sweep_id must be an integer, got {sweep_id!r}. Pass the sweep's "
            "sweep_id, as in the scan's tracks table."
        )
    searched = scan.params["smooth_points"]
    signals = [e["signal"] for e in searched if e["sweep_id"] == int(sweep_id)]
    if not signals:
        ids = sorted({e["sweep_id"] for e in searched})
        raise PlotInputError(
            f"{who}: no sweep_id={sweep_id} in this scan. It searched sweep ids {ids}; an "
            "id is the sweep's position in the input to scan_smoothing. Pass one of those."
        )
    if signal is None:
        if len(signals) > 1:
            raise PlotInputError(
                f"{who}: sweep_id={sweep_id} was searched on several signals {signals}, so "
                "signal=None is ambiguous. Pass signal= one of them."
            )
        return signals[0]
    if signal not in signals:
        raise PlotInputError(
            f"{who}: sweep_id={sweep_id} has no signal={signal!r}. Searched: {signals}. "
            "Pass signal= one of those."
        )
    return signal


def _scan_axis(scan: SmoothingScanResult, sweep_id: int, signal: str) -> str | None:
    """'T' or 'H' for one scanned sweep, from its tracks or any window's result."""
    tracks = scan.tracks
    rows = tracks[(tracks["sweep_id"] == int(sweep_id)) & (tracks["signal"] == signal)]
    axes = set(rows["swept_axis"].dropna())
    if len(axes) == 1:
        return axes.pop()
    return None


def plot_smoothing_scan(
    scan: SmoothingScanResult,
    *,
    sweep_id: int,
    signal: str | None = None,
    representation: str = "d2ydx2",
    ax: Axes | None = None,
    cmap: Any | None = None,
    **style: Any,
) -> Axes:
    """Draw one smoothed representation of a sweep at every window of a scan.

    The picture behind :func:`~calamari.scales.scan_smoothing`'s tables:
    one line per declared window, from ``scan.at(window).curve(sweep_id,
    signal)``, coloured along a sequential colour map in window order,
    narrowest first. A slope change's curvature extremum stays put as the
    window widens while its height falls, and the noise around it falls
    faster; a noise feature moves, flips sign or vanishes.

    **The curves are not normalised.** They share one unit, and the change
    of amplitude with the window *is* the information: a narrow window's
    curvature is tall and noisy, a wide one's low and smooth. Scaling each
    to one height would hide both that and the noise level each window's
    significance is measured against. When the widest window's curve is too
    flat to read beside the narrowest, set the y-limits or split the
    windows over two axes; do not rescale.

    Two windows that clamped to one point count on this sweep (see
    :func:`~calamari.scales.scan_smoothing`) draw identical curves, and
    the later one covers the earlier. ``scan.params['resolved_together']``
    lists them.

    Parameters
    ----------
    scan : calamari.scales.SmoothingScanResult
        From :func:`~calamari.scales.scan_smoothing`.
    sweep_id : int
        The sweep, as in the scan's ``tracks`` table. Required: a scan is
        usually of a family.
    signal : str, optional
        The searched signal; ``None`` takes the sweep's only one.
    representation : {'d2ydx2', 'dydx', 'y_smooth', 'dyT_dT'}, optional
        As for :func:`plot_detection_curve`. The default is the curvature,
        the representation whose significance depends most on the window.
    ax : matplotlib.axes.Axes, optional
        Draw into these axes; omitted, a new single-axes figure is created.
    cmap : str or matplotlib colormap, optional
        The sequential map the windows are coloured along, the narrowest
        window at its low end and the widest at its high end. ``None`` (the
        default) takes ``rcParams['image.cmap']``, so a
        :mod:`calamari.style` look applies its own map.
    **style
        Forwarded to ``ax.plot`` for every line. ``color=`` and ``label=``
        are refused by name: the window sets both.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into. Each line's label names its window
        (``'window 1.5 K'``), for a legend you call.

    Raises
    ------
    PlotInputError
        For an unknown ``representation``; ``'dyT_dT'`` when the scan did not
        run Fisher; a ``sweep_id`` or ``signal`` the scan did not search
        (naming those it did), or a sweep skipped at some window; ``color=``
        or ``label=`` in ``style``; and axes whose calamari label names
        another unit (see :func:`plot_detection_curve`). Every refusal comes
        before anything is drawn.
    calamari.exceptions.CurveInputError
        A requested curve is unavailable at a declared scan window; also
        a ``ValueError``.
    """
    import matplotlib as mpl

    who = "plot_smoothing_scan"
    _check_representation(representation, who)
    for keyword in ("color", "label"):
        if keyword in style:
            raise PlotInputError(
                f"{who}: {keyword}= is set per line from its window, so it cannot be "
                "passed. Choose the colours with cmap=."
            )
    signal = _scan_signal(scan, sweep_id, signal, who)
    windows = list(scan.results)
    curves = [
        _representation_xy(
            scan.at(window).curve(sweep_id, signal), representation, sweep_id, signal, who,
            "Re-run scan_smoothing",
        )
        for window in windows
    ]
    axis = _scan_axis(scan, sweep_id, signal)
    if axis is None:
        axis = _curve_axis(scan.at(windows[0]), sweep_id, signal)
    xlabel, ylabel = _detection_labels(signal, representation, axis)
    _check_axes_agree(ax, xlabel, ylabel, who)

    if isinstance(cmap, mpl.colors.Colormap):
        colormap = cmap
    else:
        colormap = mpl.colormaps[cmap if cmap is not None else mpl.rcParams["image.cmap"]]
    unit = f" {_DETECTION_AXES[axis][1]}" if axis else ""
    ax = _make_ax(ax)
    for i, (window, (x, y)) in enumerate(zip(windows, curves, strict=True)):
        position = i / (len(windows) - 1) if len(windows) > 1 else 0.5
        ax.plot(x, y, color=colormap(position), label=f"window {window:g}{unit}", **style)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_scale_tracks(
    scan: SmoothingScanResult,
    *,
    sweep_id: int,
    signal: str | None = None,
    ax: Axes | None = None,
    show_errors: bool = False,
    **style: Any,
) -> Axes:
    """Draw each tracked feature's location against the smoothing window.

    The scale-space view of a :func:`~calamari.scales.scan_smoothing` scan
    (Witkin's scale-space picture; see :mod:`calamari.scales`): the swept
    location on x, the window on y, one marker per window a feature was
    found at, and a thin line joining one track's members in window order.
    The marker is the method's :data:`~calamari.transitions.METHODS`
    marker, and the colour follows the method too. A feature that does not
    depend on the window is a vertical line through every window; a noise
    feature is usually a short or wandering one. Persistence and spread
    read at a glance; significance does not, and is in ``scan.tracks``.

    **A sign flip is drawn hollow.** Tracks are matched without regard to
    sign, so a curvature extremum that turns from 'down' to 'up' as the
    window changes stays one track. Each member whose sign differs from its
    track's ``sign_mode`` gets a hollow marker, so a track with a steady
    location but an unsteady sign cannot pass for a steady feature.

    Read it for what it is: robustness to an analysis choice, not
    evidence. Every window sees the same data, so a noise excursion can
    persist across all of them (see :mod:`calamari.scales`).

    Parameters
    ----------
    scan : calamari.scales.SmoothingScanResult
        From :func:`~calamari.scales.scan_smoothing`.
    sweep_id : int
        The sweep, as in the scan's ``tracks`` table.
    signal : str, optional
        The searched signal; ``None`` takes the sweep's only one.
    ax : matplotlib.axes.Axes, optional
        Draw into these axes; omitted, a new single-axes figure is created.
    show_errors : bool, optional
        Draw each member's location standard error (``T_err_K`` or
        ``H_err_T``) as a horizontal bar. Off by default: on a sweep with
        many short noise tracks the bars crowd the panel.
    **style
        Forwarded to ``ax.plot`` for the track lines and markers
        (``markersize=``, ``color=`` to override the per-method colour,
        ...). A sign-flipped member's hollow face is always applied.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into, with the declared windows as y-ticks. The first
        filled markers of each method carry the method's name as a legend
        label, and the first hollow ones ``'sign differs from its track'``,
        for a legend you call.

    Raises
    ------
    PlotInputError
        For a ``sweep_id`` or ``signal`` the scan did not search, naming
        those it did, and for axes whose x-label is calamari's for another
        swept axis. Every refusal comes before anything is drawn.
    """
    from calamari.transitions import METHODS

    who = "plot_scale_tracks"
    signal = _scan_signal(scan, sweep_id, signal, who)
    tracks = scan.tracks
    tracks = tracks[(tracks["sweep_id"] == int(sweep_id)) & (tracks["signal"] == signal)]
    axis = _scan_axis(scan, sweep_id, signal)
    xlabel = None if axis is None else axis_labels(*_DETECTION_AXES[axis][2])
    _check_axes_agree(ax, xlabel, None, who)

    members = scan.members
    methods = list(METHODS)
    ax = _make_ax(ax)  # after every refusal: a refused call must not leak a figure
    used: set = set()
    for _, track in tracks.iterrows():
        rows = members[members["track_id"] == track["track_id"]].sort_values("window")
        swept_t = track["swept_axis"] == "T"
        loc = rows["T_K" if swept_t else "H_T"].to_numpy(dtype=float)
        err = rows["T_err_K" if swept_t else "H_err_T"].to_numpy(dtype=float)
        window = rows["window"].to_numpy(dtype=float)
        flipped = (rows["sign"] != track["sign_mode"]).to_numpy(dtype=bool)
        method = track["method"]
        colour = _OVERLAY_COLORS[
            (methods.index(method) if method in methods else len(methods))
            % len(_OVERLAY_COLORS)
        ]
        if len(loc) > 1:
            ax.plot(loc, window, **{"color": colour, "linewidth": _TRACK_LINE_WIDTH,
                                    **style, "marker": "none"})
        if show_errors:
            ax.errorbar(loc, window, xerr=err, fmt="none",
                        ecolor=style.get("color", colour), elinewidth=_TRACK_LINE_WIDTH)
        marks = {
            "color": colour,
            "marker": METHODS[method].marker if method in METHODS else _OVERLAY_MARKERS[0],
            **style,
            "linestyle": "none",
        }
        for keep, text, extra in (
            (~flipped, _overlay_label("method", method), {}),
            (flipped, _FLIPPED_LABEL, {"markerfacecolor": "none"}),
        ):
            if keep.any():
                ax.plot(loc[keep], window[keep], label=None if text in used else text,
                        **{**marks, **extra})
                used.add(text)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    unit = _DETECTION_AXES[axis][1] if axis else None
    ax.set_ylabel(f"Smoothing window ({unit})" if unit else "Smoothing window")
    ax.set_yticks(list(scan.results))
    return ax


# ---------------------------------------------------------------------------
# A stated model through one criterion's points (calamari.boundary)
# ---------------------------------------------------------------------------

#: Opacity of the 1-sigma band :func:`plot_boundary` draws: light enough that
#: a map's colours and the points stay readable through it, dark enough to
#: see on white.
_BOUNDARY_BAND_ALPHA = 0.25

#: The note :func:`draw_map` appends to the family axis label of a
#: ``family_scale='index'`` map, where rows are sweeps, not field or
#: temperature values. :func:`plot_boundary` draws in physical coordinates,
#: so it refuses axes carrying it.
_INDEX_MAP_NOTE = " [row per sweep]"

#: The two drawable fits of a :class:`~calamari.boundary.BoundaryFit`.
_BOUNDARY_WHICH = ("stat", "total")


def plot_boundary(
    fit: BoundaryFit,
    *,
    ax: Axes | None = None,
    band: bool = True,
    which: str = "stat",
    **style: Any,
) -> Axes:
    """Draw a :func:`~calamari.boundary.fit_boundary` model over the points it was fitted to.

    Points, curve and band go in the H-T plane's fixed orientation --
    temperature on x, mu0 H in Tesla on y, as :func:`draw_map` renders a map
    -- with each point's location on its own axis. A T-swept fit (``T_K`` at
    held ``H_T``) draws horizontal error bars and the curve ``T(H)``; an
    H-swept fit (``H_T`` at held ``T_K``) draws vertical error bars and the
    curve ``H(T)``. So it composes onto a map: ``plot_boundary(bf,
    ax=mesh.axes)`` after :func:`plot_chi_map` or :func:`plot_moment_map`
    with a ``'linear'`` or ``'log'`` family scale. An ``'index'`` map places
    sweeps on rows, not at their field or temperature, so it is refused.

    The curve and its band span the used points' held range and no further
    (:attr:`~calamari.boundary.BoundaryFit.band`): beyond the points the
    curve is the assumed form speaking. To draw it there, evaluate
    :meth:`~calamari.boundary.BoundaryFit.band_at` and plot that yourself,
    where the figure's code says so.

    What is drawn: every used point, filled, with the error the fit was
    weighted by (the statistical error for ``which='stat'``, statistical and
    systematic in quadrature for ``'total'``), labelled with the criterion
    and the number of sweeps; every excluded point hollow, with the same
    kind of error bar, labelled ``'excluded by caller'``; the curve,
    labelled with the model and the criterion; and, with ``band=True``, the
    1-sigma band on the model. **Sweeps without a point get no marker**,
    because there is nothing to place: they are the rows of ``fit.sweeps``
    whose ``status`` is neither ``'used'`` nor ``'excluded by caller'``, and
    the model's value there is in their ``predicted_*`` column. A failed fit
    draws the points and no curve.

    Parameters
    ----------
    fit : calamari.boundary.BoundaryFit
        From :func:`~calamari.boundary.fit_boundary`.
    ax : matplotlib.axes.Axes, optional
        Draw into these axes; omitted, a new single-axes figure is created.
    band : bool, optional
        Draw the 1-sigma band on the model. Default True.
    which : {'stat', 'total'}, optional
        The fit with statistical errors (default), or the one with
        statistical and systematic errors in quadrature (``fit.total``,
        present only when ``yerr_sys`` was given).
    **style
        Forwarded to ``ax.plot`` for the curve. The points and the band take
        the curve's colour, so ``color=`` restyles all three.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into. Axis labels are set only where the axes have
        none, so a map's own labels stay. There is no ``legend()`` call
        (PL-1); every artist carries a label for one.

    Raises
    ------
    PlotInputError
        For ``which`` not ``'stat'`` or ``'total'``; ``'total'`` on a fit
        made without systematic errors; and axes whose label marks an
        ``'index'`` map. Every refusal comes before anything is drawn.
    """
    from calamari.transitions import METHODS

    who = "plot_boundary"
    if which not in _BOUNDARY_WHICH:
        raise PlotInputError(f"{who}: which must be one of {list(_BOUNDARY_WHICH)}, got {which!r}.")
    if which == "total":
        if fit.total is None:
            raise PlotInputError(
                f"{who}: which='total', but this fit was made without systematic errors, so "
                "there is no stat+sys fit to draw. Pass yerr_sys= to fit_boundary, or draw "
                "which='stat'."
            )
        fit = fit.total
    if ax is not None and any(
        label.endswith(_INDEX_MAP_NOTE) for label in (ax.get_xlabel(), ax.get_ylabel())
    ):
        raise PlotInputError(
            f"{who}: these axes are a family_scale='index' map, whose rows are sweeps, not "
            "field or temperature values, so a curve in physical coordinates would land on "
            "the wrong rows. Draw the map with family_scale='linear' (the default) or 'log'."
        )

    params = fit.params
    swept_t = params["swept_axis"] == "T"
    loc_col, held_col = params["location"], params["held"]
    err_col = "T_err_fit_K" if swept_t else "H_err_fit_T"
    unit = "K" if swept_t else "T"
    criterion = params["criterion"]
    method = str(criterion).split("/")[0]
    marker = METHODS[method].marker if method in METHODS else _OVERLAY_MARKERS[0]

    ax = _make_ax(ax)  # after every refusal: a refused call must not leak a figure
    curve = fit.band
    held = curve[held_col].to_numpy(dtype=float)
    value = curve[f"model_{unit}"].to_numpy(dtype=float)
    colour = style.get("color")
    if fit.fit.success and held.size:
        xy = (value, held) if swept_t else (held, value)
        (line,) = ax.plot(*xy, label=f"{params['model']} model of {criterion} locations",
                          **style)
        colour = line.get_color()
        if band:
            lower = curve[f"lower_{unit}"].to_numpy(dtype=float)
            upper = curve[f"upper_{unit}"].to_numpy(dtype=float)
            fill = ax.fill_betweenx if swept_t else ax.fill_between
            fill(held, lower, upper, color=colour, alpha=_BOUNDARY_BAND_ALPHA, linewidth=0,
                 label="1σ band on the model")
    if colour is None:
        colour = _OVERLAY_COLORS[0]

    sweeps = fit.sweeps
    for status, face in (("used", colour), ("excluded by caller", "none")):
        rows = sweeps[sweeps["status"] == status]
        if not len(rows):
            continue
        loc = rows[loc_col].to_numpy(dtype=float)
        at = rows[held_col].to_numpy(dtype=float)
        err = rows[err_col].to_numpy(dtype=float)
        if status == "used":
            label = f"{criterion} ({len(rows)} sweep{'s' if len(rows) != 1 else ''})"
        else:
            label = status
        x, y = (loc, at) if swept_t else (at, loc)
        bars = {"xerr": err} if swept_t else {"yerr": err}
        ax.errorbar(x, y, **bars, fmt=marker, color=colour, markerfacecolor=face,
                    markeredgecolor=colour, linestyle="none", label=label, zorder=3)
    if not ax.get_xlabel():
        ax.set_xlabel(axis_labels("temperature", "k"))
    if not ax.get_ylabel():
        ax.set_ylabel(axis_labels("field", "tesla"))
    return ax
