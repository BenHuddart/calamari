"""Curve and fit rendering."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from calamari._plot_data import (
    _CHI_SYMBOLS,
    _CHI_UNITS,
    PlotSeries,
    chi_series,
    fisher_series,
    moment_series,
)
from calamari.exceptions import PlotInputError, PlotTypeError, UnknownFigureFormatError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike

    from calamari.fitting import FitResult, FitWindowsResult
    from calamari.measurement import Measurement
    from calamari.sample import Sample

# ---------------------------------------------------------------------------
# Layer 2 -- drawing. Thin matplotlib wrappers over the resolution layer.
# ---------------------------------------------------------------------------


def _make_ax(ax: Axes | None) -> Axes:
    """Return ``ax``, or create a fresh single-axes figure when None.

    pyplot is imported here, not at module level: importing it selects a
    backend, and doing that at ``import calamari`` time would preempt a
    script's own ``matplotlib.use(...)`` call. Deferred to first use, the
    backend choice stays with the user.
    """
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    return ax


DEFAULT_SAVE_FORMAT = "pdf"


def save(
    target: Axes | Figure,
    path: str | os.PathLike[str],
    *,
    formats: Sequence[str] | None = None,
    close: bool = False,
    **savefig_kwargs: Any,
) -> list[Path]:
    """Write a figure to one or more files, from the axes a helper returned.

    A thin, deliberate wrapper over ``Figure.savefig``: it resolves the
    figure from an ``Axes`` (so a helper's return value can be saved
    directly), writes sibling formats from one stem, creates the output
    directory, and validates every format *before* writing anything. It
    adds nothing to the file -- no metadata, no stamp; matplotlib's own
    producer fields are left as they are. Publication suitability (vector
    output, 600 dpi for rasterised parts, TrueType-embedded fonts) comes
    from the journal style's ``rcParams`` (:mod:`calamari.style`), not
    from this function, so a figure saved without a style is saved as
    matplotlib would save it.

    The path rule is one sentence: a suffix that names a matplotlib
    format is a format, and the rest is the stem. ``save(ax, "fig1.pdf")``
    writes that file; ``save(ax, "fig1", formats=("pdf", "png"))`` writes
    ``fig1.pdf`` and ``fig1.png``; ``save(ax, "fig1.pdf", formats=("png",))``
    writes both as well. A bare stem with no ``formats`` writes
    :data:`DEFAULT_SAVE_FORMAT`; a suffix matplotlib does not know
    (``"fig1.v2"``) is part of the stem, so ``fig1.v2.pdf`` results.
    Existing files are overwritten, as ``savefig`` overwrites them: an
    analysis script re-run should replace its figures.

    Parameters
    ----------
    target : matplotlib.axes.Axes or matplotlib.figure.Figure
        What to save. An ``Axes`` saves its whole figure (every panel).
    path : str or path-like
        Output file, or output stem when ``formats`` is given.
    formats : sequence of str, optional
        Additional formats to write beside ``path``'s own, each to the
        same stem. Case-insensitive; a leading dot is tolerated.
    close : bool, optional
        Close the figure after the last file is written. Off by default so
        a notebook still shows the figure inline; a script producing many
        figures passes ``True`` to keep pyplot's figure registry small.
    **savefig_kwargs
        Forwarded to ``Figure.savefig`` unchanged (``dpi=``,
        ``transparent=``, ``bbox_inches=``, ...). ``format`` is not
        accepted -- the path and ``formats`` decide it.

    Returns
    -------
    list of pathlib.Path
        The files written, in the order written: the path's own format
        first, then ``formats`` in the order given.

    Raises
    ------
    UnknownFigureFormatError
        A requested format is not one the figure's canvas can write.
        Raised before any file is written.
    PlotTypeError
        An unsupported target or a ``format=`` keyword (use ``formats=`` or
        a suffix instead). Also a ``TypeError``; no files are written.
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    if "format" in savefig_kwargs:
        raise PlotTypeError(
            "save() does not accept format=; give the format as the path's suffix "
            "or in formats=."
        )
    fig = target
    while not isinstance(fig, Figure):
        # An Axes knows its figure; an Axes inside a SubFigure knows the
        # SubFigure, which knows the root. Stop when nothing new appears.
        parent = getattr(fig, "figure", None)
        if parent is None or parent is fig:
            raise PlotTypeError(f"save() needs an Axes or a Figure, got {type(target).__name__}")
        fig = parent

    supported = fig.canvas.get_supported_filetypes()
    path = Path(path)
    own_format = path.suffix[1:].lower()
    has_own = own_format in supported
    stem = path.with_suffix("") if has_own else path

    wanted: list[str] = [own_format] if has_own else []
    for fmt in formats or ():
        fmt = str(fmt).lower().lstrip(".")
        if fmt not in wanted:
            wanted.append(fmt)
    if not wanted:
        wanted.append(DEFAULT_SAVE_FORMAT)
    unknown = [fmt for fmt in wanted if fmt not in supported]
    if unknown:
        raise UnknownFigureFormatError(
            f"cannot save {path}: format {unknown[0]!r} is not one this backend writes; "
            f"supported: {', '.join(sorted(supported))}. Nothing was written."
        )

    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in wanted:
        # The path's own file keeps the spelling it was given (fig1.PDF stays
        # fig1.PDF); siblings take the lower-cased format as their suffix.
        out = path if (has_own and fmt == own_format) else stem.with_name(f"{stem.name}.{fmt}")
        fig.savefig(out, format=fmt, **savefig_kwargs)
        written.append(out)
    if close:
        plt.close(fig)
    return written


#: Keyword name of the private display y-transform that :func:`plot_family`
#: threads through the drawing helpers for ``offset=`` / ``baseline=``. It
#: travels inside ``**style`` and each helper pops it before anything
#: reaches matplotlib, so the helpers' public signatures do not grow a
#: parameter no caller should pass by hand.
_Y_TRANSFORM_KW = "_y_transform"


def _apply_y_transform(
    transform: Callable[[list[PlotSeries]], list[PlotSeries]] | None,
    group: list[PlotSeries],
) -> list[PlotSeries]:
    """Apply a display y-transform to one baseline group of resolved series.

    A *group* is everything that must share one baseline: all the series
    one measurement resolves to for one quantity (the branches of an M(H)
    loop, the frequencies of a multi-frequency sweep). The transform runs
    after resolution and before any axes exist, so a failure in it strands
    nothing on the figure.
    """
    return group if transform is None else transform(group)


def _draw_series(ax: Axes, series: PlotSeries, *, show_yerr: bool, style: dict) -> None:
    """Draw one PlotSeries: errorbar when it has informative errors.

    "Informative" excludes an all-zero error column: the instrument
    writes 0 where only one scan was averaged (most single-shot files --
    see ``Measurement.moment_err_emu``), and zero-length bars would add
    errorbar artists and legend handles while conveying nothing.
    """
    has_err = (
        show_yerr
        and series.yerr is not None
        and bool(np.any(np.isfinite(series.yerr) & (series.yerr > 0)))
    )
    if has_err:
        ax.errorbar(series.x, series.y, yerr=series.yerr, label=series.label, **style)
    else:
        ax.plot(series.x, series.y, label=series.label, **style)
    ax.set_xlabel(series.xlabel)
    ax.set_ylabel(series.ylabel)


def _refuse_field_unit(style: dict, name: str) -> None:
    """Refuse ``field_unit=`` on a helper whose x-axis is temperature.

    These helpers have no field axis, so they take no ``field_unit``
    parameter, and the keyword would otherwise fall through ``**style`` to
    matplotlib and die there with an ``AttributeError`` about ``Line2D``
    that names neither calamari nor the mistake. The usual route in is a
    family call (``plot_chi_vs_t_family(..., field_unit='tesla')``).
    """
    if "field_unit" in style:
        raise PlotInputError(
            f"{name} plots against temperature and has no field axis, so it takes "
            "no field_unit=. It applies to the field-axis helpers (plot_chi_vs_h, "
            "plot_mh and their families) and to chi_series/moment_series with "
            "x='field'."
        )


def _imag_series_label(base: str, imag_scale: float) -> str:
    """Append a "(x<scale>)" note to an imaginary-component label if scaled.

    ``chi''`` is typically 10-100x smaller than ``chi'`` (weaker
    out-of-phase response), so :func:`plot_chi_vs_t`'s ``imag_scale``
    exists to bring it onto a visually comparable scale to chi' -- but a
    silently rescaled curve is misleading unless the legend says so.
    """
    if imag_scale == 1.0:
        return base
    return f"{base} (x{imag_scale:g})"


def _plot_chi(
    measurement: Measurement,
    x: str,
    *,
    ax: Axes | None,
    field_unit: str = "oe",
    component: str,
    units: str,
    sample: Sample | None,
    source: str,
    inverse: bool,
    imag_scale: float,
    label: str | None,
    yerr: bool,
    style: dict,
) -> Axes:
    """Shared body of :func:`plot_chi_vs_t` and :func:`plot_chi_vs_h`.

    Every series is resolved BEFORE any axes are created or drawn: a
    resolution failure must raise cleanly, never strand a half-drawn
    figure with one mislabelled series (or leak an empty auto-created
    figure when ``ax`` was omitted).

    ``style`` may carry the private family y-transform (see
    :data:`_Y_TRANSFORM_KW`). It is applied to the plotted values -- after
    ``imag_scale`` -- once per component, so with ``component='both'``
    chi' and chi'' each get their own baseline.
    """
    style = dict(style)
    transform = style.pop(_Y_TRANSFORM_KW, None)
    if component not in ("real", "imag", "both"):
        raise PlotInputError(f"component must be 'real', 'imag', or 'both', got {component!r}")
    common = dict(x=x, field_unit=field_unit, units=units, sample=sample, inverse=inverse)

    if component == "both":
        # 'both' is inherently an AC question -- only AC data has a chi''.
        # source='dc' is refused outright, and 'auto' resolves as AC without
        # the AC-versus-DC refusal: on a file that also carries a DC moment,
        # DC could never have supplied the pair anyway, so there is no
        # genuine ambiguity to refuse over.
        if source == "dc":
            raise PlotInputError(
                "component='both' requires AC data: a DC measurement has no "
                "chi''. Use component='real' (optionally with source='dc')."
            )
        prefix = f"{label} " if label else ""
        real_symbol, real_inverse = _CHI_SYMBOLS[("ac", "real")]
        imag_symbol, imag_inverse = _CHI_SYMBOLS[("ac", "imag")]
        real = chi_series(
            measurement,
            component="real",
            source="ac",
            label=prefix + (real_inverse if inverse else real_symbol),
            **common,
        )
        imag = [
            replace(
                series,
                y=series.y * imag_scale,
                yerr=None if series.yerr is None else series.yerr * imag_scale,
                label=_imag_series_label(series.label, imag_scale),
            )
            for series in chi_series(
                measurement,
                component="imag",
                source="ac",
                label=prefix + (imag_inverse if inverse else imag_symbol),
                **common,
            )
        ]
        # One baseline per component: chi' and chi'' are different
        # quantities, so a baseline pooled across them would mean nothing.
        resolved = _apply_y_transform(transform, real) + _apply_y_transform(transform, imag)
        ax = _make_ax(ax)
        for series in resolved:
            _draw_series(ax, series, show_yerr=yerr, style=style)
        # One shared y-axis: the generic chi label, not chi'-only or
        # chi''-only (whichever series drew last would otherwise win).
        _, unit_text, inverse_unit_text = _CHI_UNITS[units]
        symbol = r"$1/\chi$" if inverse else r"$\chi$"
        ax.set_ylabel(f"{symbol} ({inverse_unit_text if inverse else unit_text})")
        return ax

    resolved = chi_series(
        measurement, component=component, source=source, label=label, **common
    )
    if component == "imag":
        resolved = [
            replace(
                series,
                y=series.y * imag_scale,
                yerr=None if series.yerr is None else series.yerr * imag_scale,
                label=(
                    None
                    if series.label is None
                    else _imag_series_label(series.label, imag_scale)
                ),
            )
            for series in resolved
        ]
    resolved = _apply_y_transform(transform, resolved)
    ax = _make_ax(ax)
    for series in resolved:
        _draw_series(ax, series, show_yerr=yerr, style=style)
    return ax


def plot_chi_vs_t(
    measurement: Measurement,
    *,
    ax: Axes | None = None,
    component: str = "real",
    units: str = "emu_per_oe",
    sample: Sample | None = None,
    source: str = "auto",
    inverse: bool = False,
    imag_scale: float = 1.0,
    label: str | None = None,
    yerr: bool = True,
    **style: Any,
) -> Axes:
    """Plot susceptibility (or 1/chi) against temperature.

    The drawing half over :func:`chi_series` -- see there for how the
    source (AC ``chi'`` vs DC ``M/H``) is chosen and stamped into the
    y-label, how units convert through ``sample``, and how a
    multi-frequency file becomes one labelled series per frequency.

    Error bars are drawn whenever the file records per-point errors
    (converted with the same factors as the values); pass ``yerr=False``
    to suppress them. An all-zero error column -- the instrument's "one
    scan averaged" convention -- is treated as no errors.

    Parameters
    ----------
    measurement : Measurement
        The loaded file.
    ax : matplotlib.axes.Axes, optional
        Draw into these axes; omitted, a new single-axes figure is
        created and the axes returned either way::

            ax = plot_chi_vs_t(m, units="molar", sample=sample)
            calamari.plot.save(ax, "chi.pdf")

    component : {'real', 'imag', 'both'}, optional
        ``'both'`` overlays chi' and chi'' as labelled series on one
        shared y-axis (use ``imag_scale`` when chi'' would be invisible
        at chi' scale). ``'both'`` is inherently AC -- a DC measurement
        has no chi'' -- so it refuses ``source='dc'``, and ``'auto'``
        resolves it as AC even on a file that also carries a DC moment.
    units, sample, source, inverse
        Forwarded to :func:`chi_series`; see there.
    imag_scale : float, optional
        Display-only multiplier on the chi'' values (and error bars);
        the series' legend label gets a ``(x<scale>)`` note so a
        rescaled curve is never mistaken for data. Default 1.0.
    label : str, optional
        Legend label (``component='both'`` and frequency splits extend
        it per series).
    yerr : bool, optional
        Draw error bars when available. Default True.
    **style
        Forwarded to ``ax.plot``/``ax.errorbar`` for every series.
        ``field_unit=`` is refused by name: this plot has no field axis
        (it belongs to :func:`plot_chi_vs_h`).

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into (artists via ``ax.lines``/``ax.containers``).
    """
    _refuse_field_unit(style, "plot_chi_vs_t")
    return _plot_chi(
        measurement, "temperature", ax=ax, component=component, units=units,
        sample=sample, source=source, inverse=inverse, imag_scale=imag_scale,
        label=label, yerr=yerr, style=style,
    )


def plot_chi_vs_h(
    measurement: Measurement,
    *,
    ax: Axes | None = None,
    field_unit: str = "oe",
    component: str = "real",
    units: str = "emu_per_oe",
    sample: Sample | None = None,
    source: str = "auto",
    inverse: bool = False,
    imag_scale: float = 1.0,
    label: str | None = None,
    yerr: bool = True,
    **style: Any,
) -> Axes:
    """Plot susceptibility (or 1/chi) against applied field.

    Identical to :func:`plot_chi_vs_t` with field on the x-axis -- the
    natural view of an isothermal chi(H) sweep. ``field_unit='oe'`` (the
    default, the file's native unit) or ``'tesla'`` (mu0 H, matching the
    family legends and the 2-D maps) picks the field unit; see
    :func:`chi_series`. Only the axis changes: the susceptibility stays in
    its own ``units``. See :func:`plot_chi_vs_t` for every other parameter.
    """
    return _plot_chi(
        measurement, "field", ax=ax, field_unit=field_unit, component=component,
        units=units, sample=sample, source=source, inverse=inverse, imag_scale=imag_scale,
        label=label, yerr=yerr, style=style,
    )


def plot_mh(
    measurement: Measurement,
    *,
    ax: Axes | None = None,
    sample: Sample | None = None,
    units: str = "emu",
    field_unit: str = "oe",
    split_branches: bool = False,
    label: str | None = None,
    yerr: bool = True,
    **style: Any,
) -> Axes:
    """Plot moment against applied field, M(H).

    Points are drawn in **file order**, never sorted by field, so a
    hysteresis loop keeps its branches; ``split_branches=True`` names
    them (``branch 1 (up)``, ...) as separate legend entries -- see
    :func:`moment_series`. Field is plotted in Oe (the native unit) unless
    ``field_unit='tesla'`` asks for mu0 H; the moment axis follows
    ``units``, with the ``muB`` forms converting through ``sample``. Error
    bars follow the same rules as :func:`plot_chi_vs_t`.

    Parameters
    ----------
    measurement : Measurement
        The loaded file.
    ax : matplotlib.axes.Axes, optional
        Draw into these axes, or omit to create a figure.
    sample : Sample, optional
        Required for ``units='muB_per_ion'``/``'muB_per_fu'``.
    units : {'emu', 'muB_per_ion', 'muB_per_fu'}, optional
        Moment axis units. Default ``'emu'``.
    field_unit : {'oe', 'tesla'}, optional
        Field axis unit: Oe, or mu0 H in Tesla. Branches split at the same
        points either way. Default ``'oe'``.
    split_branches : bool, optional
        One labelled series per monotonic sweep segment. Default False.
    label : str, optional
        Legend label (branch suffixes extend it when splitting).
    yerr : bool, optional
        Draw error bars when available. Default True.
    **style
        Forwarded to ``ax.plot``/``ax.errorbar`` for every series.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into.
    """
    transform = style.pop(_Y_TRANSFORM_KW, None)  # style is this call's own dict
    resolved = moment_series(
        measurement, x="field", field_unit=field_unit, units=units, sample=sample,
        split_branches=split_branches, label=label,
    )
    # The branches of one loop are one measurement of one quantity: they
    # take one pooled baseline, so the loop keeps its shape when shifted.
    resolved = _apply_y_transform(transform, resolved)
    ax = _make_ax(ax)  # after resolution: a failure must not leak a figure
    for series in resolved:
        _draw_series(ax, series, show_yerr=yerr, style=style)
    return ax


def plot_mt(
    measurement: Measurement,
    *,
    ax: Axes | None = None,
    sample: Sample | None = None,
    units: str = "emu",
    label: str | None = None,
    yerr: bool = True,
    **style: Any,
) -> Axes:
    """Plot moment against temperature, M(T).

    As :func:`plot_mh` but with temperature (K) on the x-axis and no
    branch splitting. (For susceptibility from a DC M(T) at fixed field,
    use :func:`plot_chi_vs_t` with ``source='dc'`` -- that is the M/H
    route, and it labels its provenance.) ``field_unit=`` is refused by
    name: this plot has no field axis.
    """
    _refuse_field_unit(style, "plot_mt")
    transform = style.pop(_Y_TRANSFORM_KW, None)  # style is this call's own dict
    resolved = moment_series(
        measurement, x="temperature", units=units, sample=sample, label=label
    )
    resolved = _apply_y_transform(transform, resolved)
    ax = _make_ax(ax)  # after resolution: a failure must not leak a figure
    for series in resolved:
        _draw_series(ax, series, show_yerr=yerr, style=style)
    return ax


def plot_fisher(
    measurement: Measurement,
    *,
    ax: Axes | None = None,
    units: str = "emu_per_oe",
    sample: Sample | None = None,
    source: str = "auto",
    smooth_window: float | None = None,
    polyorder: int = 3,
    label: str | None = None,
    **style: Any,
) -> Axes:
    """Plot Fisher's d(chi T)/dT against temperature.

    The drawing half over :func:`fisher_series` -- see there for the
    units, the source and, above all, why ``smooth_window`` should be the
    one the detection resolved. It composes with
    :func:`overlay_transitions` like every sweep plot, so the Fisher
    candidates can be marked on the curve they were found in, and a cusp
    candidate from the same detection beside them::

        r = detect_transitions(m, methods=("peak", "fisher"))
        w = r.params["per_signal"]["chi_real"]["smooth_window"]
        ax = plot_fisher(m, smooth_window=w, units="molar", sample=s)
        overlay_transitions(r, ax=ax)

    Parameters
    ----------
    measurement : Measurement
        One temperature sweep.
    ax : matplotlib.axes.Axes, optional
        Draw into these axes; omitted, a new single-axes figure is created.
    units, sample, source, smooth_window, polyorder, label
        Forwarded to :func:`fisher_series`.
    **style
        Forwarded to ``ax.plot`` for every series. ``field_unit=`` is
        refused by name: this plot has no field axis.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into.
    """
    _refuse_field_unit(style, "plot_fisher")
    resolved = fisher_series(
        measurement, units=units, sample=sample, source=source,
        smooth_window=smooth_window, polyorder=polyorder, label=label,
    )
    ax = _make_ax(ax)  # after resolution: a failure must not leak a figure
    for series in resolved:
        _draw_series(ax, series, show_yerr=False, style=style)
    return ax


def plot_fit(
    x: ArrayLike,
    result: FitResult,
    *,
    ax: Axes | None = None,
    n_dense: int = 400,
    label: str | None = None,
    **style: Any,
) -> Axes:
    """Overlay a fitted model curve.

    Evaluates ``result`` (see :mod:`calamari.fitting`) on a dense, evenly
    spaced grid spanning the range of ``x`` and plots it -- the usual
    "smooth curve through scattered data" overlay. Does not touch axis
    labels or limits, so it composes cleanly onto axes a preceding
    ``plot_*`` call has already labelled.

    Parameters
    ----------
    x : array_like
        The x-data the fit was performed over (or any range to evaluate
        the model across); only ``min(x)``/``max(x)`` are used.
    result : FitResult
        A fit result from :func:`calamari.fitting.fit <calamari.fitting.engine.fit>`,
        evaluated via :meth:`~calamari.fitting.engine.FitResult.curve`.
    ax : matplotlib.axes.Axes, optional
        The axes to overlay on -- usually the return value of the
        ``plot_*`` call that drew the data.
    n_dense : int, optional
        Number of points in the dense evaluation grid. Default 400.
    label : str, optional
        Legend label.
    **style
        Forwarded to ``ax.plot``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into.
    """
    x_arr = np.asarray(x, dtype=float)
    x_dense = np.linspace(np.nanmin(x_arr), np.nanmax(x_arr), n_dense)
    y_dense = result.curve(x_dense)
    ax = _make_ax(ax)  # after evaluation: a failure must not leak a figure
    ax.plot(x_dense, y_dense, label=label, **style)
    return ax


#: Window-edge columns :func:`plot_fit_windows` can put on x, with the axis
#: label each gets. The scan does not know the unit of x (a window over
#: temperature, field or anything else), so the label names the edge only.
_FIT_WINDOW_EDGES = {
    "x_min": "Fit window lower bound",
    "x_max": "Fit window upper bound",
}

#: How a failed window is marked: a thin grey dotted vertical line at its
#: edge. Grey and dotted so it reads as "nothing here" beside any data
#: colour; drawn with fixed style because the caller's ``**style`` describes
#: the estimates, not the gaps between them.
_FAILED_WINDOW_STYLE = {"color": "0.6", "linestyle": ":", "linewidth": 1.0}


def plot_fit_windows(
    scan: FitWindowsResult,
    parameter: str,
    *,
    x: str = "x_min",
    ax: Axes | None = None,
    **style: Any,
) -> Axes:
    """Plot one fitted parameter against the fitting window's edge.

    The picture that goes with :func:`calamari.fitting.fit_windows
    <calamari.fitting.windows.fit_windows>`: ``<parameter> +/- <parameter>_err``
    for every window, against the window's lower (or upper) bound. A
    parameter that depends on the fitting range shows up as a trend that the
    error bars do not cover. Nothing here judges whether the trend matters;
    that stays with the caller, as it does for the scan itself. A quantity
    the scan derived (``fit_windows(..., derived={"mu_eff": ...})``) plots
    the same way, with its propagated error as the bar.

    **Failed windows are skipped visibly, not silently.** A window whose fit
    failed (or that had no more points than free parameters) has no
    estimate, so it gets no point. Instead a thin grey dotted vertical line
    is drawn at its edge, so a gap in the series cannot be mistaken for a
    window that was never asked for. These lines carry no legend label and
    ignore ``**style``.

    Parameters
    ----------
    scan : FitWindowsResult
        The result of ``fit_windows``.
    parameter : str
        A model parameter or derived-quantity name, i.e. a column of
        ``scan.table`` with a matching ``<parameter>_err`` column.
    x : {'x_min', 'x_max'}, optional
        Which window edge to plot against. Default ``'x_min'``, the usual
        way to show a drift as the lower bound of the range rises.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created when omitted.
    **style
        Forwarded to ``ax.errorbar`` for the estimates (e.g. ``marker``,
        ``capsize``, ``label``).

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into. The x-label names the window edge and the
        y-label the parameter; neither knows the unit, so relabel with the
        physical quantity for a figure.

    Raises
    ------
    PlotInputError
        If ``x`` is not a window edge, or ``parameter`` is neither a parameter
        of the scanned model nor a quantity the scan derived.
    """
    if x not in _FIT_WINDOW_EDGES:
        raise PlotInputError(f"x must be one of {sorted(_FIT_WINDOW_EDGES)}, got {x!r}")
    table = scan.table
    err_column = f"{parameter}_err"
    if parameter in _FIT_WINDOW_EDGES or err_column not in table.columns:
        derived = list(scan.params.get("derived") or {})
        names = [
            c[: -len("_err")]
            for c in table.columns
            if c.endswith("_err") and c[: -len("_err")] not in derived
        ]
        raise PlotInputError(
            f"{parameter!r} is not a parameter of the scanned model "
            f"{scan.params.get('model')!r} or a quantity the scan derived; "
            f"parameters are {names}, derived quantities are {derived}."
        )

    ok = table["success"].to_numpy(dtype=bool)
    edge = table[x].to_numpy(dtype=float)
    ax = _make_ax(ax)  # after validation: a refused call must not leak a figure
    if ok.any():
        ax.errorbar(
            edge[ok],
            table[parameter].to_numpy(dtype=float)[ok],
            yerr=table[err_column].to_numpy(dtype=float)[ok],
            **style,
        )
    for value in edge[~ok]:
        ax.axvline(value, **_FAILED_WINDOW_STYLE)
    ax.set_xlabel(_FIT_WINDOW_EDGES[x])
    ax.set_ylabel(parameter)
    return ax
