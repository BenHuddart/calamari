"""Backend-free series and grid preparation; public exports live in calamari.plot."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from calamari._sweep_rules import BACKTRACK_FRACTION as _MAP_BACKTRACK_FRACTION
from calamari._sweep_rules import frequency_clusters as _frequency_clusters
from calamari._sweep_rules import select_frequency as _select_recorded_frequency
from calamari._sweep_rules import sweep_backtracking as _sweep_backtracking
from calamari.constants import OE_PER_TESLA
from calamari.exceptions import PlotInputError
from calamari.measurement import SCAN_H_TOL_OE, SCAN_T_TOL_K
from calamari.quality import (
    _OUTLIER_WARN_LIST,
    CONTINUITY_THRESHOLD,
    CONTINUITY_WINDOW,
    continuity_scores,
)
from calamari.repeats import REPEAT_FREQUENCY_RTOL

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from calamari.measurement import Measurement
    from calamari.sample import Sample


#: Quantity/units label text, one home for every axis label calamari
#: produces, so plot_mh/plot_mt/plot_chi_vs_t never hand-format a string
#: themselves (see :func:`axis_labels`). Keys are ``(quantity, units)``
#: pairs; values use matplotlib mathtext where a symbol warrants it.
_AXIS_LABELS = {
    ("field", "oe"): "Field (Oe)",
    ("field", "tesla"): r"Field, $\mu_0 H$ (T)",
    ("moment", "emu"): "Moment (emu)",
    ("moment", "muB_per_ion"): r"Moment ($\mu_B$/ion)",
    ("moment", "muB_per_fu"): r"Moment ($\mu_B$/f.u.)",
    ("temperature", "k"): "Temperature (K)",
    ("chi", "real"): r"$\chi'$ (emu/Oe)",
    ("chi", "imag"): r"$\chi''$ (emu/Oe)",
    ("chi", "both"): r"$\chi$ (emu/Oe)",
}

#: Moment units that require a :class:`~calamari.sample.Sample` to convert
#: into (physics-unit conversions need mass/molar-mass/ion-count facts that
#: live only on ``Sample`` -- see :mod:`calamari.sample`).
_SAMPLE_REQUIRED_UNITS = ("muB_per_ion", "muB_per_fu")


def axis_labels(quantity: str, units: str) -> str:
    """Return the axis-label text for a (quantity, units) pair.

    The one home for every label string calamari's plot functions use, so
    the same physical quantity is always labelled identically regardless of
    which helper produced the plot.

    Parameters
    ----------
    quantity : str
        One of ``'field'``, ``'moment'``, ``'temperature'``, ``'chi'``.
        Case-insensitive.
    units : str
        The unit/variant for that quantity: ``'oe'``/``'tesla'`` for
        ``'field'``; ``'emu'``/``'muB_per_ion'``/``'muB_per_fu'`` for
        ``'moment'``; ``'k'`` for ``'temperature'``; ``'real'``/``'imag'``
        for ``'chi'``. Matched case-sensitively first (so the mixed-case
        ``'muB_per_ion'``/``'muB_per_fu'`` unit spellings used elsewhere in
        calamari -- e.g. :func:`plot_mh`'s ``units`` parameter -- pass
        through unchanged), falling back to a case-insensitive match.

    Returns
    -------
    str
        Axis-label text, using matplotlib mathtext where a symbol
        warrants it.

    Raises
    ------
    PlotInputError
        If ``(quantity, units)`` names no known label.
    """
    quantity_key = quantity.lower()
    for candidate_units in (units, units.lower()):
        key = (quantity_key, candidate_units)
        if key in _AXIS_LABELS:
            return _AXIS_LABELS[key]
    raise PlotInputError(
        f"No axis label for quantity={quantity!r}, units={units!r}; known "
        f"combinations are {sorted(_AXIS_LABELS)}."
    )


def _require_column(series: Any, *, what: str, measurement: Measurement) -> NDArray[np.float64]:
    """Return a data column as a float ndarray, or raise a clear error.

    Every calamari accessor returns ``None`` for a column absent from the
    parsed file (see :class:`calamari.measurement.Measurement`); plotting
    that ``None`` would fail deep inside the backend with an opaque error,
    so this checks at the plot.py boundary instead and names exactly which
    column and file are the problem.
    """
    if series is None:
        raise PlotInputError(f"{measurement.path} has no {what} column to plot.")
    return np.asarray(series, dtype=float)


def _convert_moment(
    moment_emu: NDArray[np.float64], *, sample: Sample | None, units: str
) -> NDArray[np.float64]:
    """Convert a moment array (emu) to the requested plotting ``units``.

    Parameters
    ----------
    moment_emu : numpy.ndarray
        Moment in emu.
    sample : Sample or None
        Required (and used) for ``units`` other than ``'emu'``.
    units : str
        ``'emu'``, ``'muB_per_ion'``, or ``'muB_per_fu'``.

    Raises
    ------
    PlotInputError
        If ``units`` needs a ``Sample`` and none was given, or ``units`` is
        not recognised.
    """
    if units == "emu":
        return moment_emu
    if units in _SAMPLE_REQUIRED_UNITS:
        if sample is None:
            raise PlotInputError(
                f"units={units!r} requires a Sample (mass/molar mass/ion count); "
                "pass sample=... or use units='emu'."
            )
        if units == "muB_per_ion":
            return np.asarray(sample.moment_emu_to_muB_per_ion(moment_emu), dtype=float)
        return np.asarray(sample.moment_emu_to_muB_per_fu(moment_emu), dtype=float)
    raise PlotInputError(
        f"Unknown units {units!r}; expected 'emu', 'muB_per_ion', or 'muB_per_fu'."
    )


# ---------------------------------------------------------------------------
# Layer 1 -- quantity resolution. Everything about WHAT to plot happens
# here (source selection, unit conversion, housekeeping exclusion, NaN
# dropping, frequency splitting, error propagation), producing PlotSeries
# objects; the plot_* functions below only draw them. The seam is public:
# call chi_series/moment_series yourself when the drawing layer's choices
# do not fit.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlotSeries:
    """One drawable series: data arrays plus the labels that describe them.

    The unit of exchange between the resolution layer (:func:`chi_series`,
    :func:`moment_series`) and the drawing layer (:func:`plot_chi_vs_t`
    and friends). ``yerr`` is None when the file records no error for the
    quantity; a present ``yerr`` has been converted/propagated with the
    same factors as ``y``, so the two can never be in different units.
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    yerr: NDArray[np.float64] | None
    xlabel: str
    ylabel: str
    label: str | None


#: chi units vocabulary: key -> (Sample converter method or None for the
#: raw file quantity, unit text, inverse unit text). Descriptor keys match
#: the Sample method vocabulary (chi_molar, chi_mass); the axis label
#: spells the exact unit, so the shorthand never reaches a figure.
_CHI_UNITS = {
    "emu_per_oe": (None, "emu/Oe", "Oe/emu"),
    "molar": ("chi_to_chi_molar", "emu/(mol Oe)", "(mol Oe)/emu"),
    "molar_per_ion": (
        "chi_to_chi_molar_per_ion",
        r"emu/(mol$_{\mathrm{ion}}$ Oe)",
        r"(mol$_{\mathrm{ion}}$ Oe)/emu",
    ),
    "mass": ("chi_to_chi_mass", "emu/(g Oe)", "(g Oe)/emu"),
}

#: chi axis symbols by (source, component) -> (plain, inverse). The DC
#: symbol names its provenance -- a figure reader can tell an AC chi' from
#: a DC M/H without the caption's help, which is the whole point of the
#: explicit ``source`` parameter.
_CHI_SYMBOLS = {
    ("ac", "real"): (r"$\chi'$", r"$1/\chi'$"),
    ("ac", "imag"): (r"$\chi''$", r"$1/\chi''$"),
    ("dc", "real"): (r"$M/H$", r"$H/M$"),
}

#: x-axis vocabulary shared by chi_series/moment_series.
_X_AXES = ("temperature", "field")

#: Units a 1-D field axis can be drawn in (``field_unit=``) -> the unit-named
#: Measurement accessor that supplies the values (DM-2: the accessor, not a
#: factor at the call site, owns the conversion) and the unit's name for
#: messages. ``'oe'`` is the file's native unit and the default; ``'tesla'``
#: is mu0 H, the unit of the family legends, the 2-D maps and ``H_T``.
_FIELD_AXIS_UNITS = {"oe": ("field_oe", "Oe"), "tesla": ("field_tesla", "Tesla")}


def _check_field_unit(field_unit: str, x: str) -> None:
    """Refuse an unknown ``field_unit``, or one that names no axis.

    A temperature axis has no field on it, so a non-default ``field_unit``
    there is a caller error -- most likely an ``x=`` that was meant to be
    ``'field'`` -- and accepting it silently would hide that. The default
    ``'oe'`` is inert on a temperature axis and passes, exactly as
    ``split_branches=False`` does.
    """
    if field_unit not in _FIELD_AXIS_UNITS:
        raise PlotInputError(
            f"field_unit must be one of {sorted(_FIELD_AXIS_UNITS)}, got {field_unit!r}."
        )
    if x == "temperature" and field_unit != "oe":
        raise PlotInputError(
            f"field_unit={field_unit!r} applies to x='field' only: a temperature "
            "axis has no field on it."
        )


def _x_array(
    measurement: Measurement, x: str, field_unit: str = "oe"
) -> tuple[NDArray[np.float64], str]:
    """The x data for ``'temperature'``/``'field'`` plus its axis label.

    ``field_unit`` (already validated by :func:`_check_field_unit`) picks
    the field accessor and the matching label together, so values and
    label can never disagree about the unit.
    """
    if x not in _X_AXES:
        raise PlotInputError(f"x must be 'temperature' or 'field', got {x!r}")
    if x == "temperature":
        series, label = measurement.temperature, axis_labels("temperature", "k")
    else:
        series = getattr(measurement, _FIELD_AXIS_UNITS[field_unit][0])
        label = axis_labels("field", field_unit)
    return _require_column(series, what=x, measurement=measurement), label


def _measurement_rows(measurement: Measurement, *arrays):
    """Restrict aligned arrays to measurement rows (housekeeping excluded).

    The same rule the repr, ``summary()`` and :func:`phase_map_grid`
    follow: a locate scan or touchdown row is the instrument positioning
    itself, not a point of the sweep.
    """
    mask = measurement._measurement_mask().to_numpy()
    return tuple(None if a is None else a[mask] for a in arrays)


def _drop_unplottable(x, y, yerr):
    """Drop rows where x or y is not finite.

    This is what makes a combined VSM+AC file plot without ceremony: the
    accessor is NaN on the other family's rows, so those rows simply are
    not part of this series. A NaN in ``yerr`` alone keeps its row -- the
    value is real, only its error is unrecorded.
    """
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep], (None if yerr is None else yerr[keep]), keep


def _with_suffix(label: str | None, suffix: str) -> str:
    return f"{label} {suffix}" if label else suffix


def _resolve_chi_source(measurement: Measurement, component: str, source: str) -> str:
    """Which measurement a chi request reads: ``'ac'`` (chi', chi'') or ``'dc'`` (M/H).

    The ``source='auto'`` rule of :func:`chi_series`, in one place, because
    two callers need the answer: :func:`chi_series` to resolve the series,
    and :func:`plot_family` to know which of a file's points a chi plot will
    draw -- so that it warns about a flagged moment point only when M/H is
    what is plotted. An explicit ``'ac'``/``'dc'`` is returned as given
    (``component``/``source`` are assumed already validated).

    Raises
    ------
    PlotInputError
        Under ``'auto'``, when the file offers both an AC susceptibility and
        a DC moment, or neither.
    """
    if source != "auto":
        return source
    ac_chi = measurement.chi_real if component == "real" else measurement.chi_imag
    has_ac = ac_chi is not None
    # Probe DC availability without touching moment_emu, which raises on
    # a two-moment file; that refusal belongs to the caller who actually
    # chose source='dc', not to the probe.
    has_dc = (
        component == "real"
        and len(measurement.moment_columns) > 0
        and measurement.field_oe is not None
    )
    if has_ac and has_dc:
        raise PlotInputError(
            f"{measurement.path}: this file offers both an AC susceptibility "
            "and a DC moment, which are different measurements -- pass "
            "source='ac' (the chi' column) or source='dc' (M/H) explicitly, "
            "or take one family with of_type(...) first."
        )
    if has_ac:
        return "ac"
    if has_dc:
        return "dc"
    wanted = "chi_real" if component == "real" else "chi_imag"
    raise PlotInputError(
        f"{measurement.path}: no susceptibility to resolve -- {wanted} is "
        "absent" + (
            " and no populated moment+field pair offers an M/H fallback."
            if component == "real"
            else " (and chi'' has no DC counterpart)."
        )
    )


def chi_series(
    measurement: Measurement,
    *,
    x: str = "temperature",
    field_unit: str = "oe",
    component: str = "real",
    units: str = "emu_per_oe",
    sample: Sample | None = None,
    source: str = "auto",
    inverse: bool = False,
    label: str | None = None,
) -> list[PlotSeries]:
    """Resolve a susceptibility-vs-T (or vs-H) request into PlotSeries.

    The resolution half of :func:`plot_chi_vs_t`/:func:`plot_chi_vs_h` --
    everything except the drawing, returned as arrays so a caller can
    plot, fit or export them directly.

    **The source is never silently conflated.** An AC susceptibility
    (``chi'``, from the file's own AC columns) and a DC ``M/H`` are
    different measurements of related quantities. ``source='auto'`` picks
    whichever the file offers, stamps the provenance into the y-label
    (``$\\chi'$`` vs ``$M/H$``), and refuses -- naming both options --
    when a file offers both, rather than guessing.

    A file measured at more than one AC frequency returns one series per
    frequency, labelled ``f = <value> Hz``, instead of one series
    zigzagging between them.

    Parameters
    ----------
    measurement : Measurement
        The loaded file.
    x : {'temperature', 'field'}, optional
        Which axis to resolve against. Default ``'temperature'``.
    field_unit : {'oe', 'tesla'}, optional
        For ``x='field'``: the unit of ``PlotSeries.x`` and its label.
        ``'oe'`` (the default) is the file's native unit; ``'tesla'`` is
        mu0 H from :attr:`~calamari.measurement.Measurement.field_tesla`,
        the unit the family legends, the 2-D maps and the transition
        table's ``H_T`` already use. Only the axis changes: a DC ``M/H``
        is still ``moment_emu / field_oe``, in emu/Oe. Any value other than
        the default is refused with ``x='temperature'``.
    component : {'real', 'imag'}, optional
        ``'real'`` for chi' (or M/H), ``'imag'`` for chi''. (The drawing
        layer's ``component='both'`` is two calls to this function.)
    units : {'emu_per_oe', 'molar', 'molar_per_ion', 'mass'}, optional
        ``'emu_per_oe'`` is the file's own extensive quantity; the others
        need ``sample`` and convert via the ``Sample.chi_to_chi_*``
        methods. The axis label spells the exact unit.
    sample : Sample, optional
        Required for every ``units`` other than ``'emu_per_oe'``.
    source : {'auto', 'ac', 'dc'}, optional
        ``'ac'`` reads ``chi_real``/``chi_imag``; ``'dc'`` computes
        ``moment_emu / field_oe`` (``component='real'`` only -- a DC
        measurement has no out-of-phase part; rows at exactly zero field
        become NaN and are dropped, since M/H means nothing there).
        ``'auto'`` picks the one the file offers and refuses if both.
    inverse : bool, optional
        Return ``1/chi`` (for Curie-Weiss analysis): values inverted
        (``chi = 0`` becomes NaN and is dropped), errors propagated as
        ``yerr/chi**2``, symbol and unit labels reciprocated.
    label : str, optional
        Base legend label; a frequency suffix is appended when the file
        splits by frequency.

    Returns
    -------
    list of PlotSeries
        One entry per AC frequency present (one entry total for a
        single-frequency or DC file).

    Raises
    ------
    PlotInputError
        For an unknown ``x``/``field_unit``/``component``/``units``/
        ``source``; ``field_unit='tesla'`` with ``x='temperature'``; a
        ``units`` needing a Sample without one; a source the file cannot
        supply; or ``source='auto'`` when the file offers both.
    AmbiguousColumnError
        Propagated from ``moment_emu`` (``source='dc'``) when more than
        one moment column carries data -- choose with ``with_moment`` or
        ``of_type`` first, exactly as when reading the accessor yourself.
    """
    _check_field_unit(field_unit, x)
    if component not in ("real", "imag"):
        raise PlotInputError(f"component must be 'real' or 'imag', got {component!r}")
    if units not in _CHI_UNITS:
        raise PlotInputError(
            f"Unknown chi units {units!r}; expected one of {sorted(_CHI_UNITS)}."
        )
    if source not in ("auto", "ac", "dc"):
        raise PlotInputError(f"source must be 'auto', 'ac', or 'dc', got {source!r}")

    ac_chi = measurement.chi_real if component == "real" else measurement.chi_imag
    source = _resolve_chi_source(measurement, component, source)

    if source == "ac":
        y = _require_column(
            ac_chi, what=f"chi_{component}", measurement=measurement
        )
        err = (
            measurement.chi_real_err if component == "real" else measurement.chi_imag_err
        )
        yerr = None if err is None else np.asarray(err, dtype=float)
    else:
        if component != "real":
            raise PlotInputError(
                "source='dc' cannot supply chi'': a DC measurement has no "
                "out-of-phase component. Use component='real', or an AC file."
            )
        moment = _require_column(
            measurement.moment_emu, what="moment_emu", measurement=measurement
        )
        field = _require_column(
            measurement.field_oe, what="field_oe", measurement=measurement
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(field != 0.0, moment / field, np.nan)
        err = measurement.moment_err_emu
        yerr = (
            None
            if err is None
            else np.where(field != 0.0, np.asarray(err, dtype=float) / np.abs(field), np.nan)
        )

    method_name, unit_text, inverse_unit_text = _CHI_UNITS[units]
    if method_name is not None:
        if sample is None:
            raise PlotInputError(
                f"units={units!r} requires a Sample (mass/molar mass/ion count); "
                "pass sample=... or use units='emu_per_oe'."
            )
        convert = getattr(sample, method_name)
        y = np.asarray(convert(y), dtype=float)
        # The conversion is a division by a positive scalar, so the error
        # converts with the same call and the fractional error is invariant.
        if yerr is not None:
            yerr = np.asarray(convert(yerr), dtype=float)

    x_arr, xlabel = _x_array(measurement, x, field_unit)
    frequency = measurement.ac_frequency if source == "ac" else None
    freq_arr = None if frequency is None else np.asarray(frequency, dtype=float)
    x_arr, y, yerr, freq_arr = _measurement_rows(measurement, x_arr, y, yerr, freq_arr)

    if inverse:
        with np.errstate(divide="ignore", invalid="ignore"):
            if yerr is not None:
                yerr = np.where(y != 0.0, yerr / y**2, np.nan)
            y = np.where(y != 0.0, 1.0 / y, np.nan)

    x_arr, y, yerr, keep = _drop_unplottable(x_arr, y, yerr)
    freq_arr = None if freq_arr is None else freq_arr[keep]

    symbol, inverse_symbol = _CHI_SYMBOLS[(source, component)]
    ylabel = (
        f"{inverse_symbol} ({inverse_unit_text})" if inverse else f"{symbol} ({unit_text})"
    )

    distinct = (
        np.unique(freq_arr[np.isfinite(freq_arr)]) if freq_arr is not None else np.array([])
    )
    if len(distinct) <= 1:
        return [PlotSeries(x_arr, y, yerr, xlabel, ylabel, label)]

    series = []
    for f in distinct:
        rows = freq_arr == f
        series.append(
            PlotSeries(
                x_arr[rows],
                y[rows],
                None if yerr is None else yerr[rows],
                xlabel,
                ylabel,
                _with_suffix(label, f"f = {f:g} Hz"),
            )
        )
    unknown = ~np.isfinite(freq_arr)
    if unknown.any():
        series.append(
            PlotSeries(
                x_arr[unknown],
                y[unknown],
                None if yerr is None else yerr[unknown],
                xlabel,
                ylabel,
                _with_suffix(label, "f = ?"),
            )
        )
    return series


def _branch_slices(field: NDArray[np.float64]) -> list[slice]:
    """Monotonic-sweep segments of a field array, split at reversals.

    Each slice includes the turning point at both ends, so consecutive
    branches share their reversal point and draw as a connected loop.
    A zero step continues the current direction (repeated setpoints must
    not fragment a branch).
    """
    if len(field) < 2:
        return [slice(0, len(field))]
    steps = np.sign(np.diff(field))
    for i in range(1, len(steps)):
        if steps[i] == 0:
            steps[i] = steps[i - 1]
    # steps[i] is the direction leaving field[i]; a change from steps[i-1]
    # means field[i] is the turning point, so i is a branch boundary.
    turns = [i for i in range(1, len(steps)) if steps[i] != 0 and steps[i - 1] != 0
             and steps[i] != steps[i - 1]]
    bounds = [0, *turns, len(field) - 1]
    return [slice(bounds[i], bounds[i + 1] + 1) for i in range(len(bounds) - 1)]


def moment_series(
    measurement: Measurement,
    *,
    x: str = "field",
    field_unit: str = "oe",
    units: str = "emu",
    sample: Sample | None = None,
    split_branches: bool = False,
    label: str | None = None,
) -> list[PlotSeries]:
    """Resolve a moment-vs-H (or vs-T) request into PlotSeries.

    The resolution half of :func:`plot_mh`/:func:`plot_mt`. Rows are kept
    in **file order** -- never sorted by field -- so a hysteresis loop's
    branches survive; housekeeping rows are excluded and NaN rows dropped
    exactly as in :func:`chi_series`.

    Parameters
    ----------
    measurement : Measurement
        The loaded file.
    x : {'field', 'temperature'}, optional
        Which axis to resolve against. Default ``'field'``.
    field_unit : {'oe', 'tesla'}, optional
        For ``x='field'``: field in Oe (the default, the file's native
        unit) or as mu0 H in Tesla -- see :func:`chi_series`. Branches
        split at the same points in either unit. Any value other than the
        default is refused with ``x='temperature'``.
    units : {'emu', 'muB_per_ion', 'muB_per_fu'}, optional
        Moment units; the ``muB`` forms need ``sample``.
    sample : Sample, optional
        Required for the ``muB`` units.
    split_branches : bool, optional
        For ``x='field'``: return one series per monotonic sweep segment
        (virgin curve, descending branch, ascending branch, ...), each
        labelled ``branch <n> (up|down)`` so a legend can name them.
        Consecutive branches share their turning point, so they still
        draw as a connected loop. A sweep with no reversal returns one
        plainly-labelled series, so the flag is safe to set universally.
    label : str, optional
        Base legend label; the branch suffix is appended when splitting.

    Returns
    -------
    list of PlotSeries

    Raises
    ------
    PlotInputError
        For an unknown ``x``, ``field_unit`` or ``units``; a ``units``
        needing a Sample without one; ``split_branches`` or
        ``field_unit='tesla'`` with ``x='temperature'``; or a missing
        column.
    AmbiguousColumnError
        Propagated from ``moment_emu`` when more than one moment column
        carries data.
    """
    if split_branches and x != "field":
        raise PlotInputError("split_branches applies to x='field' only.")
    _check_field_unit(field_unit, x)

    moment = _require_column(
        measurement.moment_emu, what="moment_emu", measurement=measurement
    )
    y = _convert_moment(moment, sample=sample, units=units)
    err = measurement.moment_err_emu
    yerr = (
        None
        if err is None
        else _convert_moment(np.asarray(err, dtype=float), sample=sample, units=units)
    )

    x_arr, xlabel = _x_array(measurement, x, field_unit)
    x_arr, y, yerr = _measurement_rows(measurement, x_arr, y, yerr)
    x_arr, y, yerr, _ = _drop_unplottable(x_arr, y, yerr)
    ylabel = axis_labels("moment", units)

    if not split_branches:
        return [PlotSeries(x_arr, y, yerr, xlabel, ylabel, label)]

    slices = _branch_slices(x_arr)
    if len(slices) == 1:
        return [PlotSeries(x_arr, y, yerr, xlabel, ylabel, label)]
    series = []
    for n, sl in enumerate(slices, start=1):
        direction = "up" if x_arr[sl][-1] >= x_arr[sl][0] else "down"
        series.append(
            PlotSeries(
                x_arr[sl],
                y[sl],
                None if yerr is None else yerr[sl],
                xlabel,
                ylabel,
                _with_suffix(label, f"branch {n} ({direction})"),
            )
        )
    return series


#: y-axis symbols for :func:`fisher_series`, by resolved chi source. The DC
#: form spells M T/H rather than chi T for the reason :data:`_CHI_SYMBOLS`
#: gives: a figure reader must be able to tell an AC chi' from a DC M/H
#: without the caption. Upright d's, so the operator reads as an operator.
_FISHER_SYMBOLS = {
    "ac": r"$\mathrm{d}(\chi' T)/\mathrm{d}T$",
    "dc": r"$\mathrm{d}(MT/H)/\mathrm{d}T$",
}


def fisher_series(
    measurement: Measurement,
    *,
    units: str = "emu_per_oe",
    sample: Sample | None = None,
    source: str = "auto",
    smooth_window: float | None = None,
    polyorder: int = 3,
    label: str | None = None,
) -> list[PlotSeries]:
    """Resolve Fisher's d(chi T)/dT against temperature, in physical units.

    The physical-units companion of the ``'fisher'`` detection method
    (:func:`calamari.transitions.find_transitions`). Detection works on
    whatever ``y`` it is given, because a constant positive scale moves
    neither a position nor a significance; a figure, though, has an axis
    with a unit on it, and that axis should read d(chi T)/dT in the chi
    unit of the paper. This is that curve.

    Built on :func:`chi_series`, so everything it resolves is resolved the
    same way here: ``units`` and ``sample``, the AC-versus-DC ``source``
    (refused, naming both, when a file offers both) and one series per AC
    frequency. Each series' ``T*chi`` is then differentiated with
    :func:`calamari.transitions.smooth_derivative`, the Savitzky-Golay
    routine the detector uses, on the same uniform resampling of the sweep.
    Kelvin cancels (d(chi T)/dT has chi's unit), so the y-label names the
    quantity and the chi unit, and the x of each series is that uniform
    grid, not the measured temperatures.

    **Pass the detection's window.** A smoothed derivative's extremum moves
    with the smoothing, so a curve drawn at a different window from the
    detection can put its maximum visibly beside the candidate it is meant
    to support. ``smooth_window=None`` means the detector's own default (11
    grid points), which agrees with a detection run on the same sweep with
    no window; after any other detection, pass what it resolved::

        r = detect_transitions(m, methods=("peak", "fisher"))
        w = r.params["per_signal"]["chi_real"]["smooth_window"]
        ax = plot_fisher(m, smooth_window=w)
        overlay_transitions(r, ax=ax)

    Parameters
    ----------
    measurement : Measurement
        One temperature sweep (``scan_kind == 'T-scan'``). Anything else is
        refused: d(chi T)/dT is a derivative along temperature.
    units, sample, source
        Forwarded to :func:`chi_series`; see there. The DC route is M/H,
        and it is refused, as detection refuses it, when the held field is
        within :data:`~calamari.measurement.SCAN_H_TOL_OE` of zero: there
        the moment is not a response to the field.
    smooth_window : float, optional
        Savitzky-Golay window in Kelvin, as for detection. See above.
    polyorder : int, optional
        Savitzky-Golay polynomial order (default 3, detection's default).
    label : str, optional
        Base legend label; a frequency suffix is appended when the file
        splits by frequency.

    Returns
    -------
    list of PlotSeries
        One per AC frequency present (one for a single-frequency or DC
        file). ``yerr`` is ``None``: a smoothed derivative's point errors
        are correlated across the window, and bars drawn as if they were
        not would overstate what the curve knows.

    Raises
    ------
    PlotInputError
        For a sweep that is not a T-scan; a DC source at zero held field;
        a temperature at or below 0 K; or an invalid :func:`chi_series` choice.
    calamari.exceptions.CurveInputError
        A sweep :func:`~calamari.transitions.smooth_derivative` cannot
        differentiate (too few points, not monotonic in temperature), or
        a non-positive smoothing window. Also a ``ValueError``.
    calamari.exceptions.SampleValidationError
        Sample facts required for the requested normalisation are missing.
    """
    # Lazy, like the METHODS lookups below: plot reaches into transitions
    # only for the numerics it shares, never at import time.
    from calamari.transitions import smooth_derivative

    if measurement.scan_kind != "T-scan":
        raise PlotInputError(
            f"{measurement.path}: fisher_series needs a temperature sweep, and this "
            f"file's scan_kind is {measurement.scan_kind!r}. d(chi T)/dT is a "
            "derivative along temperature; split stacked sweeps first "
            "(calamari.sweeps) if this file holds several."
        )
    resolved = chi_series(
        measurement, x="temperature", component="real", units=units, sample=sample,
        source=source, label=label,
    )
    # chi_series has already refused an ambiguous 'auto', so the file offers
    # at most one source and 'auto' resolved to whichever that is.
    used = source if source != "auto" else ("ac" if measurement.chi_real is not None else "dc")
    if used == "dc":
        field = _measurement_rows(
            measurement, _require_column(measurement.field_oe, what="field_oe",
                                         measurement=measurement)
        )[0]
        field = field[np.isfinite(field)]
        held = float(np.mean(field)) if field.size else 0.0
        if abs(held) <= SCAN_H_TOL_OE:
            raise PlotInputError(
                f"{measurement.path}: fisher_series from a DC moment needs a held, "
                f"non-zero field -- M T/H stands for chi T only in a field, and this "
                f"sweep's is {held:g} Oe, within {SCAN_H_TOL_OE:g} Oe (SCAN_H_TOL_OE) "
                "of zero."
            )

    _, unit_text, _ = _CHI_UNITS[units]
    ylabel = f"{_FISHER_SYMBOLS[used]} ({unit_text})"
    out = []
    for series in resolved:
        if series.x.size and float(np.min(series.x)) <= 0:
            raise PlotInputError(
                f"{measurement.path}: Fisher's d(chi T)/dT needs absolute "
                "temperature, but the smallest temperature here is "
                f"{float(np.min(series.x)):g} K."
            )
        grid, _, derivative = smooth_derivative(
            series.x, series.x * series.y, order=1, smooth_window=smooth_window,
            polyorder=polyorder,
        )
        out.append(PlotSeries(grid, derivative, None, series.xlabel, ylabel, series.label))
    return out


# ---------------------------------------------------------------------------
# Families of curves: one sweep per measurement, overlaid and labelled by
# whichever parameter each sweep held fixed.
# ---------------------------------------------------------------------------

#: Family label parameters: key -> (accessor name, absolute tolerance or
#: None, relative tolerance or None, legend formatter). Tolerances are the
#: same numbers the rest of calamari already uses for "is this quantity
#: held?": the scan-classification tolerances for T and H, and the repeat
#: -grouping relative tolerance for frequency (a frequency is a magnitude,
#: so its tolerance is relative -- see calamari.repeats). Field legends
#: render in Tesla, matching the logbook's held-field convention.
_FAMILY_PARAMS = {
    "temperature": ("temperature", SCAN_T_TOL_K, None, lambda v: f"T = {v:.4g} K"),
    "field": ("field_oe", SCAN_H_TOL_OE, None, lambda v: f"H = {v / OE_PER_TESLA:.4g} T"),
    "frequency": ("ac_frequency", None, REPEAT_FREQUENCY_RTOL, lambda v: f"f = {v:g} Hz"),
}

#: scan_kind -> the parameter that classification says is held fixed.
#: 'mixed' and 'static/other' name no single distinguishing parameter, so
#: label_by='auto' refuses there rather than guessing.
_FAMILY_AUTO = {"H-scan": "temperature", "T-scan": "field"}


def _family_value(measurement: Measurement, param: str) -> float:
    """The fixed value of ``param`` for one sweep, or raise if it is not fixed.

    The value is the mean over measurement rows (housekeeping excluded,
    like every aggregate description); "fixed" means the measured span
    stays within the same tolerance the rest of calamari uses for that
    quantity. A sweep whose nominally-fixed parameter actually varies must
    not be silently labelled with its mean -- that is a mislabelled curve
    on a published figure.
    """
    accessor, tol_abs, tol_rel, _ = _FAMILY_PARAMS[param]
    series = measurement._measured(getattr(measurement, accessor))
    if series is None:
        raise PlotInputError(
            f"{measurement.path}: no {param} column to label this curve by. "
            "Pass labels=[...] to name the curves yourself."
        )
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        raise PlotInputError(
            f"{measurement.path}: the {param} column carries no values to "
            "label this curve by. Pass labels=[...] to name the curves yourself."
        )
    value = float(arr.mean())
    span = float(arr.max() - arr.min())
    tolerance = tol_abs if tol_abs is not None else tol_rel * abs(value)
    if span > tolerance:
        raise PlotInputError(
            f"{measurement.path}: {param} is not fixed in this sweep -- it spans "
            f"{span:.4g} against a tolerance of {tolerance:.4g}. A curve labelled "
            f"with its mean would misrepresent it. Label by a parameter the sweep "
            "holds fixed (label_by=...), or name the curves yourself (labels=[...])."
        )
    return value


# ---------------------------------------------------------------------------
# 2-D maps on the T-H plane: chi_map_grid / moment_map_grid build a
# PlotGrid; draw_map renders one; plot_chi_map / plot_moment_map are the
# resolve-then-draw one-calls.
# ---------------------------------------------------------------------------


MAP_OUTLIER_WINDOW = CONTINUITY_WINDOW

MAP_OUTLIER_THRESHOLD = CONTINUITY_THRESHOLD


@dataclass(frozen=True)
class MapOutlier:
    """One point the map glitch check flagged, with what was done to it.

    Recorded on ``PlotGrid.outliers`` (see :class:`PlotGrid`) whichever ``outliers=`` mode
    built the grid, so a figure can always be audited for what was
    excluded (``dropped=True``) or knowingly kept. ``score`` is the
    statistic compared against :data:`MAP_OUTLIER_THRESHOLD`; the
    coordinates are in the resolved series' units (K for temperature,
    Oe for field) and ``value`` in the grid's value units.
    """

    file: Path
    family_value: float
    scan_value: float
    value: float
    score: float
    dropped: bool


@dataclass(frozen=True)
class PlotGrid:
    """A resolved 2-D map on the T-H plane, in fixed orientation.

    The 2-D counterpart of :class:`PlotSeries`, and like it the unit of
    exchange between resolution (:func:`chi_map_grid`,
    :func:`moment_map_grid`) and drawing (:func:`draw_map`). The array
    orientation is fixed -- ``values[i, j]`` belongs to ``t_k[i]`` and
    ``h_oe[j]`` -- whichever parameter the input sweeps held fixed;
    ``family_axis`` records which axis carries one row per sweep (real
    sampling) and which was gridded, as provenance rather than as
    something a consumer must branch on.

    ``values`` uses NaN for cells holding no data -- deliberately a plain
    array, not a masked array (masked arrays surprise downstream
    arithmetic and do not round-trip through ``numpy.save``); the drawing
    layer masks at render time. ``counts`` says how many real points each
    cell aggregates, so "one point, taken on trust" and "twenty points,
    averaged" stay distinguishable. ``errors`` (bin mode, and only when
    every sweep supplied per-point errors) is the standard error of each
    cell mean, propagated with the same conversions as the values.
    ``sample_t``/``sample_h`` are the coordinates of every real point that
    entered the grid, for :func:`draw_map`'s sample overlay and for
    auditing any gridding decision after the fact. ``outliers`` lists
    every point the glitch check flagged (see :func:`chi_map_grid`'s
    ``outliers=``), each saying whether it was dropped or kept.
    """

    t_k: NDArray[np.float64]
    h_oe: NDArray[np.float64]
    values: NDArray[np.float64]
    counts: NDArray[np.int64]
    errors: NDArray[np.float64] | None
    value_label: str
    family_axis: str
    frequency_hz: float | None
    sample_t: NDArray[np.float64]
    sample_h: NDArray[np.float64]
    gridding: str
    outliers: tuple[MapOutlier, ...] = ()


def _map_family_param(loaded, family: str) -> str:
    """Which parameter the sweeps hold fixed ('temperature' or 'field')."""
    if family in ("temperature", "field"):
        return family
    if family != "auto":
        raise PlotInputError(
            f"family must be 'auto', 'temperature', or 'field', got {family!r}"
        )
    params = []
    for m in loaded:
        kind = m.scan_kind
        if kind not in _FAMILY_AUTO:
            raise PlotInputError(
                f"{m.path}: scan_kind is {kind!r}, so this file is not one sweep "
                "at one fixed parameter. A single file holding many isotherms "
                "splits with calamari.sweeps.split_isotherms; otherwise pass "
                "family='temperature'/'field' (with family_values=[...] if the "
                "fixed values cannot be read from the data)."
            )
        params.append(_FAMILY_AUTO[kind])
    if len(set(params)) > 1:
        raise PlotInputError(
            "The sweeps disagree about what is fixed (mixed T-scans and "
            "H-scans cannot share one map). Pass family=... to override, or "
            "split the inputs."
        )
    return params[0]


def _map_select_frequency(loaded, frequency):
    """Resolve the frequency question for a map: one value or none.

    A map is 2-D by construction; frequency is a third axis, not a
    labelling nicety. More than one distinct frequency across the sweeps
    is refused unless ``frequency=`` selects one, in which case each
    sweep is restricted to the matching rows. Returns
    ``(possibly-restricted sweeps, selected frequency or None)``.
    """
    per_sweep = []
    for m in loaded:
        freq = m.ac_frequency
        if freq is None:
            per_sweep.append(None)
            continue
        arr = np.asarray(m._measured(freq), dtype=float)
        per_sweep.append(np.unique(arr[np.isfinite(arr)]))

    pooled = sorted({float(v) for vals in per_sweep if vals is not None for v in vals})
    clusters = _frequency_clusters(np.asarray(pooled), rtol=REPEAT_FREQUENCY_RTOL)

    if frequency is None:
        if len(clusters) > 1:
            listed = ", ".join(f"{c:g}" for c in clusters)
            raise PlotInputError(
                f"These sweeps carry {len(clusters)} distinct AC frequencies "
                f"({listed} Hz); a map is one frequency at a time. Pass "
                "frequency=<value> to select one, or build one map per "
                "frequency."
            )
        return loaded, (clusters[0] if clusters else None)

    selected = _select_recorded_frequency(
        clusters, frequency, rtol=REPEAT_FREQUENCY_RTOL, error_type=PlotInputError,
    )
    restricted = []
    for m, vals in zip(loaded, per_sweep):
        if vals is None or not np.isclose(vals, selected, rtol=REPEAT_FREQUENCY_RTOL).any():
            raise PlotInputError(
                f"{m.path}: this sweep carries no rows at frequency "
                f"{selected:g} Hz, so it cannot contribute to this map."
            )
        col = np.asarray(m.ac_frequency, dtype=float)
        keep = np.isclose(col, selected, rtol=REPEAT_FREQUENCY_RTOL)
        restricted.append(m.restrict(mask=keep))
    return restricted, selected


def _map_check_monotonic(x: NDArray[np.float64], measurement) -> None:
    """Refuse a non-monotonic scan axis rather than gridding it wrongly.

    Judged by total backtracking against the span, not by raw sign
    changes -- setpoint jitter reverses by micro-steps a monotonic scan
    can carry, while a hysteresis branch retraces a span-sized distance.
    A map row must represent one branch: sorting or averaging a retraced
    sweep would mix distinct measurements.
    """
    if len(x) < 2:
        return
    span, backtrack = _sweep_backtracking(x)
    if span == 0.0:
        raise PlotInputError(
            f"{measurement.path}: the scan axis does not vary in this sweep, "
            "so it grids no map row."
        )
    if backtrack > _MAP_BACKTRACK_FRACTION * span:
        raise PlotInputError(
            f"{measurement.path}: the scan axis is not monotonic (it backtracks "
            f"by {backtrack:.4g} over a span of {span:.4g}) -- a hysteresis "
            "loop or re-swept range. Gridding it would mix branches that are "
            "different measurements. Take one branch first, e.g. "
            "calamari.sweeps.increasing_field_branch, or restrict(mask=...)."
        )


def _map_outlier_scores(
    ys: list[NDArray[np.float64]], xs: list[NDArray[np.float64]] | None = None
) -> list[NDArray[np.float64]]:
    """Glitch score for every point of every sweep in a family.

    :func:`calamari.quality.continuity_scores`, each sweep scored against its
    own scan axis, with the noise floor pooled across every sweep of the
    map -- see there and :data:`MAP_OUTLIER_THRESHOLD`.
    """
    return continuity_scores(ys, xs)


def _map_apply_outliers(rows, row_files, fam_sorted, *, what, scan_param,
                        family_param, outliers, outlier_threshold):
    """Run the glitch check on resolved rows; warn/drop/record per mode.

    Returns ``(rows, records)`` -- the rows with flagged points removed
    when ``outliers='drop'``, unchanged otherwise, and the
    :class:`MapOutlier` records for the grid either way.
    """
    if outliers not in ("warn", "drop", "ignore"):
        raise PlotInputError(
            f"outliers must be 'warn', 'drop', or 'ignore', got {outliers!r}"
        )
    if not (outlier_threshold > 0):
        raise PlotInputError(f"outlier_threshold must be positive, got {outlier_threshold!r}")

    scores = _map_outlier_scores([s.y for s in rows], [s.x for s in rows])
    records: list[MapOutlier] = []
    flagged_masks = []
    for series, file, fam_v, score in zip(rows, row_files, fam_sorted, scores):
        with np.errstate(invalid="ignore"):
            flagged = score > outlier_threshold
        flagged_masks.append(flagged)
        for i in np.where(flagged)[0]:
            records.append(MapOutlier(
                file=Path(file), family_value=float(fam_v),
                scan_value=float(series.x[i]), value=float(series.y[i]),
                score=float(score[i]), dropped=(outliers == "drop"),
            ))
    if not records:
        return rows, ()

    if outliers == "warn":
        scan_unit = "K" if scan_param == "temperature" else "Oe"
        fam_unit = "K" if family_param == "temperature" else "Oe"
        fam_sym = "T" if family_param == "temperature" else "H"
        lines = [
            f"  {r.file.name}: value {r.value:.4g} at {r.scan_value:g} {scan_unit} "
            f"({fam_sym} = {r.family_value:g} {fam_unit}), score {r.score:.0f}"
            for r in records[:_OUTLIER_WARN_LIST]
        ]
        if len(records) > _OUTLIER_WARN_LIST:
            lines.append(f"  ... and {len(records) - _OUTLIER_WARN_LIST} more")
        n = len(records)
        warnings.warn(
            f"{what}: {n} point{'s' if n > 1 else ''} depart{'' if n > 1 else 's'} "
            "from the trend of the neighbouring points by far more than those "
            f"neighbours stray from it themselves (score > {outlier_threshold:g}; quantity "
            f"{rows[0].ylabel}):\n" + "\n".join(lines) + "\n"
            "Kept as measured, so it will enter the cell it falls in and the "
            "map's min-max colour range. Pass outliers='drop' to exclude "
            "it, outliers='ignore' to keep it without this warning, or "
            "outlier_threshold=<score> to move the line; grid.outliers "
            "records the flagged points in every mode.",
            stacklevel=4,
        )
        return rows, tuple(records)
    if outliers == "ignore":
        return rows, tuple(records)

    kept_rows = []
    for series, flagged in zip(rows, flagged_masks):
        keep = ~flagged
        kept_rows.append(replace(
            series, x=series.x[keep], y=series.y[keep],
            yerr=None if series.yerr is None else series.yerr[keep],
        ))
    return kept_rows, tuple(records)


def _map_grid(
    measurements,
    resolve_one,
    *,
    family: str,
    family_values,
    scan_axis,
    n_scan,
    gridding: str,
    max_gap,
    frequency,
    outliers: str = "warn",
    outlier_threshold: float = MAP_OUTLIER_THRESHOLD,
    what: str = "map grid",
) -> PlotGrid:
    """Shared engine behind chi_map_grid / moment_map_grid."""
    from calamari.io import load

    loaded = [load(m) if isinstance(m, (str, Path)) else m for m in measurements]
    if len(loaded) < 2:
        raise PlotInputError(
            f"A map needs at least two sweeps; got {len(loaded)}. One sweep is "
            "a curve -- plot it with the 1-D helpers."
        )
    if gridding not in ("bin", "interp"):
        raise PlotInputError(f"gridding must be 'bin' or 'interp', got {gridding!r}")
    if gridding == "interp" and max_gap is None:
        raise PlotInputError(
            "gridding='interp' requires max_gap=<scan-axis units>: interpolation "
            "asserts values between samples, and how far apart two samples may "
            "be before the space between them is a hole rather than a line is a "
            "judgement about your sampling that calamari has no evidence to "
            "make for you. gridding='bin' (the default) needs no such number."
        )
    if gridding == "bin" and max_gap is not None:
        raise PlotInputError("max_gap applies to gridding='interp' only.")

    if n_scan is not None and (
        isinstance(n_scan, (bool, np.bool_))
        or not isinstance(n_scan, (int, np.integer)) or n_scan < 2
    ):
        raise PlotInputError(f"n_scan must be an integer at least 2, got {n_scan!r}.")
    if max_gap is not None:
        try:
            gap = float(max_gap)
        except (TypeError, ValueError) as exc:
            raise PlotInputError("max_gap must be a finite positive number.") from exc
        if isinstance(max_gap, (bool, np.bool_)) or not np.isfinite(gap) or gap <= 0:
            raise PlotInputError(f"max_gap must be finite and positive, got {max_gap!r}.")
        max_gap = gap
    if scan_axis is not None:
        try:
            scan_axis = np.asarray(scan_axis, dtype=float)
        except (TypeError, ValueError) as exc:
            raise PlotInputError("scan_axis must contain numeric grid centres.") from exc
        if (scan_axis.ndim != 1 or scan_axis.size < 2
                or not np.isfinite(scan_axis).all() or not (np.diff(scan_axis) > 0).all()):
            raise PlotInputError(
                "scan_axis must be one-dimensional with at least two finite, "
                "strictly increasing grid centres."
            )

    family_param = _map_family_param(loaded, family)
    scan_param = "field" if family_param == "temperature" else "temperature"
    loaded, selected_frequency = _map_select_frequency(loaded, frequency)

    if family_values is not None:
        try:
            fam_vals = np.asarray(list(family_values), dtype=float)
        except (TypeError, ValueError) as exc:
            raise PlotInputError("family_values must contain numeric coordinates.") from exc
        if fam_vals.ndim != 1 or not np.isfinite(fam_vals).all():
            raise PlotInputError("family_values must be one-dimensional and finite.")
        if len(fam_vals) != len(loaded):
            raise PlotInputError(
                f"family_values has {len(fam_vals)} entries for {len(loaded)} "
                "sweeps; they must pair one-to-one."
            )
    else:
        fam_vals = [_family_value(m, family_param) for m in loaded]

    resolved = []
    for m in loaded:
        series_list = resolve_one(m, scan_param)
        if len(series_list) != 1:
            raise PlotInputError(
                f"{m.path}: this sweep resolves to {len(series_list)} series; a "
                "map row must be exactly one. (A multi-frequency sweep needs "
                "frequency=<value>.)"
            )
        series = series_list[0]
        _map_check_monotonic(series.x, m)
        resolved.append(series)

    # One map, one quantity: every sweep must have resolved to the same
    # value label (source, units, symbol). chi_series refuses AC-vs-DC
    # ambiguity per file; this is the per-family counterpart, or a grid
    # could mix chi' rows with M/H rows under one colourbar.
    labels = {s.ylabel for s in resolved}
    if len(labels) > 1:
        first_label = resolved[0].ylabel
        offender = next(
            (m, s) for m, s in zip(loaded, resolved) if s.ylabel != first_label
        )
        raise PlotInputError(
            f"These sweeps resolve to different quantities: {loaded[0].path} "
            f"gives {first_label!r} but {offender[0].path} gives "
            f"{offender[1].ylabel!r}. One map is one quantity -- pass source=/"
            "units= explicitly so every sweep resolves the same way."
        )

    # Distinct held coordinates define distinct rows, independently of scan classification.
    order = np.argsort(fam_vals, kind="stable")
    fam_sorted = np.asarray(fam_vals, dtype=float)[order]
    rows = [resolved[i] for i in order]
    row_files = [loaded[i].path for i in order]
    if not np.isfinite(fam_sorted).all():
        raise PlotInputError("Map family coordinates must be finite.")
    for a in range(len(fam_sorted) - 1):
        if fam_sorted[a + 1] == fam_sorted[a]:
            raise PlotInputError(
                f"Two sweeps sit at the same fixed {family_param} "
                f"({fam_sorted[a]:.6g} and {fam_sorted[a + 1]:.6g}): "
                f"{row_files[a]} and {row_files[a + 1]}. One map row per "
                "setpoint -- combine them first (calamari.repeats), drop one, "
                "or, if they differ by AC frequency, build one map per "
                "frequency (frequency=...)."
            )

    # The glitch check runs on the resolved rows, before gridding: a
    # single bad reading would otherwise own one cell and, through it,
    # the whole colour range. Named per file so the reader can go and
    # look at the point rather than take the verdict on trust.
    rows, outlier_records = _map_apply_outliers(
        rows, row_files, fam_sorted, what=what, scan_param=scan_param,
        family_param=family_param, outliers=outliers,
        outlier_threshold=outlier_threshold,
    )

    # The common scan axis (always CENTRES; edges are derived at render).
    if scan_axis is not None:
        centres = np.asarray(scan_axis, dtype=float)
    else:
        lo = min(float(s.x.min()) for s in rows)
        hi = max(float(s.x.max()) for s in rows)
        # Median sweep point count, NOT the densest sweep: a grid finer
        # than most sweeps were measured maximises invented cells.
        n = n_scan if n_scan is not None else int(np.median([len(s.x) for s in rows]))
        centres = np.linspace(lo, hi, max(n, 2))
    edges = np.concatenate(
        [
            [centres[0] - (centres[1] - centres[0]) / 2],
            (centres[:-1] + centres[1:]) / 2,
            [centres[-1] + (centres[-1] - centres[-2]) / 2],
        ]
    )

    have_errors = all(s.yerr is not None for s in rows)
    values = np.full((len(rows), len(centres)), np.nan)
    counts = np.zeros((len(rows), len(centres)), dtype=np.int64)
    errors = np.full((len(rows), len(centres)), np.nan) if have_errors else None

    for i, series in enumerate(rows):
        cell = np.searchsorted(edges, series.x, side="right") - 1
        inside = (cell >= 0) & (cell < len(centres))
        cidx = cell[inside]
        np.add.at(counts[i], cidx, 1)
        if gridding == "bin":
            sums = np.zeros(len(centres))
            np.add.at(sums, cidx, series.y[inside])
            filled = counts[i] > 0
            values[i, filled] = sums[filled] / counts[i, filled]
            if have_errors:
                sq = np.zeros(len(centres))
                np.add.at(sq, cidx, series.yerr[inside] ** 2)
                # Standard error of an unweighted cell mean of n points with
                # individual errors sigma_i: sqrt(sum sigma_i^2) / n. The
                # reduction rule is deliberately recorded here and in the
                # docstring: an unweighted mean is NOT the Birge-weighted
                # combination calamari.repeats performs -- aggregate repeats
                # upstream when that treatment is wanted (RP-1).
                errors[i, filled] = np.sqrt(sq[filled]) / counts[i, filled]
        else:
            xs = series.x if series.x[0] <= series.x[-1] else series.x[::-1]
            ys = series.y if series.x[0] <= series.x[-1] else series.y[::-1]
            row = np.interp(centres, xs, ys, left=np.nan, right=np.nan)
            # Blank any centre whose bracketing samples are further apart
            # than max_gap: the space between them is a hole, not a line.
            inside_c = (centres >= xs[0]) & (centres <= xs[-1])
            hi_idx = np.searchsorted(xs, centres[inside_c], side="left").clip(1, len(xs) - 1)
            bracket = xs[hi_idx] - xs[hi_idx - 1]
            blank = np.zeros(len(centres), dtype=bool)
            blank[inside_c] = bracket > max_gap
            row[blank] = np.nan
            values[i] = row

    sample_scan = np.concatenate([s.x for s in rows])
    sample_family = np.concatenate(
        [np.full(len(s.x), v) for s, v in zip(rows, fam_sorted)]
    )
    if family_param == "temperature":
        t_k, h_oe = fam_sorted, centres
        grid_values, grid_counts, grid_errors = values, counts, errors
        sample_t, sample_h = sample_family, sample_scan
    else:
        t_k, h_oe = centres, fam_sorted
        grid_values = values.T
        grid_counts = counts.T
        grid_errors = None if errors is None else errors.T
        sample_t, sample_h = sample_scan, sample_family

    return PlotGrid(
        t_k=t_k,
        h_oe=h_oe,
        values=grid_values,
        counts=grid_counts,
        errors=grid_errors,
        value_label=rows[0].ylabel,
        family_axis=family_param,
        frequency_hz=selected_frequency,
        sample_t=sample_t,
        sample_h=sample_h,
        gridding=gridding,
        outliers=outlier_records,
    )


def chi_map_grid(
    measurements,
    *,
    component: str = "real",
    units: str = "emu_per_oe",
    sample: Sample | None = None,
    source: str = "auto",
    frequency: float | None = None,
    family: str = "auto",
    family_values=None,
    scan_axis=None,
    n_scan: int | None = None,
    gridding: str = "bin",
    max_gap: float | None = None,
    outliers: str = "warn",
    outlier_threshold: float = MAP_OUTLIER_THRESHOLD,
) -> PlotGrid:
    """Build a chi(T, H) :class:`PlotGrid` from a family of sweeps.

    Input is EITHER a set of field sweeps each at fixed temperature OR a
    set of temperature sweeps each at fixed field (auto-detected per
    sweep, unanimity required); the result lives on the same T-H plane
    either way. Each sweep's 1-D data comes from :func:`chi_series`, so
    units, the explicit AC/DC source, housekeeping exclusion and NaN
    dropping behave exactly as in the 1-D plots -- and every sweep must
    resolve to the *same* quantity, or the family is refused by name (a
    grid mixing chi' rows with M/H rows under one colourbar is the lie
    this check exists for).

    ``gridding='bin'`` (default) makes each cell the unweighted mean of
    the real points falling in it; empty cells are NaN and stay visible.
    It never asserts a value where nothing was measured, which is why it
    is the default: interior gaps, dropped-NaN samples, repeat rows and
    the M/H zero-field hole all come out as honest holes or honest
    averages with no tuning constant. ``counts`` records how many points
    each cell holds; ``errors`` (when every sweep carries per-point
    errors) is the standard error of the cell mean -- note this is an
    unweighted reduction, not the Birge-weighted combination of
    :mod:`calamari.repeats`; aggregate repeats upstream first when that
    treatment matters. ``gridding='interp'`` linearly interpolates each
    sweep onto the axis instead and **requires** ``max_gap`` (scan-axis
    units): how far apart two samples may be before the space between
    them is a hole rather than a line is a judgement about your sampling,
    and there is no evidence from which calamari could default it.

    Refused by name: fewer than two sweeps; mixed scan kinds (a single
    file of stacked isotherms splits with
    :func:`calamari.sweeps.split_isotherms`); a sweep whose fixed
    parameter drifts (see :func:`plot_family`); two sweeps at the same
    identical finite fixed value (distinct close coordinates are accepted); a non-monotonic scan axis (a hysteresis loop -- take a
    branch first); more than one AC frequency without ``frequency=``.

    Parameters largely mirror :func:`chi_series` (``component``,
    ``units``, ``sample``, ``source``) and :func:`plot_family`
    (``family``, ``family_values``); ``scan_axis`` gives explicit common
    axis centres (finite, strictly increasing, at least two), ``n_scan`` an integer
    count of at least two for the automatic axis (default:
    the median sweep point count -- the densest sweep would maximise
    invented cells). ``frequency`` selects one frequency from
    multi-frequency data and is recorded on the grid. An explicit request
    must be finite and positive and match exactly one recorded cluster within
    0.1% of its representative, with no absolute tolerance. Missing or ambiguous
    matches raise.

    **Glitches.** One bad reading -- an AC point that came back at minus
    four times its neighbours -- owns a cell and, through it, the whole
    min-max colour range, and the map renders flat. Every point is
    therefore scored against its neighbours along its own sweep
    (:func:`_map_outlier_scores`; the score is the departure from a
    straight line through the neighbours, in units of the neighbours' own
    disorder about that line -- the same score as
    :meth:`Measurement.quality() <calamari.measurement.Measurement.quality>`
    -- which is what keeps a sharp physical cusp or step from being
    flagged while still catching a glitch on a steep flank). ``outliers``
    says what to do with a point scoring above ``outlier_threshold``
    (:data:`MAP_OUTLIER_THRESHOLD`): ``'warn'`` (default) keeps it and
    warns, naming the file, the coordinates and the value so you can go
    and look; ``'drop'`` excludes it from the grid and the sample
    overlay; ``'ignore'`` keeps it silently. The flagged points are
    recorded on ``grid.outliers`` in every mode, with whether they were
    dropped, so a figure stays auditable. Dropping is never the default:
    discarding a measured point changes what the figure shows, and that
    is your call to make with the point in front of you.
    """
    def resolve_one(m, scan_param):
        return chi_series(
            m, x=scan_param, component=component, units=units, sample=sample,
            source=source,
        )

    return _map_grid(
        measurements, resolve_one, family=family, family_values=family_values,
        scan_axis=scan_axis, n_scan=n_scan, gridding=gridding, max_gap=max_gap,
        frequency=frequency, outliers=outliers, outlier_threshold=outlier_threshold,
        what="chi_map_grid",
    )


def moment_map_grid(
    measurements,
    *,
    units: str = "emu",
    sample: Sample | None = None,
    family: str = "auto",
    family_values=None,
    scan_axis=None,
    n_scan: int | None = None,
    gridding: str = "bin",
    max_gap: float | None = None,
    outliers: str = "warn",
    outlier_threshold: float = MAP_OUTLIER_THRESHOLD,
) -> PlotGrid:
    """Build an M(T, H) :class:`PlotGrid` from a family of sweeps.

    As :func:`chi_map_grid` but over :func:`moment_series`: the map of
    magnetisation itself, in ``'emu'`` or the ``muB`` units (with
    ``sample``). All the gridding, refusal, glitch-check and provenance
    behaviour is identical -- see there. A hysteresis loop is refused as
    non-monotonic; take one branch first
    (:func:`calamari.sweeps.increasing_field_branch`).
    """
    def resolve_one(m, scan_param):
        return moment_series(m, x=scan_param, units=units, sample=sample)

    return _map_grid(
        measurements, resolve_one, family=family, family_values=family_values,
        scan_axis=scan_axis, n_scan=n_scan, gridding=gridding, max_gap=max_gap,
        frequency=None, outliers=outliers, outlier_threshold=outlier_threshold,
        what="moment_map_grid",
    )


#: Rendered family rows per input sweep when interpolate_family is not
#: given an explicit n_rows: fine enough that the coarsest measured row
#: spacing spans several rendered rows (a visible gradient rather than a
#: staircase), coarse enough that the mesh stays cheap for hundreds of
#: sweeps. Purely a display resolution -- it invents no values a coarser
#: or finer grid would not.
_FAMILY_INTERP_DENSITY = 4


def interpolate_family(grid: PlotGrid, *, max_gap: float, n_rows: int | None = None) -> PlotGrid:
    """Interpolate a map's family axis: filled colour between the sweeps.

    The family-axis counterpart of ``gridding='interp'``, and the same
    bargain: interpolation asserts values *between* measurements, so
    ``max_gap`` (in family-axis units -- K for a temperature family, Oe
    for a field family) must say how far apart two sweeps may sit before
    the space between them is a hole rather than a gradient. Rows further
    apart than that stay NaN, exactly as the row-per-sweep rendering
    leaves them grey.

    Returns a new :class:`PlotGrid` whose family axis is a dense, evenly
    spaced grid over the measured range (``n_rows`` rows; default
    :data:`_FAMILY_INTERP_DENSITY` per input sweep), with each scan-axis
    column interpolated **independently** across the rows where it holds
    data. A column a sweep did not cover donates nothing across that
    sweep, so scan-axis holes are never bridged sideways; a column with
    data in fewer than two sweeps has nothing to interpolate between and
    stays NaN. Interpolation is linear in the family parameter -- for a
    family spanning decades, the row-per-sweep rendering with
    ``family_scale='log'`` is usually the more honest figure.

    The result is manufactured display data, and says so: ``counts`` is
    zero everywhere (no cell aggregates a real point), ``errors`` is
    dropped (a cell standard error has no meaning for an interpolated
    value), ``gridding`` gains a ``'+family-interp'`` suffix, and
    ``sample_t``/``sample_h`` keep every real point so the sample overlay
    still shows exactly what was measured. For a publication-style figure
    pass the result to :func:`draw_map` with ``show_samples=False`` --
    :func:`plot_chi_map` composes all of this in one call via
    ``family_gridding='interp'``.
    """
    if max_gap is None:
        raise PlotInputError(
            "interpolate_family requires max_gap=<family-axis units>: "
            "interpolation asserts values between sweeps, and how far apart "
            "two sweeps may be before the space between them is a hole rather "
            "than a gradient is a judgement about your campaign that calamari "
            "has no evidence to make for you."
        )
    if max_gap <= 0:
        raise PlotInputError(f"max_gap must be positive, got {max_gap!r}")

    family_on_t = grid.family_axis == "temperature"
    fam = grid.t_k if family_on_t else grid.h_oe
    if len(fam) < 2:
        raise PlotInputError("A family of one row has nothing to interpolate between.")
    n = n_rows if n_rows is not None else _FAMILY_INTERP_DENSITY * len(fam)
    if n < 2:
        raise PlotInputError(f"n_rows must be at least 2, got {n}")

    family_first = grid.values if family_on_t else grid.values.T  # (n_family, n_scan)
    dense = np.linspace(fam[0], fam[-1], int(n))
    out = np.full((len(dense), family_first.shape[1]), np.nan)

    for j in range(family_first.shape[1]):
        column = family_first[:, j]
        known = np.isfinite(column)
        if known.sum() < 2:
            continue
        xk = fam[known]
        yk = column[known]
        row = np.interp(dense, xk, yk, left=np.nan, right=np.nan)
        # Blank any dense row whose bracketing sweeps (in THIS column) are
        # further apart than max_gap -- but a dense row landing exactly on
        # a measured sweep carries a measured value, not an interpolated
        # one, and is never blanked. (linspace copies its endpoints, so
        # the exact comparison is sound where it matters.)
        inside = (dense >= xk[0]) & (dense <= xk[-1])
        hi = np.searchsorted(xk, dense[inside], side="left").clip(1, len(xk) - 1)
        blank = np.zeros(len(dense), dtype=bool)
        blank[inside] = (xk[hi] - xk[hi - 1]) > max_gap
        blank[np.isin(dense, xk)] = False
        row[blank] = np.nan
        out[:, j] = row

    if family_on_t:
        t_k, h_oe, values = dense, grid.h_oe, out
    else:
        t_k, h_oe, values = grid.t_k, dense, out.T

    return PlotGrid(
        t_k=t_k,
        h_oe=h_oe,
        values=values,
        counts=np.zeros(values.shape, dtype=np.int64),
        errors=None,
        value_label=grid.value_label,
        family_axis=grid.family_axis,
        frequency_hz=grid.frequency_hz,
        sample_t=grid.sample_t,
        sample_h=grid.sample_h,
        gridding=f"{grid.gridding}+family-interp",
        outliers=grid.outliers,
    )


def _family_edges(centres: NDArray[np.float64], scale: str) -> NDArray[np.float64]:
    """Cell edges for the family axis, honest about sparse spacing.

    Midpoint edges alone let one row paint colour halfway to a sweep an
    arbitrary distance away -- the family-axis version of interpolating
    across a hole. Each row's half-extent is therefore capped at half the
    median row spacing (in log space for ``scale='log'``): a row never
    claims more of the axis than the typical spacing, and sparse families
    show gaps between their rows. ``scale='index'`` needs no cap; every
    row is one unit tall by construction.
    """
    if scale == "index":
        idx = np.arange(len(centres), dtype=float)
        return np.stack([idx - 0.5, idx + 0.5], axis=1)
    work = np.log(centres) if scale == "log" else centres
    spacing = np.diff(work)
    cap = np.median(spacing) / 2
    mid = (work[:-1] + work[1:]) / 2
    lows = np.empty(len(work))
    highs = np.empty(len(work))
    lows[0] = work[0] - cap
    highs[-1] = work[-1] + cap
    lows[1:] = np.maximum(mid, work[1:] - cap)
    highs[:-1] = np.minimum(mid, work[:-1] + cap)
    edges_pairs = np.stack([lows, highs], axis=1)
    return np.exp(edges_pairs) if scale == "log" else edges_pairs


def _family_gridded(grid, family_gridding, family_max_gap, n_family):
    """Validate the family_gridding choice and apply it to a built grid."""
    if family_gridding == "rows":
        if family_max_gap is not None:
            raise PlotInputError("family_max_gap applies to family_gridding='interp' only.")
        if n_family is not None:
            raise PlotInputError("n_family applies to family_gridding='interp' only.")
        return grid
    if family_gridding != "interp":
        raise PlotInputError(
            f"family_gridding must be 'rows' or 'interp', got {family_gridding!r}"
        )
    if family_max_gap is None:
        raise PlotInputError(
            "family_gridding='interp' requires family_max_gap=<family-axis "
            "units>: interpolation asserts values between sweeps, and how far "
            "apart two sweeps may be before the space between them is a hole "
            "rather than a gradient is a judgement about your campaign that "
            "calamari has no evidence to make for you. family_gridding='rows' "
            "(the default) needs no such number."
        )
    return interpolate_family(grid, max_gap=family_max_gap, n_rows=n_family)
