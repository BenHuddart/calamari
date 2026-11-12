"""Matplotlib plotting helpers for calamari.

This package and calamari.style contain all direct matplotlib imports
(PL-1, ``docs/ARCHITECTURE.md``). Every helper takes the
measurement (or family of measurements) first and an optional ``ax=``,
and leaves figure-level concerns -- layout, colourbars, calling
``legend`` -- to the caller, so the helpers compose into multi-panel
figures.

The package has two layers with a public seam. **Resolution**
(:func:`chi_series`, :func:`moment_series`) turns a ``Measurement`` plus
a units/source request into :class:`PlotSeries` objects -- pure arrays
and label strings, no drawing. All the physics and robustness lives
there: unit conversion through :class:`~calamari.sample.Sample`,
AC-versus-DC source selection, housekeeping-row exclusion, NaN-row
dropping, per-frequency splitting, error-bar propagation. **Drawing**
(:func:`plot_chi_vs_t` and friends) resolves and then draws. Call the resolution layer yourself when the drawing layer's
choices do not fit.

Quickstart
----------
>>> import calamari
>>> m = calamari.load("run.dat")                           # doctest: +SKIP
>>> ax = calamari.plot.plot_chi_vs_t(m, component="both")  # doctest: +SKIP
>>> calamari.plot.save(ax, "chi.pdf")                       # doctest: +SKIP

Every helper takes ``ax=`` to draw into an existing figure (multi-panel
composition) and creates one via ``pyplot`` only when ``ax`` is omitted.
``**style`` kwargs are forwarded untouched to the underlying matplotlib
call (``ax.plot``/``ax.errorbar``; ``ax.pcolormesh`` for
:func:`plot_moment_map`), so anything matplotlib accepts there is
accepted here.
"""

from calamari._plot_data import (
    MAP_OUTLIER_THRESHOLD as _MAP_OUTLIER_THRESHOLD,
)
from calamari._plot_data import (
    MAP_OUTLIER_WINDOW as _MAP_OUTLIER_WINDOW,
)
from calamari._plot_data import (
    MapOutlier,
    PlotGrid,
    PlotSeries,
    axis_labels,
    chi_map_grid,
    chi_series,
    fisher_series,
    interpolate_family,
    moment_map_grid,
    moment_series,
)
from calamari.plot._curves import (
    DEFAULT_SAVE_FORMAT as _DEFAULT_SAVE_FORMAT,
)
from calamari.plot._curves import (
    plot_chi_vs_h,
    plot_chi_vs_t,
    plot_fisher,
    plot_fit,
    plot_fit_windows,
    plot_mh,
    plot_mt,
    save,
)
from calamari.plot._detection import (
    DETECTION_REPRESENTATIONS as _DETECTION_REPRESENTATIONS,
)
from calamari.plot._detection import (
    overlay_transitions,
    plot_boundary,
    plot_detection_curve,
    plot_scale_tracks,
    plot_smoothing_scan,
    plot_transitions,
)
from calamari.plot._families import (
    plot_chi_vs_h_family,
    plot_chi_vs_t_family,
    plot_family,
    plot_mh_family,
    plot_mt_family,
)
from calamari.plot._maps import (
    MAP_ROBUST_PERCENTILES as _MAP_ROBUST_PERCENTILES,
)
from calamari.plot._maps import (
    draw_map,
    plot_chi_map,
    plot_moment_map,
)

#: Neighbours on each side that a point is compared against by the map
#: glitch check -- the shared :data:`calamari.quality.CONTINUITY_WINDOW`.
MAP_OUTLIER_WINDOW = _MAP_OUTLIER_WINDOW


#: Glitch score above which a map point is reported (and, with
#: ``outliers='drop'``, excluded) -- the shared
#: :data:`calamari.quality.CONTINUITY_THRESHOLD`, whose comment records the
#: threshold rationale. One number, so a point :meth:`Measurement.quality()
#: <calamari.measurement.Measurement.quality>` flags is the point a map
#: warns about.
MAP_OUTLIER_THRESHOLD = _MAP_OUTLIER_THRESHOLD


#: Format written when :func:`save` is given a bare stem and no ``formats``.
#: PDF rather than matplotlib's PNG default because both publishers calamari
#: has styles for (APS, IOP) ask for vector figures; a PNG is a draft.
DEFAULT_SAVE_FORMAT = _DEFAULT_SAVE_FORMAT


#: Percentiles draw_map's ``clim='robust'`` sets the colour limits at.
#: 1-99 discards the extreme 2 % of cells -- enough to unseat a handful
#: of glitched or edge cells, few enough that the colourbar still covers
#: essentially all the measured structure.
MAP_ROBUST_PERCENTILES = _MAP_ROBUST_PERCENTILES


#: The smoothed representations a detection records per sweep and signal
#: (:meth:`calamari.transitions.TransitionResult.curve`) that can be drawn
#: against the swept axis, in derivative order. ``'yT'`` is recorded too, but
#: it is only the intermediate Fisher differentiates, so it is not offered.
DETECTION_REPRESENTATIONS = _DETECTION_REPRESENTATIONS


__all__ = [
    "DEFAULT_SAVE_FORMAT",
    "DETECTION_REPRESENTATIONS",
    "MAP_OUTLIER_THRESHOLD",
    "MAP_OUTLIER_WINDOW",
    "MAP_ROBUST_PERCENTILES",
    "MapOutlier",
    "PlotGrid",
    "PlotSeries",
    "axis_labels",
    "chi_map_grid",
    "chi_series",
    "draw_map",
    "fisher_series",
    "interpolate_family",
    "moment_map_grid",
    "moment_series",
    "overlay_transitions",
    "plot_boundary",
    "plot_chi_map",
    "plot_chi_vs_h",
    "plot_chi_vs_h_family",
    "plot_chi_vs_t",
    "plot_chi_vs_t_family",
    "plot_detection_curve",
    "plot_family",
    "plot_fisher",
    "plot_fit",
    "plot_fit_windows",
    "plot_mh",
    "plot_mh_family",
    "plot_moment_map",
    "plot_mt",
    "plot_mt_family",
    "plot_scale_tracks",
    "plot_smoothing_scan",
    "plot_transitions",
    "save",
]

# Preserve canonical public paths for documentation and existing pickles.
for _name in __all__:
    _value = globals()[_name]
    if callable(_value):
        _value.__module__ = __name__
del _name, _value
