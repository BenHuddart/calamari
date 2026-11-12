"""Rendering of prepared measurement grids."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from calamari._plot_data import (
    PlotGrid,
    _family_edges,
    _family_gridded,
    axis_labels,
    chi_map_grid,
    moment_map_grid,
)
from calamari.constants import OE_PER_TESLA
from calamari.exceptions import PlotInputError
from calamari.plot._curves import _make_ax
from calamari.plot._detection import overlay_transitions

if TYPE_CHECKING:
    from matplotlib.axes import Axes


MAP_ROBUST_PERCENTILES = (1.0, 99.0)

#: The flat-map diagnostic: when the central 95 % of finite cells (the
#: 2.5-97.5 percentile band) occupies less than this fraction of the
#: min-max colour range, a few extreme cells own the colourbar and the
#: structure renders as one colour. A quarter is a generous line: a map
#: dominated by one glitch typically sits near a tenth, while a
#: smoothly varying map with honest tails sits well above a half.
_CLIM_FLAT_FRACTION = 0.25
_CLIM_FLAT_BAND = (2.5, 97.5)


def draw_map(
    grid: PlotGrid,
    *,
    ax: Axes | None = None,
    family_scale: str = "linear",
    show_samples: bool = True,
    clim=None,
    robust_percentiles: tuple[float, float] = MAP_ROBUST_PERCENTILES,
    transitions=None,
    transition_kwargs: dict | None = None,
    **mesh_kwargs: Any,
):
    """Render a :class:`PlotGrid` as a colour map on the T-H plane.

    Orientation is always temperature on x and applied field (Tesla,
    mu0 H) on y -- the conventional H-T phase diagram -- whichever axis
    the sweeps ran along. Returns the ``QuadMesh``, which is the handle a
    colourbar needs (``fig.colorbar(mesh, ax=ax,
    label=grid.value_label)``); figure-level concerns stay with the
    caller per PL-1, and the axes are reachable as ``mesh.axes``.

    Honesty defaults, each overridable:

    - NaN cells render in an explicit grey, never transparent -- on a
      white background under a light-ended colormap a transparent hole is
      indistinguishable from a low value.
    - ``show_samples=True`` scatters the real sample positions over the
      mesh, so every gridding decision is visible on the figure itself.
    - Sparse family rows are capped at the median row spacing (see
      :func:`_family_edges`), leaving visible gaps rather than painting
      colour halfway to a distant sweep.
    - The axes title carries ``grid.value_label`` (and the selected
      frequency, when one was recorded), so the figure says what it shows
      without the caption's help.
    - The colour range defaults to the data's min and max, and warns when
      that range is wasted: if the central 95 % of cells occupy less than
      a quarter of it, a few extreme cells own the colourbar and the map
      renders nearly flat. The warning names the fix (``clim=``) rather
      than applying one, because clipping the range hides real values
      and that is the caller's decision.

    Parameters
    ----------
    grid : PlotGrid
        From :func:`chi_map_grid` / :func:`moment_map_grid`.
    ax : matplotlib.axes.Axes, optional
        Draw into these axes, or omit to create a figure.
    family_scale : {'linear', 'log', 'index'}, optional
        Spacing of the family axis (the one with a row per sweep).
        ``'linear'`` is physically true and the default; ``'log'`` suits
        log-spaced held fields (refused if any value is not positive --
        use ``'index'`` for families that include zero field);
        ``'index'`` gives every sweep an equal-height row with the true
        values as tick labels (the axis becomes categorical: distance on
        it means nothing).
    show_samples : bool, optional
        Overlay the real sample positions. Default True.
    clim : optional
        Colour limits. ``None`` (default): the finite cells' min and max,
        with the flat-map warning above. ``'minmax'``: the same limits,
        no warning -- you have looked and want the full range.
        ``'robust'``: the ``robust_percentiles`` of the finite cells
        (:data:`MAP_ROBUST_PERCENTILES`, 1-99 by default), which unseats
        a handful of extreme cells; cells outside the limits saturate,
        so tell the colourbar (``fig.colorbar(..., extend='both')``).
        ``(vmin, vmax)``: explicit limits. Passing ``vmin``/``vmax`` in
        ``mesh_kwargs`` is the same as an explicit pair and is refused
        alongside a non-default ``clim``.
    robust_percentiles : (float, float), optional
        The lower and upper percentiles ``clim='robust'`` uses.
    transitions : optional
        Transition candidates (a ``TransitionResult``, or its candidates
        table) to overlay after drawing -- forwarded to
        :func:`overlay_transitions` with this grid and ``family_scale``,
        so the overlay cannot be lied to about how the map was rendered.
    transition_kwargs : dict, optional
        Extra keyword arguments for :func:`overlay_transitions`.
    **mesh_kwargs
        Forwarded to ``ax.pcolormesh`` (``cmap=``, ``vmin=``, ...);
        ``shading='flat'`` with explicit edges is used and cannot be
        overridden. ``rasterized=True`` is the default: in a vector file
        the mesh is embedded as a single image at ``savefig.dpi`` (600
        under a journal style) while axes, text and the sample overlay
        stay vector. A vector mesh is thousands of adjacent filled paths,
        and PDF viewers antialias their shared edges into a visible
        hairline lattice; rasterising is the standard cure. Pass
        ``rasterized=False`` for a fully vector file.
    """
    import matplotlib as mpl

    if family_scale not in ("linear", "log", "index"):
        raise PlotInputError(
            f"family_scale must be 'linear', 'log', or 'index', got {family_scale!r}"
        )
    family_on_x = grid.family_axis == "temperature"
    fam_centres = grid.t_k if family_on_x else grid.h_oe / OE_PER_TESLA
    scan_centres = grid.h_oe / OE_PER_TESLA if family_on_x else grid.t_k
    if family_scale == "log" and (fam_centres <= 0).any():
        raise PlotInputError(
            "family_scale='log' needs every fixed value positive; this family "
            "includes zero or negative values -- use family_scale='index'."
        )

    # Family rows may be non-contiguous (capped extents): render as one
    # mesh with NaN spacer rows filling the gaps between capped cells.
    pairs = _family_edges(fam_centres, family_scale)
    fam_edges = [pairs[0, 0]]
    fam_rows: list[int | None] = [0]
    for k in range(1, len(pairs)):
        if pairs[k, 0] > pairs[k - 1, 1]:
            fam_edges.append(pairs[k - 1, 1])
            fam_rows.append(None)  # spacer
        fam_edges.append(pairs[k, 0])
        fam_rows.append(k)
    fam_edges.append(pairs[-1, 1])
    fam_edges = np.asarray(fam_edges, dtype=float)

    sc = scan_centres
    scan_edges = np.concatenate(
        [[sc[0] - (sc[1] - sc[0]) / 2], (sc[:-1] + sc[1:]) / 2,
         [sc[-1] + (sc[-1] - sc[-2]) / 2]]
    )

    family_first = grid.values if family_on_x else grid.values.T  # (n_family, n_scan)
    stacked = np.full((len(fam_rows), family_first.shape[1]), np.nan)
    for out_i, src in enumerate(fam_rows):
        if src is not None:
            stacked[out_i] = family_first[src]

    cmap = mpl.colormaps[mesh_kwargs.pop("cmap", mpl.rcParams["image.cmap"])] \
        if not isinstance(mesh_kwargs.get("cmap"), mpl.colors.Colormap) \
        else mesh_kwargs.pop("cmap")
    # Empty cells (NaN) render in explicit grey, never transparent: on a
    # light colormap a transparent hole reads as a low value. with_extremes
    # returns a modified copy, so the caller's colormap object is untouched.
    cmap = cmap.with_extremes(bad="0.75")

    vmin, vmax = _map_colour_limits(stacked, clim, robust_percentiles, mesh_kwargs)
    if vmin is not None:
        mesh_kwargs["vmin"], mesh_kwargs["vmax"] = vmin, vmax
    # Rasterise the mesh (only the mesh) in vector output: adjacent filled
    # cells antialias into hairline seams in most PDF viewers, and a map of
    # thousands of paths is also a slow, large PDF. See the docstring.
    mesh_kwargs.setdefault("rasterized", True)

    ax = _make_ax(ax)
    if family_on_x:
        mesh = ax.pcolormesh(
            fam_edges, scan_edges, np.ma.masked_invalid(stacked).T,
            shading="flat", cmap=cmap, **mesh_kwargs,
        )
    else:
        mesh = ax.pcolormesh(
            scan_edges, fam_edges, np.ma.masked_invalid(stacked),
            shading="flat", cmap=cmap, **mesh_kwargs,
        )

    if family_scale == "index":
        ticks = np.arange(len(fam_centres))
        labels = [f"{v:g}" for v in fam_centres]
        (ax.set_xticks if family_on_x else ax.set_yticks)(ticks, labels)
    elif family_scale == "log":
        (ax.set_xscale if family_on_x else ax.set_yscale)("log")

    if show_samples:
        s_t, s_h = grid.sample_t, grid.sample_h / OE_PER_TESLA
        fam_coord = s_t if family_on_x else s_h
        scan_coord = s_h if family_on_x else s_t
        if family_scale == "index":
            fam_coord = np.searchsorted(fam_centres, fam_coord).clip(0, len(fam_centres) - 1)
        if family_on_x:
            ax.scatter(fam_coord, scan_coord, s=3, c="black", alpha=0.4, linewidths=0)
        else:
            ax.scatter(scan_coord, fam_coord, s=3, c="black", alpha=0.4, linewidths=0)

    xlabel = axis_labels("temperature", "k")
    ylabel = axis_labels("field", "tesla")
    if family_scale == "index":
        note = " [row per sweep]"
        if family_on_x:
            xlabel += note
        else:
            ylabel += note
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title = grid.value_label
    if grid.frequency_hz is not None:
        title += f"   f = {grid.frequency_hz:g} Hz"
    ax.set_title(title)

    if transitions is not None:
        # The one place grid and family_scale are both in hand, so the
        # overlay cannot be lied to about how the map was rendered.
        overlay_transitions(
            transitions, ax=ax, grid=grid, family_scale=family_scale,
            **(transition_kwargs or {}),
        )
    return mesh


def _map_colour_limits(values, clim, robust_percentiles, mesh_kwargs):
    """Resolve draw_map's ``clim`` to ``(vmin, vmax)``, or ``(None, None)``.

    ``(None, None)`` means "leave it to matplotlib" -- the caller supplied
    ``vmin``/``vmax`` directly, or there is not a single finite cell to
    take limits from. The flat-map diagnostic fires only for the implicit
    default (``clim=None``): any explicit choice, including
    ``clim='minmax'``, is taken as "I have looked".
    """
    explicit = "vmin" in mesh_kwargs or "vmax" in mesh_kwargs
    if explicit and clim is not None:
        raise PlotInputError(
            "Pass either clim= or vmin=/vmax= in mesh_kwargs, not both: two "
            "statements of the colour limits cannot both be honoured."
        )
    if explicit:
        return None, None

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    if clim is None or clim == "minmax":
        vmin, vmax = float(finite.min()), float(finite.max())
        if clim is None and vmax > vmin:
            lo, hi = np.percentile(finite, _CLIM_FLAT_BAND)
            occupied = (hi - lo) / (vmax - vmin)
            if occupied < _CLIM_FLAT_FRACTION:
                warnings.warn(
                    f"draw_map: the central 95 % of cells span only "
                    f"{100 * occupied:.0f} % of the colour range "
                    f"[{vmin:.4g}, {vmax:.4g}] -- a few extreme cells set the "
                    "limits, so the map will render nearly flat. Pass "
                    "clim='robust' (limits at the "
                    f"{robust_percentiles[0]:g}/{robust_percentiles[1]:g} "
                    "percentiles; give the colourbar extend='both'), "
                    "clim=(vmin, vmax), or clim='minmax' to keep the full "
                    "range without this warning. If one glitched point is "
                    "the cause, chi_map_grid(..., outliers='drop') removes "
                    "it at the source.",
                    stacklevel=3,
                )
        return vmin, vmax
    if clim == "robust":
        lo_p, hi_p = robust_percentiles
        if not (0 <= lo_p < hi_p <= 100):
            raise PlotInputError(
                f"robust_percentiles must satisfy 0 <= lo < hi <= 100, got "
                f"{robust_percentiles!r}"
            )
        lo, hi = np.percentile(finite, [lo_p, hi_p])
        return float(lo), float(hi)
    try:
        vmin, vmax = (float(v) for v in clim)
    except (TypeError, ValueError):
        raise PlotInputError(
            f"clim must be None, 'minmax', 'robust', or a (vmin, vmax) pair, got {clim!r}"
        ) from None
    if not vmin < vmax:
        raise PlotInputError(f"clim needs vmin < vmax, got {clim!r}")
    return vmin, vmax


def plot_chi_map(measurements, *, ax=None, family_scale="linear",
                 show_samples=True, mesh_kwargs=None,
                 clim=None, robust_percentiles=MAP_ROBUST_PERCENTILES,
                 family_gridding="rows", family_max_gap=None, n_family=None,
                 transitions=None, transition_kwargs=None,
                 **grid_kwargs):
    """Resolve and draw a chi(T, H) map in one call.

    ``grid_kwargs`` go to :func:`chi_map_grid` (``component``, ``units``,
    ``sample``, ``source``, ``frequency``, ``gridding``, ``outliers``,
    ...); ``clim``/``robust_percentiles`` and ``mesh_kwargs`` (a dict,
    to ``ax.pcolormesh``) to :func:`draw_map`. Returns the ``QuadMesh``
    -- attach a colourbar with ``fig.colorbar(mesh, ax=mesh.axes,
    label=...)``; the resolved value label is on the axes title and on
    the grid (:func:`chi_map_grid` is the public seam when you want the
    arrays).

    ``family_gridding='interp'`` fills the space between sweeps by
    linear interpolation along the family axis and **requires**
    ``family_max_gap`` (family-axis units; ``n_family`` optionally sets
    the rendered row count) -- see :func:`interpolate_family` for what
    is asserted and what stays a hole. The publication-style rendering
    is that plus ``show_samples=False``; the sample overlay is the
    audit trail for what the interpolation manufactured, so leave it on
    while deciding whether to trust the figure.

    ``transitions=`` overlays detected transition candidates (see
    :func:`calamari.transitions.detect_transitions_all` and
    :func:`overlay_transitions`); ``transition_kwargs`` tunes the
    overlay. The grid and ``family_scale`` are wired through
    automatically, which is what makes the overlay safe on
    ``family_scale='index'`` maps.
    """
    grid = _family_gridded(chi_map_grid(measurements, **grid_kwargs),
                           family_gridding, family_max_gap, n_family)
    return draw_map(grid, ax=ax, family_scale=family_scale,
                    show_samples=show_samples, clim=clim,
                    robust_percentiles=robust_percentiles, transitions=transitions,
                    transition_kwargs=transition_kwargs, **(mesh_kwargs or {}))


def plot_moment_map(measurements, *, ax=None, family_scale="linear",
                    show_samples=True, mesh_kwargs=None,
                    clim=None, robust_percentiles=MAP_ROBUST_PERCENTILES,
                    family_gridding="rows", family_max_gap=None, n_family=None,
                    transitions=None, transition_kwargs=None,
                    **grid_kwargs):
    """Resolve and draw an M(T, H) map in one call; see :func:`plot_chi_map`."""
    grid = _family_gridded(moment_map_grid(measurements, **grid_kwargs),
                           family_gridding, family_max_gap, n_family)
    return draw_map(grid, ax=ax, family_scale=family_scale,
                    show_samples=show_samples, clim=clim,
                    robust_percentiles=robust_percentiles, transitions=transitions,
                    transition_kwargs=transition_kwargs, **(mesh_kwargs or {}))
