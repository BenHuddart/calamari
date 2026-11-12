"""Journal figure styles: column-width figures that match a target journal.

A publication figure has to satisfy a page it will never be drawn on:
its width is the journal's column, its lettering has to survive
reduction to that width, and its colours have to remain distinguishable
where the print edition is greyscale. Getting that right by hand means
looking up the same handful of numbers for every paper. This module
records them once per journal, as evidenced *specifications*, and turns
them into matplotlib ``rcParams`` so that every :mod:`calamari.plot`
helper -- which creates its figure through pyplot when ``ax`` is omitted
-- comes out at column width with nothing else changed.

Alongside :mod:`calamari.plot`, this is the only module permitted to
import matplotlib (PK-1, PL-1 in ``docs/ARCHITECTURE.md``); it touches
no measurement, so the analysis layers stay free of plotting concerns.

Two orthogonal choices make a style. The **journal** fixes the geometry
(single- and double-column widths and the lettering size the journal
asks for at final size); the **look** fixes everything else (font
family, tick convention, colour cycle, colour map). Every journal names
a default look, so the common call is one word:

>>> import calamari
>>> calamari.style.use("jpcm")                       # doctest: +SKIP
>>> ax = calamari.plot.plot_mh(m)                    # doctest: +SKIP
>>> calamari.plot.save(ax, "fig1.pdf")               # 8.5 cm wide, 600 dpi  # doctest: +SKIP

``"jpcm:sans"`` selects a different look; :func:`context` scopes the
style to a block; :func:`figure` builds a correctly sized multi-panel
figure. Nothing here changes what a :mod:`calamari.plot` helper draws
-- only the page it is drawn on.

The numbers are the journals' own and are cited on each
:class:`JournalSpec`. Where a publisher states a figure width rather
than a measured text width (IOP's 15 cm "large figure"), the stated
number is used and the provenance says so; a figure sized to a
publisher's stated width is what the publisher asked for, whatever the
page later turns out to measure.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cycler import cycler

from calamari.exceptions import UnknownStyleError

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "JournalSpec",
    "LookSpec",
    "JOURNALS",
    "LOOKS",
    "DEFAULT_ASPECT",
    "rc",
    "use",
    "context",
    "figure",
    "panel_labels",
    "register_journal",
    "register_look",
    "available",
]

#: Inches per centimetre. Journals specify widths in cm (or in fractions of
#: an inch, for APS); matplotlib's ``figure.figsize`` is in inches.
_IN_PER_CM = 1.0 / 2.54

#: Default width:height ratio of a single-axes figure. 4:3 is the
#: conventional proportion of a physics data panel (a golden-ratio 1.62
#: leaves too little vertical room for a y-axis label and a legend at
#: column width); a caller wanting a different shape passes ``height_cm``.
DEFAULT_ASPECT = 4.0 / 3.0

#: Font fallback chains. The first entry is the face the journal's body
#: text is set in; each chain ends in a matplotlib-bundled DejaVu face so
#: a host without the commercial fonts (Linux CI, a fresh conda env) gets
#: the closest available shape *without* a per-figure findfont warning --
#: matplotlib warns only when *no* family in the list resolves.
#:
#: Arial leads the sans chain rather than Helvetica, although Helvetica is
#: the face IOP names: on macOS the system ``Helvetica.ttc`` makes FreeType
#: fail with a division-by-zero on every glyph at 7-8 pt at screen
#: resolutions (72-100 dpi; observed with matplotlib 3.11 on macOS), which
#: is exactly the size and resolution of an inline preview. Arial is metrically identical
#: to Helvetica -- it was designed as a drop-in -- so the printed figure is
#: indistinguishable and the preview does not crash.
_SERIF = ("Times New Roman", "Times", "Nimbus Roman", "STIX Two Text", "DejaVu Serif")
_SANS = ("Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans")

#: Okabe & Ito (2008) colour-universal palette, black first. Every pair is
#: distinguishable under the common forms of colour-vision deficiency and
#: the eight have well-separated greyscale values -- the APS style guide's
#: requirement for figures that print in greyscale. Black leads so that a
#: single-series figure is simply black on white.
_OKABE_ITO = ("#000000", "#E69F00", "#56B4E9", "#009E73", "#D55E00",
              "#0072B2", "#CC79A7", "#F0E442")

#: Paul Tol's "bright" qualitative scheme (2021 technical note): colour-
#: blind safe, seven colours, designed for lines on white.
_TOL_BRIGHT = ("#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB")

#: Paul Tol's "high-contrast" scheme: three colours plus black whose
#: greyscale values are also maximally separated, for figures that must
#: read in monochrome with no line-style variation.
_TOL_HIGH_CONTRAST = ("#004488", "#DDAA33", "#BB5566", "#000000")

#: Line-style cycle paired with the colour cycle in the ``classic`` look, so
#: curves stay distinguishable in a greyscale print (APS: "use different
#: line styles (dashed, solid, etc.)" for colour-online figures).
_LINESTYLES = ("-", "--", ":", "-.", "-", "--", ":", "-.")


@dataclass(frozen=True)
class JournalSpec:
    """The page geometry a journal asks figures to be prepared for.

    Frozen: a specification is a record of what a publisher states, and a
    caller wanting a different width passes it to :func:`rc` rather than
    editing the record. ``sources`` records the provenance of the dimensions
    so a figure's geometry can be traced to its specification.

    Attributes
    ----------
    name : str
        Registry key, lower-case (``"prb"``, ``"jpcm"``).
    publisher : str
        Human-readable publisher and journal family.
    single_cm, double_cm : float
        Figure widths for a one-column and a full-width figure, in cm.
    font_pt : float
        Lettering size at final figure size, in points. Tick labels and
        legends are set one point smaller.
    default_look : str
        Key into :data:`~calamari.style.LOOKS` used when a spec string names no look.
    aliases : tuple of str
        Other names resolving to this journal (``"prl"`` for the APS
        geometry, say).
    sources : tuple of str
        Where each number came from.
    """

    name: str
    publisher: str
    single_cm: float
    double_cm: float
    font_pt: float
    default_look: str = "classic"
    aliases: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()

    def width_in(self, width: str | float) -> float:
        """Resolve ``'single'``/``'double'``/a width in cm to inches."""
        if isinstance(width, str):
            if width == "single":
                return self.single_cm * _IN_PER_CM
            if width == "double":
                return self.double_cm * _IN_PER_CM
            raise UnknownStyleError(
                f"width must be 'single', 'double', or a number of centimetres, got {width!r}"
            )
        if not width > 0:
            raise UnknownStyleError(f"width in cm must be positive, got {width!r}")
        return float(width) * _IN_PER_CM


@dataclass(frozen=True)
class LookSpec:
    """Everything about a style that is not page geometry.

    Attributes
    ----------
    name : str
        Registry key.
    description : str
        One line for :func:`available` and the docs.
    family : str
        ``"serif"`` or ``"sans-serif"``; the fallback chain is chosen to
        match.
    mathtext : str
        matplotlib ``mathtext.fontset`` -- ``"stix"`` is metrically
        compatible with Times, ``"stixsans"`` with Helvetica, so
        ``$\\chi'$`` in a label matches the surrounding lettering.
    ticks_in, all_sides, minor_ticks : bool
        Tick direction, whether the top/right spines carry ticks too, and
        whether minor ticks are shown.
    axes_linewidth : float
        Spine and tick width in points. APS's floor for any line is 0.5 pt.
    colors : tuple of str
        Colour cycle.
    linestyles : tuple of str or None
        Paired line-style cycle, or None for solid lines throughout.
    cmap : str
        ``image.cmap`` for :func:`calamari.plot.draw_map` and friends; both
        defaults are perceptually uniform and monotonic in greyscale.
    """

    name: str
    description: str
    family: str
    mathtext: str
    ticks_in: bool
    all_sides: bool
    minor_ticks: bool
    axes_linewidth: float
    colors: tuple[str, ...]
    linestyles: tuple[str, ...] | None
    cmap: str
    extra_rc: dict[str, Any] = field(default_factory=dict)


#: Registered journals by name; extend with :func:`register_journal`.
JOURNALS: dict[str, JournalSpec] = {}
#: Registered looks by name; extend with :func:`register_look`.
LOOKS: dict[str, LookSpec] = {}
#: Alias -> journal name, maintained by :func:`register_journal`.
_ALIASES: dict[str, str] = {}


def register_journal(spec: JournalSpec) -> None:
    """Add (or replace) a journal in the registry, with its aliases.

    The extension point for a journal not shipped here: build a
    :class:`JournalSpec` from the publisher's author guidelines, cite them
    in ``sources``, and register it; ``calamari.style.use(spec.name)`` then
    works like the built-ins. An alias already pointing at a *different*
    journal is refused -- silently re-pointing ``"prl"`` would change the
    width of every figure a script makes without any visible edit.
    """
    # Names and aliases are keyed lower-case, matching the case-insensitive
    # lookup in _resolve -- a spec registered as "JPCM" must be reachable as
    # use("jpcm") and must clash with an existing "jpcm", not sit beside it.
    name = spec.name.strip().lower()
    aliases = tuple(a.strip().lower() for a in spec.aliases)
    for alias in aliases:
        owner = _ALIASES.get(alias)
        if owner is not None and owner != name:
            raise UnknownStyleError(
                f"alias {alias!r} already names journal {owner!r}; pick another alias for "
                f"{name!r} or remove it from {owner!r} first."
            )
    JOURNALS[name] = spec
    for alias in aliases:
        _ALIASES[alias] = name


def register_look(spec: LookSpec) -> None:
    """Add (or replace) a look; ``use("jpcm:<name>")`` then selects it."""
    LOOKS[spec.name.strip().lower()] = spec


register_journal(JournalSpec(
    name="prb",
    publisher="American Physical Society -- Physical Review B/L/Letters, PRX, PRMaterials",
    single_cm=8.6,
    double_cm=17.8,
    font_pt=8.0,
    aliases=("prl", "prx", "prmaterials", "aps"),
    sources=(
        "APS Journals Style Guide (Nov 2024), S. Figures: single-column width 8.5 cm or "
        "3 3/8 in; lettering >= 2 mm high and data points >= 1 mm at final size; lines "
        ">= 0.18 mm (0.5 pt); units in parentheses; greyscale-distinct colours.",
        "REVTeX 4.2 class source, 10pt module: \\textwidth 42.5 pc (510 pt = 18.0 cm), "
        "\\columnsep 1.5 pc, so one column is 246 pt = 8.68 cm; 17.8 cm (7 in) is the "
        "conventional full-width figure and matches APS's Information for Contributors.",
        "8 pt lettering: a 2 mm cap height at final size is a nominal size of about 8 pt "
        "(cap height is roughly 0.7 of the nominal size for Times).",
    ),
))

register_journal(JournalSpec(
    name="jpcm",
    publisher="IOP Publishing -- Journal of Physics: Condensed Matter and sister journals",
    single_cm=8.5,
    double_cm=15.0,
    font_pt=8.0,
    aliases=("iop",),
    sources=(
        "IOP Publishing, 'Figures' (publishingsupport.iopscience.iop.org/questions/"
        "figures-journal-articles): 'Aim for text sizes of 8 to 12 pt at the final figure "
        "size: typically 8.5cm for a small/single-column figure and 15cm for a "
        "large/double-column figure'; fonts restricted to Times, Helvetica, Courier or "
        "Symbol; parts labelled by a lower-case letter in parentheses.",
        "15 cm is IOP's *stated* large-figure width, not a measured text width; the "
        "typeset two-column page was not available to measure.",
    ),
))

register_look(LookSpec(
    name="classic",
    description="Times + STIX mathtext; ticks inward on all four sides with minors; "
                "Okabe-Ito colours paired with a line-style cycle; viridis maps.",
    family="serif", mathtext="stix",
    ticks_in=True, all_sides=True, minor_ticks=True, axes_linewidth=0.6,
    colors=_OKABE_ITO, linestyles=_LINESTYLES, cmap="viridis",
))

register_look(LookSpec(
    name="sans",
    description="Arial/Helvetica + STIX-sans mathtext; ticks inward on all four sides with minors; "
                "Paul Tol 'bright' colours, solid lines; viridis maps.",
    family="sans-serif", mathtext="stixsans",
    ticks_in=True, all_sides=True, minor_ticks=True, axes_linewidth=0.6,
    colors=_TOL_BRIGHT, linestyles=None, cmap="viridis",
))

register_look(LookSpec(
    name="minimal",
    description="Times + STIX mathtext; ticks outward on the left and bottom only, no minors; "
                "Paul Tol 'high-contrast' colours; cividis maps.",
    family="serif", mathtext="stix",
    ticks_in=False, all_sides=False, minor_ticks=False, axes_linewidth=0.5,
    colors=_TOL_HIGH_CONTRAST, linestyles=None, cmap="cividis",
))


def available() -> dict[str, list[str]]:
    """Registered journal and look names, for discovery and error messages."""
    return {"journals": sorted(JOURNALS), "looks": sorted(LOOKS)}


def _resolve(spec: str, look: str | None) -> tuple[JournalSpec, LookSpec]:
    """Parse ``"journal"`` or ``"journal:look"`` against the registries.

    An unknown name is refused with the registered names listed -- a
    misspelt journal must not fall back to matplotlib's defaults, since a
    6.4-inch figure submitted as a column figure is exactly the failure
    this module exists to prevent.
    """
    if not isinstance(spec, str) or not spec:
        raise UnknownStyleError(f"style must be a name like 'jpcm' or 'prb:sans', got {spec!r}")
    journal_name, sep, look_in_spec = spec.partition(":")
    if sep and look is not None and look_in_spec.strip().lower() != look.strip().lower():
        raise UnknownStyleError(
            f"look given twice and differently: {spec!r} says {look_in_spec!r}, look= says {look!r}"
        )
    key = journal_name.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in JOURNALS:
        raise UnknownStyleError(
            f"unknown journal {journal_name!r}; registered: {sorted(JOURNALS)} "
            f"(aliases: {sorted(_ALIASES)}). Add one with calamari.style.register_journal."
        )
    journal = JOURNALS[key]
    look_name = (look_in_spec if sep else look) or journal.default_look
    look_key = look_name.strip().lower()
    if look_key not in LOOKS:
        raise UnknownStyleError(
            f"unknown look {look_name!r}; registered: {sorted(LOOKS)}."
        )
    return journal, LOOKS[look_key]


def rc(
    spec: str,
    *,
    width: str | float = "single",
    look: str | None = None,
    height_cm: float | None = None,
) -> dict[str, Any]:
    """The rcParams for a journal and look, as a plain dict.

    This is the whole style; :func:`use`, :func:`context` and
    :func:`figure` are thin wrappers over it. Returned as a dict so it can
    be inspected, diffed, or merged with a caller's own overrides
    (``{**calamari.style.rc("prb"), "lines.linewidth": 1.5}``).

    Parameters
    ----------
    spec : str
        ``"jpcm"``, ``"prb"``, an alias, or ``"journal:look"``.
    width : {'single', 'double'} or float
        Figure width -- one column, the full text width, or a value in cm
        (for a 1.5-column APS figure, say).
    look : str, optional
        Look name; overrides the journal's default. Equivalent to the
        ``:look`` suffix.
    height_cm : float, optional
        Figure height in cm. Default is ``width / DEFAULT_ASPECT``.

    Returns
    -------
    dict
        Keys and values acceptable to ``matplotlib.rcParams.update``.
    """
    journal, lk = _resolve(spec, look)
    w_in = journal.width_in(width)
    h_in = (height_cm * _IN_PER_CM) if height_cm is not None else w_in / DEFAULT_ASPECT
    if not h_in > 0:
        raise UnknownStyleError(f"height_cm must be positive, got {height_cm!r}")
    pt = journal.font_pt
    lw = lk.axes_linewidth
    prop_cycle = cycler(color=list(lk.colors))
    if lk.linestyles is not None:
        n = min(len(lk.colors), len(lk.linestyles))
        prop_cycle = cycler(color=list(lk.colors[:n])) + cycler(linestyle=list(lk.linestyles[:n]))

    params: dict[str, Any] = {
        # -- page geometry and output --------------------------------------
        "figure.figsize": (w_in, h_in),
        "figure.dpi": 100,            # screen preview; the file is what matters
        "savefig.dpi": 600,           # APS: "preferably 600 dpi or higher"
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.format": "pdf",      # both publishers want vector; PNG is a draft
        # Embed fonts as TrueType (Type 42), not matplotlib's Type 3 default:
        # Type 3 is rejected by some submission systems and leaves the text
        # unselectable and unsearchable in the PDF. Type 42 files are larger.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # -- lettering ---------------------------------------------------------
        "font.size": pt,
        "axes.labelsize": pt,
        "axes.titlesize": pt,
        "xtick.labelsize": pt - 1,
        "ytick.labelsize": pt - 1,
        "legend.fontsize": pt - 1,
        "legend.title_fontsize": pt - 1,
        "mathtext.fontset": lk.mathtext,
        "text.usetex": False,
        "axes.formatter.use_mathtext": True,   # 10^4 not 1e4 on offset labels
        "axes.formatter.limits": (-3, 4),
        "axes.unicode_minus": True,
        # -- lines and markers (APS floors: 0.5 pt lines, 1 mm points) --------
        "axes.linewidth": lw,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "lines.markeredgewidth": 0.6,
        "errorbar.capsize": 1.5,
        "patch.linewidth": lw,
        # -- ticks --------------------------------------------------------------
        "xtick.direction": "in" if lk.ticks_in else "out",
        "ytick.direction": "in" if lk.ticks_in else "out",
        "xtick.top": lk.all_sides,
        "ytick.right": lk.all_sides,
        "xtick.minor.visible": lk.minor_ticks,
        "ytick.minor.visible": lk.minor_ticks,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.width": lw, "ytick.major.width": lw,
        "xtick.minor.width": lw, "ytick.minor.width": lw,  # APS floor applies to minors too
        "xtick.major.pad": 3.0, "ytick.major.pad": 3.0,
        # -- colour ------------------------------------------------------------
        "axes.prop_cycle": prop_cycle,
        "image.cmap": lk.cmap,
        # -- legend --------------------------------------------------------------
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.borderaxespad": 0.4,
        # -- nothing decorative --------------------------------------------------
        "axes.grid": False,
    }
    if lk.family == "serif":
        params["font.family"] = "serif"
        params["font.serif"] = list(_SERIF)
    else:
        params["font.family"] = "sans-serif"
        params["font.sans-serif"] = list(_SANS)
    params.update(lk.extra_rc)
    return params


def use(spec: str, **kwargs: Any) -> None:
    """Apply a journal style to the current matplotlib session.

    Global, like ``matplotlib.style.use``: every figure created afterwards
    -- including the ones :mod:`calamari.plot` helpers create when ``ax``
    is omitted -- comes out at the journal's column width. Keyword
    arguments are those of :func:`rc`. To undo, ``matplotlib.rcdefaults()``
    or prefer :func:`context`.
    """
    import matplotlib as mpl

    mpl.rcParams.update(rc(spec, **kwargs))


@contextmanager
def context(spec: str, **kwargs: Any) -> Iterator[None]:
    """Apply a journal style inside a ``with`` block only.

    The right tool in a notebook or a script that also makes screen
    figures: rcParams are restored on exit, however the block ends.
    Keyword arguments are those of :func:`rc`.

    >>> with calamari.style.context("prb", width="double"):    # doctest: +SKIP
    ...     fig, axes = plt.subplots(1, 2)                      # doctest: +SKIP
    """
    import matplotlib as mpl

    with mpl.rc_context(rc(spec, **kwargs)):
        yield


def figure(
    spec: str,
    *,
    width: str | float = "single",
    look: str | None = None,
    height_cm: float | None = None,
    nrows: int = 1,
    ncols: int = 1,
    **subplots_kwargs: Any,
) -> tuple[Figure, Any]:
    """Create a journal-sized figure and axes, styled, in one call.

    ``plt.subplots`` under :func:`context`, returning what it returns. The
    figure keeps its size after the context ends, but artists added later
    from *outside* a style context take the session's rcParams -- so build
    the whole figure inside one :func:`context` block when mixing helpers
    with your own ``ax.plot`` calls, or call :func:`use` once.

    The default height gives every panel :data:`DEFAULT_ASPECT`, so a
    two-across double-column figure is one panel tall; ``layout='constrained'`` is on unless
    ``subplots_kwargs`` says otherwise, because tick labels at 7 pt on an
    8.5 cm figure clip without it.

    Parameters
    ----------
    spec, width, look, height_cm
        As for :func:`rc`. ``height_cm`` is the height of the *whole*
        figure.
    nrows, ncols : int
        Panel grid, forwarded to ``plt.subplots``.
    **subplots_kwargs
        Forwarded to ``plt.subplots`` (``sharex=``, ``gridspec_kw=``, ...).

    Returns
    -------
    tuple
        ``(fig, axes)`` exactly as ``plt.subplots`` returns them: a single
        ``Axes`` for a 1x1 grid, otherwise an array of them.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if not (isinstance(nrows, int) and nrows >= 1 and isinstance(ncols, int) and ncols >= 1):
        raise UnknownStyleError(
            f"nrows and ncols must be positive integers, got {nrows!r}, {ncols!r}"
        )
    params = rc(spec, width=width, look=look, height_cm=height_cm)
    if height_cm is None:
        # Each panel keeps DEFAULT_ASPECT: a 1x2 double-column figure is
        # the height of one single-column panel, not of a double-width one.
        w_in = params["figure.figsize"][0]
        params["figure.figsize"] = (w_in, (w_in / ncols) / DEFAULT_ASPECT * nrows)
    subplots_kwargs.setdefault("layout", "constrained")
    with mpl.rc_context(params):
        fig, axes = plt.subplots(nrows, ncols, **subplots_kwargs)
    return fig, axes


def panel_labels(
    axes: Sequence[Axes] | Any,
    *,
    loc: tuple[float, float] = (0.03, 0.96),
    fmt: str = "({})",
    start: str = "a",
    **text_kwargs: Any,
) -> list[Any]:
    """Letter a figure's panels ``(a)``, ``(b)``, ... in axes coordinates.

    Both APS and IOP want multi-part figures labelled by a lower-case
    letter in parentheses; this puts one in each panel's top-left corner,
    at the lettering size, and returns the ``Text`` artists so a caller can
    move or restyle one. ``axes`` may be a single ``Axes``, a list, or the
    array ``plt.subplots`` returns (flattened in C order, so a 2x2 grid
    reads a, b across the top row).

    Parameters
    ----------
    axes
        The panels, in labelling order.
    loc : (float, float)
        Position in axes fraction; default just inside the top-left.
    fmt : str
        Format with one ``{}`` for the letter.
    start : str
        First letter.
    **text_kwargs
        Forwarded to ``Axes.text`` (``color='white'`` over a dark map).
    """
    import numpy as np

    flat = list(np.atleast_1d(np.asarray(axes, dtype=object)).ravel())
    kwargs = {"va": "top", "ha": "left"}
    kwargs.update(text_kwargs)
    return [
        ax.text(loc[0], loc[1], fmt.format(chr(ord(start) + i)), transform=ax.transAxes, **kwargs)
        for i, ax in enumerate(flat)
    ]
