"""Tests for calamari.style: journal geometry, looks, and the entry points.

The style module is rcParams in, rcParams out, so most tests read the
dict :func:`calamari.style.rc` returns; the drawing tests confirm that a
:mod:`calamari.plot` helper creating its own figure under a style context
really does come out at the journal's column width, and that every look
renders every label path to PNG (a fontset that cannot set ``$\\chi'$``
only fails at draw time).
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; must precede the pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import calamari  # noqa: E402
from calamari import plot, style  # noqa: E402
from calamari.exceptions import UnknownStyleError  # noqa: E402
from calamari.measurement import Measurement  # noqa: E402

CM = 2.54


@pytest.fixture(autouse=True)
def _restore_rcparams():
    with matplotlib.rc_context():
        yield


@pytest.fixture
def mh_measurement() -> Measurement:
    field = np.linspace(-5000.0, 5000.0, 21)
    return Measurement(
        path=Path("synthetic.dat"), data_format="MPMS3", measurement_type="DC",
        file_role="summary", title="synthetic", header={},
        data=pd.DataFrame({"Magnetic Field (Oe)": field, "Moment (emu)": np.tanh(field / 2000)}),
    )


# ---------------------------------------------------------------------------
# geometry: the numbers the journals state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec, width, expected_cm",
    [
        ("prb", "single", 8.6),
        ("prb", "double", 17.8),
        ("jpcm", "single", 8.5),
        ("jpcm", "double", 15.0),
        ("prl", "single", 8.6),  # alias resolves to the APS geometry
        ("iop", "single", 8.5),
        ("prb", 12.0, 12.0),  # explicit cm, e.g. a 1.5-column APS figure
    ],
)
def test_rc_width_matches_journal(spec, width, expected_cm):
    w_in, _ = style.rc(spec, width=width)["figure.figsize"]
    assert w_in * CM == pytest.approx(expected_cm)


def test_rc_default_height_follows_aspect():
    w, h = style.rc("jpcm")["figure.figsize"]
    assert w / h == pytest.approx(style.DEFAULT_ASPECT)


def test_rc_explicit_height():
    _, h = style.rc("jpcm", height_cm=5.0)["figure.figsize"]
    assert h * CM == pytest.approx(5.0)


def test_rc_lettering_is_journal_font_size_with_smaller_ticks():
    params = style.rc("prb")
    assert params["font.size"] == 8.0
    assert params["axes.labelsize"] == 8.0
    assert params["xtick.labelsize"] == 7.0
    assert params["legend.fontsize"] == 7.0


def test_rc_respects_aps_line_floor():
    """APS: every line at least 0.5 pt at final size."""
    for look in style.LOOKS:
        params = style.rc("prb", look=look)
        assert params["axes.linewidth"] >= 0.5
        assert params["lines.linewidth"] >= 0.5
        assert params["xtick.major.width"] >= 0.5
        assert params["xtick.minor.width"] >= 0.5


def test_rc_saves_at_600_dpi_tight():
    params = style.rc("jpcm")
    assert params["savefig.dpi"] == 600
    assert params["savefig.bbox"] == "tight"


def test_font_chains_end_in_a_bundled_face():
    """A host without Times/Helvetica must fall back silently, not warn."""
    assert style.rc("prb")["font.serif"][-1] == "DejaVu Serif"
    assert style.rc("prb:sans")["font.sans-serif"][-1] == "DejaVu Sans"


# ---------------------------------------------------------------------------
# looks
# ---------------------------------------------------------------------------


def test_default_look_is_classic_for_both_journals():
    for name in ("prb", "jpcm"):
        assert style.JOURNALS[name].default_look == "classic"
        assert style.rc(name)["font.family"] == "serif"
        assert style.rc(name)["mathtext.fontset"] == "stix"


def test_look_suffix_and_keyword_agree():
    assert style.rc("jpcm:sans") == style.rc("jpcm", look="sans")
    assert style.rc("JPCM:Sans") == style.rc("jpcm", look="sans")  # case-insensitive


def test_look_given_twice_but_equal_up_to_case_is_fine():
    assert style.rc("jpcm:sans", look="Sans") == style.rc("jpcm:sans")


def test_register_journal_normalises_case(monkeypatch):
    monkeypatch.setattr(style, "JOURNALS", dict(style.JOURNALS))
    monkeypatch.setattr(style, "_ALIASES", dict(style._ALIASES))
    style.register_journal(style.JournalSpec(
        name="MixedCase", publisher="x", single_cm=8.0, double_cm=16.0, font_pt=8.0,
        aliases=("MC",),
    ))
    assert style.rc("mixedcase")["figure.figsize"][0] * CM == pytest.approx(8.0)
    assert style.rc("mc") == style.rc("MIXEDCASE")
    with pytest.raises(UnknownStyleError, match="prl"):
        style.register_journal(style.JournalSpec(
            name="other", publisher="x", single_cm=8.0, double_cm=16.0, font_pt=8.0,
            aliases=("PRL",),
        ))


@pytest.mark.parametrize("grid", [dict(nrows=0), dict(ncols=0), dict(nrows=-1), dict(ncols=1.5)])
def test_figure_refuses_bad_grid(grid):
    with pytest.raises(UnknownStyleError, match="positive integers"):
        style.figure("prb", **grid)


def test_look_given_twice_and_differently_is_refused():
    with pytest.raises(UnknownStyleError, match="twice"):
        style.rc("jpcm:sans", look="classic")


def test_classic_look_pairs_linestyles_with_colours():
    """Greyscale survival: the classic cycle varies line style as well as colour."""
    keys = style.rc("prb")["axes.prop_cycle"].keys
    assert keys == {"color", "linestyle"}
    assert style.rc("prb:sans")["axes.prop_cycle"].keys == {"color"}


def test_classic_look_leads_with_black():
    first = next(iter(style.rc("prb")["axes.prop_cycle"]))
    assert first["color"].lower() == "#000000"


def test_sans_look_switches_family_and_mathtext():
    params = style.rc("prb:sans")
    assert params["font.family"] == "sans-serif"
    assert params["mathtext.fontset"] == "stixsans"


def test_minimal_look_ticks_out_left_bottom_only():
    params = style.rc("prb:minimal")
    assert params["xtick.direction"] == "out"
    assert params["xtick.top"] is False
    assert params["ytick.right"] is False
    assert params["xtick.minor.visible"] is False
    params = style.rc("prb:classic")
    assert params["xtick.direction"] == "in"
    assert params["xtick.top"] is True
    assert params["xtick.minor.visible"] is True


def test_map_colormaps_are_greyscale_monotonic():
    """Both default maps must read in a greyscale print: luminance
    strictly increasing along the map."""
    for look in style.LOOKS:
        cmap = matplotlib.colormaps[style.rc("prb", look=look)["image.cmap"]]
        rgb = cmap(np.linspace(0, 1, 64))[:, :3]
        lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
        assert np.all(np.diff(lum) > 0), look


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def test_unknown_journal_lists_registered_names():
    with pytest.raises(UnknownStyleError, match="jpcm") as info:
        style.rc("nature")
    assert "register_journal" in str(info.value)
    assert isinstance(info.value, ValueError)
    assert isinstance(info.value, calamari.CalamariError)


def test_unknown_look_lists_registered_names():
    with pytest.raises(UnknownStyleError, match="classic"):
        style.rc("prb:fancy")


@pytest.mark.parametrize("width", ["wide", 0.0, -3.0])
def test_bad_width_refused(width):
    with pytest.raises(UnknownStyleError):
        style.rc("prb", width=width)


def test_bad_height_refused():
    with pytest.raises(UnknownStyleError):
        style.rc("prb", height_cm=0.0)


@pytest.mark.parametrize("spec", ["", None, 3])
def test_non_string_spec_refused(spec):
    with pytest.raises(UnknownStyleError):
        style.rc(spec)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def test_register_journal_then_use(monkeypatch):
    monkeypatch.setattr(style, "JOURNALS", dict(style.JOURNALS))
    monkeypatch.setattr(style, "_ALIASES", dict(style._ALIASES))
    spec = style.JournalSpec(
        name="testj", publisher="Test", single_cm=9.0, double_cm=18.0, font_pt=9.0,
        default_look="sans", aliases=("tj",), sources=("made up for the test",),
    )
    style.register_journal(spec)
    params = style.rc("tj", width="double")
    assert params["figure.figsize"][0] * CM == pytest.approx(18.0)
    assert params["font.size"] == 9.0
    assert params["font.family"] == "sans-serif"
    assert "testj" in style.available()["journals"]


def test_register_journal_refuses_alias_clash(monkeypatch):
    monkeypatch.setattr(style, "JOURNALS", dict(style.JOURNALS))
    monkeypatch.setattr(style, "_ALIASES", dict(style._ALIASES))
    spec = style.JournalSpec(
        name="other", publisher="x", single_cm=8.0, double_cm=16.0, font_pt=8.0, aliases=("prl",)
    )
    with pytest.raises(UnknownStyleError, match="prl"):
        style.register_journal(spec)
    assert style._ALIASES["prl"] == "prb"  # untouched


def test_register_look_then_use(monkeypatch):
    monkeypatch.setattr(style, "LOOKS", dict(style.LOOKS))
    base = style.LOOKS["classic"]
    from dataclasses import replace

    style.register_look(replace(base, name="bold", extra_rc={"lines.linewidth": 2.0}))
    assert style.rc("prb:bold")["lines.linewidth"] == 2.0


# ---------------------------------------------------------------------------
# entry points: use / context / figure / panel_labels
# ---------------------------------------------------------------------------


def test_use_is_global_and_context_restores():
    before = list(plt.rcParams["figure.figsize"])
    with style.context("jpcm"):
        assert plt.rcParams["figure.figsize"][0] * CM == pytest.approx(8.5)
    assert list(plt.rcParams["figure.figsize"]) == before
    style.use("prb", width="double")
    assert plt.rcParams["figure.figsize"][0] * CM == pytest.approx(17.8)


def test_plot_helper_creates_column_width_figure_under_context(mh_measurement):
    """The whole point: ax=None figures from calamari.plot come out at column width."""
    with style.context("jpcm"):
        ax = plot.plot_mh(mh_measurement)
    try:
        w, h = ax.figure.get_size_inches()
        assert w * CM == pytest.approx(8.5)
        assert w / h == pytest.approx(style.DEFAULT_ASPECT)
        assert ax.xaxis.get_label().get_fontsize() == 8.0
    finally:
        plt.close(ax.figure)


def test_figure_double_column_two_panels_is_one_panel_tall():
    fig, axes = style.figure("prb", width="double", ncols=2)
    try:
        w, h = fig.get_size_inches()
        assert w * CM == pytest.approx(17.8)
        assert (w / 2) / h == pytest.approx(style.DEFAULT_ASPECT)
        assert axes.shape == (2,)
        assert fig.get_layout_engine() is not None  # constrained layout on
    finally:
        plt.close(fig)


def test_figure_forwards_subplots_kwargs():
    fig, axes = style.figure("jpcm", nrows=2, sharex=True, layout=None)
    try:
        assert axes.shape == (2,)
        assert axes[1].get_shared_x_axes().joined(axes[0], axes[1])
        assert fig.get_layout_engine() is None
    finally:
        plt.close(fig)


def test_panel_labels_letters_in_order_and_returns_artists():
    fig, axes = style.figure("prb", width="double", nrows=2, ncols=2)
    try:
        texts = style.panel_labels(axes, color="red")
        assert [t.get_text() for t in texts] == ["(a)", "(b)", "(c)", "(d)"]
        assert texts[0].get_color() == "red"
        assert texts[0].axes is axes[0, 0] and texts[1].axes is axes[0, 1]
        (single,) = style.panel_labels(axes[1, 1], start="e", fmt="{}.")
        assert single.get_text() == "e."
    finally:
        plt.close(fig)


@pytest.mark.parametrize("look", sorted(style.LOOKS))
def test_every_look_renders_every_label_path(look, mh_measurement):
    """Force a real draw under each fontset: chi', mu_B and the 10^n offset
    text all have to set in the look's mathtext font."""
    chi = Measurement(
        path=Path("s.dat"), data_format="MPMS3", measurement_type="AC", file_role="summary",
        title="s", header={},
        data=pd.DataFrame({
            "Temperature (K)": np.linspace(2, 300, 20),
            "AC X' (emu/Oe)": 1e-4 / np.linspace(2, 300, 20),
            "AC X'' (emu/Oe)": 1e-6 / np.linspace(2, 300, 20),
        }),
    )
    with style.context("prb", look=look):
        fig, (a1, a2) = plt.subplots(1, 2)
        plot.plot_chi_vs_t(chi, ax=a1, component="both", imag_scale=10.0)
        a1.legend()
        plot.plot_mh(
            mh_measurement, ax=a2, units="muB_per_ion",
            sample=calamari.Sample(name="s", mass_mg=10.0, molar_mass_g_per_mol=100.0,
                                   magnetic_ions_per_fu=1),
        )
        style.panel_labels((a1, a2))
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=100)
    plt.close(fig)
    assert buffer.getbuffer().nbytes > 0
