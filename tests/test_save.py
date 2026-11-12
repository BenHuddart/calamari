"""Tests for calamari.plot.save: the path rule, sibling formats, and the guarantees.

``save`` is a thin wrapper over ``Figure.savefig``, so the tests are about
what the wrapper adds -- resolving a figure from an ``Axes``, the
suffix-or-stem rule, validating every format before writing anything,
creating the output directory, the returned paths, and ``close`` -- and
about what it deliberately does not add (a metadata stamp). The figures
are plain matplotlib: nothing here depends on a measurement.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; must precede the pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import calamari  # noqa: E402
from calamari import plot, style  # noqa: E402
from calamari.exceptions import CalamariError, UnknownFigureFormatError  # noqa: E402


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0], label="line")
    ax.set_xlabel("Temperature (K)")
    yield ax
    plt.close(fig)


def test_suffix_names_the_format_and_nothing_else_is_written(ax, tmp_path):
    out = plot.save(ax, tmp_path / "fig1.pdf")
    assert out == [tmp_path / "fig1.pdf"]
    assert out[0].read_bytes().startswith(b"%PDF")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fig1.pdf"]


def test_figure_is_accepted_as_well_as_axes(ax, tmp_path):
    assert isinstance(ax.figure, Figure)
    out = plot.save(ax.figure, tmp_path / "fig1.png")
    assert out == [tmp_path / "fig1.png"]
    assert out[0].read_bytes().startswith(b"\x89PNG")


def test_formats_write_siblings_at_one_stem_in_order(ax, tmp_path):
    out = plot.save(ax, tmp_path / "fig1", formats=("pdf", "png", "svg"))
    assert out == [tmp_path / "fig1.pdf", tmp_path / "fig1.png", tmp_path / "fig1.svg"]
    assert all(p.stat().st_size > 0 for p in out)


def test_suffix_and_formats_combine_without_duplicates(ax, tmp_path):
    # The path's own format comes first; a repeat of it (any case, with or
    # without a dot) is written once.
    out = plot.save(ax, tmp_path / "fig1.pdf", formats=("PNG", ".pdf", "png"))
    assert out == [tmp_path / "fig1.pdf", tmp_path / "fig1.png"]


def test_bare_stem_defaults_to_pdf(ax, tmp_path):
    assert plot.DEFAULT_SAVE_FORMAT == "pdf"
    out = plot.save(ax, tmp_path / "fig1")
    assert out == [tmp_path / "fig1.pdf"]


def test_unknown_suffix_is_part_of_the_stem(ax, tmp_path):
    out = plot.save(ax, tmp_path / "fig1.v2")
    assert out == [tmp_path / "fig1.v2.pdf"]


def test_own_suffix_keeps_its_spelling(ax, tmp_path):
    out = plot.save(ax, tmp_path / "fig1.PDF", formats=("png",))
    assert out == [tmp_path / "fig1.PDF", tmp_path / "fig1.png"]
    assert out[0].read_bytes().startswith(b"%PDF")


def test_unknown_format_raises_before_anything_is_written(ax, tmp_path):
    target = tmp_path / "out"
    with pytest.raises(UnknownFigureFormatError, match="'pgn'") as info:
        plot.save(ax, target / "fig1.pdf", formats=("png", "pgn"))
    assert "Nothing was written" in str(info.value)
    assert "pdf" in str(info.value)  # the supported list is in the message
    assert not target.exists()  # not even the directory
    assert isinstance(info.value, CalamariError)
    assert isinstance(info.value, ValueError)


def test_missing_parent_directories_are_created(ax, tmp_path):
    out = plot.save(ax, tmp_path / "figures" / "paper" / "fig1.pdf")
    assert out[0].exists()


def test_existing_file_is_overwritten(ax, tmp_path):
    target = tmp_path / "fig1.png"
    target.write_bytes(b"stale")
    plot.save(ax, target)
    assert target.read_bytes().startswith(b"\x89PNG")


def test_format_kwarg_is_refused(ax, tmp_path):
    with pytest.raises(TypeError, match="format="):
        plot.save(ax, tmp_path / "fig1", format="pdf")


def test_wrong_target_type_is_a_type_error(tmp_path):
    with pytest.raises(TypeError, match="Axes or a Figure"):
        plot.save("not an axes", tmp_path / "fig1.pdf")  # type: ignore[arg-type]


def test_figure_stays_open_by_default_and_closes_on_request(tmp_path):
    fig, ax = plt.subplots()
    plot.save(ax, tmp_path / "a.png")
    assert fig.number in plt.get_fignums()
    plot.save(ax, tmp_path / "b.png", close=True)
    assert fig.number not in plt.get_fignums()


def test_savefig_kwargs_are_forwarded(ax, tmp_path):
    from PIL import Image  # Pillow is a hard dependency of matplotlib (pillow>=9)

    ax.figure.set_size_inches(2.0, 1.0)
    (out,) = plot.save(ax, tmp_path / "fig1.png", dpi=50, bbox_inches=None)
    with Image.open(out) as image:
        assert image.size == (100, 50)


def test_no_metadata_is_added_to_the_file(ax, tmp_path):
    # The maintainer's decision: save() stamps nothing. The PNG carries only
    # what matplotlib itself writes by default (its Software tag).
    from PIL import Image  # Pillow is a hard dependency of matplotlib (pillow>=9)

    (out,) = plot.save(ax, tmp_path / "fig1.png")
    with Image.open(out) as image:
        keys = set(image.info)
    assert "calamari" not in " ".join(f"{k}={v}" for k, v in image.info.items()).lower()
    assert keys <= {"Software", "dpi"}


def test_journal_style_embeds_truetype_fonts_and_defaults_to_pdf(tmp_path):
    for spec in style.available()["journals"]:
        params = style.rc(spec)
        assert params["pdf.fonttype"] == 42
        assert params["ps.fonttype"] == 42
        assert params["savefig.format"] == "pdf"
    with style.context("jpcm"):
        fig, ax = plt.subplots()
        ax.plot([0.0, 1.0], [0.0, 1.0])
        ax.set_xlabel(plot.axis_labels("chi", "real"))  # mathtext: exercises a second font
        (out,) = plot.save(ax, tmp_path / "fig1.pdf")
        # A bare savefig under the style also lands as PDF.
        fig.savefig(tmp_path / "bare")
    plt.close(fig)
    pdf = out.read_bytes()
    assert b"/Type3" not in pdf
    assert b"/TrueType" in pdf or b"/Type0" in pdf
    assert (tmp_path / "bare.pdf").read_bytes().startswith(b"%PDF")


def test_unstyled_matplotlib_default_is_type3_so_the_style_matters(tmp_path):
    # Guards the rationale: without the style matplotlib writes Type 3
    # fonts. If a future matplotlib changes its default this test tells us
    # the rcParams above have become redundant rather than silently so.
    with matplotlib.rc_context({"pdf.fonttype": 3}):
        fig, ax = plt.subplots()
        ax.plot([0.0, 1.0], [0.0, 1.0])
        ax.set_xlabel("Temperature (K)")
        (out,) = plot.save(ax, tmp_path / "fig1.pdf")
    plt.close(fig)
    assert matplotlib.rcParamsDefault["pdf.fonttype"] == 3
    assert b"/Type3" in out.read_bytes()


def test_top_level_namespace_reaches_save():
    assert calamari.plot.save is plot.save
