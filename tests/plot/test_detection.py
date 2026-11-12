"""Candidate overlays, detection diagnostics and boundary drawing."""

from __future__ import annotations

import re
import warnings
from io import BytesIO
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import QuadMesh

from calamari import _plot_data as plot_data
from calamari import plot
from calamari.plot import _detection as plot_detection

from ._helpers import (
    _TESLA_LABEL,
    _ac_tscan_measurements,
    _chi_vs_h_measurement,
    _detected,
    _field_candidates,
    _line_xy,
    _toy_afm_file,
)


def test_overlay_on_map_places_points_and_states_thresholds(tmp_path):
    """Map-mode overlay draws one stroked marker per candidate at (T, H)
    and the legend title carries the robust() thresholds -- the figure
    states its own inclusion criterion."""
    ms, result = _detected(tmp_path)
    mesh = plot.plot_chi_map(ms, transitions=result)
    ax = mesh.axes
    legend = ax.get_legend()
    assert legend is not None
    assert "significance" in legend.get_title().get_text()
    plt.close("all")


def test_overlay_index_scale_snaps_to_rows(tmp_path):
    """On a family_scale='index' map the family coordinate is a row index;
    the one-call path wires grid and scale through so points land on rows,
    not at physical values."""
    ms, result = _detected(tmp_path)
    mesh = plot.plot_chi_map(
        ms, family_scale="index", transitions=result,
        transition_kwargs={"legend": False},
    )
    # Family axis is field (T-scans) on y: overlaid y-values must be the
    # row indices 0..2, not the physical fields 0.01..1.0 T.
    overlaid = [
        line for line in mesh.axes.get_lines() if line.get_marker() not in ("", "None")
    ]
    ys = np.concatenate([line.get_ydata() for line in overlaid])
    assert set(np.round(ys).astype(int)) <= {0, 1, 2}
    plt.close("all")


def test_overlay_returns_artists_keyed_by_group(tmp_path):
    ms, result = _detected(tmp_path)
    grid = plot.chi_map_grid(ms)
    fig, ax = plt.subplots()
    artists = plot.overlay_transitions(result, ax=ax, grid=grid)
    assert ("peak", None) in artists
    assert artists[("peak", None)]
    plt.close("all")


def test_overlay_1d_marks_swept_coordinate(tmp_path):
    """1-D mode (no grid) draws vertical rules at the candidates' swept
    coordinates on a sweep plot -- the audit view."""
    ms, result = _detected(tmp_path)
    low_field = result.candidates[result.candidates["H_T"] < 0.05]
    ax = plot.plot_chi_vs_t(ms[0])
    artists = plot.overlay_transitions(low_field, ax=ax)
    lines = [a for group in artists.values() for a in group]
    assert lines
    assert lines[0].get_xdata()[0] == pytest.approx(14.0, abs=0.3)
    plt.close("all")


def test_overlay_family_scale_without_grid_refused(tmp_path):
    _, result = _detected(tmp_path)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="grid"):
        plot.overlay_transitions(result, ax=ax, family_scale="index")
    plt.close("all")


def _rule_positions(artists) -> list[float]:
    return sorted(line.get_xdata()[0] for group in artists.values() for line in group)


def test_overlay_1d_field_rules_follow_field_unit(ax):
    """A rule lands at the candidate's field in the unit the sweep was drawn
    in; temperature rules are unaffected by field_unit."""
    plot.plot_chi_vs_h(_chi_vs_h_measurement(), ax=ax, field_unit="tesla")
    artists = plot.overlay_transitions(_field_candidates(), ax=ax, field_unit="tesla")
    assert _rule_positions(artists) == pytest.approx([0.3, 5.0])

    fig, ax_oe = plt.subplots()
    try:
        plot.plot_chi_vs_h(_chi_vs_h_measurement(), ax=ax_oe)
        artists = plot.overlay_transitions(_field_candidates(), ax=ax_oe, field_unit="oe")
        assert _rule_positions(artists) == pytest.approx([5.0, 3000.0])
    finally:
        plt.close(fig)

_CHI_VS_H_PLOTTERS = {
    "single": lambda ax, unit: plot.plot_chi_vs_h(
        _chi_vs_h_measurement(), ax=ax, field_unit=unit
    ),
    "family": lambda ax, unit: plot.plot_chi_vs_h_family(
        [_chi_vs_h_measurement()], ax=ax, field_unit=unit
    ),
}


@pytest.mark.parametrize("plotter", sorted(_CHI_VS_H_PLOTTERS))
@pytest.mark.parametrize(("unit", "expected"), [("tesla", [0.3, 5.0]), ("oe", [5.0, 3000.0])])
def test_overlay_1d_default_follows_calamaris_own_field_label(ax, plotter, unit, expected):
    """With field_unit left unset, the unit is read from the label calamari
    wrote -- a record of the plotting call's field_unit, not a guess. On
    main the default was 'oe', so a Tesla plot refused it and made the
    caller repeat field_unit='tesla'."""
    _CHI_VS_H_PLOTTERS[plotter](ax, unit)
    artists = plot.overlay_transitions(_field_candidates(), ax=ax)
    assert _rule_positions(artists) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("plotted", "passed", "match"),
    [
        ("tesla", "oe", r"this axis is in Tesla.*Pass field_unit='tesla'"),
        ("oe", "tesla", r"this axis is in Oe.*Pass field_unit='oe'"),
    ],
)
def test_overlay_1d_refuses_field_unit_that_contradicts_the_axis(ax, plotted, passed, match):
    plot.plot_chi_vs_h(_chi_vs_h_measurement(), ax=ax, field_unit=plotted)
    n_lines = len(ax.lines)
    with pytest.raises(ValueError, match=match):
        plot.overlay_transitions(_field_candidates(), ax=ax, field_unit=passed)
    assert len(ax.lines) == n_lines  # refused before anything was drawn


@pytest.mark.parametrize(("unit", "expected"), [("tesla", [0.3, 5.0]), ("oe", [5.0, 3000.0])])
def test_overlay_1d_relabelled_axis_takes_field_unit_at_its_word(ax, unit, expected):
    """A label calamari did not write is not second-guessed in either
    direction -- the caller who relabelled owns the unit."""
    plot.plot_chi_vs_h(_chi_vs_h_measurement(), ax=ax, field_unit=unit)
    ax.set_xlabel("B (T)" if unit == "tesla" else "H (Oe)")
    artists = plot.overlay_transitions(_field_candidates(), ax=ax, field_unit=unit)
    assert _rule_positions(artists) == pytest.approx(expected)


@pytest.mark.parametrize("label", ["B (T)", ""])
def test_overlay_1d_default_refuses_a_field_rule_on_a_foreign_label(ax, label):
    """On a label calamari did not write the axis unit is unknowable, and a
    field rule's two placements are 10^4 apart. On main the default silently
    assumed Oe: a relabelled Tesla plot got its rule at 3000 on a 0-1 axis."""
    plot.plot_chi_vs_h(_chi_vs_h_measurement(), ax=ax, field_unit="tesla")
    ax.set_xlabel(label)
    n_lines = len(ax.lines)
    shown = re.escape(f'x-label "{label}"') if label else "it has no x-label"
    with pytest.raises(
        ValueError,
        match=rf"cannot read the field axis unit.*{shown}.*"
        r"Pass field_unit='oe' or field_unit='tesla', whichever the curve was plotted in",
    ):
        plot.overlay_transitions(_field_candidates(), ax=ax)
    assert len(ax.lines) == n_lines  # refused before anything was drawn


def test_overlay_1d_default_needs_no_unit_for_temperature_rules(ax):
    """T-swept rules need no field unit, so a chi(T)-style overlay works with
    the default on any label -- the refusal is only for a rule it cannot place."""
    ax.set_xlabel("my temperature")
    t_only = _field_candidates().query("swept_axis == 'T'")
    assert _rule_positions(plot.overlay_transitions(t_only, ax=ax)) == pytest.approx([5.0])

    fig, ax2 = plt.subplots()
    try:
        ax2.set_xlabel("my temperature")
        mixed = plot.overlay_transitions(_field_candidates(), ax=ax2, x="temperature")
        assert _rule_positions(mixed) == pytest.approx([5.0])
    finally:
        plt.close(fig)


def test_overlay_field_unit_value_checked_even_with_nothing_to_draw(ax):
    with pytest.raises(
        ValueError, match=r"field_unit must be one of \['oe', 'tesla'\], or None.*got 'gauss'"
    ):
        plot.overlay_transitions(_field_candidates().iloc[:0], ax=ax, field_unit="gauss")
    ax.set_xlabel("B (T)")
    assert plot.overlay_transitions(_field_candidates().iloc[:0], ax=ax) == {}


def test_overlay_map_mode_field_unit(tmp_path):
    """A map's field axis is always Tesla: the default and 'tesla' (simply
    true) place identical points, so one field_unit='tesla' can serve every
    overlay in a script; 'oe' names no map axis and is refused. On main
    'tesla' was refused too."""
    ms, result = _detected(tmp_path)
    grid = plot.chi_map_grid(ms)
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match=r"field_unit='oe' cannot apply to a map.*Tesla"):
            plot.overlay_transitions(result, ax=ax, grid=grid, field_unit="oe")
        assert not ax.containers and not ax.lines  # refused before anything was drawn
        placed = []
        for unit in (None, "tesla"):
            artists = plot.overlay_transitions(result, ax=ax, grid=grid, field_unit=unit)
            # Each group is one errorbar container; lines[0] carries the markers.
            placed.append(
                np.concatenate(
                    [c.lines[0].get_xydata() for group in artists.values() for c in group]
                )
            )
        assert len(placed[0]) > 0
        np.testing.assert_array_equal(placed[0], placed[1])
    finally:
        plt.close(fig)


def test_overlay_bare_dataframe_loses_threshold_line(tmp_path):
    """A hand-filtered DataFrame carries no robust() record, so the legend
    title has no thresholds -- documented, and the reason .robust() beats a
    pandas filter for publication figures."""
    ms, result = _detected(tmp_path)
    grid = plot.chi_map_grid(ms)
    fig, ax = plt.subplots()
    plot.overlay_transitions(result.candidates, ax=ax, grid=grid)
    assert "significance" not in ax.get_legend().get_title().get_text()
    plt.close("all")

# Cross-sweep linking


def _linked(tmp_path, tol_k=0.6):
    """A small campaign whose cusp drifts, linked into chains."""
    from calamari.transitions import detect_transitions_all

    ms = _ac_tscan_measurements(
        tmp_path,
        [(100.0, 14.0), (2000.0, 13.7), (5000.0, 13.4), (10000.0, 13.1)],
    )
    return ms, detect_transitions_all(ms).link(t_tol_k=tol_k)


def _drawn_lines(ax):
    """Line2D artists that actually draw a line (errorbar's marker artists
    have linestyle 'None' and are not connectors)."""
    return [
        line
        for line in ax.get_lines()
        if str(line.get_linestyle()).lower() not in ("none", " ")
        and len(line.get_xdata()) >= 2
    ]


def test_overlay_connect_draws_one_line_per_multi_member_chain(tmp_path):
    """connect='chain_id' traces each chain across the map: one line per
    chain with two or more members, and none for the singletons."""
    ms, linked = _linked(tmp_path)
    grid = plot.chi_map_grid(ms)
    fig, ax = plt.subplots()
    before = len(_drawn_lines(ax))
    artists = plot.overlay_transitions(linked, ax=ax, grid=grid, connect="chain_id")
    expected = int((linked.chains["n"] >= 2).sum())
    assert expected >= 1
    assert len(_drawn_lines(ax)) - before == expected
    connectors = [a for key, group in artists.items() if key[0] == "connect" for a in group]
    assert len(connectors) == expected
    for line in connectors:
        # Family order (field on y for a family of T-scans), and under the
        # markers: the line is the claim, the markers are the measurement.
        assert np.all(np.diff(line.get_ydata()) >= 0)
        assert line.get_zorder() < 2
    plt.close("all")


def test_overlay_connect_ignored_in_1d_mode(tmp_path):
    """A 1-D overlay marks one sweep with vertical rules; a line joining
    other sweeps' candidates would have nothing to run along."""
    ms, linked = _linked(tmp_path)
    ax = plot.plot_chi_vs_t(ms[0])
    before = len(_drawn_lines(ax))
    artists = plot.overlay_transitions(
        linked.candidates[linked.candidates["H_T"] < 0.05], ax=ax, connect="chain_id"
    )
    assert not any(key[0] == "connect" for key in artists)
    # axvline artists are two-vertex lines, so count the connectors by key,
    # not by artist: what matters is that nothing extra was drawn.
    added = len(_drawn_lines(ax)) - before
    assert added == sum(len(group) for group in artists.values())
    plt.close("all")


def test_overlay_connect_refuses_unknown_column(tmp_path):
    ms, linked = _linked(tmp_path)
    grid = plot.chi_map_grid(ms)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="chain_id"):
        plot.overlay_transitions(linked, ax=ax, grid=grid, connect="chains")
    plt.close("all")


def test_overlay_connect_refuses_unknown_column_in_1d_mode(tmp_path):
    """Only the LINE is map-only. An unknown column is a caller error in
    either mode -- and its usual cause is a result that was never linked,
    so a 1-D overlay that accepted the argument in silence would hide
    exactly the mistake worth reporting."""
    ms, linked = _linked(tmp_path)
    ax = plot.plot_chi_vs_t(ms[0])
    with pytest.raises(ValueError, match="chain_id"):
        plot.overlay_transitions(linked.candidates, ax=ax, connect="chains")
    plt.close("all")


def _many_chains_frame(n_chains=15):
    """A candidates frame with more chains than the palette has colours."""
    return pd.DataFrame(
        {
            "T_K": np.linspace(6.0, 26.0, n_chains),
            "H_T": np.tile([0.01, 0.2, 1.0], n_chains // 3 + 1)[:n_chains],
            "T_err_K": 0.05,
            "H_err_T": np.nan,
            "method": "peak",
            "signal": "chi_real",
            "sign": "max",
            "swept_axis": "T",
            "chain_id": np.arange(n_chains),
        }
    )


def test_overlay_color_by_chain_id_does_not_explode_the_legend(tmp_path):
    """Past the palette length the colours repeat, so per-value entries
    would be both unreadable and wrong: they are dropped, and the shape
    encoding still gets its (deduplicated) entries."""
    ms, _ = _linked(tmp_path)
    frame = _many_chains_frame()
    grid = plot.chi_map_grid(ms)
    fig, ax = plt.subplots()
    plot.overlay_transitions(frame, ax=ax, grid=grid, color_by="chain_id")
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert labels == ["signal extremum"]  # the marker encoding only
    plt.close("all")


def test_overlay_few_colors_still_get_legend_entries(tmp_path):
    """The suppression is a size cut-off, not a rule about chains: a
    handful of chains still names each one."""
    ms, _ = _linked(tmp_path)
    frame = _many_chains_frame(n_chains=3)
    grid = plot.chi_map_grid(ms)
    fig, ax = plt.subplots()
    plot.overlay_transitions(frame, ax=ax, grid=grid, color_by="chain_id")
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert len(labels) == 3
    assert all("signal extremum" in label for label in labels)
    plt.close("all")

# Detection diagnostics: plot_detection_curve, plot_smoothing_scan,
# plot_scale_tracks

#: Kink family for the diagnostic helpers: two rising slopes meeting at a
#: softplus-rounded corner, the upper five times shallower. Invented numbers.
_KINK_FIELDS_T = (0.5, 1.0)

_KINK_WINDOWS_K = (1.0, 1.5, 2.0, 3.0)


def _kink_corner(h_tesla: float) -> float:
    return 9.6 - 0.6 * h_tesla


def _kink_file(path: Path, h_tesla: float, rng) -> Path:
    from calamari import simulate

    t = np.linspace(2.0, 22.0, 121)
    corner, rounding = _kink_corner(h_tesla), 0.25
    softplus = rounding * np.logaddexp(0.0, (t - corner) / rounding)
    chi = 4.0e-6 + 1.5e-7 * (t - corner) - 1.2e-7 * softplus
    return simulate.write_mpms3_file(
        path, app="SQUID AC", temperature_K=t, field_oe=np.full_like(t, h_tesla * 1.0e4),
        chi_real=chi + rng.normal(scale=1.5e-9, size=t.shape),
        ac_frequency_hz=np.full_like(t, 997.0),
    )


@pytest.fixture(scope="module")
def kink_sweeps(tmp_path_factory):
    from calamari.io import load

    root = tmp_path_factory.mktemp("kinks")
    rng = np.random.default_rng(20260928)
    return [load(_kink_file(root / f"kink_{h:g}T.dat", h, rng)) for h in _KINK_FIELDS_T]


@pytest.fixture(scope="module")
def kink_scan(kink_sweeps):
    from calamari.scales import scan_smoothing

    return scan_smoothing(
        kink_sweeps, windows=_KINK_WINDOWS_K, track_tol_k=0.5,
        methods=("peak", "derivative", "kink"), criterion="kink", t_window_k=(4.0, 18.0),
    )


@pytest.fixture(scope="module")
def field_sweep(tmp_path_factory):
    """A chi'(H) isotherm with a peak at mu0 H = 1.5 T (15000 Oe)."""
    from calamari import simulate
    from calamari.io import load

    h_oe = np.linspace(0.0, 30000.0, 121)
    chi = simulate.synthetic_chi_peak(
        h_oe, centre=15000.0, width=2000.0, height=1e-4, background=1e-3, noise=1e-7,
        rng=np.random.default_rng(7),
    )
    path = simulate.write_mpms3_file(
        tmp_path_factory.mktemp("iso") / "iso.dat", app="SQUID AC",
        temperature_K=np.full_like(h_oe, 3.0), field_oe=h_oe, chi_real=chi,
        ac_frequency_hz=np.full_like(h_oe, 997.0),
    )
    return load(path)


def _rules_and_markers(ax):
    """(rules, markers) drawn after the curve: they alternate, rule first."""
    marks = ax.lines[1:]
    return marks[0::2], marks[1::2]


def _render(ax) -> None:
    buffer = BytesIO()
    ax.figure.savefig(buffer, format="png")
    assert buffer.getbuffer().nbytes > 0


def test_plot_detection_curve_draws_the_stored_curve_and_every_candidate(kink_scan):
    """The curve is the result's own, and there is one rule and one registry
    marker per candidate row of that sweep and signal, on the curve."""
    from calamari.transitions import METHODS

    result = kink_scan.at(2.0)
    rows = result.candidates[result.candidates["sweep_id"] == 1]
    assert len(rows) > 1  # noise candidates too: the count is a real check
    ax = plot.plot_detection_curve(result, sweep_id=1, representation="d2ydx2", color="C2")
    try:
        curve = result.curve(1)
        x, y = _line_xy(ax.lines[0])
        np.testing.assert_array_equal(x, curve["x"])
        np.testing.assert_array_equal(y, curve["d2ydx2"])
        assert ax.lines[0].get_color() == "C2"
        rules, markers = _rules_and_markers(ax)
        assert len(rules) == len(markers) == len(rows)
        for rule, marker, (_, row) in zip(rules, markers, rows.iterrows(), strict=True):
            assert rule.get_xdata()[0] == pytest.approx(row["T_K"])
            assert rule.get_linestyle() == "--"
            mx, my = _line_xy(marker)
            assert mx[0] == pytest.approx(row["T_K"])
            assert my[0] == pytest.approx(np.interp(row["T_K"], curve["x"], curve["d2ydx2"]))
            assert marker.get_marker() == METHODS[row["method"]].marker
        assert ax.get_xlabel() == "Temperature (K)"
        assert ax.get_ylabel() == r"$\mathrm{d}^2\chi'/\mathrm{d}T^2$ (emu/(Oe K$^2$))"
        _render(ax)
    finally:
        plt.close(ax.figure)


def test_plot_detection_curve_labels_each_method_and_sign_once(kink_scan):
    result = kink_scan.at(2.0)
    rows = result.candidates[result.candidates["sweep_id"] == 0]
    ax = plot.plot_detection_curve(result, sweep_id=0, representation="dydx")
    try:
        _, markers = _rules_and_markers(ax)
        labels = [m.get_label() for m in markers if not m.get_label().startswith("_")]
        expected = {
            f"{plot_detection._overlay_label('method', m)} ({s})"
            for m, s in zip(rows["method"], rows["sign"], strict=True)
        }
        assert sorted(labels) == sorted(expected)  # each pair once, none repeated
        assert "curvature extremum (down)" in labels
        assert ax.get_ylabel() == r"$\mathrm{d}\chi'/\mathrm{d}T$ (emu/(Oe K))"
    finally:
        plt.close(ax.figure)


def test_plot_detection_curve_draws_the_views_candidates_only(kink_scan):
    """A robust() view draws what survived; candidates=False draws the curve alone."""
    kept = kink_scan.at(2.0).robust()
    n_kept = int((kept.candidates["sweep_id"] == 1).sum())
    assert n_kept == 1
    fig, (a, b) = plt.subplots(1, 2)
    try:
        plot.plot_detection_curve(kept, sweep_id=1, ax=a)
        rules, _ = _rules_and_markers(a)
        assert len(rules) == n_kept
        (row,) = kept.candidates[kept.candidates["sweep_id"] == 1].itertuples()
        assert (row.method, row.sign) == ("kink", "down")
        assert abs(row.T_K - _kink_corner(_KINK_FIELDS_T[1])) < 0.1
        plot.plot_detection_curve(kept, sweep_id=1, ax=b, candidates=False)
        assert len(b.lines) == 1
    finally:
        plt.close(fig)


def test_plot_detection_curve_composes_with_the_raw_sweep_on_one_scale(kink_sweeps, kink_scan):
    """The default plot_chi_vs_t composition: same label, and the smoothed
    curve lies on the data (emu/Oe, K), not orders of magnitude away."""
    fig, ax = plt.subplots()
    try:
        plot.plot_chi_vs_t(kink_sweeps[0], ax=ax, linestyle="none", marker=".")
        raw_label = ax.get_ylabel()
        plot.plot_detection_curve(kink_scan.at(2.0), sweep_id=0, ax=ax)
        assert ax.get_ylabel() == raw_label == r"$\chi'$ (emu/Oe)"
        t, chi = _line_xy(ax.lines[0])
        x, y = _line_xy(ax.lines[1])
        np.testing.assert_allclose(y, np.interp(x, t, chi), rtol=5e-3)
        _render(ax)
    finally:
        plt.close(fig)


def test_plot_detection_curve_refuses_a_converted_raw_sweep(kink_sweeps, kink_scan, sample):
    """chi per mole under the smoothed emu/Oe curve would be one axis, two
    scales: refused, naming the unit to draw the raw sweep in."""
    fig, ax = plt.subplots()
    try:
        plot.plot_chi_vs_t(kink_sweeps[0], ax=ax, units="molar", sample=sample)
        n_lines = len(ax.lines)
        with pytest.raises(ValueError, match=r"units='emu_per_oe'"):
            plot.plot_detection_curve(kink_scan.at(2.0), sweep_id=0, ax=ax)
        assert len(ax.lines) == n_lines  # nothing drawn before the refusal
        # A label the caller wrote is theirs, and is not second-guessed.
        ax.set_ylabel("susceptibility (arb.)")
        plot.plot_detection_curve(kink_scan.at(2.0), sweep_id=0, ax=ax)
    finally:
        plt.close(fig)


def test_plot_detection_curve_field_rules_land_on_the_tesla_axis(field_sweep):
    """An H-sweep's curve and candidates are both in mu0 H (Tesla), so the
    rule lands at H_T on the curve's own axis, beside plot_chi_vs_h drawn in
    Tesla -- and the Oe default, 10^4 away, is refused by name."""
    from calamari.transitions import detect_transitions

    result = detect_transitions(field_sweep, methods=("peak",)).robust()
    (row,) = result.candidates.itertuples()
    assert row.swept_axis == "H"
    assert row.H_T == pytest.approx(1.5, abs=0.05)
    fig, (tesla, oe) = plt.subplots(1, 2)
    try:
        plot.plot_chi_vs_h(field_sweep, ax=tesla, field_unit="tesla")
        plot.plot_detection_curve(result, ax=tesla)
        rules, markers = tesla.lines[2::2], tesla.lines[3::2]
        assert [r.get_xdata()[0] for r in rules] == pytest.approx([row.H_T])
        assert _line_xy(markers[0])[0][0] == pytest.approx(row.H_T)
        curve_x, _ = _line_xy(tesla.lines[1])
        assert curve_x.max() == pytest.approx(3.0)
        assert tesla.get_xlabel() == _TESLA_LABEL
        plot.plot_chi_vs_h(field_sweep, ax=oe)
        with pytest.raises(ValueError, match=r"field_unit='tesla'"):
            plot.plot_detection_curve(result, ax=oe)
    finally:
        plt.close(fig)


def test_plot_detection_curve_refusals_leak_no_figure(kink_scan):
    from calamari.transitions import find_transitions

    result = kink_scan.at(2.0)
    before = plt.get_fignums()
    with pytest.raises(ValueError, match=r"unknown representation='d3ydx3'.*'y_smooth', "
                                         r"'dydx', 'd2ydx2', 'dyT_dT'"):
        plot.plot_detection_curve(result, representation="d3ydx3")
    with pytest.raises(ValueError, match=r"no Fisher curve.*methods=\(\.\.\., 'fisher'\)"):
        plot.plot_detection_curve(result, representation="dyT_dT")
    with pytest.raises(ValueError, match=r"no sweep_id=7.*sweep ids \[0, 1\]"):
        plot.plot_detection_curve(result, sweep_id=7)
    with pytest.raises(ValueError, match=r"no curve for signal='chi_imag'"):
        plot.plot_detection_curve(result, signal="chi_imag")
    x = np.linspace(1.0, 10.0, 50)
    array_level = find_transitions(x, np.abs(x - 5.0), methods=("peak",))
    with pytest.raises(ValueError, match="array-level find_transitions result"):
        plot.plot_detection_curve(array_level)
    assert plt.get_fignums() == before


def test_plot_detection_curve_draws_fisher_when_it_ran(tmp_path):
    from calamari.transitions import detect_transitions

    result = detect_transitions(_toy_afm_file(tmp_path), methods=("peak", "fisher"))
    ax = plot.plot_detection_curve(result, representation="dyT_dT")
    try:
        np.testing.assert_array_equal(_line_xy(ax.lines[0])[1], result.curve()["dyT_dT"])
        assert ax.get_ylabel() == f"{plot_data._FISHER_SYMBOLS['ac']} (emu/Oe)"
        _render(ax)
    finally:
        plt.close(ax.figure)


def test_detection_labels_render_as_mathtext(ax):
    for signal in plot_detection._DETECTION_SIGNAL_UNITS:
        for representation in plot.DETECTION_REPRESENTATIONS:
            for axis in (*plot_detection._DETECTION_AXES, None):
                xlabel, ylabel = plot_detection._detection_labels(signal, representation, axis)
                ax.set_xlabel(xlabel or "")
                ax.set_ylabel(ylabel)
                ax.figure.canvas.draw()
    ax.set_ylabel("Smoothing window (K)")
    ax.figure.canvas.draw()


def test_plot_smoothing_scan_one_unscaled_line_per_window_in_colour_order(kink_scan):
    import matplotlib as mpl

    ax = plot.plot_smoothing_scan(kink_scan, sweep_id=1, cmap="viridis", linewidth=0.7)
    try:
        assert len(ax.lines) == len(_KINK_WINDOWS_K)
        assert [line.get_label() for line in ax.lines] == [
            "window 1 K", "window 1.5 K", "window 2 K", "window 3 K"
        ]
        for line, window in zip(ax.lines, _KINK_WINDOWS_K, strict=True):
            curve = kink_scan.at(window).curve(1)
            np.testing.assert_array_equal(_line_xy(line)[1], curve["d2ydx2"])  # not normalised
            assert line.get_linewidth() == 0.7
        viridis = mpl.colormaps["viridis"]
        assert mpl.colors.to_rgba(ax.lines[0].get_color()) == viridis(0.0)
        assert mpl.colors.to_rgba(ax.lines[-1].get_color()) == viridis(1.0)
        assert ax.get_ylabel() == r"$\mathrm{d}^2\chi'/\mathrm{d}T^2$ (emu/(Oe K$^2$))"
        _render(ax)
    finally:
        plt.close(ax.figure)


def test_plot_smoothing_scan_default_map_is_the_styles(kink_scan):
    import matplotlib as mpl

    with mpl.rc_context({"image.cmap": "cividis"}):
        ax = plot.plot_smoothing_scan(kink_scan, sweep_id=0, representation="y_smooth")
    try:
        assert mpl.colors.to_rgba(ax.lines[0].get_color()) == mpl.colormaps["cividis"](0.0)
        assert ax.get_ylabel() == r"$\chi'$ (emu/Oe)"
    finally:
        plt.close(ax.figure)


def test_plot_smoothing_scan_refusals_leak_no_figure(kink_scan):
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="unknown representation"):
        plot.plot_smoothing_scan(kink_scan, sweep_id=0, representation="curvature")
    with pytest.raises(ValueError, match=r"no Fisher curve.*Re-run scan_smoothing"):
        plot.plot_smoothing_scan(kink_scan, sweep_id=0, representation="dyT_dT")
    with pytest.raises(ValueError, match=r"no sweep_id=5.*sweep ids \[0, 1\]"):
        plot.plot_smoothing_scan(kink_scan, sweep_id=5)
    with pytest.raises(ValueError, match=r"no signal='moment_emu'.*\['chi_real'\]"):
        plot.plot_smoothing_scan(kink_scan, sweep_id=0, signal="moment_emu")
    for keyword in ("color", "label"):
        with pytest.raises(ValueError, match=f"{keyword}= is set per line"):
            plot.plot_smoothing_scan(kink_scan, sweep_id=0, **{keyword: "x"})
    assert plt.get_fignums() == before


def _marker_points(ax, hollow: bool) -> int:
    return sum(
        len(line.get_xdata())
        for line in ax.lines
        if line.get_linestyle() == "None"
        and (line.get_markerfacecolor() == "none") == hollow
    )


def test_plot_scale_tracks_plots_every_member_of_the_sweep(kink_scan):
    tracks = kink_scan.tracks[kink_scan.tracks["sweep_id"] == 1]
    members = kink_scan.members[kink_scan.members["track_id"].isin(tracks["track_id"])]
    signs = members.merge(tracks[["track_id", "sign_mode"]], on="track_id")
    n_flipped = int((signs["sign"] != signs["sign_mode"]).sum())
    ax = plot.plot_scale_tracks(kink_scan, sweep_id=1, show_errors=True)
    try:
        assert _marker_points(ax, hollow=False) + _marker_points(ax, hollow=True) == len(members)
        assert _marker_points(ax, hollow=True) == n_flipped
        # The planted kink: one curvature track standing through every window.
        (kink,) = tracks[
            (tracks["method"] == "kink")
            & ((tracks["T_K_median"] - _kink_corner(_KINK_FIELDS_T[1])).abs() < 0.1)
        ].itertuples()
        assert kink.persistence == 1.0
        assert list(ax.get_yticks()) == list(_KINK_WINDOWS_K)
        assert ax.get_xlabel() == "Temperature (K)"
        assert ax.get_ylabel() == "Smoothing window (K)"
        labels = {line.get_label() for line in ax.lines if not line.get_label().startswith("_")}
        assert "curvature extremum" in labels
        assert ax.containers  # show_errors=True drew the location bars
        _render(ax)
    finally:
        plt.close(ax.figure)


def _hand_scan(members: pd.DataFrame, tracks: pd.DataFrame, windows, axis="T"):
    """A SmoothingScanResult built by hand, so a sign flip is certain."""
    from calamari.scales import SmoothingScanResult

    return SmoothingScanResult(
        results={float(w): None for w in windows},
        members=members,
        tracks=tracks,
        params={"smooth_points": [{"sweep_id": 0, "signal": "chi_real"}]},
    )


def test_plot_scale_tracks_draws_a_sign_flip_hollow():
    """Three members of one kink track, 'down', 'down', 'up': the modal sign
    is 'down', so exactly the 'up' member is hollow, in its window's row."""
    windows = (1.0, 2.0, 3.0)
    members = pd.DataFrame(
        {
            "track_id": [0, 0, 0],
            "window": list(windows),
            "T_K": [9.0, 9.1, 9.3],
            "H_T": [1.0] * 3,
            "T_err_K": [0.1] * 3,
            "H_err_T": [np.nan] * 3,
            "sign": ["down", "down", "up"],
        }
    )
    tracks = pd.DataFrame(
        {
            "track_id": [0],
            "sweep_id": [0],
            "swept_axis": ["T"],
            "signal": ["chi_real"],
            "method": ["kink"],
            "sign_mode": ["down"],
        }
    )
    ax = plot.plot_scale_tracks(_hand_scan(members, tracks, windows), sweep_id=0)
    try:
        line, filled, hollow = ax.lines
        np.testing.assert_array_equal(_line_xy(line)[0], [9.0, 9.1, 9.3])
        np.testing.assert_array_equal(_line_xy(filled)[1], [1.0, 2.0])
        np.testing.assert_array_equal(_line_xy(hollow), [[9.3], [3.0]])
        assert hollow.get_markerfacecolor() == "none"
        assert filled.get_markerfacecolor() != "none"
        assert filled.get_marker() == hollow.get_marker() == "D"  # METHODS['kink']
        assert hollow.get_label() == "sign differs from its track"
        assert filled.get_label() == "curvature extremum"
    finally:
        plt.close(ax.figure)


def test_plot_scale_tracks_refusals(kink_scan):
    before = plt.get_fignums()
    with pytest.raises(ValueError, match=r"no sweep_id=9.*sweep ids \[0, 1\]"):
        plot.plot_scale_tracks(kink_scan, sweep_id=9)
    with pytest.raises(ValueError, match="sweep_id must be an integer"):
        plot.plot_scale_tracks(kink_scan, sweep_id="0")
    assert plt.get_fignums() == before
    fig, ax = plt.subplots()
    try:
        ax.set_xlabel(plot.axis_labels("field", "tesla"))
        with pytest.raises(ValueError, match="wrong scale"):
            plot.plot_scale_tracks(kink_scan, sweep_id=0, ax=ax)
    finally:
        plt.close(fig)

# plot_boundary


def _boundary_points(rng, *, n=8, swept="T"):
    """One point per sweep on an invented straight boundary."""
    held = np.linspace(0.0, 1.4, n) if swept == "T" else np.linspace(2.0, 9.0, n)
    loc = 14.0 - 2.0 * held if swept == "T" else 7.0 - 0.5 * held
    loc = loc + rng.normal(scale=0.05, size=n)
    cols = ("H_T", "T_K", "T_err_K") if swept == "T" else ("T_K", "H_T", "H_err_T")
    return pd.DataFrame({"sweep_id": range(n), cols[0]: held, cols[1]: loc, cols[2]: 0.05,
                         "criterion": "kink/down"})


def _quiet_boundary(*args, **kwargs):
    from calamari.boundary import fit_boundary

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fit_boundary(*args, **kwargs)


def test_plot_boundary_draws_points_curve_and_band_in_map_orientation():
    points = _boundary_points(np.random.default_rng(301))
    bf = _quiet_boundary(points, "linear", swept_axis="T", exclude=[2])
    ax = plot.plot_boundary(bf)
    try:
        curve = ax.lines[0]
        x, y = _line_xy(curve)
        # A T-swept fit: temperature on x, the held field on y, as draw_map.
        np.testing.assert_allclose(y, bf.band["H_T"])
        np.testing.assert_allclose(x, bf.band["model_K"])
        assert curve.get_label() == "linear model of kink/down locations"
        band = ax.collections[0]
        assert band.get_label() == "1σ band on the model"
        used, excluded = ax.containers
        ux, uy = _line_xy(used.lines[0])
        keep = bf.sweeps["status"] == "used"
        np.testing.assert_allclose(ux, bf.sweeps.loc[keep, "T_K"])
        np.testing.assert_allclose(uy, bf.sweeps.loc[keep, "H_T"])
        assert used.has_xerr and not used.has_yerr
        assert used.get_label() == "kink/down (7 sweeps)"
        assert used.lines[0].get_marker() == "D"  # METHODS['kink'].marker
        assert used.lines[0].get_markerfacecolor() != "none"
        # The excluded point is hollow, at its own place.
        assert excluded.get_label() == "excluded by caller"
        assert excluded.lines[0].get_markerfacecolor() == "none"
        np.testing.assert_allclose(_line_xy(excluded.lines[0])[1], [points.loc[2, "H_T"]])
        assert ax.get_xlabel() == plot.axis_labels("temperature", "k")
        assert ax.get_ylabel() == plot.axis_labels("field", "tesla")
        assert ax.get_legend() is None  # PL-1: the caller calls legend()
        ax.legend()
        buffer = BytesIO()
        ax.figure.savefig(buffer, format="png")
        assert buffer.getvalue()
    finally:
        plt.close(ax.figure)


def test_plot_boundary_h_swept_draws_vertical_errors_and_h_of_t(ax):
    points = _boundary_points(np.random.default_rng(302), swept="H")
    points["criterion"] = "derivative/rise"
    bf = _quiet_boundary(points, "linear", swept_axis="H")
    plot.plot_boundary(bf, ax=ax, band=False, color="k")
    x, y = _line_xy(ax.lines[0])
    np.testing.assert_allclose(x, bf.band["T_K"])
    np.testing.assert_allclose(y, bf.band["model_T"])
    assert "1σ band on the model" not in [c.get_label() for c in ax.collections]  # band=False
    (used,) = ax.containers
    assert used.has_yerr and not used.has_xerr
    assert used.lines[0].get_marker() == "^"  # METHODS['derivative'].marker
    assert matplotlib.colors.same_color(used.lines[0].get_color(), "k")


def test_plot_boundary_which_total_and_refusals():
    points = _boundary_points(np.random.default_rng(303))
    bf = _quiet_boundary(points, "linear", swept_axis="T", yerr_sys=0.2)
    ax = plot.plot_boundary(bf, which="total")
    try:
        np.testing.assert_allclose(_line_xy(ax.lines[0])[0], bf.total.band["model_K"])
        (used,) = ax.containers
        np.testing.assert_allclose(
            np.ptp(used[2][0].get_segments()[0][:, 0]), 2 * np.hypot(0.05, 0.2)
        )
    finally:
        plt.close(ax.figure)
    before = plt.get_fignums()
    plain = _quiet_boundary(points, "linear", swept_axis="T")
    with pytest.raises(ValueError, match=r"which='total'.*Pass yerr_sys="):
        plot.plot_boundary(plain, which="total")
    with pytest.raises(ValueError, match="which must be one of"):
        plot.plot_boundary(plain, which="sys")
    assert plt.get_fignums() == before


def test_plot_boundary_composes_onto_a_map_and_refuses_an_index_map(tmp_path):
    ms, result = _detected(tmp_path)
    chosen = result.select(swept_axis="T", method="peak", sign="max", t_window_k=(8.0, 18.0))
    bf = _quiet_boundary(chosen, "linear")
    mesh = plot.plot_chi_map(ms)
    try:
        plot.plot_boundary(bf, ax=mesh.axes)
        (used,) = mesh.axes.containers
        x, y = _line_xy(used.lines[0])
        np.testing.assert_allclose(x, chosen.candidates["T_K"])
        np.testing.assert_allclose(y, chosen.candidates["H_T"])
        assert mesh.axes.get_ylabel() == plot.axis_labels("field", "tesla")
    finally:
        plt.close("all")
    mesh = plot.plot_chi_map(ms, family_scale="index")
    try:
        with pytest.raises(ValueError, match="family_scale='index' map"):
            plot.plot_boundary(bf, ax=mesh.axes)
        assert not mesh.axes.containers
    finally:
        plt.close("all")


def test_plot_boundary_failed_fit_draws_points_and_no_curve(ax):
    points = _boundary_points(np.random.default_rng(304), n=2)
    bf = _quiet_boundary(points, "linear", swept_axis="T")
    assert not bf.fit.success
    plot.plot_boundary(bf, ax=ax)
    assert not ax.lines or all(line.get_label().startswith("_") for line in ax.lines)
    assert "1σ band on the model" not in [c.get_label() for c in ax.collections]
    (used,) = ax.containers
    assert used.get_label() == "kink/down (2 sweeps)"


@pytest.mark.parametrize("unit, scale", [("tesla", 1.0), ("oe", 10_000.0)])
def test_plot_transitions_physical_coordinates_and_errors(ax, unit, scale):
    frame = _field_candidates().assign(T_err_K=[np.nan, 0.2], H_err_T=[0.03, np.nan])
    original = frame.copy(deep=True)
    returned = plot.plot_transitions(frame, ax=ax, field_unit=unit)
    assert returned is ax
    assert not ax.images and not any(isinstance(c, QuadMesh) for c in ax.collections)
    container = ax.containers[0]
    line, _, bars = container.lines
    np.testing.assert_allclose(line.get_xdata(), [2, 5])
    np.testing.assert_allclose(line.get_ydata(), np.array([0.3, 0.1]) * scale)
    assert bars[0].get_segments()[0].size == 0
    np.testing.assert_allclose(bars[0].get_segments()[1],
                               [[4.8, 0.1 * scale], [5.2, 0.1 * scale]])
    np.testing.assert_allclose(bars[1].get_segments()[0],
                               [[2, 0.27 * scale], [2, 0.33 * scale]])
    assert bars[1].get_segments()[1].size == 0
    assert ax.get_xlabel() == plot.axis_labels("temperature", "k")
    assert ax.get_ylabel() == plot.axis_labels("field", unit)
    assert line.get_linestyle() == "None"
    pd.testing.assert_frame_equal(frame, original)


def test_plot_transitions_grouping_styles_and_no_errors(ax):
    frame = pd.concat([_field_candidates()] * 2, ignore_index=True)
    frame["method"] = ["peak", "derivative", "peak", "derivative"]
    frame["signal"] = ["chi_real", "chi_real", "chi_imag", "chi_imag"]
    plot.plot_transitions(
        frame, ax=ax, color_by="signal", markers={"peak": "s"},
        colors={"chi_real": "red", "chi_imag": "blue"}, markersize=9, capsize=4,
    )
    assert len(ax.containers) == 4
    lines = [c.lines[0] for c in ax.containers]
    assert [line.get_marker() for line in lines] == ["s", "s", "^", "^"]
    assert [line.get_markerfacecolor() for line in lines] == ["blue", "red"] * 2
    assert all(line.get_markersize() == 9 for line in lines)
    assert len(ax.get_legend().get_texts()) == 4
    assert all(line.get_linestyle() == "None" for line in lines)
    assert all(segment.size == 0 for c in ax.containers
               for bars in c.lines[2] for segment in bars.get_segments())


def test_plot_transitions_result_sequence_and_created_axes(tmp_path):
    _, result = _detected(tmp_path)
    ax = plot.plot_transitions([result, result])
    try:
        assert sum(len(c.lines[0].get_xdata()) for c in ax.containers) == 2 * len(result.candidates)
        assert "significance" in ax.get_legend().get_title().get_text()
        ax.figure.savefig(BytesIO(), format="png")
    finally:
        plt.close(ax.figure)


def test_plot_transitions_errors_and_legend_can_be_hidden(ax):
    frame = _field_candidates().assign(T_err_K=0.2, H_err_T=0.03)
    plot.plot_transitions(frame, ax=ax, show_errors=False, legend=False)
    assert not ax.containers[0].has_xerr
    assert not ax.containers[0].has_yerr
    assert ax.get_legend() is None


def test_plot_transitions_connections_require_named_group(ax):
    frame = pd.concat([_field_candidates()] * 2, ignore_index=True)
    frame["chain_id"] = [0, 1, 0, 1]
    frame["T_K"] = [4, 5, 2, 3]
    plot.plot_transitions(frame, ax=ax, connect="chain_id", show_errors=False)
    connections = [line for line in ax.lines if line.get_linestyle() != "None"]
    assert len(connections) == 2
    np.testing.assert_allclose(connections[0].get_xdata(), [2, 4])
    np.testing.assert_allclose(connections[1].get_xdata(), [3, 5])


def test_plot_transitions_empty_and_invalid_inputs(ax):
    empty = _field_candidates().iloc[:0]
    with pytest.raises(ValueError, match="field_unit must be"):
        plot.plot_transitions(empty, ax=ax, field_unit="gauss")
    with pytest.raises(ValueError, match="physical coordinate columns.*H_T"):
        plot.plot_transitions(empty.drop(columns="H_T"), ax=ax)
    with pytest.raises(ValueError, match="connect=.*not a candidates column"):
        plot.plot_transitions(_field_candidates(), ax=ax, connect="chain_id")
    assert not ax.lines
    assert plot.plot_transitions(empty, ax=ax) is ax
    assert not ax.containers


def test_plot_transitions_normalises_matplotlib_style_aliases(ax):
    plot.plot_transitions(
        _field_candidates(), ax=ax, mfc="red", mec="blue", ms=8, mew=2, ls="--", lw=3,
    )
    line = ax.containers[0].lines[0]
    assert line.get_markerfacecolor() == "red"
    assert line.get_markeredgecolor() == "blue"
    assert line.get_markersize() == 8
    assert line.get_markeredgewidth() == 2
    assert line.get_linewidth() == 3
    assert line.get_linestyle() == "None"  # connections still require connect=


@pytest.mark.parametrize("column", ["method", "frequency_hz"])
@pytest.mark.parametrize("missing", [np.nan, None, pd.NA])
def test_plot_transitions_retains_missing_encoding_groups(ax, column, missing):
    frame = _field_candidates()
    frame[column] = pd.Series([missing, "peak" if column == "method" else 997.0], dtype=object)
    options = (
        {"marker_by": column, "markers": {None: "s"}}
        if column == "method"
        else {"color_by": column, "colors": {None: "red"}}
    )
    plot.plot_transitions(frame, ax=ax, show_errors=False, **options)
    lines = [c.lines[0] for c in ax.containers]
    assert sum(len(line.get_xdata()) for line in lines) == len(frame)
    missing_line = next(line for line in lines if 2.0 in line.get_xdata())
    if column == "method":
        assert missing_line.get_marker() == "s"
    else:
        assert missing_line.get_markerfacecolor() == "red"
    assert any("missing" in label.get_text() for label in ax.get_legend().get_texts())


def test_plot_transitions_all_missing_frequency_with_connections(ax):
    frame = _field_candidates().assign(frequency_hz=np.nan, chain_id=0)
    plot.plot_transitions(frame, ax=ax, color_by="frequency_hz", connect="chain_id")
    assert len(ax.containers) == 1
    assert len(ax.containers[0].lines[0].get_xdata()) == 2
    connections = [line for line in ax.lines if line.get_linestyle() != "None"]
    assert len(connections) == 1
    assert connections[0].get_color() == ax.containers[0].lines[0].get_markerfacecolor()


def test_overlay_missing_encoding_groups(ax):
    frame = _field_candidates().assign(frequency_hz=[np.nan, 997.0])
    artists = plot.overlay_transitions(frame, ax=ax, field_unit="tesla", color_by="frequency_hz")
    assert ("peak", None) in artists
    assert sum(len(group) for group in artists.values()) == 2


def test_plot_transitions_unifies_null_group_keys(ax):
    frame = pd.concat([_field_candidates()] * 2, ignore_index=True)
    frame["frequency_hz"] = pd.Series([None, np.nan, pd.NA, None], dtype=object)
    plot.plot_transitions(frame, ax=ax, color_by="frequency_hz", colors={np.nan: "red"})
    assert len(ax.containers) == 1
    line = ax.containers[0].lines[0]
    assert len(line.get_xdata()) == len(frame)
    assert line.get_markerfacecolor() == "red"
