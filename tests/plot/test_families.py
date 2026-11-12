"""Measurement families, display offsets and screening."""

from __future__ import annotations

import warnings
from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import QuadMesh

from calamari import plot
from calamari.constants import OE_PER_TESLA
from calamari.measurement import Measurement

from ._helpers import (
    _TESLA_LABEL,
    _line_xy,
    _measurement,
)

# plot_family and the per-plot family aliases


def _iso_chi_sweep(t_k: float, chi0: float = 1.0) -> Measurement:
    """One isothermal chi(H) sweep at t_k."""
    h = np.linspace(0.0, 5000.0, 11)
    return _measurement(
        **{
            "Temperature (K)": np.full(11, t_k),
            "Magnetic Field (Oe)": h,
            "AC X' (emu/Oe)": chi0 / (1.0 + (h / 2000.0) ** 2),
        }
    )


def _fixed_field_mt(h_oe: float) -> Measurement:
    """One M(T) sweep at fixed field h_oe."""
    t = np.linspace(2.0, 100.0, 11)
    return _measurement(
        **{
            "Temperature (K)": t,
            "Magnetic Field (Oe)": np.full(11, h_oe),
            "Moment (emu)": 1e-3 / (t + 5.0),
        }
    )


def test_family_chi_vs_h_labels_and_sorting(ax):
    sweeps = [_iso_chi_sweep(10.0), _iso_chi_sweep(2.0), _iso_chi_sweep(5.0)]
    plot.plot_chi_vs_h_family(sweeps, ax=ax)
    assert [line.get_label() for line in ax.lines] == ["T = 2 K", "T = 5 K", "T = 10 K"]
    assert ax.get_xlabel() == "Field (Oe)"


def test_family_mt_labels_field_in_tesla(ax):
    sweeps = [_fixed_field_mt(50000.0), _fixed_field_mt(1000.0)]
    plot.plot_mt_family(sweeps, ax=ax)
    assert [line.get_label() for line in ax.lines] == ["H = 0.1 T", "H = 5 T"]


def test_family_label_by_frequency(ax):
    def sweep(freq):
        t = np.linspace(2.0, 30.0, 8)
        return _measurement(
            **{
                "Temperature (K)": t,
                "AC X' (emu/Oe)": 1.0 / t,
                "AC Frequency (Hz)": np.full(8, freq),
            }
        )

    plot.plot_chi_vs_t_family([sweep(997.0), sweep(97.0)], ax=ax, label_by="frequency")
    assert [line.get_label() for line in ax.lines] == ["f = 97 Hz", "f = 997 Hz"]


def test_family_accepts_paths(tmp_path, ax):
    from calamari import simulate

    for t_k in (2.0, 5.0):
        h = np.linspace(0.0, 2000.0, 6)
        simulate.write_mpms3_file(
            tmp_path / f"sweep_{t_k:g}K.dat",
            app="SQUID VSM",
            temperature_K=np.full(6, t_k),
            field_oe=h,
            vsm_moment_emu=1e-3 * h / 2000.0,
        )
    paths = sorted(tmp_path.glob("*.dat"))
    plot.plot_mh_family(paths, ax=ax)
    assert [line.get_label() for line in ax.lines] == ["T = 2 K", "T = 5 K"]


def test_family_and_map_accept_a_catalog(tmp_path):
    """A Catalog iterates as its files' absolute paths, so the family and
    map helpers -- which already load paths -- take one directly, with no
    intermediate list comprehension."""
    from calamari import build_catalog, simulate

    for h_oe in (1000.0, 2000.0):
        t = np.linspace(2.0, 8.0, 7)
        simulate.write_mpms3_file(
            tmp_path / f"chiT_{h_oe:g}Oe.dat",
            app="SQUID AC",
            temperature_K=t,
            field_oe=np.full(7, h_oe),
            chi_real=1e-3 / t,
            ac_frequency_hz=np.full(7, 997.0),
            ac_drive_oe=np.full(7, 3.0),
        )
    cat = build_catalog(tmp_path).sort_by("H_min_Oe")

    ax_family = plot.plot_chi_vs_t_family(cat)
    assert [line.get_label() for line in ax_family.lines] == ["H = 0.1 T", "H = 0.2 T"]
    plt.close(ax_family.figure)

    mesh = plot.plot_chi_map(cat)
    assert isinstance(mesh, QuadMesh)
    plt.close(mesh.axes.figure)


def test_family_refuses_unfixed_parameter_by_name(ax):
    drifting = _iso_chi_sweep(5.0)
    drifting.data["Temperature (K)"] = np.linspace(4.0, 7.0, 11)  # spans 3 K
    # Under 'auto' the drift is caught even earlier: the sweep classifies
    # as 'mixed', which names no fixed parameter at all.
    with pytest.raises(ValueError, match="scan_kind is 'mixed'"):
        plot.plot_chi_vs_h_family([_iso_chi_sweep(2.0), drifting], ax=ax)
    # Forcing the label parameter reaches the span check, which names the
    # file and the drift.
    with pytest.raises(ValueError, match="temperature is not fixed"):
        plot.plot_chi_vs_h_family(
            [_iso_chi_sweep(2.0), drifting], ax=ax, label_by="temperature"
        )
    assert len(ax.lines) == 0  # refused before drawing anything


def test_family_labels_escape_hatch(ax):
    drifting = _iso_chi_sweep(5.0)
    drifting.data["Temperature (K)"] = np.linspace(4.0, 7.0, 11)
    plot.plot_chi_vs_h_family(
        [_iso_chi_sweep(2.0), drifting], ax=ax, labels=["cooldown 1", "cooldown 2"]
    )
    assert [line.get_label() for line in ax.lines] == ["cooldown 1", "cooldown 2"]


def test_family_refuses_indistinguishable_curves(ax):
    with pytest.raises(ValueError, match="do not differ"):
        plot.plot_chi_vs_h_family([_iso_chi_sweep(5.0), _iso_chi_sweep(5.0)], ax=ax)


def test_family_refuses_mixed_scan_kinds(ax):
    with pytest.raises(ValueError, match="disagree about what is fixed"):
        plot.plot_family(
            [_iso_chi_sweep(5.0), _fixed_field_mt(1000.0)], plot.plot_mh, ax=ax
        )


def test_family_cmap_colours_monotonic(ax):
    sweeps = [_iso_chi_sweep(10.0), _iso_chi_sweep(2.0), _iso_chi_sweep(5.0)]
    plot.plot_chi_vs_h_family(sweeps, ax=ax, cmap="viridis")
    colours = [line.get_color() for line in ax.lines]
    assert len(set(map(tuple, colours))) == 3  # distinct, sequential colours


def test_family_cmap_and_color_conflict(ax):
    with pytest.raises(ValueError, match="not both"):
        plot.plot_chi_vs_h_family(
            [_iso_chi_sweep(2.0), _iso_chi_sweep(5.0)], ax=ax, cmap="viridis", color="C0"
        )


def test_family_sort_false_keeps_given_order(ax):
    plot.plot_chi_vs_h_family(
        [_iso_chi_sweep(10.0), _iso_chi_sweep(2.0)], ax=ax, sort=False
    )
    assert [line.get_label() for line in ax.lines] == ["T = 10 K", "T = 2 K"]


def test_family_kwargs_forward_to_plot(ax, sample):
    plot.plot_chi_vs_h_family(
        [_iso_chi_sweep(2.0), _iso_chi_sweep(5.0)], ax=ax, units="molar", sample=sample
    )
    assert ax.get_ylabel() == r"$\chi'$ (emu/(mol Oe))"


def test_family_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        plot.plot_family([], plot.plot_mh)


def test_family_composes_with_frequency_split(ax):
    """A two-frequency member of a T-labelled family gets both suffixes."""
    h = np.linspace(0.0, 2000.0, 6)
    two_freq = _measurement(
        **{
            "Temperature (K)": np.full(12, 5.0),
            "Magnetic Field (Oe)": np.repeat(h, 2),
            "AC X' (emu/Oe)": np.tile([1.0, 2.0], 6),
            "AC Frequency (Hz)": np.tile([97.0, 997.0], 6),
        }
    )
    plot.plot_chi_vs_h_family([_iso_chi_sweep(2.0), two_freq], ax=ax)
    labels = [line.get_label() for line in ax.lines]
    assert labels == ["T = 2 K", "T = 5 K f = 97 Hz", "T = 5 K f = 997 Hz"]


def test_family_labels_length_mismatch_raises(ax):
    with pytest.raises(ValueError, match="one-to-one"):
        plot.plot_chi_vs_h_family([_iso_chi_sweep(2.0)], ax=ax, labels=["a", "b"])

# plot_family waterfalls: offset= / baseline=

_WF_H = np.linspace(0.0, 5000.0, 11)  # 500 Oe steps: x0 = 1250 lies between points


def _drifting_iso(t_k: float, **extra) -> Measurement:
    """A chi(H) isotherm whose background rises with temperature, so every
    member of a family has a different baseline under every mode."""
    chi = 0.1 * t_k + 1.0 / (1.0 + (_WF_H / 2000.0) ** 2) - 0.02 * t_k * _WF_H / 5000.0
    return _measurement(
        **{
            "Temperature (K)": np.full(len(_WF_H), t_k),
            "Magnetic Field (Oe)": _WF_H,
            "AC X' (emu/Oe)": chi,
            **extra,
        }
    )


def _expected_baseline(x, y, baseline):
    """The baseline computed independently of calamari, on a plain sweep."""
    if baseline is None:
        return 0.0
    if baseline == "start":
        return y[np.argmin(x)]
    if baseline == "min":
        return y.min()
    if isinstance(baseline, tuple):
        lo, hi = baseline
        return y[(x >= lo) & (x <= hi)].mean()
    return np.interp(baseline, x, y)


@pytest.mark.parametrize(
    ("baseline", "offset"),
    [(None, 0.25)]
    + [
        (baseline, offset)
        for baseline in ("start", "min", 1250.0, (2000.0, 3000.0))
        for offset in (None, 0.25)
    ],
)
def test_family_waterfall_draws_y_minus_baseline_plus_offset(ax, baseline, offset):
    temps = [2.0, 5.0, 10.0]
    sweeps = [_drifting_iso(t) for t in temps]
    plot.plot_chi_vs_h_family(sweeps, ax=ax, baseline=baseline, offset=offset)

    assert len(ax.lines) == 3
    step = 0.0 if offset is None else offset
    for i, (line, m) in enumerate(zip(ax.lines, sweeps)):
        raw = m.chi_real.to_numpy()
        x, y = _line_xy(line)
        np.testing.assert_allclose(x, _WF_H)  # x untouched
        b = _expected_baseline(_WF_H, raw, baseline)
        np.testing.assert_allclose(y, raw - b + i * step, rtol=0, atol=1e-12)
    suffix = (" (baseline-subtracted)" if baseline is not None else "") + (
        " (offset for clarity)" if offset else ""
    )
    assert ax.get_ylabel() == r"$\chi'$ (emu/Oe)" + suffix


def test_family_waterfall_is_display_only(ax):
    """The measurements and the resolution layer still give measured values."""
    m = _drifting_iso(5.0)
    before = m.data.copy()
    plot.plot_chi_vs_h_family([_drifting_iso(2.0), m], ax=ax, baseline="min", offset=1.0)
    pd.testing.assert_frame_equal(m.data, before)
    (series,) = plot.chi_series(m, x="field")
    np.testing.assert_allclose(series.y, m.chi_real.to_numpy())


def test_family_waterfall_stacks_in_ascending_fixed_value(ax):
    """Given out of order, the step still follows temperature, not input
    position; with labels= (no sort) it follows the given order."""
    sweeps = [_drifting_iso(10.0), _drifting_iso(2.0), _drifting_iso(5.0)]
    plot.plot_chi_vs_h_family(sweeps, ax=ax, baseline="start", offset=1.0)
    assert [line.get_label() for line in ax.lines] == ["T = 2 K", "T = 5 K", "T = 10 K"]
    assert [_line_xy(line)[1][0] for line in ax.lines] == pytest.approx([0.0, 1.0, 2.0])

    fig, ax2 = plt.subplots()
    try:
        plot.plot_chi_vs_h_family(
            sweeps, ax=ax2, labels=["a", "b", "c"], baseline="start", offset=1.0
        )
        assert [line.get_label() for line in ax2.lines] == ["a", "b", "c"]
        assert [_line_xy(line)[1][0] for line in ax2.lines] == pytest.approx([0.0, 1.0, 2.0])
    finally:
        plt.close(fig)


def test_family_waterfall_error_bars_unchanged(ax):
    """The bars are the points' own: same half-length, centred on the
    shifted value (the baseline is treated as exact)."""
    err = np.linspace(0.01, 0.05, len(_WF_H))
    sweeps = [
        _drifting_iso(t, **{"AC X' Std Err. (emu/Oe)": err}) for t in (2.0, 5.0)
    ]
    plot.plot_chi_vs_h_family(sweeps, ax=ax, baseline="start", offset=0.5)
    assert len(ax.containers) == 2
    for i, (container, m) in enumerate(zip(ax.containers, sweeps)):
        raw = m.chi_real.to_numpy()
        shifted = raw - raw[0] + 0.5 * i
        data_line, _, (bars,) = container.lines
        np.testing.assert_allclose(_line_xy(data_line)[1], shifted, atol=1e-12)
        segments = np.asarray(bars.get_segments())
        lows, highs = segments[:, 0, 1], segments[:, 1, 1]
        np.testing.assert_allclose((highs - lows) / 2.0, err, atol=1e-12)
        np.testing.assert_allclose((highs + lows) / 2.0, shifted, atol=1e-12)


def test_family_waterfall_with_cmap(ax):
    sweeps = [_drifting_iso(t) for t in (2.0, 5.0, 10.0)]
    plot.plot_chi_vs_h_family(sweeps, ax=ax, cmap="viridis", baseline="start", offset=1.0)
    colours = [tuple(line.get_color()) for line in ax.lines]
    assert len(set(colours)) == 3
    assert colours[0] == pytest.approx(tuple(matplotlib.colormaps["viridis"](0.0)))
    assert [_line_xy(line)[1][0] for line in ax.lines] == pytest.approx([0.0, 1.0, 2.0])


def test_family_waterfall_baseline_in_plotted_units(ax, sample):
    """The baseline and the step are in the unit drawn, after units=/sample=."""
    sweeps = [_drifting_iso(2.0), _drifting_iso(5.0)]
    plot.plot_chi_vs_h_family(
        sweeps, ax=ax, units="molar", sample=sample, baseline=(2000.0, 3000.0), offset=3.0
    )
    for i, (line, m) in enumerate(zip(ax.lines, sweeps)):
        molar = sample.chi_to_chi_molar(m.chi_real.to_numpy())
        b = _expected_baseline(_WF_H, molar, (2000.0, 3000.0))
        np.testing.assert_allclose(_line_xy(line)[1], molar - b + 3.0 * i, atol=1e-9)


def test_family_waterfall_both_components_get_their_own_baseline(ax):
    """chi' and chi'' each start at zero (their own 'start' baseline), and
    one measurement's pair shares one offset."""
    sweeps = [
        _drifting_iso(t, **{"AC X'' (emu/Oe)": 0.3 + 0.01 * t * _WF_H / 5000.0})
        for t in (2.0, 5.0)
    ]
    plot.plot_chi_vs_h_family(sweeps, ax=ax, component="both", baseline="start", offset=1.0)
    starts = [_line_xy(line)[1][0] for line in ax.lines]
    assert starts == pytest.approx([0.0, 0.0, 1.0, 1.0])
    imag = sweeps[1].chi_imag.to_numpy()
    np.testing.assert_allclose(_line_xy(ax.lines[3])[1], imag - imag[0] + 1.0, atol=1e-12)
    assert ax.get_ylabel() == r"$\chi$ (emu/Oe) (baseline-subtracted) (offset for clarity)"


def _mh_loop(t_k: float) -> Measurement:
    """A virgin + down + up loop whose moment carries a T-dependent offset."""
    field = np.concatenate(
        [np.linspace(0.0, 1000.0, 5), np.linspace(750.0, -1000.0, 8), np.linspace(-750.0, 1000.0, 8)]
    )
    branch = np.concatenate([np.zeros(5), np.ones(8), -np.ones(8)])  # hysteresis
    moment = 1e-3 * (np.tanh(field / 400.0) + 0.2 * branch) + 1e-4 * t_k
    return _measurement(
        **{
            "Temperature (K)": np.full(len(field), t_k),
            "Magnetic Field (Oe)": field,
            "Moment (emu)": moment,
        }
    )


@pytest.mark.parametrize("baseline", ["start", "min", 300.0, (-500.0, 500.0)])
def test_family_waterfall_pools_one_baseline_across_branches(ax, baseline):
    """Every branch of one loop moves by the same amount, and that amount
    does not depend on split_branches -- a display choice."""
    loops = [_mh_loop(2.0), _mh_loop(5.0)]
    # Virgin, down and up branches; each shares its turning point with the next.
    segments = [slice(0, 5), slice(4, 13), slice(12, 21)]
    plot.plot_mh_family(loops, ax=ax, split_branches=True, baseline=baseline, offset=5e-3)
    assert len(ax.lines) == 6  # three branches per loop
    fig, ax_whole = plt.subplots()
    try:
        plot.plot_mh_family(loops, ax=ax_whole, baseline=baseline, offset=5e-3)
        for i, m in enumerate(loops):
            raw, field = m.moment_emu.to_numpy(), m.field_oe.to_numpy()
            # The pooled baseline, computed from the whole loop, each point once.
            if baseline == "start":
                b = raw[field == field.min()].mean()
            elif baseline == "min":
                b = raw.min()
            elif baseline == 300.0:
                b = np.mean([
                    np.interp(300.0, np.sort(field[s]), raw[s][np.argsort(field[s])])
                    for s in segments
                ])
            else:
                b = raw[(field >= -500.0) & (field <= 500.0)].mean()
            shift = 5e-3 * i - b
            np.testing.assert_allclose(_line_xy(ax_whole.lines[i])[1], raw + shift, atol=1e-15)
            for line, s in zip(ax.lines[3 * i : 3 * i + 3], segments):
                np.testing.assert_allclose(_line_xy(line)[1], raw[s] + shift, atol=1e-15)
    finally:
        plt.close(fig)


def test_family_waterfall_aliases_forward(ax):
    plot.plot_mt_family(
        [_fixed_field_mt(1000.0), _fixed_field_mt(50000.0)], ax=ax, baseline="min", offset=1e-4
    )
    ys = [_line_xy(line)[1] for line in ax.lines]
    assert [y.min() for y in ys] == pytest.approx([0.0, 1e-4])

    def sweep(freq):
        t = np.linspace(2.0, 30.0, 8)
        return _measurement(
            **{
                "Temperature (K)": t,
                "AC X' (emu/Oe)": 1.0 / t,
                "AC Frequency (Hz)": np.full(8, freq),
            }
        )

    fig, ax2 = plt.subplots()
    try:
        plot.plot_chi_vs_t_family(
            [sweep(997.0), sweep(97.0)], ax=ax2, label_by="frequency", offset=0.1
        )
        np.testing.assert_allclose(
            _line_xy(ax2.lines[1])[1] - _line_xy(ax2.lines[0])[1], 0.1
        )
    finally:
        plt.close(fig)


def test_family_waterfall_suffix_only_when_something_is_offset(ax):
    plot.plot_chi_vs_h_family([_drifting_iso(2.0), _drifting_iso(5.0)], ax=ax, offset=0.0)
    assert ax.get_ylabel() == r"$\chi'$ (emu/Oe)"
    fig, ax2 = plt.subplots()
    try:
        plot.plot_chi_vs_h_family([_drifting_iso(2.0)], ax=ax2, offset=1.0)
        assert ax2.get_ylabel() == r"$\chi'$ (emu/Oe)"
    finally:
        plt.close(fig)


def test_family_waterfall_refuses_foreign_plot_callable(ax):
    def my_plot(m, ax=None, label=None, **style):
        return plot.plot_chi_vs_h(m, ax=ax, label=label, **style)

    for kwargs in ({"offset": 1.0}, {"baseline": "start"}):
        with pytest.raises(ValueError, match="calamari's own plot helpers.*my_plot"):
            plot.plot_family([_drifting_iso(2.0), _drifting_iso(5.0)], my_plot, ax=ax, **kwargs)
    assert len(ax.lines) == 0
    # Without offset/baseline a foreign callable is as welcome as ever.
    plot.plot_family([_drifting_iso(2.0), _drifting_iso(5.0)], my_plot, ax=ax)
    assert len(ax.lines) == 2


def test_family_waterfall_refuses_x0_off_the_curve(ax):
    with pytest.raises(ValueError, match=r"x0 = 6000 lies outside the curve.*0 to 5000"):
        plot.plot_chi_vs_h_family([_drifting_iso(2.0), _drifting_iso(5.0)], ax=ax, baseline=6000.0)
    assert len(ax.lines) == 0  # the failing curve drew nothing


def test_family_waterfall_refuses_empty_window(ax):
    # 1100-1400 Oe falls between the 1000 and 1500 Oe points.
    with pytest.raises(ValueError, match=r"window \(1100, 1400\) contains no plotted point"):
        plot.plot_chi_vs_h_family(
            [_drifting_iso(2.0), _drifting_iso(5.0)], ax=ax, baseline=(1100.0, 1400.0)
        )
    assert len(ax.lines) == 0


def test_family_forwards_field_unit(ax):
    sweeps = [_iso_chi_sweep(5.0), _iso_chi_sweep(2.0)]
    plot.plot_chi_vs_h_family(sweeps, ax=ax, field_unit="tesla")
    assert [line.get_label() for line in ax.lines] == ["T = 2 K", "T = 5 K"]
    for line in ax.lines:
        np.testing.assert_allclose(_line_xy(line)[0], np.linspace(0.0, 0.5, 11))
    assert ax.get_xlabel() == _TESLA_LABEL

    fig, ax2 = plt.subplots()
    try:
        plot.plot_mh_family(
            [_mh_loop(2.0), _mh_loop(5.0)], ax=ax2, field_unit="tesla", split_branches=True
        )
        assert len(ax2.lines) == 6
        np.testing.assert_allclose(
            _line_xy(ax2.lines[0])[0], _mh_loop(2.0).field_tesla.to_numpy()[:5]
        )
        assert ax2.get_xlabel() == _TESLA_LABEL
    finally:
        plt.close(fig)


def test_family_temperature_helpers_refuse_field_unit(ax):
    for family, sweeps in (
        (plot.plot_mt_family, [_fixed_field_mt(1000.0), _fixed_field_mt(50000.0)]),
        (plot.plot_chi_vs_t_family, [_fixed_field_mt(1000.0), _fixed_field_mt(50000.0)]),
    ):
        with pytest.raises(ValueError, match="plots against temperature"):
            family(sweeps, ax=ax, field_unit="tesla")
    assert len(ax.lines) == 0


@pytest.mark.parametrize(
    ("oe_baseline", "tesla_baseline"),
    [(1250.0, 0.125), ((2000.0, 3000.0), (0.2, 0.3))],
)
def test_family_waterfall_baseline_in_tesla(ax, oe_baseline, tesla_baseline):
    """x0 and windows are read in the plotted x unit, so a Tesla family takes
    them in Tesla and draws the same shifted curves as the Oe family."""
    sweeps = [_drifting_iso(2.0), _drifting_iso(5.0)]
    plot.plot_chi_vs_h_family(
        sweeps, ax=ax, field_unit="tesla", baseline=tesla_baseline, offset=0.25
    )
    fig, ax_oe = plt.subplots()
    try:
        plot.plot_chi_vs_h_family(sweeps, ax=ax_oe, baseline=oe_baseline, offset=0.25)
        for tesla_line, oe_line in zip(ax.lines, ax_oe.lines):
            np.testing.assert_allclose(
                _line_xy(tesla_line)[1], _line_xy(oe_line)[1], rtol=0, atol=1e-12
            )
            np.testing.assert_allclose(_line_xy(tesla_line)[0], _WF_H / OE_PER_TESLA)
    finally:
        plt.close(fig)


def test_family_waterfall_refuses_oe_x0_on_a_tesla_axis(ax):
    """An x0 left in Oe on a Tesla family is off the curve, and the refusal
    quotes the curve's span in T -- the unit it must be given in."""
    with pytest.raises(
        ValueError,
        match=r"x0 = 20000 lies outside the curve.*0 to 0\.5 on the x-axis, "
        + r"Field, \$\\mu_0 H\$ \(T\)",
    ):
        plot.plot_chi_vs_h_family(
            [_drifting_iso(2.0), _drifting_iso(5.0)], ax=ax, field_unit="tesla",
            baseline=20000.0,
        )
    assert len(ax.lines) == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"baseline": "median"}, "not a known mode"),
        ({"baseline": True}, "baseline must be"),
        ({"baseline": np.nan}, "x0 must be finite"),
        ({"baseline": (3000.0, 1000.0)}, "x_lo < x_hi"),
        ({"baseline": (1000.0, 2000.0, 3000.0)}, "baseline must be"),
        ({"offset": np.inf}, "offset must be"),
        ({"offset": "1"}, "offset must be"),
    ],
)
def test_family_waterfall_refuses_malformed_values(ax, kwargs, match):
    with pytest.raises(ValueError, match=match):
        plot.plot_chi_vs_h_family([_drifting_iso(2.0), _drifting_iso(5.0)], ax=ax, **kwargs)
    assert len(ax.lines) == 0


def test_family_waterfall_renders_to_png(ax):
    """The suffixed y-label must still be valid mathtext at render time."""
    sweeps = [
        _drifting_iso(t, **{"AC X'' (emu/Oe)": np.full(len(_WF_H), 0.1 * t)})
        for t in (2.0, 5.0, 10.0)
    ]
    plot.plot_chi_vs_h_family(
        sweeps, ax=ax, component="both", imag_scale=10.0, cmap="viridis",
        baseline=(0.0, 1000.0), offset=0.5,
    )
    ax.legend()
    buffer = BytesIO()
    ax.figure.savefig(buffer, format="png")
    assert buffer.getbuffer().nbytes > 0

# plot_family glitch screening: outliers= / outlier_threshold=

#: The isotherms below: 0-4 T in 0.1 T steps, a 1e-4 chi' feature at 1.5 T
#: on a 1e-3 background, and -- on the glitched ones -- one reading 3e-4 low
#: at row 30 (3 T), on the flat tail. Under baseline='min' that one reading
#: would lift every clean point by three times the feature's height.
_GL_H = np.linspace(0.0, 40000.0, 41)

_GL_ROW = 30

_GL_DEPTH = 3e-4

_GL_FEATURE = 1e-4


def _gl_load(path):
    from calamari.io import load as _load

    return _load(path)


def _glitch_iso_file(path, t_k, *, glitch=True, extra_columns=None):
    """One chi'(H) isotherm at t_k, glitched at row 30 unless glitch=False.

    Seeded per temperature, so each file's scatter -- and the glitch's
    score -- is fixed whichever tests ran first.
    """
    from calamari import simulate

    rng = np.random.default_rng(20260928 + int(round(t_k * 10)))
    chi = simulate.synthetic_chi_peak(
        _GL_H, centre=15000.0, width=2000.0, height=_GL_FEATURE,
        background=1e-3 + 1e-5 * t_k, noise=1e-7, rng=rng,
    )
    if glitch:
        chi[_GL_ROW] -= _GL_DEPTH
    return simulate.write_mpms3_file(
        path, app="SQUID AC", temperature_K=np.full_like(_GL_H, t_k), field_oe=_GL_H,
        chi_real=chi, ac_frequency_hz=np.full_like(_GL_H, 997.0),
        extra_columns=extra_columns,
    )


def _glitch_family(tmp_path):
    """[glitched 5 K, clean 2 K]: deliberately not in draw order."""
    return [
        _gl_load(_glitch_iso_file(tmp_path / "iso_5K.dat", 5.0)),
        _gl_load(_glitch_iso_file(tmp_path / "iso_2K.dat", 2.0, glitch=False)),
    ]


def _recorded(call):
    """Run ``call`` and return the warnings it raised."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        call()
    return record


def test_family_glitch_premise(tmp_path):
    """quality() flags exactly the planted point, far above the line, and
    nothing on the clean isotherm."""
    glitched, clean = _glitch_family(tmp_path)
    flagged = glitched.quality().flagged
    assert list(flagged.index) == [_GL_ROW]
    assert flagged["continuity_score"].iloc[0] > 100.0
    assert clean.quality().flagged.empty


def test_family_glitch_drop_keeps_it_out_of_the_baseline(tmp_path, ax):
    """The regression: baseline='min' took the glitch as the curve's zero,
    lifting every clean point by the glitch's depth -- three times the
    feature. Dropped, the curve is its clean points minus their own minimum."""
    glitched, clean = _glitch_family(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a drop is the caller's decision: no warning
        plot.plot_chi_vs_h_family([glitched, clean], ax=ax, baseline="min", outliers="drop")
    assert [line.get_label() for line in ax.lines] == ["T = 2 K", "T = 5 K"]
    x, y = _line_xy(ax.lines[1])
    kept = np.delete(glitched.chi_real.to_numpy(), _GL_ROW)
    np.testing.assert_allclose(x, np.delete(_GL_H, _GL_ROW))
    np.testing.assert_allclose(y, kept - kept.min(), rtol=0, atol=1e-15)
    # The clean member is untouched by its neighbour's glitch.
    raw = clean.chi_real.to_numpy()
    np.testing.assert_allclose(_line_xy(ax.lines[0])[1], raw - raw.min(), rtol=0, atol=1e-15)


def test_family_glitch_warn_is_the_default_and_names_the_point_and_the_baseline(tmp_path, ax):
    glitched, clean = _glitch_family(tmp_path)
    record = _recorded(lambda: plot.plot_chi_vs_h_family([glitched, clean], ax=ax, baseline="min"))
    assert len(record) == 1  # one warning per family, not per curve
    assert record[0].category is UserWarning
    message = str(record[0].message)
    assert message.startswith("plot_family: 1 point drawn departs")
    value = glitched.chi_real[_GL_ROW]
    assert f"iso_5K.dat: chi_real {value:.4g} at H = 30000 Oe (row {_GL_ROW}), score" in message
    assert "iso_2K.dat" not in message
    assert "can set the baseline (baseline='min')" in message
    for fix in ("outliers='drop'", "outliers='ignore'", "outlier_threshold=", "m.quality()"):
        assert fix in message
    # Kept as measured: the glitch is the baseline, and the clean points sit
    # the glitch's depth above zero -- the figure the warning is about.
    y = _line_xy(ax.lines[1])[1]
    assert y[_GL_ROW] == 0.0
    assert np.delete(y, _GL_ROW).min() > 2.5 * _GL_FEATURE


def test_family_glitch_ignore_is_silent_and_draws_what_warn_draws(tmp_path, ax):
    glitched, clean = _glitch_family(tmp_path)
    _recorded(lambda: plot.plot_chi_vs_h_family([glitched, clean], ax=ax, baseline="min"))
    fig, ax_ignore = plt.subplots()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plot.plot_chi_vs_h_family(
                [glitched, clean], ax=ax_ignore, baseline="min", outliers="ignore"
            )
        for warned, ignored in zip(ax.lines, ax_ignore.lines, strict=True):
            np.testing.assert_array_equal(_line_xy(warned)[1], _line_xy(ignored)[1])
    finally:
        plt.close(fig)


def test_family_warning_without_baseline_does_not_mention_one(tmp_path, ax):
    glitched, clean = _glitch_family(tmp_path)
    (warning,) = _recorded(lambda: plot.plot_chi_vs_h_family([glitched, clean], ax=ax))
    assert "baseline" not in str(warning.message)


def test_family_clean_family_is_silent(tmp_path, ax):
    ms = [
        _gl_load(_glitch_iso_file(tmp_path / f"c{t:g}.dat", t, glitch=False))
        for t in (2.0, 5.0)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot.plot_chi_vs_h_family(ms, ax=ax, baseline="min", offset=1e-4)
    assert len(ax.lines) == 2


def test_family_warning_coordinate_follows_field_unit(tmp_path, ax):
    glitched, clean = _glitch_family(tmp_path)
    (warning,) = _recorded(
        lambda: plot.plot_chi_vs_h_family([glitched, clean], ax=ax, field_unit="tesla")
    )
    assert f"at H = 3 T (row {_GL_ROW})" in str(warning.message)


def test_family_outlier_threshold_moves_the_line(tmp_path, ax):
    glitched, clean = _glitch_family(tmp_path)
    score = glitched.quality().flagged["continuity_score"].iloc[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot.plot_chi_vs_h_family([glitched, clean], ax=ax, outlier_threshold=2 * score)
    fig, ax2 = plt.subplots()
    try:
        (warning,) = _recorded(
            lambda: plot.plot_chi_vs_h_family([glitched, clean], ax=ax2, outlier_threshold=5)
        )
        assert "continuity score > 5," in str(warning.message)
    finally:
        plt.close(fig)


def test_family_warning_caps_the_list(tmp_path, ax):
    ms = [
        _gl_load(_glitch_iso_file(tmp_path / f"g{t:g}.dat", float(t)))
        for t in range(2, 14)
    ]
    (warning,) = _recorded(lambda: plot.plot_chi_vs_h_family(ms, ax=ax))
    message = str(warning.message)
    assert message.startswith("plot_family: 12 points drawn depart ")
    assert message.count(f"(row {_GL_ROW})") == 10
    assert "  ... and 2 more" in message
    # Listed in draw order, the order of the legend.
    assert message.index("g2.dat") < message.index("g3.dat") < message.index("g11.dat")


def test_family_warning_points_at_the_caller(tmp_path, ax):
    """stacklevel reaches past the alias as well as plot_family itself."""
    glitched, clean = _glitch_family(tmp_path)
    (via_alias,) = _recorded(lambda: plot.plot_chi_vs_h_family([glitched, clean], ax=ax))
    (direct,) = _recorded(lambda: plot.plot_family([glitched, clean], plot.plot_chi_vs_h, ax=ax))
    assert via_alias.filename == __file__
    assert direct.filename == __file__


@pytest.mark.parametrize("caller_module", ["workflow", "calamari.plotting_helpers"])
def test_family_warning_keeps_external_caller_with_similar_module_name(tmp_path, ax, caller_module):
    namespace = {
        "__name__": caller_module, "plot": plot, "ax": ax,
        "measurements": _glitch_family(tmp_path),
    }
    code = compile("plot.plot_chi_vs_h_family(measurements, ax=ax)", "external_workflow.py", "exec")
    (warning,) = _recorded(lambda: exec(code, namespace))
    assert warning.filename == "external_workflow.py"


def test_family_screening_works_with_a_foreign_plot_callable(tmp_path, ax):
    """Screening acts on the Measurement before plot is called, so -- unlike
    offset/baseline -- it reaches a callable calamari knows nothing about."""
    received = []

    def my_plot(m, ax=None, label=None, **style):
        received.append(m)
        return plot.plot_chi_vs_h(m, ax=ax, label=label, **style)

    glitched, clean = _glitch_family(tmp_path)
    plot.plot_family([glitched, clean], my_plot, ax=ax, outliers="drop")
    assert [len(m.data) for m in received] == [41, 40]  # 2 K clean, then 5 K
    assert glitched.chi_real[_GL_ROW] not in _line_xy(ax.lines[1])[1]
    assert len(glitched.data) == 41  # the input is untouched (DM-1)
    received.clear()
    (warning,) = _recorded(lambda: plot.plot_family([glitched, clean], my_plot, ax=ax))
    # No known axis for a foreign callable: the sweep's own (an H-scan).
    assert f"at H = 30000 Oe (row {_GL_ROW})" in str(warning.message)


def test_family_reports_pair_with_the_measurements_as_given(tmp_path, ax):
    """Reports align with the input order; sorting (5 K after 2 K) happens
    after pairing, so each report still drops from its own curve."""
    glitched, clean = _glitch_family(tmp_path)
    reports = [glitched.quality(), clean.quality()]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # reports are an explicit drop: no warning
        plot.plot_chi_vs_h_family([glitched, clean], ax=ax, outliers=reports, baseline="min")
    assert [line.get_label() for line in ax.lines] == ["T = 2 K", "T = 5 K"]
    assert [len(line.get_xdata()) for line in ax.lines] == [41, 40]
    kept = np.delete(glitched.chi_real.to_numpy(), _GL_ROW)
    np.testing.assert_allclose(_line_xy(ax.lines[1])[1], kept - kept.min(), atol=1e-15)


def test_family_reports_in_the_wrong_order_are_refused(tmp_path, ax):
    glitched, clean = _glitch_family(tmp_path)
    with pytest.raises(ValueError, match=r"outliers\[0\].*iso_2K\.dat.*another file"):
        plot.plot_chi_vs_h_family(
            [glitched, clean], ax=ax, outliers=[clean.quality(), glitched.quality()]
        )
    assert len(ax.lines) == 0


def test_family_report_count_must_match(tmp_path, ax):
    glitched, clean = _glitch_family(tmp_path)
    with pytest.raises(ValueError, match="1 QualityReports for 2 measurements.*one-to-one"):
        plot.plot_chi_vs_h_family([glitched, clean], ax=ax, outliers=[glitched.quality()])
    assert len(ax.lines) == 0


def test_family_report_of_another_view_is_refused(tmp_path, ax):
    """A report describes the view it was taken on. restrict() renumbers the
    rows -- as do the isotherms split_isotherms cuts from one file, which all
    share its path -- so a report of the whole file is refused for a view."""
    glitched, clean = _glitch_family(tmp_path)
    view = glitched.restrict(h_range=(5000.0, 40000.0))
    with pytest.raises(ValueError, match=r"outliers\[0\].*restrict\(\)"):
        plot.plot_chi_vs_h_family(
            [view, clean], ax=ax, outliers=[glitched.quality(), clean.quality()]
        )
    assert len(ax.lines) == 0
    # Scored as passed, the view screens itself: its own labels, its own drop.
    plot.plot_chi_vs_h_family([view, clean], ax=ax, outliers=[view.quality(), clean.quality()])
    plot.plot_chi_vs_h_family([view, clean], ax=ax, outliers="drop")
    assert [len(line.get_xdata()) for line in ax.lines] == [41, len(view.data) - 1] * 2


def test_family_reports_for_paths_describe_the_loaded_files(tmp_path, ax):
    paths = [
        _glitch_iso_file(tmp_path / "iso_5K.dat", 5.0),
        _glitch_iso_file(tmp_path / "iso_2K.dat", 2.0, glitch=False),
    ]
    reports = [_gl_load(p).quality() for p in paths]
    plot.plot_chi_vs_h_family(paths, ax=ax, outliers=reports)
    assert [len(line.get_xdata()) for line in ax.lines] == [41, 40]
    with pytest.raises(ValueError, match="another file"):
        plot.plot_chi_vs_h_family(paths, ax=ax, outliers=reports[::-1])


def test_family_reports_with_a_threshold_are_refused(tmp_path, ax):
    glitched, clean = _glitch_family(tmp_path)
    with pytest.raises(ValueError, match="at its own threshold"):
        plot.plot_chi_vs_h_family(
            [glitched, clean], ax=ax, outliers=[glitched.quality(), clean.quality()],
            outlier_threshold=5.0,
        )
    assert len(ax.lines) == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"outliers": "bogus"}, "outliers must be 'warn', 'drop', 'ignore', or a sequence"),
        ({"outliers": None}, "outliers must be"),
        ({"outliers": True}, "outliers must be"),
        ({"outliers": ["drop", "drop"]}, r"outliers\[0\] is a str, not a QualityReport"),
        ({"outlier_threshold": 0.0}, "outlier_threshold must be"),
        ({"outlier_threshold": -5.0}, "outlier_threshold must be"),
        ({"outlier_threshold": float("nan")}, "outlier_threshold must be"),
        ({"outlier_threshold": True}, "outlier_threshold must be"),
        ({"outlier_threshold": "20"}, "outlier_threshold must be"),
    ],
)
def test_family_outlier_keywords_refused_before_drawing(tmp_path, ax, kwargs, match):
    glitched, clean = _glitch_family(tmp_path)
    with pytest.raises(ValueError, match=match):
        plot.plot_chi_vs_h_family([glitched, clean], ax=ax, **kwargs)
    assert len(ax.lines) == 0


def test_family_single_report_is_refused_by_name(tmp_path, ax):
    glitched, clean = _glitch_family(tmp_path)
    with pytest.raises(ValueError, match="one QualityReport per measurement"):
        plot.plot_chi_vs_h_family([glitched, clean], ax=ax, outliers=glitched.quality())
    assert len(ax.lines) == 0


def _unscreenable_iso_file(path, t_k):
    """A clean isotherm plus Min./Max. Temperature columns in degC: quality()
    refuses the unit (it reads those for its spread evidence), while the
    chi' curve -- which never reads them -- draws fine."""
    t = np.full_like(_GL_H, t_k)
    return _glitch_iso_file(
        path, t_k, glitch=False,
        extra_columns={"Min. Temperature (degC)": t - 273.15, "Max. Temperature (degC)": t - 273.15},
    )


def test_family_unscreenable_premise(tmp_path):
    from calamari.exceptions import UnknownUnitError

    with pytest.raises(UnknownUnitError, match="degC"):
        _gl_load(_unscreenable_iso_file(tmp_path / "degc.dat", 2.0)).quality()


def test_family_unscreenable_warn_draws_as_measured_and_says_so_once(tmp_path, ax):
    ms = [
        _gl_load(_unscreenable_iso_file(tmp_path / f"degc_{t:g}K.dat", t)) for t in (2.0, 5.0)
    ]
    record = _recorded(lambda: plot.plot_chi_vs_h_family(ms, ax=ax, baseline="min"))
    assert len(record) == 1
    message = str(record[0].message)
    assert message.startswith("plot_family: 2 curves not screened for bad points")
    assert "degc_2K.dat: UnknownUnitError" in message and "degc_5K.dat" in message
    assert "outliers='ignore'" in message
    assert record[0].filename == __file__
    fig, ax_ignore = plt.subplots()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plot.plot_chi_vs_h_family(ms, ax=ax_ignore, baseline="min", outliers="ignore")
        for warned, ignored in zip(ax.lines, ax_ignore.lines, strict=True):
            np.testing.assert_array_equal(_line_xy(warned)[1], _line_xy(ignored)[1])
    finally:
        plt.close(fig)


def test_family_unscreenable_drop_raises_before_drawing(tmp_path, ax):
    """A removal that cannot be honoured must not pass as a cleaned figure."""
    from calamari.exceptions import UnknownUnitError

    glitched, _ = _glitch_family(tmp_path)
    bad = _gl_load(_unscreenable_iso_file(tmp_path / "degc.dat", 2.0))
    with pytest.raises(ValueError, match="degc.dat: outliers='drop'.*could not be screened") as info:
        plot.plot_chi_vs_h_family([glitched, bad], ax=ax, outliers="drop")
    assert isinstance(info.value.__cause__, UnknownUnitError)
    message = str(info.value)
    assert "outliers='ignore'" in message and "restrict(mask=" in message
    assert len(ax.lines) == 0


def _combined_glitch_file(path):
    """A VSM+AC isotherm (one VSM and one AC row per setpoint) whose one
    glitch is in the VSM moment; chi' is clean."""
    from calamari import simulate

    rng = np.random.default_rng(20260929)
    n = len(_GL_H)
    moment = 1e-3 * np.tanh(_GL_H / 20000.0) + rng.normal(scale=1e-8, size=n)
    moment[_GL_ROW] -= 3e-4
    chi = simulate.synthetic_chi_peak(
        _GL_H, centre=15000.0, width=2000.0, height=_GL_FEATURE, background=1e-3,
        noise=1e-7, rng=rng,
    )
    vsm = np.arange(2 * n) % 2 == 0
    return simulate.write_mpms3_file(
        path, app="MPMS3 w/ AC,1.0,1.1", temperature_K=np.full(2 * n, 5.0),
        field_oe=np.repeat(_GL_H, 2),
        vsm_moment_emu=np.where(vsm, np.repeat(moment, 2), np.nan),
        chi_real=np.where(vsm, np.nan, np.repeat(chi, 2)),
        chi_imag=np.where(vsm, np.nan, 1e-6),
        ac_frequency_hz=np.where(vsm, np.nan, 997.0),
        transport_action=np.where(vsm, 1.0, 5.0),
    )


def test_family_warns_only_about_the_family_the_curve_draws(tmp_path, ax):
    """A glitch in a combined file's VSM rows is not on a chi' figure (chi'
    is NaN on those rows), so it is not warned about there -- but it is on
    the M(H) figure, and on a foreign callable's, which could draw anything."""
    m = _gl_load(_combined_glitch_file(tmp_path / "combined.dat"))
    flagged = m.quality().flagged
    assert list(flagged["family"]) == ["VSM"] and list(flagged.index) == [2 * _GL_ROW]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot.plot_chi_vs_h_family([m], ax=ax, source="ac")
        plot.plot_chi_vs_h_family([m], ax=ax, source="ac", component="both")
    (mh,) = _recorded(lambda: plot.plot_mh_family([m], ax=ax))
    assert f"moment_emu {m.moment_emu[2 * _GL_ROW]:.4g} at H = 30000 Oe" in str(mh.message)

    def my_plot(m, ax=None, label=None, **style):
        return plot.plot_chi_vs_h(m, ax=ax, label=label, source="ac", **style)

    (foreign,) = _recorded(lambda: plot.plot_family([m], my_plot, ax=ax))
    assert "moment_emu" in str(foreign.message)


def test_family_chi_auto_resolving_to_m_over_h_screens_the_moment(tmp_path, ax):
    """source='auto' on a DC file draws M/H, so a moment glitch is on the
    figure -- resolved by the rule chi_series uses, not guessed."""
    from calamari import simulate

    t = np.linspace(2.0, 42.0, 41)
    paths = []
    for h_oe in (1000.0, 2000.0):
        rng = np.random.default_rng(int(h_oe))
        moment = 1e-3 * h_oe / 1000.0 / (t + 5.0) + rng.normal(scale=1e-9, size=len(t))
        if h_oe == 1000.0:
            moment[_GL_ROW] -= 1e-5
        paths.append(simulate.write_mpms3_file(
            tmp_path / f"mt_{h_oe:g}Oe.dat", app="SQUID VSM", temperature_K=t,
            field_oe=np.full_like(t, h_oe), vsm_moment_emu=moment,
        ))
    (warning,) = _recorded(lambda: plot.plot_chi_vs_t_family(paths, ax=ax))
    assert ax.get_ylabel().startswith("$M/H$")
    assert "mt_1000Oe.dat: moment_emu" in str(warning.message)
    assert f"at T = {t[_GL_ROW]:g} K (row {_GL_ROW})" in str(warning.message)
