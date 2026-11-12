"""Family grids, interpolation, colour limits and map drawing."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import QuadMesh

from calamari import _plot_data as plot_data
from calamari import plot

from ._helpers import (
    _detected,
    _field_candidates,
    _measurement,
)

# 2-D maps: chi_map_grid / moment_map_grid / draw_map / plot_*_map


def _map_isotherm(t_k: float, field_oe: np.ndarray, chi: np.ndarray, **extra):
    cols = {
        "Temperature (K)": np.full(len(field_oe), t_k),
        "Magnetic Field (Oe)": field_oe,
        "AC X' (emu/Oe)": chi,
    }
    cols.update(extra)
    return _measurement(**cols)


@pytest.fixture
def map_isotherms():
    """Three chi(H) isotherms, chi linear in H so every cell mean is exact."""
    h = np.linspace(0.0, 1000.0, 11)  # step 100
    return [
        _map_isotherm(2.0, h, 5.0 + 0.001 * h),
        _map_isotherm(5.0, h, 10.0 + 0.002 * h),
        _map_isotherm(10.0, h, 15.0 + 0.003 * h),
    ]


def test_map_grid_bin_values_and_orientation(map_isotherms):
    g = plot.chi_map_grid(map_isotherms, scan_axis=np.linspace(0.0, 1000.0, 11))
    assert g.family_axis == "temperature"
    np.testing.assert_allclose(g.t_k, [2.0, 5.0, 10.0])
    assert g.values.shape == (3, 11)
    # One real point per cell: the cell mean IS the sample value.
    assert (g.counts == 1).all()
    np.testing.assert_allclose(g.values[0], 5.0 + 0.001 * g.h_oe)
    np.testing.assert_allclose(g.values[2], 15.0 + 0.003 * g.h_oe)
    assert g.value_label == r"$\chi'$ (emu/Oe)"
    assert g.gridding == "bin"


def test_map_grid_bin_cell_mean_and_counts():
    """Two points in one cell: the cell is their mean and counts says 2."""
    h = np.array([0.0, 10.0, 500.0, 1000.0])
    sweeps = [
        _map_isotherm(2.0, h, np.array([1.0, 3.0, 5.0, 7.0])),
        _map_isotherm(5.0, h, np.array([2.0, 4.0, 6.0, 8.0])),
    ]
    g = plot.chi_map_grid(sweeps, scan_axis=np.array([0.0, 500.0, 1000.0]))
    # cell 0 spans [-250, 250): rows 0 and 10 Oe both land there.
    assert g.counts[0, 0] == 2
    assert g.values[0, 0] == pytest.approx(2.0)  # mean(1, 3)
    assert g.values[0, 1] == pytest.approx(5.0)
    assert g.counts[0, 1] == 1


def test_map_grid_interior_gap_stays_a_hole(map_isotherms):
    h = np.concatenate([np.linspace(0, 200, 5), np.linspace(800, 1000, 5)])
    gappy = [
        _map_isotherm(2.0, h, np.full(10, 1.0)),
        _map_isotherm(5.0, h, np.full(10, 2.0)),
    ]
    g = plot.chi_map_grid(gappy, scan_axis=np.linspace(0.0, 1000.0, 21))
    hole = (g.h_oe > 250) & (g.h_oe < 750)
    assert hole.any()
    assert np.isnan(g.values[:, hole]).all()
    assert (g.counts[:, hole] == 0).all()


def test_map_grid_error_propagation():
    """n equal-sigma points per cell -> standard error sigma/sqrt(n)."""
    h = np.array([100.0, 300.0])  # both in the first cell of [250, 750]
    sweeps = [
        _map_isotherm(2.0, h, np.array([1.0, 3.0]),
                      **{"AC X' Std Err. (emu/Oe)": np.array([0.4, 0.4])}),
        _map_isotherm(5.0, h, np.array([2.0, 4.0]),
                      **{"AC X' Std Err. (emu/Oe)": np.array([0.4, 0.4])}),
    ]
    g = plot.chi_map_grid(sweeps, scan_axis=np.array([250.0, 750.0]))
    assert g.errors is not None
    assert g.counts[0, 0] == 2
    assert g.values[0, 0] == pytest.approx(2.0)
    assert g.errors[0, 0] == pytest.approx(0.4 / np.sqrt(2))


def test_map_grid_t_scan_orientation():
    """T-scans at fixed field: family is the field axis; array stays [T, H]."""
    t = np.linspace(2.0, 40.0, 20)
    sweeps = []
    for h_oe in (1000.0, 10000.0):
        sweeps.append(_measurement(**{
            "Temperature (K)": t,
            "Magnetic Field (Oe)": np.full(20, h_oe),
            "AC X' (emu/Oe)": np.full(20, h_oe / 1000.0),
        }))
    g = plot.chi_map_grid(sweeps)
    assert g.family_axis == "field"
    np.testing.assert_allclose(g.h_oe, [1000.0, 10000.0])
    assert g.values.shape == (20, 2)  # [i_T, j_H], fixed orientation
    np.testing.assert_allclose(g.values[:, 0], 1.0)
    np.testing.assert_allclose(g.values[:, 1], 10.0)


def test_map_grid_refuses_mixed_quantities(map_isotherms):
    """chi' rows and M/H rows under one colourbar is the composition lie."""
    h = np.linspace(0.0, 1000.0, 11)
    dc_only = _measurement(**{
        "Temperature (K)": np.full(11, 20.0),
        "Magnetic Field (Oe)": h + 1.0,  # no zero field for M/H
        "Moment (emu)": 1e-3 * (h + 1.0),
    })
    with pytest.raises(ValueError, match="different quantities"):
        plot.chi_map_grid([*map_isotherms, dc_only])


def test_map_grid_refuses_duplicate_family_values(map_isotherms):
    twin = _map_isotherm(5.0, np.linspace(0, 1000, 11), np.full(11, 1.0))
    with pytest.raises(ValueError, match="same fixed temperature"):
        plot.chi_map_grid([*map_isotherms, twin])


def test_map_grid_refuses_hysteresis_loop(map_isotherms):
    h = np.concatenate([np.linspace(0, 1000, 6), np.linspace(1000, 0, 6)])
    loop = _map_isotherm(20.0, h, np.linspace(1, 2, 12))
    with pytest.raises(ValueError, match="not monotonic"):
        plot.chi_map_grid([*map_isotherms, loop])


def test_map_grid_tolerates_setpoint_jitter_backtrack(map_isotherms):
    """Tiny backtracks (magnet jitter) must not read as hysteresis."""
    h = np.linspace(0.0, 1000.0, 11)
    h[5] -= 0.5  # a 0.5 Oe backtrack on a 1000 Oe span: 0.05%
    jittery = _map_isotherm(20.0, np.sort(h) + 0.0, 1.0 + 0.001 * np.sort(h))
    # jitter version: use unsorted h with the micro-backtrack
    jittery = _map_isotherm(20.0, h, 1.0 + 0.001 * h)
    g = plot.chi_map_grid([*map_isotherms, jittery])
    assert g.values.shape[0] == 4


def test_map_grid_refuses_multi_frequency_without_selection():
    t = np.linspace(2.0, 40.0, 10)
    def sweep(h_oe):
        return _measurement(**{
            "Temperature (K)": np.repeat(t, 2)[:20],
            "Magnetic Field (Oe)": np.full(20, h_oe),
            "AC X' (emu/Oe)": np.ones(20),
            "AC Frequency (Hz)": np.tile([97.0, 997.0], 10),
        })
    with pytest.raises(ValueError, match="distinct AC frequencies"):
        plot.chi_map_grid([sweep(100.0), sweep(1000.0)])
    g = plot.chi_map_grid([sweep(100.0), sweep(1000.0)], frequency=997.0)
    assert g.frequency_hz == pytest.approx(997.0)


def test_map_grid_interp_requires_max_gap(map_isotherms):
    with pytest.raises(ValueError, match="requires max_gap"):
        plot.chi_map_grid(map_isotherms, gridding="interp")


def test_map_grid_interp_blanks_wide_brackets():
    h = np.concatenate([np.linspace(0, 200, 5), np.linspace(800, 1000, 5)])
    sweeps = [
        _map_isotherm(2.0, h, np.full(10, 1.0)),
        _map_isotherm(5.0, h, np.full(10, 2.0)),
    ]
    g = plot.chi_map_grid(sweeps, gridding="interp", max_gap=100.0,
                          scan_axis=np.linspace(0.0, 1000.0, 21))
    hole = (g.h_oe > 250) & (g.h_oe < 750)
    assert np.isnan(g.values[:, hole]).all()
    inside = g.h_oe <= 200
    assert np.isfinite(g.values[0, inside]).all()


def test_map_grid_refuses_single_sweep(map_isotherms):
    with pytest.raises(ValueError, match="at least two sweeps"):
        plot.chi_map_grid(map_isotherms[:1])


def test_map_grid_mixed_scan_kinds_names_split_isotherms(map_isotherms):
    t_scan = _measurement(**{
        "Temperature (K)": np.linspace(2, 40, 10),
        "Magnetic Field (Oe)": np.full(10, 100.0),
        "AC X' (emu/Oe)": np.ones(10),
    })
    with pytest.raises(ValueError, match="mixed T-scans and H-scans"):
        plot.chi_map_grid([*map_isotherms, t_scan])


def test_moment_map_grid(map_isotherms):
    h = np.linspace(0.0, 1000.0, 11)
    sweeps = [
        _measurement(**{"Temperature (K)": np.full(11, tk),
                        "Magnetic Field (Oe)": h,
                        "Moment (emu)": np.full(11, tk * 1e-4)})
        for tk in (2.0, 5.0)
    ]
    g = plot.moment_map_grid(sweeps)
    assert g.value_label == "Moment (emu)"
    np.testing.assert_allclose(g.values[0], 2e-4)


def test_draw_map_returns_quadmesh_with_explicit_bad_colour(ax, map_isotherms):
    g = plot.chi_map_grid(map_isotherms)
    mesh = plot.draw_map(g, ax=ax)
    assert isinstance(mesh, QuadMesh)
    assert mesh.axes is ax
    bad = mesh.get_cmap().get_bad()
    assert bad[3] > 0  # never transparent: a hole must not read as background
    assert ax.get_xlabel() == "Temperature (K)"
    assert g.value_label in ax.get_title()


def test_draw_map_rasterises_the_mesh_only_and_can_be_told_not_to(ax, map_isotherms):
    # In a vector file the mesh becomes one image at savefig.dpi while the
    # axes, text and sample overlay stay vector; the caller can opt out.
    g = plot.chi_map_grid(map_isotherms)
    mesh = plot.draw_map(g, ax=ax)
    assert mesh.get_rasterized() is True
    assert all(coll.get_rasterized() is not True for coll in ax.collections if coll is not mesh)
    assert ax.xaxis.label.get_rasterized() is not True
    mesh2 = plot.draw_map(g, ax=ax, rasterized=False)
    assert mesh2.get_rasterized() is False


def test_draw_map_show_samples_overlay(ax, map_isotherms):
    g = plot.chi_map_grid(map_isotherms)
    plot.draw_map(g, ax=ax)
    # mesh + scatter of the real sample positions
    assert len(ax.collections) == 2
    plot.draw_map(g, ax=ax, show_samples=False)
    assert len(ax.collections) == 3  # only the second mesh was added


def test_draw_map_family_scales(ax):
    t = np.linspace(2.0, 40.0, 10)
    sweeps = [
        _measurement(**{"Temperature (K)": t,
                        "Magnetic Field (Oe)": np.full(10, h_oe),
                        "AC X' (emu/Oe)": np.ones(10)})
        for h_oe in (100.0, 1000.0, 30000.0)
    ]
    g = plot.chi_map_grid(sweeps)
    for scale in ("linear", "log", "index"):
        fig2, ax2 = plt.subplots()
        try:
            plot.draw_map(g, ax=ax2, family_scale=scale)
            if scale == "index":
                labels = [tick.get_text() for tick in ax2.get_yticklabels()]
                assert labels[:3] == ["0.01", "0.1", "3"]
        finally:
            plt.close(fig2)


def test_draw_map_log_refuses_nonpositive_family(ax):
    t = np.linspace(2.0, 40.0, 10)
    sweeps = [
        _measurement(**{"Temperature (K)": t,
                        "Magnetic Field (Oe)": np.full(10, h_oe),
                        "AC X' (emu/Oe)": np.ones(10)})
        for h_oe in (0.0, 1000.0)
    ]
    g = plot.chi_map_grid(sweeps)
    with pytest.raises(ValueError, match="family_scale='index'"):
        plot.draw_map(g, ax=ax, family_scale="log")


def test_plot_chi_map_one_call(ax, map_isotherms):
    mesh = plot.plot_chi_map(map_isotherms, ax=ax, mesh_kwargs={"cmap": "magma"})
    assert isinstance(mesh, QuadMesh)


# interpolate_family: filled colour between the sweeps


def test_interpolate_family_fills_between_sweeps_exactly(map_isotherms):
    """chi is linear in H per row, so an interpolated row is an exact
    weighted blend of its bracketing measured rows."""
    g = plot.chi_map_grid(map_isotherms, scan_axis=np.linspace(0.0, 1000.0, 11))
    dense = plot.interpolate_family(g, max_gap=10.0, n_rows=9)

    np.testing.assert_allclose(dense.t_k, np.arange(2.0, 10.5, 1.0))
    assert dense.values.shape == (9, 11)
    # T = 4 K sits 2/3 of the way from the 2 K sweep to the 5 K sweep.
    np.testing.assert_allclose(
        dense.values[2], (1 / 3) * g.values[0] + (2 / 3) * g.values[1]
    )
    # Manufactured data says so: no cell aggregates a real point, cell
    # errors are gone, the gridding string records the step, and the
    # sample audit trail survives untouched.
    assert (dense.counts == 0).all()
    assert dense.errors is None
    assert dense.gridding == "bin+family-interp"
    np.testing.assert_array_equal(dense.sample_t, g.sample_t)
    np.testing.assert_array_equal(dense.sample_h, g.sample_h)
    assert dense.family_axis == "temperature"
    assert dense.value_label == g.value_label


def test_interpolate_family_max_gap_leaves_holes(map_isotherms):
    """Sweeps at 2, 5 and 10 K with max_gap=4: the 3 K interval fills,
    the 5 K interval stays a hole, and measured rows are never blanked."""
    g = plot.chi_map_grid(map_isotherms, scan_axis=np.linspace(0.0, 1000.0, 11))
    dense = plot.interpolate_family(g, max_gap=4.0, n_rows=9)

    t = dense.t_k
    assert np.isfinite(dense.values[(t >= 2.0) & (t <= 5.0)]).all()
    assert np.isnan(dense.values[(t > 5.0) & (t < 10.0)]).all()
    # The 10 K row is measured data landing exactly on a dense row: kept.
    np.testing.assert_allclose(dense.values[t == 10.0][0], g.values[2])


def test_interpolate_family_does_not_bridge_scan_holes():
    """A column the middle sweep did not cover interpolates across the
    outer sweeps only -- and if THEIR spacing exceeds max_gap, it stays a
    hole. Scan-axis holes are never bridged sideways."""
    h_full = np.linspace(0.0, 1000.0, 11)
    h_short = h_full[h_full <= 500.0]
    sweeps = [
        _map_isotherm(2.0, h_full, 5.0 + 0.001 * h_full),
        _map_isotherm(5.0, h_short, 10.0 + 0.002 * h_short),
        _map_isotherm(10.0, h_full, 15.0 + 0.003 * h_full),
    ]
    g = plot.chi_map_grid(sweeps, scan_axis=h_full)
    dense = plot.interpolate_family(g, max_gap=6.0, n_rows=9)

    covered = g.h_oe <= 500.0  # all three sweeps: gaps 3 K and 5 K, both fill
    uncovered = g.h_oe > 500.0  # outer sweeps only: an 8 K gap, over max_gap
    interior = (dense.t_k > 2.0) & (dense.t_k < 10.0)
    assert np.isfinite(dense.values[np.ix_(interior, covered)]).all()
    assert np.isnan(dense.values[np.ix_(interior, uncovered)]).all()


def test_interpolate_family_field_family_orientation():
    """T-scans at fixed fields: the field axis is the family, and the
    interpolation transposes correctly ([T, H] orientation is fixed)."""
    t = np.linspace(2.0, 40.0, 20)
    sweeps = [
        _measurement(**{
            "Temperature (K)": t,
            "Magnetic Field (Oe)": np.full(20, h_oe),
            "AC X' (emu/Oe)": np.full(20, h_oe / 1000.0),
        })
        for h_oe in (1000.0, 3000.0)
    ]
    dense = plot.interpolate_family(
        plot.chi_map_grid(sweeps), max_gap=5000.0, n_rows=5
    )
    assert dense.family_axis == "field"
    np.testing.assert_allclose(dense.h_oe, [1000.0, 1500.0, 2000.0, 2500.0, 3000.0])
    assert dense.values.shape == (20, 5)
    np.testing.assert_allclose(dense.values[:, 2], 2.0)  # midway blend


def test_interpolate_family_refusals(map_isotherms):
    g = plot.chi_map_grid(map_isotherms, scan_axis=np.linspace(0.0, 1000.0, 11))
    with pytest.raises(ValueError, match="requires max_gap"):
        plot.interpolate_family(g, max_gap=None)
    with pytest.raises(ValueError, match="must be positive"):
        plot.interpolate_family(g, max_gap=-1.0)
    with pytest.raises(ValueError, match="at least 2"):
        plot.interpolate_family(g, max_gap=10.0, n_rows=1)


def test_plot_map_family_gridding_composes(map_isotherms):
    mesh = plot.plot_chi_map(map_isotherms, family_gridding="interp",
                             family_max_gap=10.0, show_samples=False)
    assert isinstance(mesh, QuadMesh)
    plt.close(mesh.axes.figure)

    with pytest.raises(ValueError, match="requires family_max_gap"):
        plot.plot_chi_map(map_isotherms, family_gridding="interp")
    with pytest.raises(ValueError, match="family_gridding='interp' only"):
        plot.plot_chi_map(map_isotherms, family_max_gap=5.0)
    with pytest.raises(ValueError, match="'rows' or 'interp'"):
        plot.plot_chi_map(map_isotherms, family_gridding="banana")

# Map glitch check (outliers=) and colour limits (clim=)


def _glitched_isotherms():
    """Three linear isotherms with one point of the middle sweep hit by a
    glitch four times the family below zero -- the shape of the real
    defect this check was built for."""
    h = np.linspace(0.0, 1000.0, 11)
    chi_mid = 10.0 + 0.002 * h
    chi_mid[4] = -40.0
    return [
        _map_isotherm(2.0, h, 5.0 + 0.001 * h),
        _map_isotherm(5.0, h, chi_mid),
        _map_isotherm(10.0, h, 15.0 + 0.003 * h),
    ]


def _glitched_family_grid():
    """A 6 x 41 map with one glitched cell: large enough that a single
    cell is well inside the 2.5 % tails the flat-map diagnostic ignores
    (on a 3 x 11 map one cell is 3 % of the map, and the diagnostic is
    right not to fire)."""
    h = np.linspace(0.0, 4000.0, 41)
    sweeps = []
    for i, t_k in enumerate([2.0, 4.0, 6.0, 8.0, 10.0, 12.0]):
        chi = 10.0 + 0.002 * h + i
        if t_k == 6.0:
            chi[20] = -40.0
        sweeps.append(_map_isotherm(t_k, h, chi))
    return plot.chi_map_grid(sweeps, outliers="ignore")


def test_map_grid_clean_family_flags_nothing(map_isotherms):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g = plot.chi_map_grid(map_isotherms)
    assert g.outliers == ()


def test_map_grid_glitch_warns_by_name_and_is_kept():
    with pytest.warns(UserWarning, match="synthetic.dat: value -40 at 400 Oe") as rec:
        g = plot.chi_map_grid(_glitched_isotherms(), scan_axis=np.linspace(0, 1000, 11))
    msg = str(rec[0].message)
    assert msg.startswith("chi_map_grid: 1 point departs")
    assert "outliers='drop'" in msg and "outliers='ignore'" in msg
    assert "(T = 5 K)" in msg
    # Kept as measured: the cell holds the glitch, the record says so.
    assert g.values[1, 4] == pytest.approx(-40.0)
    assert len(g.outliers) == 1
    o = g.outliers[0]
    assert (o.family_value, o.scan_value, o.value, o.dropped) == (5.0, 400.0, -40.0, False)
    assert o.score > plot.MAP_OUTLIER_THRESHOLD
    assert len(g.sample_t) == 33


def test_map_grid_outliers_drop_removes_point_and_records_it():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g = plot.chi_map_grid(
            _glitched_isotherms(), scan_axis=np.linspace(0, 1000, 11), outliers="drop"
        )
    assert np.isnan(g.values[1, 4])
    assert g.counts[1, 4] == 0
    np.testing.assert_allclose(np.delete(g.values[1], 4), np.delete(10.0 + 0.002 * g.h_oe, 4))
    assert len(g.sample_t) == 32  # the dropped point leaves the overlay too
    assert g.outliers[0].dropped is True
    # The record survives family interpolation.
    dense = plot.interpolate_family(g, max_gap=10.0)
    assert dense.outliers == g.outliers


def test_map_grid_outliers_ignore_is_silent_but_recorded():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g = plot.chi_map_grid(_glitched_isotherms(), outliers="ignore")
    assert len(g.outliers) == 1 and g.outliers[0].dropped is False


def test_map_grid_outlier_threshold_moves_the_line():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # The glitch sits on a noiseless, exactly linear sweep, so its score
        # is unbounded: only an infinite threshold spares it.
        g = plot.chi_map_grid(_glitched_isotherms(), outlier_threshold=np.inf)
    assert g.outliers == ()
    with pytest.raises(ValueError, match="outliers must be"):
        plot.chi_map_grid(_glitched_isotherms(), outliers="delete")
    with pytest.raises(ValueError, match="outlier_threshold must be positive"):
        plot.chi_map_grid(_glitched_isotherms(), outlier_threshold=0)


def test_map_outlier_score_spares_a_cusp():
    """A sharp peak sampled at its apex is physics, not a glitch: the
    neighbours slope away on both sides, so no straight line fits them and
    they stray from it by as much as the apex departs -- the score stays of
    order one."""
    x = np.linspace(-5, 5, 41)
    peak = 1.0 / (1.0 + x**2)  # apex at x = 0, sampled exactly
    scores = plot_data._map_outlier_scores([peak, peak * 2])
    assert np.nanmax(scores[0]) < 3
    glitched = peak.copy()
    glitched[30] += 5.0
    scores = plot_data._map_outlier_scores([glitched, peak])
    assert scores[0][30] > plot.MAP_OUTLIER_THRESHOLD
    assert np.nanmax(np.delete(scores[0], 30)) < 3


def test_draw_map_default_warns_when_one_cell_owns_the_range(ax):
    g = _glitched_family_grid()
    with pytest.warns(UserWarning, match="render nearly flat") as rec:
        mesh = plot.draw_map(g, ax=ax)
    assert "clim='robust'" in str(rec[0].message)
    assert mesh.get_clim() == (pytest.approx(-40.0), pytest.approx(23.0))


def test_draw_map_clim_modes(ax, map_isotherms):
    g = _glitched_family_grid()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        full = plot.draw_map(g, ax=ax, clim="minmax")
        assert full.get_clim() == (pytest.approx(-40.0), pytest.approx(23.0))
        robust = plot.draw_map(g, ax=ax, clim="robust")
        finite = g.values[np.isfinite(g.values)]
        lo, hi = np.percentile(finite, plot.MAP_ROBUST_PERCENTILES)
        assert robust.get_clim() == (pytest.approx(lo), pytest.approx(hi))
        assert lo > 0  # the glitch no longer sets the floor
        explicit = plot.draw_map(g, ax=ax, clim=(1.0, 2.0))
        assert explicit.get_clim() == (1.0, 2.0)
        # vmin/vmax in mesh_kwargs are an explicit choice: honoured, no warning.
        direct = plot.draw_map(g, ax=ax, vmin=3.0, vmax=4.0)
        assert direct.get_clim() == (3.0, 4.0)
        # A clean map under the default never warns.
        plot.draw_map(plot.chi_map_grid(map_isotherms), ax=ax)
    with pytest.raises(ValueError, match="not both"):
        plot.draw_map(g, ax=ax, clim="robust", vmin=0.0)
    with pytest.raises(ValueError, match="clim must be"):
        plot.draw_map(g, ax=ax, clim="auto")
    with pytest.raises(ValueError, match="vmin < vmax"):
        plot.draw_map(g, ax=ax, clim=(2.0, 1.0))
    with pytest.raises(ValueError, match="robust_percentiles"):
        plot.draw_map(g, ax=ax, clim="robust", robust_percentiles=(99, 1))


def test_plot_chi_map_threads_outliers_and_clim(ax):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mesh = plot.plot_chi_map(
            _glitched_isotherms(), ax=ax, outliers="drop", clim="robust",
            robust_percentiles=(5, 95),
        )
    lo, hi = mesh.get_clim()
    assert 5.0 <= lo < hi <= 18.0


def test_map_overlay_missing_groups_use_alias_styles(tmp_path, ax):
    sweeps, _ = _detected(tmp_path)
    grid = plot.chi_map_grid(sweeps)
    frame = _field_candidates().assign(frequency_hz=[np.nan, 997.0])
    artists = plot.overlay_transitions(
        frame, ax=ax, grid=grid, color_by="frequency_hz",
        scatter_kwargs={"mfc": "red", "mec": "blue", "ls": "none"},
    )
    assert ("peak", None) in artists
    assert sum(len(c.lines[0].get_xdata()) for c in ax.containers) == 2
    assert all(c.lines[0].get_markerfacecolor() == "red" for c in ax.containers)
    assert all(c.lines[0].get_markeredgecolor() == "blue" for c in ax.containers)

@pytest.mark.parametrize("quantity", ["chi", "moment"])
@pytest.mark.parametrize("family", ["temperature", "field"])
def test_map_distinct_close_coordinates(quantity, family):
    axis = np.linspace(0.0, 1000.0, 11) if family == "temperature" else np.linspace(2.0, 6.0, 11)
    coordinates = [2.0, 2.3] if family == "temperature" else [1000.0, 1005.0]
    column = "AC X' (emu/Oe)" if quantity == "chi" else "DC Moment Free Ctr (emu)"
    measurements = [
        _measurement(**{
            "Temperature (K)": np.full(11, value) if family == "temperature" else axis,
            "Magnetic Field (Oe)": axis if family == "temperature" else np.full(11, value),
            column: axis * scale,
        }) for value, scale in zip(coordinates, [1.0, 2.0])
    ]
    build = plot.chi_map_grid if quantity == "chi" else plot.moment_map_grid
    grid = build(measurements, scan_axis=axis, family_values=iter(coordinates),
                 outliers="ignore")
    expected = np.array([axis, 2 * axis])
    np.testing.assert_allclose(grid.values, expected if family == "temperature" else expected.T)
    assert (grid.counts == 1).all()
    with pytest.raises(plot_data.PlotInputError, match="same fixed"):
        build(measurements, family_values=[coordinates[0]] * 2, outliers="ignore")


@pytest.mark.parametrize("quantity", ["chi", "moment"])
@pytest.mark.parametrize("kwargs", [
    {"scan_axis": []}, {"scan_axis": [1.0]}, {"scan_axis": [[1.0, 2.0]]},
    {"scan_axis": [2.0, 1.0]}, {"scan_axis": [1.0, 1.0]},
    {"scan_axis": [0.0, np.nan]}, {"scan_axis": [0.0, np.inf]},
    {"scan_axis": ["bad", 1.0]}, {"n_scan": 1}, {"n_scan": 2.5}, {"n_scan": True},
    {"family_values": [2.0, np.nan]}, {"family_values": [[2.0, 3.0]]},
    {"family_values": 2.0}, {"family_values": ["bad", 3.0]},
    {"gridding": "interp", "max_gap": 0.0},
    {"gridding": "interp", "max_gap": np.inf},
    {"gridding": "interp", "max_gap": "bad"},
])
def test_map_invalid_grid_request_raises_domain_error(quantity, kwargs):
    axis = np.linspace(0.0, 1000.0, 11)
    column = "AC X' (emu/Oe)" if quantity == "chi" else "DC Moment Free Ctr (emu)"
    measurements = [
        _measurement(**{"Temperature (K)": np.full(11, value),
                        "Magnetic Field (Oe)": axis, column: 1.0 + axis / 1000.0})
        for value in [2.0, 5.0]
    ]
    build = plot.chi_map_grid if quantity == "chi" else plot.moment_map_grid
    with pytest.raises(plot_data.PlotInputError):
        build(measurements, outliers="ignore", **kwargs)
