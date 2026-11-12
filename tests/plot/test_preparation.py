"""Prepared quantities, units, labels and uncertainty propagation."""

from __future__ import annotations

import numpy as np
import pytest

from calamari import _plot_data as plot_data
from calamari import plot
from calamari.constants import OE_PER_TESLA
from calamari.exceptions import AmbiguousColumnError

from ._helpers import (
    _N,
    _TESLA_LABEL,
    _chi_vs_h_measurement,
    _measurement,
    _toy_afm_file,
)

# axis_labels


def test_axis_labels_known_combinations():
    assert plot.axis_labels("field", "oe") == "Field (Oe)"
    assert plot.axis_labels("moment", "emu") == "Moment (emu)"
    assert plot.axis_labels("temperature", "K") == "Temperature (K)"
    assert r"\mu_B" in plot.axis_labels("moment", "muB_per_ion")
    assert r"\chi'" in plot.axis_labels("chi", "real")


def test_axis_labels_unknown_combination_raises():
    with pytest.raises(ValueError):
        plot.axis_labels("field", "furlongs")


def test_every_axis_label_renders_as_mathtext(ax):
    """An invalid mathtext string only fails at draw time -- force one."""
    labels = list(plot_data._AXIS_LABELS.values())
    for _, unit_text, inverse_unit_text in plot_data._CHI_UNITS.values():
        labels.extend([unit_text, inverse_unit_text])
    for symbol, inverse_symbol in plot_data._CHI_SYMBOLS.values():
        labels.extend([symbol, inverse_symbol])
    for label in labels:
        ax.set_xlabel(label)
        ax.figure.canvas.draw()

# chi_series (resolution layer)


def test_chi_series_ac_real(chi_measurement):
    (series,) = plot.chi_series(chi_measurement)
    np.testing.assert_allclose(series.y, chi_measurement.chi_real.to_numpy())
    np.testing.assert_allclose(series.x, chi_measurement.temperature.to_numpy())
    assert series.yerr is None
    assert series.xlabel == "Temperature (K)"
    assert series.ylabel == r"$\chi'$ (emu/Oe)"
    assert series.label is None


def test_chi_series_units_molar_matches_sample_method(chi_measurement, sample):
    (series,) = plot.chi_series(chi_measurement, units="molar", sample=sample)
    expected = sample.chi_to_chi_molar(chi_measurement.chi_real.to_numpy())
    np.testing.assert_allclose(series.y, expected)
    assert series.ylabel == r"$\chi'$ (emu/(mol Oe))"


def test_chi_series_units_require_sample(chi_measurement):
    with pytest.raises(ValueError, match="requires a Sample"):
        plot.chi_series(chi_measurement, units="mass")


def test_chi_series_unknown_units_raises(chi_measurement):
    with pytest.raises(ValueError, match="Unknown chi units"):
        plot.chi_series(chi_measurement, units="banana")


def test_chi_series_dc_source_is_m_over_h(mt_measurement):
    (series,) = plot.chi_series(mt_measurement)  # auto resolves to dc: no chi columns
    expected = (
        mt_measurement.moment_emu.to_numpy() / mt_measurement.field_oe.to_numpy()
    )
    np.testing.assert_allclose(series.y, expected)
    assert series.ylabel == r"$M/H$ (emu/Oe)"  # provenance stamped into the label


def test_chi_series_dc_drops_zero_field_rows():
    m = _measurement(
        **{
            "Temperature (K)": np.array([2.0, 3.0, 4.0]),
            "Magnetic Field (Oe)": np.array([1000.0, 0.0, 1000.0]),
            "Moment (emu)": np.array([1e-3, 1e-3, 2e-3]),
        }
    )
    (series,) = plot.chi_series(m, source="dc")
    np.testing.assert_allclose(series.x, [2.0, 4.0])  # H=0 row gone, not inf
    np.testing.assert_allclose(series.y, [1e-6, 2e-6])


def test_chi_series_dc_cannot_supply_imag(mt_measurement):
    with pytest.raises(ValueError, match="out-of-phase"):
        plot.chi_series(mt_measurement, source="dc", component="imag")


def test_chi_series_auto_refuses_when_both_sources_available(chi_measurement):
    data = dict(chi_measurement.data)
    data["Moment (emu)"] = np.full(_N, 1.0e-3)
    data["Magnetic Field (Oe)"] = np.full(_N, 100.0)
    both = _measurement(**data)
    with pytest.raises(ValueError, match="source='ac'"):
        plot.chi_series(both)
    # Naming the source resolves it, either way.
    (ac,) = plot.chi_series(both, source="ac")
    (dc,) = plot.chi_series(both, source="dc")
    assert ac.ylabel != dc.ylabel


def test_chi_series_nothing_to_resolve_raises():
    m = _measurement(**{"Temperature (K)": np.array([2.0, 3.0])})
    with pytest.raises(ValueError, match="no susceptibility"):
        plot.chi_series(m)


def test_chi_series_inverse_values_errors_and_labels(sample):
    t = np.array([10.0, 20.0, 40.0])
    chi = np.array([0.05, 0.025, 0.0])  # a zero: 1/chi must drop it, not inf
    err = np.array([0.001, 0.001, 0.001])
    m = _measurement(
        **{
            "Temperature (K)": t,
            "AC X' (emu/Oe)": chi,
            "AC X' Std Err. (emu/Oe)": err,
        }
    )
    (series,) = plot.chi_series(m, inverse=True)
    np.testing.assert_allclose(series.x, [10.0, 20.0])  # chi=0 row dropped
    np.testing.assert_allclose(series.y, [20.0, 40.0])
    np.testing.assert_allclose(series.yerr, err[:2] / chi[:2] ** 2)  # d(1/y) = dy/y^2
    assert series.ylabel == r"$1/\chi'$ (Oe/emu)"


def test_chi_series_yerr_from_error_column(chi_measurement):
    data = dict(chi_measurement.data)
    data["AC X' Std Err. (emu/Oe)"] = np.full(_N, 0.01)
    m = _measurement(**data)
    (series,) = plot.chi_series(m)
    np.testing.assert_allclose(series.yerr, np.full(_N, 0.01))


def test_chi_series_splits_by_frequency():
    t = np.tile(np.array([10.0, 20.0, 30.0]), 2)
    freq = np.repeat([97.0, 997.0], 3)
    m = _measurement(
        **{
            "Temperature (K)": t,
            "AC X' (emu/Oe)": np.linspace(1.0, 6.0, 6),
            "AC Frequency (Hz)": freq,
        }
    )
    series = plot.chi_series(m, label="run 1")
    assert [s.label for s in series] == ["run 1 f = 97 Hz", "run 1 f = 997 Hz"]
    np.testing.assert_allclose(series[0].y, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(series[1].y, [4.0, 5.0, 6.0])


def test_chi_series_single_frequency_stays_one_series(chi_measurement):
    data = dict(chi_measurement.data)
    data["AC Frequency (Hz)"] = np.full(_N, 997.0)
    (series,) = plot.chi_series(_measurement(**data), label="run 1")
    assert series.label == "run 1"  # no pointless "f = ..." on a one-frequency file


def test_chi_series_excludes_housekeeping_rows():
    m = _measurement(
        **{
            "Temperature (K)": np.array([5.0, 5.0, 8.5]),
            "AC X' (emu/Oe)": np.array([0.05, 0.05, np.nan]),
            "Transport Action": np.array([5.0, 5.0, 3.0]),
        }
    )
    (series,) = plot.chi_series(m)
    np.testing.assert_allclose(series.x, [5.0, 5.0])  # the locate row is not a point


def test_chi_series_drops_other_family_nan_rows():
    """Combined VSM+AC shape: chi is NaN on the VSM rows; they are simply
    not part of this series, no of_type gymnastics required."""
    m = _measurement(
        **{
            "Temperature (K)": np.array([2.0, 2.0, 3.0, 3.0]),
            "AC X' (emu/Oe)": np.array([np.nan, 1.0, np.nan, 2.0]),
        }
    )
    (series,) = plot.chi_series(m)
    np.testing.assert_allclose(series.x, [2.0, 3.0])
    np.testing.assert_allclose(series.y, [1.0, 2.0])


def test_chi_series_vs_field():
    m = _measurement(
        **{
            "Temperature (K)": np.full(4, 2.0),
            "Magnetic Field (Oe)": np.array([0.0, 100.0, 200.0, 300.0]),
            "AC X' (emu/Oe)": np.array([1.0, 2.0, 3.0, 4.0]),
        }
    )
    (series,) = plot.chi_series(m, x="field")
    np.testing.assert_allclose(series.x, [0.0, 100.0, 200.0, 300.0])
    assert series.xlabel == "Field (Oe)"

# moment_series (resolution layer)


def test_moment_series_emu_with_paired_error(mh_measurement):
    data = dict(mh_measurement.data)
    data["M. Std. Err. (emu)"] = np.full(_N, 1.0e-5)
    (series,) = plot.moment_series(_measurement(**data))
    np.testing.assert_allclose(series.y, mh_measurement.moment_emu.to_numpy())
    np.testing.assert_allclose(series.yerr, np.full(_N, 1.0e-5))
    assert series.xlabel == "Field (Oe)"
    assert series.ylabel == "Moment (emu)"


def test_moment_series_muB_converts_value_and_error(mh_measurement, sample):
    data = dict(mh_measurement.data)
    data["M. Std. Err. (emu)"] = np.full(_N, 1.0e-5)
    (series,) = plot.moment_series(
        _measurement(**data), units="muB_per_ion", sample=sample
    )
    expected_y = sample.moment_emu_to_muB_per_ion(mh_measurement.moment_emu.to_numpy())
    np.testing.assert_allclose(series.y, expected_y)
    np.testing.assert_allclose(
        series.yerr, sample.moment_emu_to_muB_per_ion(np.full(_N, 1.0e-5))
    )


def test_moment_series_split_branches_loop():
    field = np.array([0.0, 500.0, 1000.0, 500.0, 0.0, -500.0, -1000.0, -500.0, 0.0, 500.0, 1000.0])
    moment = np.linspace(0.0, 1.0, len(field))
    m = _measurement(**{"Magnetic Field (Oe)": field, "Moment (emu)": moment})

    series = plot.moment_series(m, split_branches=True, label="5 K")
    assert [s.label for s in series] == [
        "5 K branch 1 (up)",
        "5 K branch 2 (down)",
        "5 K branch 3 (up)",
    ]
    np.testing.assert_allclose(series[0].x, [0.0, 500.0, 1000.0])
    np.testing.assert_allclose(series[1].x, [1000.0, 500.0, 0.0, -500.0, -1000.0])
    np.testing.assert_allclose(series[2].x, [-1000.0, -500.0, 0.0, 500.0, 1000.0])
    # Turning points are shared, so the branches draw as a connected loop.
    assert series[0].x[-1] == series[1].x[0]
    assert series[1].x[-1] == series[2].x[0]


def test_moment_series_split_branches_no_reversal_is_plain(mh_measurement):
    (series,) = plot.moment_series(mh_measurement, split_branches=True, label="run")
    assert series.label == "run"  # no branch annotation invented for a plain sweep


def test_moment_series_split_branches_needs_field_axis(mt_measurement):
    with pytest.raises(ValueError, match="split_branches"):
        plot.moment_series(mt_measurement, x="temperature", split_branches=True)


def test_moment_series_ambiguous_moment_propagates():
    m = _measurement(
        **{
            "Magnetic Field (Oe)": np.array([0.0, 100.0]),
            "Moment (emu)": np.array([1e-3, np.nan]),
            "DC Moment Free Ctr (emu)": np.array([np.nan, 1e-3]),
        }
    )
    with pytest.raises(AmbiguousColumnError):
        plot.moment_series(m)


def test_chi_series_field_unit_tesla_converts_x_and_label_only():
    m = _chi_vs_h_measurement()
    (oe,) = plot.chi_series(m, x="field")
    (tesla,) = plot.chi_series(m, x="field", field_unit="tesla")
    np.testing.assert_allclose(tesla.x, m.field_oe.to_numpy() / OE_PER_TESLA)
    np.testing.assert_allclose(tesla.x, m.field_tesla.to_numpy())
    assert tesla.xlabel == _TESLA_LABEL
    np.testing.assert_array_equal(tesla.y, oe.y)  # the y axis is not touched
    assert tesla.ylabel == oe.ylabel


def test_chi_series_field_unit_default_is_oe():
    m = _chi_vs_h_measurement()
    (default,) = plot.chi_series(m, x="field")
    (oe,) = plot.chi_series(m, x="field", field_unit="oe")
    np.testing.assert_array_equal(default.x, m.field_oe.to_numpy())
    np.testing.assert_array_equal(oe.x, default.x)
    assert default.xlabel == oe.xlabel == "Field (Oe)"


def test_chi_series_dc_in_tesla_still_divides_by_oe(mh_measurement):
    """Only the axis changes unit: M/H stays emu/Oe, computed from field_oe."""
    (oe,) = plot.chi_series(mh_measurement, x="field", source="dc")
    (tesla,) = plot.chi_series(mh_measurement, x="field", source="dc", field_unit="tesla")
    np.testing.assert_array_equal(tesla.y, oe.y)
    np.testing.assert_allclose(tesla.x, oe.x / OE_PER_TESLA)
    assert tesla.ylabel == oe.ylabel == "$M/H$ (emu/Oe)"


def test_moment_series_field_unit_tesla(mh_measurement):
    (series,) = plot.moment_series(mh_measurement, field_unit="tesla")
    np.testing.assert_allclose(series.x, mh_measurement.field_oe.to_numpy() / OE_PER_TESLA)
    assert series.xlabel == _TESLA_LABEL
    np.testing.assert_array_equal(series.y, mh_measurement.moment_emu.to_numpy())
    (default,) = plot.moment_series(mh_measurement)
    assert default.xlabel == "Field (Oe)"


def test_moment_series_split_branches_in_tesla():
    """Dividing by a positive constant keeps every reversal (and every zero
    step), so the branches split at exactly the same rows in either unit."""
    field = np.array([0.0, 500.0, 1000.0, 500.0, 0.0, -500.0, -1000.0, -500.0, 0.0, 500.0, 1000.0])
    m = _measurement(
        **{"Magnetic Field (Oe)": field, "Moment (emu)": np.linspace(0.0, 1.0, len(field))}
    )
    oe = plot.moment_series(m, split_branches=True, label="5 K")
    tesla = plot.moment_series(m, split_branches=True, label="5 K", field_unit="tesla")
    assert [s.label for s in tesla] == [s.label for s in oe] == [
        "5 K branch 1 (up)",
        "5 K branch 2 (down)",
        "5 K branch 3 (up)",
    ]
    for o, t in zip(oe, tesla):
        np.testing.assert_allclose(t.x, o.x / OE_PER_TESLA)
        np.testing.assert_array_equal(t.y, o.y)
        assert t.xlabel == _TESLA_LABEL


def test_fisher_series_matches_a_hand_computation_in_molar_units(tmp_path, sample):
    """y is d(T chi_molar)/dT by the detector's own Savitzky-Golay routine,
    on the uniform grid it resamples to, labelled in the chi unit."""
    from calamari.transitions import smooth_derivative

    m = _toy_afm_file(tmp_path)
    (series,) = plot.fisher_series(m, units="molar", sample=sample, smooth_window=1.5)
    t = m.temperature.to_numpy(dtype=float)
    chi_molar = np.asarray(sample.chi_to_chi_molar(m.chi_real.to_numpy(dtype=float)))
    grid, _, expected = smooth_derivative(t, t * chi_molar, smooth_window=1.5)
    np.testing.assert_allclose(series.x, grid)
    np.testing.assert_allclose(series.y, expected, rtol=1e-12)
    assert series.yerr is None
    assert series.ylabel == r"$\mathrm{d}(\chi' T)/\mathrm{d}T$ (emu/(mol Oe))"
    assert series.xlabel == "Temperature (K)"
    # The molar curve is the emu/Oe curve scaled by the sample's constant.
    (raw,) = plot.fisher_series(m, smooth_window=1.5)
    assert raw.ylabel.endswith("(emu/Oe)")
    ratio = np.asarray(sample.chi_to_chi_molar(1.0))
    np.testing.assert_allclose(series.y, raw.y * ratio, rtol=1e-9)


def test_fisher_series_default_window_matches_detection(tmp_path):
    """With no window on either side, the curve drawn IS the curve the
    detector searched, and the Fisher maximum sits on its peak.

    Within half a window, not on the grid point: the candidate's location
    is a local quadratic's vertex, and this peak is lopsided (d(chi T)/dT
    drops steeply at T_N), so the vertex leans to the gentle side."""
    from calamari.transitions import detect_transitions

    m = _toy_afm_file(tmp_path)
    result = detect_transitions(m, methods=("peak", "fisher"))
    (series,) = plot.fisher_series(m)
    np.testing.assert_allclose(
        series.y, result.curves[(str(m.path), "chi_real")]["dyT_dT"], rtol=1e-12
    )
    near = (series.x > 7.0) & (series.x < 13.0)
    t_max = float(series.x[near][np.argmax(series.y[near])])
    fisher = result.robust().candidates.query("method == 'fisher' and sign == 'max'")
    half_window = result.params["per_signal"]["chi_real"]["smooth_window"] / 2
    assert abs(float(fisher["T_K"].iloc[0]) - t_max) <= half_window


def test_fisher_series_dc_moment_labels_its_provenance(tmp_path):
    m = _toy_afm_file(tmp_path, dc=True, field_oe=2000.0)
    (series,) = plot.fisher_series(m)
    assert series.ylabel == r"$\mathrm{d}(MT/H)/\mathrm{d}T$ (emu/Oe)"
    ac = _toy_afm_file(tmp_path, name="ac.dat")
    (ac_series,) = plot.fisher_series(ac)
    # The same chi behind both files: M/H recovers it, so the curves agree
    # (to the precision the file format writes values at).
    np.testing.assert_allclose(series.y, ac_series.y, rtol=1e-5)


def test_fisher_series_splits_by_frequency(tmp_path):
    m = _toy_afm_file(tmp_path, frequencies=(11.0, 997.0))
    out = plot.fisher_series(m, label="run")
    assert [s.label for s in out] == ["run f = 11 Hz", "run f = 997 Hz"]


def test_fisher_series_refusals(tmp_path):
    with pytest.raises(ValueError, match="needs a held, non-zero field"):
        plot.fisher_series(_toy_afm_file(tmp_path, name="zero.dat", dc=True, field_oe=0.0))
    with pytest.raises(ValueError, match="needs a temperature sweep.*'H-scan'"):
        plot.fisher_series(
            _measurement(
                **{
                    "Temperature (K)": np.full(_N, 5.0),
                    "Magnetic Field (Oe)": np.linspace(0.0, 5000.0, _N),
                    "Moment (emu)": np.linspace(0.0, 1e-3, _N),
                }
            )
        )
    with pytest.raises(ValueError, match="absolute temperature.*smallest temperature here is -1"):
        plot.fisher_series(
            _measurement(
                **{
                    "Temperature (K)": np.linspace(-1.0, 20.0, _N),
                    "Magnetic Field (Oe)": np.full(_N, 1000.0),
                    "AC X' (emu/Oe)": np.linspace(1.0, 2.0, _N),
                }
            )
        )
    with pytest.raises(ValueError, match="requires a Sample"):
        plot.fisher_series(_toy_afm_file(tmp_path, name="ac.dat"), units="molar")


def test_fisher_labels_render_as_mathtext(ax):
    for symbol in plot_data._FISHER_SYMBOLS.values():
        for _, unit_text, _ in plot_data._CHI_UNITS.values():
            ax.set_ylabel(f"{symbol} ({unit_text})")
            ax.figure.canvas.draw()
