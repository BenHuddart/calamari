"""Curve plotting, fitting displays and figure export."""

from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pytest

from calamari import plot
from calamari.fitting import fit, fit_windows
from calamari.sample import mu_eff_from_curie_constant
from calamari.simulate import curie_weiss_chi

from ._helpers import (
    _N,
    _TESLA_LABEL,
    _chi_vs_h_measurement,
    _line_xy,
    _measurement,
    _toy_afm_file,
)

# drawing layer


def test_plot_mh_emu_units(ax, mh_measurement):
    result = plot.plot_mh(mh_measurement, ax=ax, label="run 1")

    assert result is ax
    assert len(ax.lines) == 1
    x, y = _line_xy(ax.lines[0])
    np.testing.assert_allclose(y, mh_measurement.moment_emu.to_numpy())
    assert ax.lines[0].get_label() == "run 1"
    assert ax.get_xlabel() == "Field (Oe)"
    assert ax.get_ylabel() == "Moment (emu)"


def test_plot_mh_creates_figure_when_ax_omitted(mh_measurement):
    ax = plot.plot_mh(mh_measurement)
    try:
        assert len(ax.lines) == 1
        assert ax.figure is not None
    finally:
        plt.close(ax.figure)


def test_plot_mh_preserves_hysteresis_file_order(ax):
    """M(H) is drawn in file order, never sorted by field -- a hysteresis
    loop's two branches must not be scrambled into one zigzag."""
    field_oe = np.array([0.0, 500.0, 1000.0, 500.0, 0.0, -500.0, -1000.0])
    moment = np.array([0.0, 1.0, 2.0, 1.8, 1.5, 0.5, -2.0]) * 1e-3
    m = _measurement(**{"Magnetic Field (Oe)": field_oe, "Moment (emu)": moment})

    plot.plot_mh(m, ax=ax)
    x, y = _line_xy(ax.lines[0])
    np.testing.assert_allclose(x, field_oe)  # verbatim order
    np.testing.assert_allclose(y, moment)


def test_plot_mh_split_branches_draws_labelled_lines(ax):
    field = np.array([0.0, 1000.0, 0.0, -1000.0, 0.0, 1000.0])
    m = _measurement(
        **{"Magnetic Field (Oe)": field, "Moment (emu)": np.linspace(0, 1, 6)}
    )
    plot.plot_mh(m, ax=ax, split_branches=True)
    assert len(ax.lines) == 3
    assert "branch 1 (up)" in ax.lines[0].get_label()


def test_plot_mh_muB_per_ion_without_sample_raises(ax, mh_measurement):
    with pytest.raises(ValueError):
        plot.plot_mh(mh_measurement, ax=ax, units="muB_per_ion")


def test_plot_mh_missing_moment_column_raises(ax):
    m = _measurement(**{"Magnetic Field (Oe)": np.linspace(0, 100, 5)})
    with pytest.raises(ValueError):
        plot.plot_mh(m, ax=ax)


def test_plot_mh_draws_error_bars_when_informative(ax, mh_measurement):
    data = dict(mh_measurement.data)
    data["M. Std. Err. (emu)"] = np.full(_N, 1.0e-5)
    m = _measurement(**data)

    plot.plot_mh(m, ax=ax)
    assert len(ax.containers) == 1  # an ErrorbarContainer, not a bare line


def test_plot_mh_yerr_false_suppresses_error_bars(ax, mh_measurement):
    data = dict(mh_measurement.data)
    data["M. Std. Err. (emu)"] = np.full(_N, 1.0e-5)
    m = _measurement(**data)

    plot.plot_mh(m, ax=ax, yerr=False)
    assert len(ax.containers) == 0
    assert len(ax.lines) == 1


def test_plot_mh_all_zero_errors_treated_as_absent(ax, mh_measurement):
    """The instrument writes 0 where one scan was averaged; zero-length
    bars would add errorbar artists conveying nothing."""
    data = dict(mh_measurement.data)
    data["M. Std. Err. (emu)"] = np.zeros(_N)
    m = _measurement(**data)

    plot.plot_mh(m, ax=ax)
    assert len(ax.containers) == 0
    assert len(ax.lines) == 1


def test_plot_mt_emu_units(ax, mt_measurement):
    plot.plot_mt(mt_measurement, ax=ax)
    x, y = _line_xy(ax.lines[0])
    np.testing.assert_allclose(x, mt_measurement.temperature.to_numpy())
    np.testing.assert_allclose(y, mt_measurement.moment_emu.to_numpy())
    assert ax.get_xlabel() == "Temperature (K)"


def test_plot_chi_vs_t_real_component(ax, chi_measurement):
    result = plot.plot_chi_vs_t(chi_measurement, ax=ax)
    assert result is ax
    _, y = _line_xy(ax.lines[0])
    np.testing.assert_allclose(y, chi_measurement.chi_real.to_numpy())
    assert ax.get_ylabel() == r"$\chi'$ (emu/Oe)"


def test_plot_chi_vs_t_dc_provenance_in_label(ax, mt_measurement):
    plot.plot_chi_vs_t(mt_measurement, ax=ax)  # auto -> dc (no chi columns)
    assert ax.get_ylabel() == r"$M/H$ (emu/Oe)"


def test_plot_chi_vs_t_inverse_molar(ax, chi_measurement, sample):
    plot.plot_chi_vs_t(chi_measurement, ax=ax, units="molar", sample=sample, inverse=True)
    _, y = _line_xy(ax.lines[0])
    expected = 1.0 / sample.chi_to_chi_molar(chi_measurement.chi_real.to_numpy())
    np.testing.assert_allclose(y, expected)
    assert ax.get_ylabel() == r"$1/\chi'$ ((mol Oe)/emu)"


def test_plot_chi_vs_t_invalid_component_raises(ax, chi_measurement):
    with pytest.raises(ValueError):
        plot.plot_chi_vs_t(chi_measurement, ax=ax, component="bogus")


def test_plot_chi_vs_t_imag_label_untouched_when_none(ax, chi_measurement):
    plot.plot_chi_vs_t(chi_measurement, ax=ax, component="imag")
    assert ax.lines[0].get_label().startswith("_")  # legend-excluded, no invented note


def test_plot_chi_vs_t_imag_scale_scales_data_and_label(ax, chi_measurement):
    plot.plot_chi_vs_t(
        chi_measurement, ax=ax, component="imag", imag_scale=10.0, label="run 1"
    )
    _, y = _line_xy(ax.lines[0])
    np.testing.assert_allclose(y, 10.0 * chi_measurement.chi_imag.to_numpy())
    assert ax.lines[0].get_label() == "run 1 (x10)"


def test_plot_chi_vs_t_both_emits_two_labelled_series(ax, chi_measurement):
    plot.plot_chi_vs_t(chi_measurement, ax=ax, component="both")

    assert len(ax.lines) == 2
    _, y_real = _line_xy(ax.lines[0])
    _, y_imag = _line_xy(ax.lines[1])
    np.testing.assert_allclose(y_real, chi_measurement.chi_real.to_numpy())
    np.testing.assert_allclose(y_imag, chi_measurement.chi_imag.to_numpy())
    assert ax.lines[0].get_label() == r"$\chi'$"
    assert ax.lines[1].get_label() == r"$\chi''$"
    assert ax.get_ylabel() == r"$\chi$ (emu/Oe)"  # shared axis: the generic label


def test_plot_chi_vs_t_both_imag_scale_reflected_in_label_and_data(ax, chi_measurement):
    plot.plot_chi_vs_t(
        chi_measurement, ax=ax, component="both", imag_scale=5.0, label="run 1"
    )
    _, y_imag = _line_xy(ax.lines[1])
    assert ax.lines[0].get_label() == "run 1 " + r"$\chi'$"
    assert ax.lines[1].get_label() == "run 1 " + r"$\chi''$" + " (x5)"
    np.testing.assert_allclose(y_imag, 5.0 * chi_measurement.chi_imag.to_numpy())


def test_plot_chi_vs_h_field_axis(ax):
    m = _measurement(
        **{
            "Temperature (K)": np.full(4, 2.0),
            "Magnetic Field (Oe)": np.array([0.0, 100.0, 200.0, 300.0]),
            "AC X' (emu/Oe)": np.array([1.0, 2.0, 3.0, 4.0]),
        }
    )
    plot.plot_chi_vs_h(m, ax=ax)
    x, _ = _line_xy(ax.lines[0])
    np.testing.assert_allclose(x, [0.0, 100.0, 200.0, 300.0])
    assert ax.get_xlabel() == "Field (Oe)"


@pytest.mark.parametrize(
    "call",
    [
        lambda m, ax: plot.chi_series(m, x="field", field_unit="T"),
        lambda m, ax: plot.moment_series(m, field_unit="T"),
        lambda m, ax: plot.plot_chi_vs_h(m, ax=ax, source="dc", field_unit="T"),
        lambda m, ax: plot.plot_mh(m, ax=ax, field_unit="T"),
    ],
    ids=["chi_series", "moment_series", "plot_chi_vs_h", "plot_mh"],
)
def test_field_unit_unknown_value_refused(ax, mh_measurement, call):
    with pytest.raises(ValueError, match=r"field_unit must be one of \['oe', 'tesla'\], got 'T'"):
        call(mh_measurement, ax)
    assert len(ax.lines) == 0 and len(ax.containers) == 0


def test_field_unit_refused_on_temperature_axis(mt_measurement, chi_measurement):
    with pytest.raises(ValueError, match="applies to x='field' only"):
        plot.chi_series(chi_measurement, x="temperature", field_unit="tesla")
    with pytest.raises(ValueError, match="applies to x='field' only"):
        plot.moment_series(mt_measurement, x="temperature", field_unit="tesla")
    # The default names no unit worth refusing, as split_branches=False.
    plot.chi_series(chi_measurement, x="temperature", field_unit="oe")
    plot.moment_series(mt_measurement, x="temperature", field_unit="oe")


def test_temperature_plots_refuse_field_unit_by_name(ax, mt_measurement, chi_measurement):
    """These helpers take no field_unit; on main the keyword reached matplotlib
    and died with an AttributeError naming Line2D, not the mistake."""
    calls = [
        ("plot_chi_vs_t", lambda: plot.plot_chi_vs_t(chi_measurement, ax=ax, field_unit="tesla")),
        ("plot_mt", lambda: plot.plot_mt(mt_measurement, ax=ax, field_unit="tesla")),
        ("plot_fisher", lambda: plot.plot_fisher(chi_measurement, ax=ax, field_unit="oe")),
    ]
    for name, call in calls:
        with pytest.raises(ValueError, match=f"{name} plots against temperature.*no field_unit"):
            call()
    assert len(ax.lines) == 0 and len(ax.containers) == 0


def test_plot_chi_vs_h_field_unit_tesla(ax):
    m = _chi_vs_h_measurement()
    plot.plot_chi_vs_h(m, ax=ax, field_unit="tesla")
    x, y = _line_xy(ax.lines[0])
    np.testing.assert_allclose(x, [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_array_equal(y, m.chi_real.to_numpy())
    assert ax.get_xlabel() == _TESLA_LABEL


def test_plot_chi_vs_h_both_components_in_tesla(ax):
    m = _measurement(**dict(_chi_vs_h_measurement().data, **{"AC X'' (emu/Oe)": np.ones(5)}))
    plot.plot_chi_vs_h(m, ax=ax, component="both", field_unit="tesla")
    assert len(ax.lines) == 2
    for line in ax.lines:
        np.testing.assert_allclose(_line_xy(line)[0], [0.0, 0.25, 0.5, 0.75, 1.0])
    assert ax.get_xlabel() == _TESLA_LABEL


def test_plot_mh_field_unit_tesla_with_split_branches(ax):
    field = np.array([0.0, 5000.0, 10000.0, 5000.0, 0.0])
    m = _measurement(**{"Magnetic Field (Oe)": field, "Moment (emu)": np.arange(5.0)})
    plot.plot_mh(m, ax=ax, field_unit="tesla", split_branches=True)
    assert [line.get_label() for line in ax.lines] == ["branch 1 (up)", "branch 2 (down)"]
    np.testing.assert_allclose(_line_xy(ax.lines[0])[0], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(_line_xy(ax.lines[1])[0], [1.0, 0.5, 0.0])
    assert ax.get_xlabel() == _TESLA_LABEL


def test_plot_chi_vs_t_multifrequency_two_series(ax):
    t = np.tile(np.array([10.0, 20.0, 30.0]), 2)
    m = _measurement(
        **{
            "Temperature (K)": t,
            "AC X' (emu/Oe)": np.linspace(1.0, 6.0, 6),
            "AC Frequency (Hz)": np.repeat([97.0, 997.0], 3),
        }
    )
    plot.plot_chi_vs_t(m, ax=ax)
    assert len(ax.lines) == 2
    assert ax.lines[0].get_label() == "f = 97 Hz"
    assert ax.lines[1].get_label() == "f = 997 Hz"

# plot_fit


def test_plot_fit_draws_dense_curve(ax):
    x = np.linspace(0.0, 10.0, 20)
    y = 2.5 * x - 1.0
    result = fit("linear", x, y)
    assert result.success

    returned = plot.plot_fit(x, result, ax=ax, n_dense=50, label="fit")

    assert returned is ax
    assert len(ax.lines) == 1
    x_dense, y_dense = _line_xy(ax.lines[0])
    assert len(x_dense) == 50 and len(y_dense) == 50
    assert x_dense[0] == pytest.approx(0.0)
    assert x_dense[-1] == pytest.approx(10.0)
    np.testing.assert_allclose(y_dense, 2.5 * x_dense - 1.0, atol=1e-6)
    assert ax.lines[0].get_label() == "fit"

    # plot_fit is a pure overlay: it must not touch axis labels/limits.
    assert ax.get_xlabel() == "" and ax.get_ylabel() == ""


def test_curie_weiss_workflow_composes(ax, chi_measurement, sample):
    """The two-line Curie-Weiss analysis the inverse flag exists for."""
    plot.plot_chi_vs_t(chi_measurement, ax=ax, units="molar", sample=sample, inverse=True)
    inv_chi = 1.0 / sample.chi_to_chi_molar(chi_measurement.chi_real.to_numpy())
    result = fit("linear", chi_measurement.temperature.to_numpy(), inv_chi)
    plot.plot_fit(chi_measurement.temperature, result, ax=ax, label="Curie-Weiss")
    assert len(ax.lines) == 2
    assert ax.get_ylabel() == r"$1/\chi'$ ((mol Oe)/emu)"

# plot_fit_windows


def _window_scan():
    """A Curie-Weiss window scan with one underdetermined window in it."""
    T = np.linspace(10.0, 300.0, 59)
    rng = np.random.default_rng(41)
    chi = curie_weiss_chi(T, 0.5, -20.0, 2e-4) * (1 + 0.005 * rng.standard_normal(T.size))
    windows = [(20.0, 300.0), (60.0, 300.0), (290.0, 295.0), (100.0, 250.0)]
    return fit_windows("curie_weiss", T, chi, windows=windows)


def test_plot_fit_windows_draws_estimates_with_error_bars(ax):
    scan = _window_scan()
    ok = scan.table[scan.table["success"]]
    assert len(ok) == 3  # the (290, 295) window holds two points: a failure row

    returned = plot.plot_fit_windows(scan, "theta", ax=ax, marker="s", label="scan")

    assert returned is ax
    assert len(ax.containers) == 1
    container = ax.containers[0]
    data_line, _, (bars,) = container.lines
    x, y = _line_xy(data_line)
    np.testing.assert_array_equal(x, ok["x_min"].to_numpy())
    np.testing.assert_array_equal(y, ok["theta"].to_numpy())
    segments = bars.get_segments()
    assert len(segments) == len(ok)
    for (x0, lo), (x1, hi) in (tuple(map(tuple, s)) for s in segments):
        i = int(np.flatnonzero(ok["x_min"].to_numpy() == x0)[0])
        assert x1 == x0
        assert lo == pytest.approx(ok["theta"].iloc[i] - ok["theta_err"].iloc[i])
        assert hi == pytest.approx(ok["theta"].iloc[i] + ok["theta_err"].iloc[i])
    # **style reaches the estimates.
    assert data_line.get_marker() == "s"
    assert container.get_label() == "scan"
    assert ax.get_xlabel() == "Fit window lower bound"
    assert ax.get_ylabel() == "theta"


def test_plot_fit_windows_marks_failed_windows_without_a_point(ax):
    scan = _window_scan()
    plot.plot_fit_windows(scan, "C", ax=ax, x="x_max", color="C3")
    failed_x = scan.table.loc[~scan.table["success"], "x_max"].to_numpy()
    data_line = ax.containers[0].lines[0]
    assert not np.isin(failed_x, data_line.get_xdata()).any()
    markers = [line for line in ax.lines if line is not data_line]
    assert len(markers) == len(failed_x) == 1
    np.testing.assert_array_equal(markers[0].get_xdata(), [failed_x[0]] * 2)
    assert markers[0].get_linestyle() == ":"
    # The failure marker ignores **style and stays out of the legend.
    assert markers[0].get_color() != "C3"
    assert markers[0].get_label().startswith("_")
    assert ax.get_xlabel() == "Fit window upper bound"


def test_plot_fit_windows_all_failed_draws_markers_only(ax):
    T = np.linspace(10.0, 300.0, 59)
    scan = fit_windows(
        "curie_weiss", T, curie_weiss_chi(T, 0.5, -20.0), windows=[(10.0, 15.0), (20.0, 25.0)]
    )
    plot.plot_fit_windows(scan, "theta", ax=ax)
    assert ax.containers == []
    assert len(ax.lines) == 2
    assert ax.get_ylabel() == "theta"


def test_plot_fit_windows_refuses_unknown_parameter_or_edge():
    scan = _window_scan()
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="not a parameter.*'C', 'theta', 'chi0'"):
        plot.plot_fit_windows(scan, "mu_eff")
    with pytest.raises(ValueError, match="not a parameter"):
        plot.plot_fit_windows(scan, "x_min")
    with pytest.raises(ValueError, match="x must be one of"):
        plot.plot_fit_windows(scan, "theta", x="n_points")
    assert plt.get_fignums() == before  # a refused call leaks no figure


def test_plot_fit_windows_plots_a_derived_quantity(ax):
    # A derived column has an _err partner like any parameter, so it plots
    # the same way, with its propagated error as the bar.
    T = np.linspace(10.0, 300.0, 59)
    rng = np.random.default_rng(41)
    chi = curie_weiss_chi(T, 0.5, -20.0, 2e-4) * (1 + 0.005 * rng.standard_normal(T.size))
    windows = [(20.0, 300.0), (60.0, 300.0), (290.0, 295.0), (100.0, 250.0)]
    scan = fit_windows(
        "curie_weiss", T, chi, windows=windows,
        derived={"mu_eff": lambda p: mu_eff_from_curie_constant(p["C"])},
    )
    ok = scan.table[scan.table["success"]]
    plot.plot_fit_windows(scan, "mu_eff", ax=ax)
    data_line, _, (bars,) = ax.containers[0].lines
    np.testing.assert_array_equal(_line_xy(data_line)[1], ok["mu_eff"].to_numpy())
    segments = bars.get_segments()
    assert len(segments) == len(ok)
    for segment, value, err in zip(segments, ok["mu_eff"], ok["mu_eff_err"]):
        (_, lo), (_, hi) = segment
        assert err > 0.0
        assert lo == pytest.approx(value - err)
        assert hi == pytest.approx(value + err)
    assert len(ax.lines) == 2  # the estimates and one failed-window marker
    assert ax.get_ylabel() == "mu_eff"
    # An unknown name lists both kinds of plottable column.
    with pytest.raises(
        ValueError, match=r"parameters are \['C', 'theta', 'chi0'\], derived quantities are "
        r"\['mu_eff'\]"
    ):
        plot.plot_fit_windows(scan, "mu")


def test_plot_fit_windows_renders_to_png():
    scan = _window_scan()
    ax = plot.plot_fit_windows(scan, "chi0", capsize=2)
    try:
        buffer = BytesIO()
        ax.figure.savefig(buffer, format="png")
        assert buffer.getbuffer().nbytes > 0
    finally:
        plt.close(ax.figure)

# rendering smoke test


def test_full_figure_renders_to_png(ax, chi_measurement, mh_measurement, sample):
    """Force a real draw of every label path (mathtext included): a label
    string matplotlib cannot parse only fails at render time."""
    plot.plot_chi_vs_t(
        chi_measurement, ax=ax, component="both", imag_scale=10.0, label="run"
    )
    ax.legend()
    ax2 = plot.plot_mh(mh_measurement, sample=sample, units="muB_per_ion")
    try:
        for figure in (ax.figure, ax2.figure):
            buffer = BytesIO()
            figure.savefig(buffer, format="png")
            assert buffer.getbuffer().nbytes > 0
    finally:
        plt.close(ax2.figure)

# component='both' requires AC data.


def test_plot_chi_both_on_dc_file_raises_before_drawing(ax, mt_measurement):
    """Regression: 'both' on a DC-only file used to draw the real series
    (M/H data under an AC chi' legend label) and then raise on the imag
    call, stranding half-drawn, mislabelled axes."""
    with pytest.raises(ValueError):
        plot.plot_chi_vs_t(mt_measurement, ax=ax, component="both")
    assert len(ax.lines) == 0  # nothing stranded


def test_plot_chi_both_refuses_source_dc(ax, chi_measurement):
    with pytest.raises(ValueError, match="requires AC data"):
        plot.plot_chi_vs_t(chi_measurement, ax=ax, component="both", source="dc")
    assert len(ax.lines) == 0


def test_plot_chi_both_resolves_as_ac_on_mixed_file(ax, chi_measurement):
    """A file carrying AC chi columns AND a DC moment must not trip the
    AC-versus-DC auto-refusal for 'both': DC could never supply the pair."""
    data = dict(chi_measurement.data)
    data["Moment (emu)"] = np.full(_N, 1.0e-3)
    data["Magnetic Field (Oe)"] = np.full(_N, 100.0)
    plot.plot_chi_vs_t(_measurement(**data), ax=ax, component="both")
    assert len(ax.lines) == 2
    assert ax.lines[0].get_label() == r"$\chi'$"


def test_plot_chi_both_missing_imag_raises_before_drawing(ax):
    """chi' present, chi'' absent: resolve-before-draw means clean failure."""
    m = _measurement(
        **{
            "Temperature (K)": np.linspace(2.0, 30.0, 5),
            "AC X' (emu/Oe)": np.full(5, 0.05),
        }
    )
    with pytest.raises(ValueError):
        plot.plot_chi_vs_t(m, ax=ax, component="both")
    assert len(ax.lines) == 0


def test_plot_fisher_composes_with_overlay_and_renders_to_png(tmp_path, sample):
    from calamari.transitions import detect_transitions

    m = _toy_afm_file(tmp_path)
    result = detect_transitions(m, methods=("peak", "fisher")).robust()
    window = result.params["per_signal"]["chi_real"]["smooth_window"]
    ax = plot.plot_fisher(
        m, smooth_window=window, units="molar", sample=sample, label="toy", color="C1"
    )
    try:
        (line,) = ax.lines
        assert line.get_label() == "toy"
        assert ax.get_ylabel() == r"$\mathrm{d}(\chi' T)/\mathrm{d}T$ (emu/(mol Oe))"
        artists = plot.overlay_transitions(result, ax=ax, x="temperature")
        assert {key[0] for key in artists} == {"peak", "fisher"}
        buffer = BytesIO()
        ax.figure.savefig(buffer, format="png")
        assert buffer.getbuffer().nbytes > 0
    finally:
        plt.close(ax.figure)


def test_plot_fisher_refusal_leaks_no_figure(tmp_path):
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        plot.plot_fisher(_toy_afm_file(tmp_path, dc=True, field_oe=0.0))
    assert plt.get_fignums() == before
