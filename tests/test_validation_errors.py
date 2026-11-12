"""Deliberate validation is catchable without changing built-in compatibility."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import calamari
from calamari import plot
from calamari.exceptions import (
    CalamariError,
    FitParameterError,
    PlotInputError,
    PlotTypeError,
    QualityInputError,
    UnknownModelError,
)


@pytest.mark.parametrize(
    "kwargs", [{"initial": {"typo": 1}}, {"fixed": {"typo": 1}}, {"limits": {"typo": (0, 1)}}]
)
@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_unknown_fit_parameter_is_a_calamari_key_error(kwargs, windowed, empty):
    x = np.array([]) if empty else np.arange(5.0)
    call = calamari.fit_windows if windowed else calamari.fit
    if windowed:
        kwargs = {**kwargs, "windows": [(0, 4)]}
    with pytest.raises(FitParameterError, match="typo") as caught:
        call("linear", x, 2 * x + 1, **kwargs)
    assert isinstance(caught.value, CalamariError) and isinstance(caught.value, KeyError)


@pytest.mark.parametrize("windowed", [False, True])
def test_unknown_model_is_a_calamari_key_error(windowed):
    call = calamari.fit_windows if windowed else calamari.fit
    kwargs = {"windows": [(0, 1)]} if windowed else {}
    with pytest.raises(UnknownModelError, match="no_such_model") as caught:
        call("no_such_model", [], [], **kwargs)
    assert isinstance(caught.value, CalamariError) and isinstance(caught.value, KeyError)


@pytest.fixture
def measurement(mpms3_ac_file):
    return calamari.load(mpms3_ac_file.path)


@pytest.mark.parametrize(
    "operation",
    [
        lambda m: plot.axis_labels("typo", "k"),
        lambda m: plot.chi_series(m, component="typo"),
        lambda m: plot.moment_series(m, x="typo"),
        lambda m: plot.fisher_series(m, units="typo"),
        lambda m: plot.chi_map_grid([m], family="typo"),
        lambda m: plot.moment_map_grid([m], family="typo"),
        lambda m: plot.plot_family([], plot.plot_chi_vs_t),
        lambda m: plot.plot_chi_vs_t(m, component="typo"),
        lambda m: plot.plot_chi_vs_t_family([m], offset=float("nan")),
        lambda m: plot.plot_detection_curve(None, representation="typo"),
        lambda m: plot.plot_boundary(None, which="typo"),
    ],
)
def test_plot_validation_is_a_calamari_value_error(measurement, operation):
    with pytest.raises(PlotInputError) as caught:
        operation(measurement)
    assert isinstance(caught.value, CalamariError) and isinstance(caught.value, ValueError)


def test_shared_outlier_policy_is_catchable_from_plotting(measurement):
    with pytest.raises(QualityInputError, match="outliers must be") as caught:
        plot.plot_chi_vs_t_family([measurement], outliers=None)
    assert isinstance(caught.value, CalamariError) and isinstance(caught.value, ValueError)


@pytest.mark.parametrize("kwargs", [{}, {"format": "png"}])
def test_save_type_errors_keep_builtin_compatibility(tmp_path, kwargs):
    with pytest.raises(PlotTypeError) as caught:
        plot.save(object(), tmp_path / "figure.png", **kwargs)
    assert isinstance(caught.value, CalamariError) and isinstance(caught.value, TypeError)
    assert not list(tmp_path.iterdir())


def test_third_party_drawing_and_filesystem_errors_propagate(tmp_path, monkeypatch):
    fig, ax = plt.subplots()
    try:
        failure = ValueError("renderer failure")

        def reject(*args, **kwargs):
            raise failure

        monkeypatch.setattr(ax, "plot", reject)
        with pytest.raises(ValueError) as caught:
            plot.plot_fit([0, 1], calamari.fit("linear", np.arange(5.0), np.arange(5.0)), ax=ax)
        assert caught.value is failure
        assert not isinstance(caught.value, CalamariError)
        obstruction = tmp_path / "file"
        obstruction.write_text("obstruction")
        with pytest.raises(FileExistsError) as caught:
            plot.save(fig, obstruction / "figure.pdf")
        assert not isinstance(caught.value, CalamariError)
    finally:
        plt.close(fig)


def test_caller_model_and_derived_errors_propagate():
    from calamari.fitting import ModelDefinition

    failure = KeyError("caller callback")

    def reject(*args):
        raise failure

    x = np.arange(5.0)
    result = calamari.fit("linear", x, x)
    with pytest.raises(KeyError) as caught:
        result.derive(reject)
    assert caught.value is failure
    with pytest.raises(KeyError) as caught:
        calamari.fit(
            ModelDefinition("caller", reject, ("slope",), {"slope": 1.0}, "Synthetic failing model"),
            x,
            x,
        )
    assert caught.value is failure
    sparse = calamari.fit("linear", [], [])
    assert not sparse.success and sparse.n_points == 0


@pytest.mark.parametrize(
    "operation",
    [
        lambda m: plot.fisher_series(m, smooth_window=-1),
        lambda m: plot.plot_detection_curve(
            calamari.detect_transitions(m, methods=("peak",), outliers="ignore"), sweep_id=100
        ),
    ],
)
def test_shared_curve_preparation_and_selection_are_catchable(measurement, operation):
    from calamari.exceptions import CurveInputError

    with pytest.raises(CurveInputError) as caught:
        operation(measurement)
    assert isinstance(caught.value, CalamariError) and isinstance(caught.value, ValueError)


def test_mismatched_quality_report_keeps_plot_context(measurement, tmp_path):
    from dataclasses import replace

    report = measurement.quality()
    other = replace(measurement, path=tmp_path / "other_synthetic.dat")
    with pytest.raises(PlotInputError, match=r"outliers\[0\]") as caught:
        plot.plot_chi_vs_t_family([other], outliers=[report])
    assert isinstance(caught.value.__cause__, QualityInputError)


def test_grid_preparation_and_rendering_validation(measurement, mpms3_ac_file, tmp_path):
    source = mpms3_ac_file
    path = calamari.simulate.write_mpms3_file(
        tmp_path / "second_synthetic.dat",
        temperature_K=source.temperature_K,
        field_oe=source.field_oe + 1000,
        chi_real=source.chi_real,
        ac_frequency_hz=source.frequency_hz,
    )
    grid = plot.chi_map_grid([measurement, calamari.load(path)], outliers="ignore")
    with pytest.raises(PlotInputError, match="family_scale"):
        plot.draw_map(grid, family_scale="typo")
    with pytest.raises(PlotInputError):
        plot.interpolate_family(grid, max_gap=-1)


def test_unexpected_report_validator_error_propagates(measurement, monkeypatch):
    from calamari.plot import _families

    report = measurement.quality()
    failure = ValueError("upstream failure")

    def reject(*args):
        raise failure

    monkeypatch.setattr(_families, "_check_report_matches", reject)
    with pytest.raises(ValueError) as caught:
        plot.plot_chi_vs_t_family([measurement], outliers=[report])
    assert caught.value is failure
    assert not isinstance(caught.value, CalamariError)

@pytest.mark.parametrize("operation", ["model", "sweep", "smoothing"])
def test_analysis_validation_keeps_common_and_builtin_handlers(operation):
    from calamari.fitting import MODELS, register_model
    from calamari.sweeps import segment_by_constant

    calls = {
        "model": lambda: register_model(MODELS["linear"]),
        "sweep": lambda: segment_by_constant([1.0, 2.0], tol=-1.0),
        "smoothing": lambda: calamari.scan_smoothing([], windows=(1.0, -1.0, 2.0)),
    }
    with pytest.raises(CalamariError) as caught:
        calls[operation]()
    assert isinstance(caught.value, ValueError)
