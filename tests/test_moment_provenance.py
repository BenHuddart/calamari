"""Resolved moment provenance, using only invented synthetic measurements."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from calamari import detect_transitions, detect_transitions_all, find_transitions, load, simulate
from calamari.exceptions import AmbiguousColumnError
from calamari.scales import scan_smoothing


@pytest.fixture
def moment_views(tmp_path):
    t = np.linspace(2.0, 30.0, 141)
    y = np.exp(-((t - 9.0) / 2.0) ** 2)
    path = simulate.write_mpms3_file(
        tmp_path / "synthetic.dat", app="SQUID DC", temperature_K=t,
        field_oe=np.full_like(t, 1000.0), dc_moment_emu=y,
        dc_moment_err_emu=np.full_like(t, 0.01),
        extra_columns={
            "DC Moment Fixed Ctr (emu)": y[::-1],
            "DC Moment Err Fixed Ctr (emu)": np.full_like(t, 0.02),
        },
    )
    m = load(path)
    return m, [m.with_moment(base) for base in ("DC Moment Free Ctr", "DC Moment Fixed Ctr")]


def _detect(m, **kwargs):
    return detect_transitions(
        m, signal="moment_emu", methods="peak", outliers="ignore", noise=0.01,
        smooth_window=1.0, **kwargs,
    )


def _source(column, unit="emu", error=None):
    return {"column": column, "native_unit": unit, "canonical_unit": "emu", "error": error}


def _batch_sources(result):
    return {
        entry["sweep_id"]: entry["per_signal"]["moment_emu"]["moment_source"]
        for entry in result.params["per_sweep"]
    }


def test_distinct_moment_sources_leave_detection_unchanged(moment_views):
    m, views = moment_views
    before = m.data.copy(deep=True)
    header = deepcopy(m.header)
    for view, centre, base in zip(
        views, (9.0, 23.0), ("Free", "Fixed"), strict=True,
    ):
        result = _detect(view)
        record = result.params["per_signal"]["moment_emu"]["moment_source"]
        assert record == _source(
            f"DC Moment {base} Ctr (emu)", error={
                "column": f"DC Moment Err {base} Ctr (emu)",
                "native_unit": "emu", "canonical_unit": "emu",
            },
        )
        # Allow roundoff at the planted centre, far below the 0.2 K grid step.
        assert result.candidates["T_K"].tolist() == pytest.approx([centre], rel=0, abs=1e-12)
        # The same arrays at the array-level boundary reproduce coordinates,
        # errors and scores: provenance must not alter the calculation.
        expected = find_transitions(
            view.temperature, view.moment_emu, y_err=view.moment_err_emu,
            methods="peak", noise=0.01, smooth_window=1.0, x_name="T", x_unit="K",
        )
        for column in ("T_K", "T_K_err", "significance"):
            actual = "T_err_K" if column == "T_K_err" else column
            np.testing.assert_array_equal(result.candidates[actual], expected.candidates[column])
    ambiguous = replace(m, data=m.data.assign(**{"Moment (emu)": views[0].moment_emu}))
    with pytest.raises(AmbiguousColumnError):
        _detect(ambiguous)
    pd.testing.assert_frame_equal(m.data, before)
    assert m.header == header
    assert m.moment_base is None
    original = _detect(views[0])
    params = deepcopy(original.params)
    chosen = original.robust(min_significance=0).select(
        swept_axis="T", method="peak", sign="max", t_window_k=(5.0, 15.0),
    ).link(t_tol_k=1.0)
    assert chosen.all().params["per_signal"] == params["per_signal"]
    assert original.params == params


@pytest.mark.parametrize("unit", ["emu", "Am2"])
@pytest.mark.parametrize("explicit", [False, True])
def test_native_units_and_automatic_resolution(tmp_path, unit, explicit):
    t = np.linspace(2.0, 30.0, 141)
    y = np.exp(-((t - 9.0) / 2.0) ** 2)
    factor = 1000.0 if unit == "Am2" else 1.0
    path = simulate.write_mpms3_file(
        tmp_path / "units.dat", app="SQUID DC", unit_system=unit,
        temperature_K=t, field_oe=np.full_like(t, 0.1 if unit == "Am2" else 1000.0),
        dc_moment_emu=y / factor, dc_moment_err_emu=np.full_like(t, 0.01 / factor),
    )
    m = load(path)
    if explicit:
        m = m.with_moment("DC Moment Free Ctr")
    result = _detect(m)
    assert result.params["per_signal"]["moment_emu"]["moment_source"] == _source(
        f"DC Moment Free Ctr ({unit})", unit, error={
            "column": f"DC Moment Err Free Ctr ({unit})",
            "native_unit": unit, "canonical_unit": "emu",
        },
    )
    expected = find_transitions(
        t, y, y_err=np.full_like(t, 0.01), methods="peak", noise=0.01,
        smooth_window=1.0, x_name="T", x_unit="K",
    )
    assert result.candidates["H_T"].tolist() == [0.1]
    # The synthetic writer rounds exported values; compare with the ideal
    # input within that precision rather than claiming bitwise equality.
    for column, actual in (("T_K", "T_K"), ("T_K_err", "T_err_K"),
                           ("significance", "significance")):
        np.testing.assert_allclose(result.candidates[actual], expected.candidates[column], rtol=1e-9)


def test_missing_units_and_errors_are_explicit(moment_views):
    _, views = moment_views
    m = views[0]
    # A synthetic, hand-edited header with no suffix uses the accessor's
    # established canonical-unit assumption; provenance must retain absence.
    m.data = m.data.rename(columns={"DC Moment Free Ctr (emu)": "DC Moment Free Ctr"})
    m.data["DC Moment Err Free Ctr (emu)"] = np.nan
    result = _detect(m)
    assert result.params["per_signal"]["moment_emu"]["moment_source"] == _source(
        "DC Moment Free Ctr", None,
    )


def test_batch_and_transformations_preserve_each_view(moment_views):
    _, views = moment_views
    result = detect_transitions_all(
        views, signal="moment_emu", methods="peak", outliers="ignore", noise=0.01,
        smooth_window=1.0,
    )
    expected = {i: _detect(view).params["per_signal"]["moment_emu"]["moment_source"]
                for i, view in enumerate(views)}
    assert expected[0] != expected[1]
    before = deepcopy(result.params)
    assert result.candidates["T_K"].tolist() == pytest.approx([9.0, 23.0], rel=0, abs=1e-12)
    assert result.candidates["measurement_file"].nunique() == 1
    filtered = result.robust(min_significance=1e10)
    assert filtered.candidates.empty
    selected = result.select(swept_axis="T", method="peak", sign="max", t_window_k=(5.0, 25.0))
    linked = selected.link(t_tol_k=1.0).robust(min_significance=0)
    for transformed in (result, filtered, filtered.all(), selected, linked, linked.all()):
        assert _batch_sources(transformed) == expected
    assert result.params == before


def test_smoothing_lookup_and_reference_preserve_sources(moment_views):
    _, views = moment_views
    scan = scan_smoothing(
        views, windows=(1.0, 1.5, 2.0), track_tol_k=0.5, signal="moment_emu",
        methods="peak", noise=0.01, outliers="ignore",
    )
    before = {window: deepcopy(result.params) for window, result in scan.results.items()}
    expected = _batch_sources(scan.at(1.0))
    for window in (1.0, 1.5, 2.0):
        assert _batch_sources(scan.at(window)) == expected
    reference = scan.reference(1.5)
    assert _batch_sources(reference) == expected
    assert _batch_sources(reference.robust(min_significance=0).all()) == expected
    for window, record in reference.params["smoothing"]["results"].items():
        assert record["params"]["per_sweep"] == before[window]["per_sweep"]
    assert {window: result.params for window, result in scan.results.items()} == before


def test_ac_and_array_searches_have_no_moment_source(tmp_path):
    t = np.linspace(2.0, 30.0, 141)
    y = np.exp(-((t - 9.0) / 2.0) ** 2)
    m = load(simulate.write_mpms3_file(
        tmp_path / "ac.dat", app="SQUID AC", temperature_K=t,
        field_oe=np.full_like(t, 1000.0), chi_real=y,
        ac_frequency_hz=np.full_like(t, 997.0),
    ))
    ac = detect_transitions(m, signal="chi_real", outliers="ignore", noise=0.01)
    assert "moment_source" not in ac.params["per_signal"]["chi_real"]
    assert "moment_source" not in find_transitions(t, y, methods="peak").params


def test_mpms3_source_spelling_and_unitless_error(tmp_path):
    t = np.linspace(2.0, 30.0, 141)
    y = np.exp(-((t - 9.0) / 2.0) ** 2)
    m = load(simulate.write_mpms3_file(
        tmp_path / "mpms3.dat", app="SQUID DC", temperature_K=t,
        field_oe=np.full_like(t, 1000.0), dc_moment_emu=y,
    ))
    m.data = m.data.rename(columns={"DC Moment Free Ctr (emu)": "DC Moment Free Ctr(EMU)"})
    m.data["DC Moment Err Free Ctr"] = np.full_like(t, 0.01)
    record = _detect(m).params["per_signal"]["moment_emu"]["moment_source"]
    assert record == _source("DC Moment Free Ctr(EMU)", "EMU", error={
        "column": "DC Moment Err Free Ctr", "native_unit": None, "canonical_unit": "emu",
    })
