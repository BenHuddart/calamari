"""Tests for calamari.quality: per-point screening of a measurement.

Synthetic AC/DC and VSM/AC scans include a planted anomaly with a manifest.
Metadata covers nonzero statuses, harmless status departures and range changes;
a fixed "nonzero = bad" policy would fail these cases. All values are invented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import calamari
from calamari import _plot_data as plot_data
from calamari import quality, simulate

#: Fixed seed for the whole campaign; the manifest records it.
_SEED = 20260927

#: Arbitrary relative signal scale of each piece -- the two campaigns are
#: nominally equivalent, differing only by how much material each piece holds.
#: Invented numbers; no sample mass is written anywhere.
_SCALES = {"piece_a": 1.0, "piece_b": 0.62}

#: Descending settled setpoints, ~2-60 K at a nominal 0.5 K.
_T_SETPOINTS = np.arange(60.0, 1.99, -0.5)

#: The field scan: fixed temperature, 0-7 T in 250 Oe steps.
_FIELD_SCAN_T = 2.0
_FIELD_SETPOINTS = np.arange(0.0, 70_000.1, 250.0)

#: Where the field scan's smooth chi'(H) has its steep (but real, and smooth)
#: step -- a transition the screen must NOT flag -- and its width.
_STEP_FIELD_OE = 32_000.0
_STEP_WIDTH_OE = 600.0

#: AC row (0-based, among the field scan's AC rows) given the injected anomaly.
_ANOMALY_AC_INDEX = 137


def _spreads(rng, n):
    """Within-point temperature spreads: 1-5 mK, occasionally 10-25 mK."""
    spread = rng.uniform(0.001, 0.005, n)
    wide = rng.random(n) < 0.04
    spread[wide] = rng.uniform(0.010, 0.025, wide.sum())
    return spread


def _bracket(rng, centre, spread):
    """Min/max around each average: min <= average <= max, max - min = spread."""
    below = rng.uniform(0.2, 0.8, len(centre)) * spread
    return centre - below, centre - below + spread


def _timestamps(rng, n, start=3.8e9):
    """Adjacent points 1-3 minutes apart."""
    return start + np.cumsum(rng.uniform(60.0, 180.0, n))


def _chi_of_T(T, scale):
    """A smooth antiferromagnet-like chi'(T) in emu/Oe: Curie-Weiss above a
    rounded cusp near 20 K."""
    curie_weiss = 2.0e-3 / (T + 5.0)
    below = 2.0e-3 / 25.0 * (0.8 + 0.2 * (T / 20.0) ** 2)
    weight = 0.5 * (1 + np.tanh((T - 20.0) / 1.0))
    return scale * (weight * curie_weiss + (1 - weight) * below)


def _chi_of_H(H, scale):
    """A smooth chi'(H) at fixed T with one steep real step (a metamagnetic
    transition) -- shaped so that its neighbours disagree across the step."""
    base = 9.0e-5 / (1 + (H / 40_000.0) ** 2)
    step = 3.0e-5 * 0.5 * (1 + np.tanh((H - _STEP_FIELD_OE) / _STEP_WIDTH_OE))
    return scale * (base + step)


def _write_temperature_scan(path, rng, scale):
    """Settled, descending T-scan with an AC row and a DC row per setpoint."""
    n_set = len(_T_SETPOINTS)
    T = np.repeat(_T_SETPOINTS, 2) + rng.normal(0, 0.0005, 2 * n_set)
    is_ac = np.tile([True, False], n_set)
    n = len(T)
    H = np.full(n, 1000.0)
    t_lo, t_hi = _bracket(rng, T, _spreads(rng, n))

    chi = _chi_of_T(T, scale)
    chi_err = 0.002 * chi
    chi_meas = chi + rng.normal(0, 1, n) * chi_err
    moment = chi * H
    moment_err = 0.002 * np.abs(moment)
    moment_meas = moment + rng.normal(0, 1, n) * moment_err
    nan = np.full(n, np.nan)

    simulate.write_mpms3_file(
        path,
        app="MPMS3 w/ AC,1.0,1.1",
        sample_material="SyntheticMaterial",
        sample_mass_mg=None,
        temperature_K=T,
        field_oe=H,
        dc_moment_emu=np.where(is_ac, np.nan, moment_meas),
        dc_moment_err_emu=np.where(is_ac, np.nan, moment_err),
        chi_real=np.where(is_ac, chi_meas, np.nan),
        chi_imag=np.where(is_ac, 0.05 * chi_meas, np.nan),
        ac_frequency_hz=np.where(is_ac, 997.0, np.nan),
        transport_action=np.where(is_ac, 5.0, 6.0),
        extra_columns={
            "Time Stamp (sec)": _timestamps(rng, n),
            "Min. Temperature (K)": t_lo,
            "Max. Temperature (K)": t_hi,
            "Min. Field (Oe)": H,
            "Max. Field (Oe)": H,
            "Measurement Number": np.ones(n),
            "AC X' Std Err. (emu/Oe)": np.where(is_ac, chi_err, np.nan),
            "AC Range": np.where(is_ac, 10.0, np.nan),
            "Range": np.where(is_ac, np.nan, 1000.0),
            "DC Free Fit": np.where(is_ac, np.nan, rng.uniform(0.975, 0.983, n)),
            "DC Fixed Fit": np.where(is_ac, np.nan, rng.uniform(0.94, 0.95, n)),
            "Measure Status (code)": np.where(is_ac, np.nan, 0.0),
            "SQUID Status (code)": np.where(is_ac, np.nan, 0.0),
            "Temp. Status (code)": np.ones(n),
            "Field Status (code)": np.ones(n),
            "Chamber Status (code)": np.ones(n),
            "Unused Mode Column": nan,
        },
    )
    return path


def _write_field_scan(path, rng, scale, *, anomaly):
    """Combined VSM + AC field scan at fixed T; returns the anomaly's row label
    in ``data`` (or None)."""
    n_set = len(_FIELD_SETPOINTS)
    H = np.repeat(_FIELD_SETPOINTS, 2)
    is_ac = np.tile([False, True], n_set)
    n = len(H)
    T = np.full(n, _FIELD_SCAN_T) + rng.normal(0, 0.0003, n)
    t_lo, t_hi = _bracket(rng, T, _spreads(rng, n))

    chi = _chi_of_H(H, scale)
    chi_err = 0.003 * chi
    # The observed sweep-start effect: the first AC point's error is tens of
    # times its neighbours' -- evidence a screen must not treat as a gate.
    chi_err[1] *= 30.0
    chi_meas = chi + rng.normal(0, 1, n) * chi_err / np.where(np.arange(n) == 1, 30.0, 1.0)

    moment = scale * 6.0 * np.tanh(H / 20_000.0)
    moment_err = 0.002 * np.maximum(np.abs(moment), 1e-3)
    moment_meas = moment + rng.normal(0, 1, n) * moment_err

    ac_rows = np.flatnonzero(is_ac)
    anomaly_row = None
    if anomaly:
        # An isolated interior point whose chi' changes sign, with a visibly
        # larger error than its neighbours and NO status flag.
        anomaly_row = int(ac_rows[_ANOMALY_AC_INDEX])
        chi_meas[anomaly_row] = -0.6 * chi[anomaly_row]
        chi_err[anomaly_row] *= 8.0

    measure_status = np.where(is_ac, np.nan, 1.0)
    squid_status = np.where(is_ac, np.nan, 0.0)
    measure_count = np.where(is_ac, np.nan, 28.0)
    # The observed first-point departure: status codes off the file's usual
    # values on a point whose moment is fine.
    measure_status[0], squid_status[0], measure_count[0] = 0.0, 68.0, 27.0

    simulate.write_mpms3_file(
        path,
        app="MPMS3 w/ AC,1.0,1.1",
        sample_material="SyntheticMaterial",
        sample_mass_mg=None,
        temperature_K=T,
        field_oe=H,
        vsm_moment_emu=np.where(is_ac, np.nan, moment_meas),
        vsm_moment_err_emu=np.where(is_ac, np.nan, moment_err),
        chi_real=np.where(is_ac, chi_meas, np.nan),
        chi_imag=np.where(is_ac, 0.05 * np.abs(chi_meas), np.nan),
        ac_frequency_hz=np.where(is_ac, 113.0, np.nan),
        transport_action=np.where(is_ac, 5.0, 1.0),
        extra_columns={
            "Time Stamp (sec)": _timestamps(rng, n),
            "Min. Temperature (K)": t_lo,
            "Max. Temperature (K)": t_hi,
            "Min. Field (Oe)": H,
            "Max. Field (Oe)": H,
            "Measurement Number": np.ones(n),
            "AC X' Std Err. (emu/Oe)": np.where(is_ac, chi_err, np.nan),
            "AC Range": np.where(is_ac, 10.0, np.nan),
            "Range": np.where(is_ac, np.nan, 1000.0),
            "Measure Status (code)": measure_status,
            "SQUID Status (code)": squid_status,
            "Measure Count": measure_count,
            "Temp. Status (code)": np.ones(n),
            "Field Status (code)": np.ones(n),
            "Chamber Status (code)": np.ones(n),
        },
    )
    return anomaly_row


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    """Two synthetic campaigns and the manifest of what is deliberately bad."""
    root = tmp_path_factory.mktemp("campaign")
    rng = np.random.default_rng(_SEED)
    manifest = {"seed": _SEED, "root": root, "campaigns": {}, "anomaly": None}
    for piece, scale in _SCALES.items():
        (root / piece).mkdir()
        files = []
        for k in range(2):
            path = root / piece / f"tscan_{k + 1}.dat"
            _write_temperature_scan(path, rng, scale)
            files.append(path)
        path = root / piece / "field_scan.dat"
        row = _write_field_scan(path, rng, scale, anomaly=(piece == "piece_a"))
        files.append(path)
        if row is not None:
            manifest["anomaly"] = {
                "file": path,
                "row": row,
                "family": "AC",
                "label": "isolated chi' sign flip, enlarged error, no status flag",
            }
        manifest["campaigns"][piece] = {"scale": scale, "files": files}
    return manifest


def _all_files(campaign):
    return [f for c in campaign["campaigns"].values() for f in c["files"]]


# ---------------------------------------------------------------------------
# The campaign: exactly the truth is flagged, and nothing innocent
# ---------------------------------------------------------------------------


def test_flags_exactly_the_injected_anomaly(campaign):
    flagged = {
        (path, label)
        for path in _all_files(campaign)
        for label in calamari.load(path).quality().flagged.index
    }
    truth = campaign["anomaly"]
    assert flagged == {(truth["file"], truth["row"])}


def test_the_flag_carries_its_evidence(campaign):
    truth = campaign["anomaly"]
    m = calamari.load(truth["file"])
    row = m.quality().rows.loc[truth["row"]]
    assert row["family"] == "AC"
    assert row["signal"] == "chi_real"
    assert row["value"] < 0
    assert row["continuity_score"] > quality.CONTINUITY_THRESHOLD
    assert row["err_ratio"] > 4  # its error is several times its neighbours'
    assert row["T_spread_ratio"] < 10  # the temperature did not move unusually
    assert row["status_departure"] == ""  # and the instrument raised nothing


def test_report_indexes_the_files_own_rows(campaign):
    truth = campaign["anomaly"]
    m = calamari.load(truth["file"])
    raw = m.data.loc[m.quality().flagged.index]
    assert raw["AC X' (emu/Oe)"].iloc[0] < 0


def test_occasional_wide_temperature_spreads_are_not_flags(campaign):
    """10-25 mK spreads are a stability figure in a settled scan, not a
    reason to distrust the value."""
    rows = pd.concat(calamari.load(p).quality().rows for p in _all_files(campaign))
    wide = rows[rows["T_spread_K"] > 0.010]
    assert len(wide) > 5
    assert not wide["flagged"].any()


def test_first_point_error_spike_is_evidence_not_a_flag(campaign):
    m = calamari.load(campaign["campaigns"]["piece_b"]["files"][-1])
    ac = m.quality().rows.query("family == 'AC'")
    first = ac.iloc[0]
    assert first["err_ratio"] > 10
    assert not first["flagged"]


def test_real_transition_step_is_not_flagged(campaign):
    """Across a steep real step the neighbours disagree among themselves, so
    the score stays small -- the property that separates physics from glitch."""
    m = calamari.load(campaign["campaigns"]["piece_b"]["files"][-1])
    rows = m.quality().rows.query("family == 'AC'")
    field = m.field_oe.loc[rows.index]
    near = rows[(field - _STEP_FIELD_OE).abs() < 3 * _STEP_WIDTH_OE]
    assert len(near) >= 5
    assert near["continuity_score"].max() < quality.CONTINUITY_THRESHOLD / 4


def test_status_codes_read_against_the_files_usual_value(campaign):
    """Measure Status 1 on VSM rows and Temp. Status 1 everywhere are normal;
    only the first VSM point, whose codes depart from its file's, is noted."""
    m = calamari.load(campaign["campaigns"]["piece_a"]["files"][-1])
    rows = m.quality().rows
    noted = rows[rows["status_departure"] != ""]
    assert list(noted.index) == [0]
    text = noted.loc[0, "status_departure"]
    assert "SQUID Status (code)=68 (usual 0)" in text
    assert "Measure Status (code)=0 (usual 1)" in text
    assert "Measure Count=27 (usual 28)" in text
    assert "Temp. Status" not in text
    assert not noted.loc[0, "flagged"]


def test_families_are_scored_separately(campaign):
    """A combined run interleaves VSM and AC rows. Each family is its own
    series, in the file's own row labels."""
    m = calamari.load(campaign["campaigns"]["piece_b"]["files"][-1])
    rows = m.quality().rows
    assert set(rows["family"]) == {"VSM", "AC"}
    assert set(rows.loc[rows["family"] == "VSM", "signal"]) == {"moment_emu"}
    assert not rows["flagged"].any()
    assert rows.index.equals(m.data.index)


def test_temperature_scans_pair_dc_fit_quality(campaign):
    m = calamari.load(campaign["campaigns"]["piece_a"]["files"][0])
    rows = m.quality().rows
    dc = rows[rows["family"] == "DC"]
    np.testing.assert_allclose(dc["fit_quality"], m.data.loc[dc.index, "DC Free Fit"])
    assert rows.loc[rows["family"] == "AC", "fit_quality"].isna().all()


# ---------------------------------------------------------------------------
# Screening a directory
# ---------------------------------------------------------------------------


def test_catalog_surfaces_the_bad_file(campaign):
    cat = calamari.build_catalog(campaign["root"])
    bad = cat.where(n_flagged=1)
    assert bad.paths == [campaign["anomaly"]["file"]]
    worst = cat.sort_by("max_continuity_score", ascending=False)
    assert worst.paths[0] == campaign["anomaly"]["file"]
    assert set(cat["n_flagged"]) == {0, 1}


def test_summary_counts_match_the_report(campaign):
    m = calamari.load(campaign["anomaly"]["file"])
    report = m.quality().summary()
    summary = m.summary()
    assert summary["n_flagged"] == report["n_flagged"] == 1
    assert summary["max_continuity_score"] == report["max_continuity_score"]
    assert summary["n_status_departures"] == report["n_status_departures"] == 1


def test_unscreenable_file_summarises_as_none_not_zero(tmp_path):
    """A raw file has nothing to score; 0 would claim it was screened clean."""
    path = simulate.write_mpms3_raw_file(
        tmp_path / "scan.rw.dat",
        raw_position_mm=np.linspace(-15, 15, 50),
        raw_voltage_v=np.linspace(-1, 1, 50),
    )
    m = calamari.load(path)
    assert len(m.quality().rows) == 0
    assert list(m.quality().rows.columns) == list(quality.QUALITY_COLUMNS)
    s = m.summary()
    assert s["n_flagged"] is None
    assert s["max_continuity_score"] is None
    assert s["n_status_departures"] is None


# ---------------------------------------------------------------------------
# Pairing and splitting rules
# ---------------------------------------------------------------------------


def _dc_file(tmp_path):
    n = 12
    T = np.linspace(2.0, 13.0, n)
    free = 1e-3 * (1 + 0.01 * np.arange(n))
    return simulate.write_mpms3_file(
        tmp_path / "dc.dat",
        temperature_K=T,
        field_oe=np.full(n, 1000.0),
        dc_moment_emu=free,
        dc_moment_err_emu=np.full(n, 1e-6),
        extra_columns={
            "DC Moment Fixed Ctr (emu)": 0.99 * free,
            "DC Moment Err Fixed Ctr (emu)": np.full(n, 1e-6),
            "DC Free Fit": np.full(n, 0.98),
            "DC Fixed Fit": np.full(n, 0.93),
        },
    )


def test_fit_quality_follows_the_moment_in_use(tmp_path):
    """DC Free Fit belongs to the free-centre moment and DC Fixed Fit to the
    fixed-centre one; one column for both would mix two measurements."""
    m = calamari.load(_dc_file(tmp_path))
    np.testing.assert_allclose(m.quality().rows["fit_quality"], 0.98)
    fixed = m.with_moment("DC Moment Fixed Ctr").quality().rows
    np.testing.assert_allclose(fixed["fit_quality"], 0.93)
    np.testing.assert_allclose(fixed["value"], 0.99 * m.moment_emu)


def test_ac_frequencies_are_scored_separately(tmp_path):
    """Interleaved frequencies are separate series. Scored as one, each
    point's neighbours would include the other frequency's values, whose
    offset sets their spread -- and a real glitch in either series drowns in
    it. Split, the glitch stands out."""
    rng = np.random.default_rng(_SEED)
    n = 80
    T = np.repeat(np.linspace(2.0, 21.5, n // 2), 2)
    freq = np.tile([10.0, 1000.0], n // 2)
    chi = 1e-4 * (1 - 0.002 * T) * np.where(freq == 10.0, 1.0, 0.7)
    chi *= 1 + rng.normal(0, 0.001, n)
    glitch = 40  # a 10 Hz row, mid-sweep
    chi[glitch] *= 1.15
    m = calamari.load(
        simulate.write_mpms3_file(
            tmp_path / "ac.dat",
            temperature_K=T,
            field_oe=np.zeros(n),
            chi_real=chi,
            chi_imag=0.01 * chi,
            ac_frequency_hz=freq,
        )
    )
    rows = m.quality().rows
    assert set(rows["frequency_hz"]) == {10.0, 1000.0}
    assert list(rows.index[rows["flagged"]]) == [glitch]
    mixed = quality.continuity_scores([chi])[0]
    assert mixed[glitch] < quality.CONTINUITY_THRESHOLD  # hidden when mixed


def test_threshold_is_the_callers_to_move(campaign):
    m = calamari.load(campaign["anomaly"]["file"])
    report = m.quality()
    second = report.rows["continuity_score"].nlargest(2).iloc[1]
    looser = m.quality(threshold=second / 2)
    assert len(looser.flagged) >= 2
    assert looser.threshold == pytest.approx(second / 2)
    with pytest.raises(ValueError, match="threshold must be positive"):
        m.quality(threshold=0)


def test_report_repr_is_one_line(campaign):
    text = repr(calamari.load(campaign["anomaly"]["file"]).quality())
    assert "\n" not in text
    assert "1 of" in text and "points flagged" in text


def test_map_check_uses_the_same_score():
    """One scorer: a point quality() flags is the point a map warns about."""
    from calamari import plot

    assert plot.MAP_OUTLIER_THRESHOLD == quality.CONTINUITY_THRESHOLD
    y = np.sin(np.linspace(0, 3, 30))
    y[12] += 5
    np.testing.assert_array_equal(
        plot_data._map_outlier_scores([y])[0], quality.continuity_scores([y])[0]
    )
