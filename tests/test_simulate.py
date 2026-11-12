"""Tests for calamari.simulate: synthetic Quantum Design files and signals.

These tests check the simulate module in its own right (header fidelity,
validation, determinism) rather than parsing -- parsing correctness against
these same synthetic files is exercised in test_io.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from calamari import io as cio
from calamari import simulate
from calamari.exceptions import AmbiguousColumnError

# ---------------------------------------------------------------------------
# mpms3_lines / write_mpms3_file
# ---------------------------------------------------------------------------


def test_mpms3_lines_header_signature():
    lines = simulate.mpms3_lines(temperature_K=[5.0, 10.0], field_oe=[0.0, 100.0])
    assert lines[0] == "[Header]"
    assert "; MPMS3 Data File" in lines
    assert any(ln.startswith("BYAPP,SQUID") for ln in lines)


def test_mpms3_lines_info_ordering_is_reversed():
    """MPMS3 INFO lines are value-first, KEY-last -- the reversed dialect."""
    lines = simulate.mpms3_lines(
        sample_mass_mg=10.0,
        sample_material="SyntheticSample",
        temperature_K=[5.0],
        field_oe=[0.0],
    )
    assert "INFO,10.0,SAMPLE_MASS" in lines
    assert "INFO,SyntheticSample,SAMPLE_MATERIAL" in lines


def test_mpms3_lines_epoch_is_labview_magnitude():
    """The written FILEOPENTIME epoch is ~3.6e9 (LabVIEW, seconds since
    1904), not Unix magnitude (~1.6e9 for a 2020 date) -- the trap the
    parser must be robust to (see calamari.io.quantum_design module
    docstring)."""
    lines = simulate.mpms3_lines(
        date="1/29/2020", time_str="10:00 AM", temperature_K=[5.0], field_oe=[0.0]
    )
    fileopentime = next(ln for ln in lines if ln.startswith("FILEOPENTIME,"))
    epoch = float(fileopentime.split(",", 1)[1].split()[0])
    assert epoch > 3.0e9
    # Unix epoch for the same date/time is ~1.58e9; LabVIEW is offset by
    # ~2.08e9 s (1904 -> 1970), so the two must differ by roughly that.
    unix_epoch = 1_580_292_000.0
    assert abs((epoch - unix_epoch) - 2_082_844_800.0) < 1.0


def test_mpms3_lines_optional_fields_omitted_when_none():
    lines = simulate.mpms3_lines(
        sample_material=None,
        sample_mass_mg=None,
        sample_shape=None,
        sample_comment=None,
        temperature_K=[5.0],
        field_oe=[0.0],
    )
    assert not any("SAMPLE_MASS" in ln for ln in lines)
    assert not any("SAMPLE_MATERIAL" in ln for ln in lines)
    assert not any("SAMPLE_SHAPE" in ln for ln in lines)
    assert not any("SAMPLE_COMMENT" in ln for ln in lines)


def test_mpms3_lines_columns_match_supplied_arrays_only():
    lines = simulate.mpms3_lines(
        temperature_K=[5.0, 6.0], field_oe=[0.0, 1.0], vsm_moment_emu=[1e-3, 2e-3]
    )
    header_row = lines[lines.index("[Data]") + 1]
    columns = header_row.split(",")
    assert columns == ["Temperature (K)", "Magnetic Field (Oe)", "Moment (emu)"]


def test_mpms3_lines_all_optional_columns():
    lines = simulate.mpms3_lines(
        temperature_K=[5.0],
        field_oe=[0.0],
        vsm_moment_emu=[1e-3],
        chi_real=[2e-3],
        chi_imag=[3e-3],
        ac_frequency_hz=[10.0],
    )
    header_row = lines[lines.index("[Data]") + 1]
    assert header_row == (
        "Temperature (K),Magnetic Field (Oe),Moment (emu),"
        "AC X' (emu/Oe),AC X'' (emu/Oe),AC Frequency (Hz)"
    )


def test_mpms3_lines_ac_drive_oe_column():
    lines = simulate.mpms3_lines(
        temperature_K=[5.0],
        field_oe=[0.0],
        chi_real=[2e-3],
        ac_drive_oe=[0.3],
    )
    header_row = lines[lines.index("[Data]") + 1]
    assert header_row == (
        "Temperature (K),Magnetic Field (Oe),AC X' (emu/Oe),AC Drive (Oe)"
    )
    data_row = lines[lines.index("[Data]") + 2]
    assert data_row.split(",")[-1] == "0.3"


def test_mpms3_lines_mismatched_length_raises():
    with pytest.raises(ValueError):
        simulate.mpms3_lines(temperature_K=[5.0, 6.0], field_oe=[0.0])
    with pytest.raises(ValueError):
        simulate.mpms3_lines(
            temperature_K=[5.0, 6.0], field_oe=[0.0, 1.0], vsm_moment_emu=[1e-3]
        )


def test_write_mpms3_file_round_trips_through_loader(tmp_path):
    path = simulate.write_mpms3_file(
        tmp_path / "sample.dat",
        app="SQUID DC",
        sample_material="SyntheticSample",
        sample_mass_mg=10.0,
        temperature_K=[5.0, 6.0, 7.0],
        field_oe=[0.0, 100.0, 200.0],
        vsm_moment_emu=[1e-3, 2e-3, 3e-3],
    )
    assert path.exists()
    m = cio.load(path)
    assert m.data_format == "MPMS3"
    # BYAPP says "SQUID DC" but the file carries a VSM `Moment` column, and the
    # rows decide (IO-4): the header names an application, not the run's content.
    assert m.measurement_type == "VSM"
    assert m.sample_name == "SyntheticSample"
    assert m.sample_mass_mg == pytest.approx(10.0)
    np.testing.assert_allclose(m.moment_emu, [1e-3, 2e-3, 3e-3])


# ---------------------------------------------------------------------------
# VSM vs DC-scan fidelity
#
# An MPMS3 DC scan writes `DC Moment Free Ctr`, never a bare `Moment` --
# that spelling belongs to the VSM application.
# ---------------------------------------------------------------------------


def test_the_two_moments_are_written_under_different_column_names():
    """The whole point of the split: they are different measurements."""
    vsm = simulate.mpms3_lines(
        temperature_K=[5.0], field_oe=[0.0], vsm_moment_emu=[1e-3]
    )
    dc = simulate.mpms3_lines(
        temperature_K=[5.0], field_oe=[0.0], dc_moment_emu=[1e-3]
    )
    assert vsm[vsm.index("[Data]") + 1].split(",")[-1] == "Moment (emu)"
    assert dc[dc.index("[Data]") + 1].split(",")[-1] == "DC Moment Free Ctr (emu)"


def test_each_moment_carries_its_own_error_column(tmp_path):
    """Each names its own error column, and `moment_err_emu` pairs them --
    previously reachable only by hand-writing the spelling in extra_columns."""
    vsm = cio.load(
        simulate.write_mpms3_file(
            tmp_path / "vsm.dat",
            temperature_K=[5.0, 5.0],
            field_oe=[0.0, 100.0],
            vsm_moment_emu=[1e-3, 2e-3],
            vsm_moment_err_emu=[1e-5, 2e-5],
        )
    )
    assert "M. Std. Err. (emu)" in vsm.data.columns
    np.testing.assert_allclose(vsm.moment_err_emu, [1e-5, 2e-5])

    dc = cio.load(
        simulate.write_mpms3_file(
            tmp_path / "dc.dat",
            temperature_K=[5.0, 5.0],
            field_oe=[0.0, 100.0],
            dc_moment_emu=[1e-3, 2e-3],
            dc_moment_err_emu=[1e-5, 2e-5],
        )
    )
    assert "DC Moment Err Free Ctr (emu)" in dc.data.columns
    np.testing.assert_allclose(dc.moment_err_emu, [1e-5, 2e-5])


def test_a_combined_vsm_and_dc_run_is_now_expressible(tmp_path):
    """The shape this module could not previously build: two moment families
    in one file, each blank on the other's rows, told apart by Transport
    Action. `moment_emu` refuses to choose between them, which is correct."""
    vsm_rows = np.arange(6) % 2 == 0
    m = cio.load(
        simulate.write_mpms3_file(
            tmp_path / "combined.dat",
            app="MPMS3 w/ AC,1.0,1.1",
            temperature_K=np.full(6, 5.0),
            field_oe=np.repeat([0.0, 100.0, 200.0], 2),
            vsm_moment_emu=np.where(vsm_rows, 1e-3, np.nan),
            dc_moment_emu=np.where(vsm_rows, np.nan, 2e-3),
            transport_action=np.where(vsm_rows, 1.0, 6.0),
        )
    )
    assert m.moment_columns == ("Moment (emu)", "DC Moment Free Ctr (emu)")
    with pytest.raises(AmbiguousColumnError):
        m.moment_emu
    np.testing.assert_allclose(m.with_moment("Moment").moment_emu.dropna(), 1e-3)
    np.testing.assert_allclose(
        m.with_moment("DC Moment Free Ctr").moment_emu.dropna(), 2e-3
    )


def test_transport_action_is_written_as_its_own_column(tmp_path):
    """Real summary files carry housekeeping rows, so a fixture needs to be
    able to write one without reaching for extra_columns."""
    m = cio.load(
        simulate.write_mpms3_file(
            tmp_path / "ta.dat",
            temperature_K=[5.0, 5.0, 10.0],
            field_oe=[0.0, 100.0, 100.0],
            vsm_moment_emu=[1e-3, 2e-3, np.nan],
            transport_action=[1.0, 1.0, 3.0],
        )
    )
    assert "Transport Action" in m.data.columns
    assert m.summary()["n_measurements"] == 2


def test_mpms3_rejects_ambiguous_moment_keyword():
    with pytest.raises(TypeError, match="unexpected keyword argument 'moment_emu'"):
        simulate.mpms3_lines(temperature_K=[5.0], field_oe=[0.0], moment_emu=[1e-3])


def test_extra_columns_may_not_duplicate_a_named_parameters_column():
    """Two columns of one name is a file no instrument writes, and silently
    accepting it would make which one the parser reads depend on order."""
    with pytest.raises(ValueError, match="already have their own parameter"):
        simulate.mpms3_lines(
            temperature_K=[5.0],
            field_oe=[0.0],
            vsm_moment_emu=[1e-3],
            extra_columns={"Moment (emu)": [2e-3]},
        )


# ---------------------------------------------------------------------------
# MPMS3 generator and locale handling
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# curie_weiss_chi
# ---------------------------------------------------------------------------


def test_curie_weiss_chi_theta_zero_is_curie_law():
    T_K = np.array([10.0, 20.0, 100.0])
    chi = simulate.curie_weiss_chi(T_K, C=2.0, theta_K=0.0)
    np.testing.assert_allclose(chi, 2.0 / T_K)


def test_curie_weiss_chi_background_offset():
    T_K = np.array([50.0])
    chi = simulate.curie_weiss_chi(T_K, C=1.0, theta_K=-5.0, chi0=0.01)
    np.testing.assert_allclose(chi, 1.0 / 55.0 + 0.01)


# ---------------------------------------------------------------------------
# synthetic_mh_curve
# ---------------------------------------------------------------------------


def test_synthetic_mh_curve_zero_field_zero_moment():
    m = simulate.synthetic_mh_curve(np.array([0.0]), m_sat_emu=1.0, h_knee_oe=100.0)
    np.testing.assert_allclose(m, [0.0])


def test_synthetic_mh_curve_saturates():
    m = simulate.synthetic_mh_curve(np.array([1.0e6]), m_sat_emu=2.5, h_knee_oe=100.0)
    np.testing.assert_allclose(m, [2.5], rtol=1e-6)


def test_synthetic_mh_curve_is_odd():
    field_oe = np.array([50.0, 500.0, 5000.0])
    m_pos = simulate.synthetic_mh_curve(field_oe, m_sat_emu=1.3, h_knee_oe=800.0)
    m_neg = simulate.synthetic_mh_curve(-field_oe, m_sat_emu=1.3, h_knee_oe=800.0)
    np.testing.assert_allclose(m_neg, -m_pos)


def test_synthetic_mh_curve_noise_requires_rng():
    with pytest.raises(ValueError):
        simulate.synthetic_mh_curve(np.array([1.0]), m_sat_emu=1.0, h_knee_oe=1.0, noise=0.1)


def test_synthetic_mh_curve_noise_is_deterministic_given_rng():
    field_oe = np.linspace(-1000.0, 1000.0, 11)
    m1 = simulate.synthetic_mh_curve(
        field_oe, m_sat_emu=1.0, h_knee_oe=300.0, noise=0.05, rng=np.random.default_rng(42)
    )
    m2 = simulate.synthetic_mh_curve(
        field_oe, m_sat_emu=1.0, h_knee_oe=300.0, noise=0.05, rng=np.random.default_rng(42)
    )
    np.testing.assert_array_equal(m1, m2)
    assert not np.allclose(m1, simulate.synthetic_mh_curve(field_oe, m_sat_emu=1.0, h_knee_oe=300.0))


# ---------------------------------------------------------------------------
# synthetic_chi_peak
# ---------------------------------------------------------------------------


def test_synthetic_chi_peak_at_centre():
    y = simulate.synthetic_chi_peak(
        np.array([5.0]), centre=5.0, width=1.0, height=10.0, background=2.0
    )
    np.testing.assert_allclose(y, [12.0])


def test_synthetic_chi_peak_decays_away_from_centre():
    x = np.array([0.0, 1000.0])
    y = simulate.synthetic_chi_peak(x, centre=0.0, width=1.0, height=10.0, background=1.0)
    assert y[1] == pytest.approx(1.0, abs=1e-9)
    assert y[0] == pytest.approx(11.0)


def test_synthetic_chi_peak_noise_requires_rng():
    with pytest.raises(ValueError):
        simulate.synthetic_chi_peak(
            np.array([0.0]), centre=0.0, width=1.0, height=1.0, noise=0.1
        )


def test_synthetic_chi_peak_noise_is_deterministic_given_rng():
    x = np.linspace(-5.0, 5.0, 11)
    y1 = simulate.synthetic_chi_peak(
        x, centre=0.0, width=1.0, height=5.0, noise=0.1, rng=np.random.default_rng(7)
    )
    y2 = simulate.synthetic_chi_peak(
        x, centre=0.0, width=1.0, height=5.0, noise=0.1, rng=np.random.default_rng(7)
    )
    np.testing.assert_array_equal(y1, y2)


# ---------------------------------------------------------------------------
# locale independence
# ---------------------------------------------------------------------------


def test_epoch_time_token_rejects_out_of_range_hours():
    """The hand parse must keep strptime's %I contract (hours 1-12): the
    % 12 arithmetic would otherwise turn "0:15 AM" silently into
    midnight instead of failing loudly.
    """
    with pytest.raises(ValueError, match="hour must be 1-12"):
        simulate.mpms3_lines(
            time_str="0:15 AM",
            temperature_K=np.array([1.0]),
            field_oe=np.array([1.0]),
            chi_real=np.array([1.0]),
        )


def test_header_epoch_parse_is_locale_independent():
    """Regression: the header epoch used strptime('%p'), which fails outright
    in locales whose AM/PM designators are empty (en_GB among them). Any
    host that calls locale.setlocale(LC_ALL, "") -- Sphinx does, which is
    how a documentation build first hit it -- exposed the
    bug; a bare test process runs in the C locale and never did.
    """
    import locale

    saved = locale.setlocale(locale.LC_TIME)
    try:
        locale.setlocale(locale.LC_TIME, "en_GB.UTF-8")
    except locale.Error:
        pytest.skip("en_GB.UTF-8 locale not generated on this system")
    try:
        # The header writer uses _epoch_seconds.
        lines = simulate.mpms3_lines(
            temperature_K=np.array([1.0]),
            field_oe=np.array([1.0]),
            chi_real=np.array([1.0]),
        )
        assert any("FILEOPENTIME" in line for line in lines)
    finally:
        locale.setlocale(locale.LC_TIME, saved)
