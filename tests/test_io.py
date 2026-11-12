"""Tests for calamari.io: content-sniffing dispatch and the Quantum Design
MPMS3 parser, exercised entirely against synthetic files (see
tests/conftest.py and calamari.simulate -- no real instrument data is ever
committed to this repository).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calamari import io as cio
from calamari import simulate
from calamari.constants import OE_PER_TESLA
from calamari.exceptions import (
    AmbiguousColumnError,
    CalamariError,
    MeasurementFamilyError,
    UnknownFormatError,
    UnknownUnitError,
    UnsupportedFileError,
)
from calamari.measurement import Measurement
from calamari.repeats import AggregatedMeasurement, aggregate_repeats

# ---------------------------------------------------------------------------
# Dialect sniffing
# ---------------------------------------------------------------------------


def test_sniff_format_mpms3(mpms3_dc_file):
    assert cio.sniff_format(mpms3_dc_file.path) == "MPMS3"
    assert cio.is_supported_file(mpms3_dc_file.path)


def test_sniff_format_rejects_header_less_dat(tmp_path):
    """A plain CSV-style .dat with no [Header] marker is not a recognised
    MPMS dialect -- format detection must sniff content, not the extension."""
    path = tmp_path / "not_mpms.dat"
    path.write_text("Time,Value\n0,1\n1,2\n", encoding="utf-8")
    assert cio.sniff_format(path) is None
    assert not cio.is_supported_file(path)


def test_sniff_format_rejects_mpms_log_style_file(tmp_path):
    """MPMS Log Data (.env-style) files share the [Header]/[Data] shape but
    carry no MPMS3 measurement signature, and must be rejected."""
    path = tmp_path / "log.dat"
    path.write_text(
        "[Header]\n"
        "TITLE,MPMS Log Data\n"
        "INFO,NAME,Environment\n"
        "[Data]\n"
        "Time,Temperature (K)\n"
        "0,300.0\n",
        encoding="utf-8",
    )
    assert cio.sniff_format(path) is None
    assert not cio.is_supported_file(path)


# ---------------------------------------------------------------------------
# Reversed MPMS3 INFO ordering / header field mapping
# ---------------------------------------------------------------------------


def test_mpms3_reversed_info_lands_in_right_fields(mpms3_dc_file):
    m = cio.load(mpms3_dc_file.path)
    assert m.header["sample_mass"] == pytest.approx(10.0)
    assert m.sample_mass_mg == pytest.approx(10.0)
    assert m.sample_name == "SyntheticSample"
    assert m.header["info"]["sample_mass"] == "10.0"
    assert m.header["info"]["sample_material"] == "SyntheticSample"


# ---------------------------------------------------------------------------
# Date extraction beats the LabVIEW epoch
# ---------------------------------------------------------------------------


def test_mpms3_date_iso_correct_despite_labview_epoch(mpms3_dc_file):
    m = cio.load(mpms3_dc_file.path)
    assert m.date_iso == "2020-01-29"
    # The raw epoch is LabVIEW-magnitude (~3.6e9), nowhere near a sane Unix
    # timestamp for 2020 (~1.58e9) -- date_iso must not be derived from it.
    assert m.date_epoch > 3.0e9


# ---------------------------------------------------------------------------
# Column accessors return ground truth, on MPMS3
# ---------------------------------------------------------------------------


def test_mpms3_dc_accessors_match_ground_truth(mpms3_dc_file):
    m = cio.load(mpms3_dc_file.path)
    np.testing.assert_allclose(m.temperature, mpms3_dc_file.temperature_K)
    np.testing.assert_allclose(m.field_oe, mpms3_dc_file.field_oe)
    np.testing.assert_allclose(m.moment_emu, mpms3_dc_file.moment_emu)


def test_mpms3_ac_accessors_match_ground_truth(mpms3_ac_file):
    m = cio.load(mpms3_ac_file.path)
    np.testing.assert_allclose(m.temperature, mpms3_ac_file.temperature_K)
    np.testing.assert_allclose(m.chi_real, mpms3_ac_file.chi_real)
    np.testing.assert_allclose(m.chi_imag, mpms3_ac_file.chi_imag)
    np.testing.assert_allclose(m.ac_frequency, mpms3_ac_file.frequency_hz)


def test_mpms3_ac_susceptibility_is_used_directly_not_derived(mpms3_ac_file):
    """MPMS3 records a susceptibility, so chi_* must return it as written --
    dividing it by the drive amplitude again would be wrong."""
    m = cio.load(mpms3_ac_file.path)
    np.testing.assert_allclose(m.chi_real, mpms3_ac_file.chi_real)
    np.testing.assert_allclose(m.chi_imag, mpms3_ac_file.chi_imag)
    # MPMS3 writes moment amplitude/phase, not resolved moment components.


def test_susceptibility_column_wins_over_moment_column(tmp_path):
    """Dispatch is on which column is present, not on data_format. A file
    carrying both forms must use the instrument's own susceptibility rather
    than re-deriving one."""
    path = simulate.write_mpms3_file(
        tmp_path / "both.dat",
        app="SQUID AC",
        temperature_K=[2.0, 3.0],
        field_oe=[0.0, 0.0],
        chi_real=[1.0e-4, 2.0e-4],
        ac_drive_oe=[0.5, 0.5],
        extra_columns={"m' (emu)": [9.9, 9.9]},  # would derive to nonsense
    )
    m = cio.load(path)
    np.testing.assert_allclose(m.chi_real, [1.0e-4, 2.0e-4])


def test_field_tesla_matches_field_oe_over_1e4(mpms3_dc_file):
    m = cio.load(mpms3_dc_file.path)
    np.testing.assert_allclose(m.field_tesla, m.field_oe / OE_PER_TESLA)


# ---------------------------------------------------------------------------
# AC drive-amplitude accessor (recorded MPMS3 spelling:
# MPMS3 'AC Drive (Oe)')
# ---------------------------------------------------------------------------


def test_mpms3_ac_drive_oe_matches_ground_truth(mpms3_ac_file):
    m = cio.load(mpms3_ac_file.path)
    np.testing.assert_allclose(m.ac_drive_oe, mpms3_ac_file.ac_drive_oe)


def test_ac_drive_oe_none_when_absent(mpms3_dc_file):
    """A DC file (no AC columns of any kind supplied) has no drive column."""
    m = cio.load(mpms3_dc_file.path)
    assert m.ac_drive_oe is None


# ---------------------------------------------------------------------------
# Missing-column accessors return None
# ---------------------------------------------------------------------------


def test_dc_file_has_no_ac_columns(mpms3_dc_file):
    m = cio.load(mpms3_dc_file.path)
    assert m.chi_real is None
    assert m.chi_imag is None
    assert m.ac_frequency is None


def test_ac_file_has_no_dc_moment_column(mpms3_ac_file):
    m = cio.load(mpms3_ac_file.path)
    assert m.moment_emu is None


# ---------------------------------------------------------------------------
# AC vs DC inference, including the BYAPP-lies-about-AC trap
# ---------------------------------------------------------------------------


def test_mpms3_measurement_type_from_byapp(mpms3_dc_file, mpms3_ac_file):
    assert cio.load(mpms3_dc_file.path).measurement_type == "DC"
    assert cio.load(mpms3_ac_file.path).measurement_type == "AC"


def test_mpms3_byapp_ac_but_only_dc_data_is_classified_dc(tmp_path):
    """MPMS3's BYAPP often says 'SQUID AC' even for DC hysteresis runs; the
    parser must verify against which columns actually carry data rather
    than trusting BYAPP at face value."""
    path = simulate.write_mpms3_file(
        tmp_path / "mislabelled.dat",
        app="SQUID AC",
        temperature_K=[5.0, 5.0, 5.0],
        field_oe=[-100.0, 0.0, 100.0],
        dc_moment_emu=[-1e-3, 0.0, 1e-3],
    )
    m = cio.load(path)
    assert m.measurement_type == "DC"


# ---------------------------------------------------------------------------
# scan_kind classification
# ---------------------------------------------------------------------------


def test_scan_kind_t_sweep(tmp_path):
    path = simulate.write_mpms3_file(
        tmp_path / "tsweep.dat",
        temperature_K=np.linspace(2.0, 300.0, 10),
        field_oe=np.full(10, 1000.0),
        vsm_moment_emu=np.linspace(1e-3, 2e-3, 10),
    )
    m = cio.load(path)
    assert m.scan_kind == "T-scan"


def test_scan_kind_h_sweep(tmp_path):
    path = simulate.write_mpms3_file(
        tmp_path / "hsweep.dat",
        temperature_K=np.full(10, 5.0),
        field_oe=np.linspace(-10_000.0, 10_000.0, 10),
        vsm_moment_emu=np.linspace(-1e-3, 1e-3, 10),
    )
    m = cio.load(path)
    assert m.scan_kind == "H-scan"


def test_scan_kind_static(tmp_path):
    path = simulate.write_mpms3_file(
        tmp_path / "static.dat",
        temperature_K=np.full(10, 5.0),
        field_oe=np.full(10, 1000.0),
        vsm_moment_emu=np.linspace(1e-3, 1.01e-3, 10),
    )
    m = cio.load(path)
    assert m.scan_kind == "static/other"


def test_scan_kind_mixed(tmp_path):
    path = simulate.write_mpms3_file(
        tmp_path / "mixed.dat",
        temperature_K=np.linspace(2.0, 300.0, 10),
        field_oe=np.linspace(-10_000.0, 10_000.0, 10),
        vsm_moment_emu=np.linspace(-1e-3, 1e-3, 10),
    )
    m = cio.load(path)
    assert m.scan_kind == "mixed"


# ---------------------------------------------------------------------------
# Housekeeping rows (Transport Action 2/3/4)
#
# Summary exports can contain off-setpoint locate rows with no moment.
# The fixture below reproduces that structural fact using invented values.
# ---------------------------------------------------------------------------


def _housekeeping_tail(tmp_path, name="locate_tail.dat", *, measurement_number=None):
    """An invented isothermal sweep followed by an off-setpoint locate row."""
    n = 8
    temperature = np.append(np.full(n, 7.0), 30.0)
    field = np.append(np.linspace(0.0, 800.0, n), 1500.0)
    moment = np.append(np.linspace(0.0, 1e-3, n), np.nan)
    action = np.append(np.full(n, 1.0), 3.0)

    extra = {}
    if measurement_number is not None:
        extra["Measurement Number"] = measurement_number

    return simulate.write_mpms3_file(
        tmp_path / name,
        app="MPMS3 w/ AC,1.0,1.1",
        temperature_K=temperature,
        field_oe=field,
        vsm_moment_emu=moment,
        transport_action=action,
        extra_columns=extra,
    )


def test_housekeeping_row_does_not_change_the_scan_kind(tmp_path):
    """The defect this guards: one locate scan parked off setpoint made an
    isothermal field sweep classify as 'mixed'."""
    m = cio.load(_housekeeping_tail(tmp_path))
    assert m.scan_kind == "H-scan"


def test_housekeeping_row_does_not_widen_the_reported_spans(tmp_path):
    """Spans describe the measurement, so a row the transport merely passed
    through must not appear in them."""
    m = cio.load(_housekeeping_tail(tmp_path))
    summary = m.summary()
    assert summary["T_max"] == pytest.approx(7.0)
    assert summary["H_max_Oe"] == pytest.approx(800.0)
    assert "T 7-7 K" in repr(m)


def test_n_points_counts_rows_and_n_measurements_excludes_housekeeping(tmp_path):
    """`n_points` keeps its documented meaning -- rows -- because silently
    redefining a published column is worse than adding one."""
    m = cio.load(_housekeeping_tail(tmp_path))
    summary = m.summary()
    assert summary["n_points"] == 9
    assert summary["n_measurements"] == 8


def test_n_measurements_equals_n_points_without_housekeeping_rows(tmp_path):
    """Without housekeeping rows, all parsed rows count as measurements."""
    path = simulate.write_mpms3_file(
        tmp_path / "ordinary.dat",
        temperature_K=np.full(5, 5.0),
        field_oe=np.linspace(0.0, 1000.0, 5),
        vsm_moment_emu=np.linspace(0.0, 1e-3, 5),
    )
    summary = cio.load(path).summary()
    assert summary["n_points"] == summary["n_measurements"] == 5


def test_housekeeping_row_does_not_inflate_max_repeats(tmp_path):
    """Whether a housekeeping row carries a counter value is unknown -- no
    real file has shown one -- so both shapes are pinned. A continuing value
    would extend a genuine pair into a phantom triple."""
    counter = np.array([1, 2, 1, 2, 1, 2, 1, 2, 3], dtype=float)
    m = cio.load(_housekeeping_tail(tmp_path, "counter.dat", measurement_number=counter))
    assert m.max_repeats() == 2

    blank = np.array([1, 2, 1, 2, 1, 2, 1, 2, np.nan])
    blank_m = cio.load(
        _housekeeping_tail(tmp_path, "blank.dat", measurement_number=blank)
    )
    assert blank_m.max_repeats() == 2


def test_housekeeping_row_stays_in_data_and_in_the_accessors(tmp_path):
    """The row is excluded from descriptions, never from the data. Dropping it
    would break the length alignment repeats and restrict(mask=...) rely on."""
    m = cio.load(_housekeeping_tail(tmp_path))
    assert len(m.data) == 9
    assert len(m.temperature) == 9
    assert len(m.field_oe) == 9
    # The instrument leaves the moment blank on such a row, so the quantity
    # accessors already exclude it without any masking of ours.
    assert m.moment_emu.notna().sum() == 8


def test_aggregated_measurement_also_excludes_housekeeping_rows(tmp_path):
    """AggregatedMeasurement inherits summary()/_repr_bits, so it misreported
    identically. Aggregation's keep-first rule preserves Transport Action on
    the singleton group, so the fix carries over with no code of its own."""
    # Four field setpoints measured twice each, then the locate scan. The
    # repeats have to be real: a counter proposing rows at different setpoints
    # is refused by repeat_groups, and rightly.
    setpoints = np.repeat([0.0, 1000.0, 2000.0, 3000.0], 2)
    m = cio.load(
        simulate.write_mpms3_file(
            tmp_path / "agg.dat",
            app="MPMS3 w/ AC,1.0,1.1",
            temperature_K=np.append(np.full(8, 1.8), 10.0),
            field_oe=np.append(setpoints, 100.0),
            vsm_moment_emu=np.append(np.linspace(0.0, 1e-3, 8), np.nan),
            transport_action=np.append(np.full(8, 1.0), 3.0),
            extra_columns={
                "Measurement Number": np.array([1, 2, 1, 2, 1, 2, 1, 2, np.nan]),
            },
        )
    )
    aggregated = aggregate_repeats(m)
    assert aggregated.scan_kind == "H-scan"
    assert aggregated.summary()["T_max"] == pytest.approx(1.8)


# ---------------------------------------------------------------------------
# Measurement families
#
# A combined run can alternate VSM and AC rows at each field setpoint.
# Both families must remain visible regardless of their interleave order.
# ---------------------------------------------------------------------------


def _combined_run(tmp_path, name="combined.dat", *, n=6, transport=True):
    """A VSM + AC run: alternating rows, each populating only its own family."""
    vsm_rows = np.arange(n) % 2 == 0
    kwargs = {}
    if transport:
        kwargs["transport_action"] = np.where(vsm_rows, 1.0, 5.0)
    return simulate.write_mpms3_file(
        tmp_path / name,
        app="MPMS3 w/ AC,1.0,1.1",
        temperature_K=np.full(n, 5.0),
        field_oe=np.repeat(np.linspace(0.0, 1000.0, n // 2), 2),
        vsm_moment_emu=np.where(vsm_rows, 1.0e-3, np.nan),
        chi_real=np.where(vsm_rows, np.nan, 3.0e-5),
        ac_drive_oe=np.where(vsm_rows, np.nan, 1.0),
        **kwargs,
    )


def test_a_combined_run_reports_both_families(tmp_path):
    """The defect this closes: such a file reported a single family."""
    m = cio.load(_combined_run(tmp_path))
    assert m.measurement_type == "VSM+AC"
    assert m.measurement_types == ("VSM", "AC")


def test_the_composite_label_uses_a_fixed_precedence_not_file_order(tmp_path):
    """A combined run's interleave order flips between setpoints, so file
    order would label two identical runs differently and split one experiment
    across two catalog groups."""
    ac_first = simulate.write_mpms3_file(
        tmp_path / "ac_first.dat",
        app="MPMS3 w/ AC,1.0,1.1",
        temperature_K=np.full(4, 5.0),
        field_oe=np.repeat([0.0, 100.0], 2),
        vsm_moment_emu=[np.nan, 1e-3, np.nan, 1e-3],
        chi_real=[3e-5, np.nan, 3e-5, np.nan],
        ac_drive_oe=[1.0, np.nan, 1.0, np.nan],
        transport_action=[5.0, 1.0, 5.0, 1.0],
    )
    assert cio.load(ac_first).measurement_type == "VSM+AC"


def test_of_type_returns_one_family_with_its_own_label(tmp_path):
    """Each half is an ordinary Measurement whose own accessors resolve, and
    which no longer claims to hold rows it does not."""
    m = cio.load(_combined_run(tmp_path))
    vsm, ac = m.of_type("VSM"), m.of_type("AC")

    assert (len(vsm.data), len(ac.data)) == (3, 3)
    assert vsm.measurement_type == "VSM"
    assert ac.measurement_type == "AC"
    assert vsm.measurement_types == ("VSM",)

    assert vsm.moment_emu.notna().sum() == 3
    assert vsm.chi_real is None
    assert ac.chi_real.notna().sum() == 3
    assert ac.moment_emu is None
    assert len(m.data) == 6  # the original is untouched (DM-1)


def test_split_by_type_returns_every_family_in_precedence_order(tmp_path):
    m = cio.load(_combined_run(tmp_path))
    parts = m.split_by_type()
    assert list(parts) == ["VSM", "AC"]
    assert {k: len(v.data) for k, v in parts.items()} == {"VSM": 3, "AC": 3}


def test_split_by_type_on_a_single_family_file_returns_one_entry(mpms3_ac_file):
    """So a caller written for combined runs needs no special case."""
    parts = cio.load(mpms3_ac_file.path).split_by_type()
    assert list(parts) == ["AC"]


def test_of_type_refuses_a_family_the_file_does_not_have(tmp_path):
    """Returning an empty Measurement would hide the mistake -- an empty sweep
    looks like a sweep that measured nothing, not like a wrong question."""
    m = cio.load(_combined_run(tmp_path))
    with pytest.raises(MeasurementFamilyError) as excinfo:
        m.of_type("DC")
    message = str(excinfo.value)
    assert "'VSM'" in message and "'AC'" in message
    assert "measurement_types" in message


def test_housekeeping_rows_are_in_no_family_bucket(tmp_path):
    """They are not a measurement, so they are neither a family of their own
    nor silently folded into a real one."""
    m = cio.load(
        simulate.write_mpms3_file(
            tmp_path / "hk.dat",
            temperature_K=[1.8, 1.8, 10.0],
            field_oe=[0.0, 100.0, 100.0],
            vsm_moment_emu=[1e-3, 2e-3, np.nan],
            transport_action=[1.0, 1.0, 3.0],
        )
    )
    assert m.measurement_types == ("VSM",)
    assert len(m.of_type("VSM").data) == 2
    assert len(m.data) == 3


def test_an_unknown_transport_action_code_raises_only_when_asked(tmp_path):
    """The file still loads and its data is still readable: refusing the whole
    file over a question the caller may never ask would be the wrong trade."""
    path = simulate.write_mpms3_file(
        tmp_path / "code9.dat",
        temperature_K=[5.0, 5.0],
        field_oe=[0.0, 100.0],
        vsm_moment_emu=[1e-3, 2e-3],
        transport_action=[1.0, 9.0],
    )
    m = cio.load(path)  # loads fine
    np.testing.assert_allclose(m.moment_emu, [1e-3, 2e-3])
    with pytest.raises(MeasurementFamilyError, match="9"):
        m.measurement_types


def test_a_code_that_contradicts_its_own_columns_raises(tmp_path):
    """One of the two is wrong and calamari cannot tell which."""
    m = cio.load(
        simulate.write_mpms3_file(
            tmp_path / "mismatch.dat",
            app="MPMS3 w/ AC,1.0,1.1",
            temperature_K=[5.0, 5.0],
            field_oe=[0.0, 100.0],
            vsm_moment_emu=[1e-3, 2e-3],
            transport_action=[1.0, 5.0],  # row 1 says AC, but carries a moment
        )
    )
    with pytest.raises(MeasurementFamilyError, match="disagree|populates"):
        m.measurement_types


def test_a_row_populating_nothing_does_not_contradict_its_code(tmp_path):
    """Silence is not disagreement. A blank row is routine -- an AC row of a
    combined run populates no moment at all."""
    m = cio.load(_combined_run(tmp_path))
    assert m.measurement_types == ("VSM", "AC")


def test_families_fall_back_to_the_columns_without_transport_action(tmp_path):
    """Populated quantities classify families when no transport codes are present."""
    m = cio.load(_combined_run(tmp_path, "no_ta.dat", transport=False))
    assert m.measurement_types == ("VSM", "AC")


def test_a_bare_moment_column_is_a_vsm_measurement(tmp_path):
    """The VSM `Moment` column is distinct from the DC-scan moment columns."""
    vsm = cio.load(
        simulate.write_mpms3_file(
            tmp_path / "vsm.dat", app="SQUID VSM",
            temperature_K=[5.0, 5.0], field_oe=[0.0, 100.0],
            vsm_moment_emu=[1e-3, 2e-3],
        )
    )
    dc = cio.load(
        simulate.write_mpms3_file(
            tmp_path / "dc.dat", app="SQUID DC",
            temperature_K=[5.0, 5.0], field_oe=[0.0, 100.0],
            dc_moment_emu=[1e-3, 2e-3],
        )
    )
    assert vsm.measurement_type == "VSM"
    assert dc.measurement_type == "DC"


def test_the_label_cannot_go_stale_under_restrict(tmp_path):
    """A hand-rolled row subset must not keep advertising families it no
    longer holds -- `measurement_types` is derived and would correct itself,
    leaving the two accessors disagreeing."""
    m = cio.load(_combined_run(tmp_path))
    ac_only = m.restrict(mask=(m.row_families == "AC").to_numpy())
    assert ac_only.measurement_type == "AC"
    assert ac_only.measurement_types == ("AC",)


@pytest.mark.parametrize("include_ac", [False, True])
def test_of_type_resolves_the_two_moment_ambiguity(tmp_path, include_ac):
    """Explicit family selection resolves two moment channels, also with AC present."""
    codes = np.tile([1.0, 6.0, 5.0] if include_ac else [1.0, 6.0], 3)
    m = cio.load(
        simulate.write_mpms3_file(
            tmp_path / "combined.dat",
            app="MPMS3 w/ AC,1.0,1.1",
            temperature_K=np.full(codes.size, 11.0),
            field_oe=np.repeat([-250.0, 0.0, 250.0], 3 if include_ac else 2),
            vsm_moment_emu=np.where(codes == 1, 0.003, np.nan),
            dc_moment_emu=np.where(codes == 6, 0.005, np.nan),
            chi_real=np.where(codes == 5, 0.0007, np.nan) if include_ac else None,
            transport_action=codes,
        )
    )
    assert m.measurement_types == (("DC", "VSM", "AC") if include_ac else ("DC", "VSM"))
    with pytest.raises(AmbiguousColumnError):
        m.moment_emu
    np.testing.assert_allclose(m.of_type("VSM").moment_emu, [0.003] * 3)
    np.testing.assert_allclose(m.of_type("DC").moment_emu, [0.005] * 3)
    if include_ac:
        ac = m.of_type("AC")
        np.testing.assert_allclose(ac.chi_real, [0.0007] * 3)
        assert ac.moment_emu is None


def test_moment_method_names_how_the_moment_was_measured(tmp_path, mpms3_ac_file):
    """Both a VSM measurement and a DC scan yield a moment in emu; the
    accessor names its unit but not its provenance, and this names that."""
    vsm = cio.load(
        simulate.write_mpms3_file(
            tmp_path / "v.dat", temperature_K=[5.0], field_oe=[0.0],
            vsm_moment_emu=[1e-3],
        )
    )
    dc = cio.load(
        simulate.write_mpms3_file(
            tmp_path / "d.dat", temperature_K=[5.0], field_oe=[0.0],
            dc_moment_emu=[1e-3],
        )
    )
    assert vsm.moment_method == "VSM"
    assert dc.moment_method == "DC"
    assert cio.load(mpms3_ac_file.path).moment_method is None  # no moment at all
    # The vocabulary is the family vocabulary, so the two compose.
    assert vsm.of_type(vsm.moment_method).measurement_type == "VSM"


def test_moment_method_follows_with_moment_and_refuses_to_guess(tmp_path):
    """If it answered while `moment_emu` refused, the two accessors could
    describe different measurements."""
    vsm_rows = np.arange(4) % 2 == 0
    m = cio.load(
        simulate.write_mpms3_file(
            tmp_path / "both.dat",
            app="MPMS3 w/ AC,1.0,1.1",
            temperature_K=np.full(4, 5.0),
            field_oe=np.repeat([0.0, 100.0], 2),
            vsm_moment_emu=np.where(vsm_rows, 1e-3, np.nan),
            dc_moment_emu=np.where(vsm_rows, np.nan, 2e-3),
        )
    )
    with pytest.raises(AmbiguousColumnError):
        m.moment_method
    assert m.with_moment("Moment").moment_method == "VSM"
    assert m.with_moment("DC Moment Free Ctr").moment_method == "DC"


def _unreadable_code_file(tmp_path, name="code9.dat", app="MPMS3 w/ AC,1.0,1.1"):
    """A file whose Transport Action carries a code calamari has no family for."""
    return simulate.write_mpms3_file(
        tmp_path / name,
        app=app,
        temperature_K=[5.0, 6.0],
        field_oe=[0.0, 100.0],
        vsm_moment_emu=[1.0e-3, 2.0e-3],
        transport_action=[1.0, 9.0],
    )


def test_restrict_does_not_raise_on_a_family_question_nobody_asked(tmp_path):
    """`restrict` is general row-selection. A file whose codes calamari cannot
    read still loads, so it must still be filterable by temperature or field --
    otherwise the "raises only when asked" contract holds for `load` and then
    breaks on the next ordinary call."""
    m = cio.load(_unreadable_code_file(tmp_path))
    assert len(m.restrict(t_range=(4.0, 7.0)).data) == 2
    assert len(m.restrict(h_range=(0.0, 50.0)).data) == 1
    assert len(m.restrict(mask=[True, False]).data) == 1
    # ...and the family question itself still refuses, when actually asked.
    with pytest.raises(MeasurementFamilyError):
        m.measurement_types


def test_a_failed_classification_does_not_fall_back_to_the_header(tmp_path):
    """'unknown', not the BYAPP guess. Falling through would let a file whose
    rows plainly carry a VSM moment be catalogued 'AC' because its header names
    the AC application -- the exact substitution IO-4 exists to prevent."""
    m = cio.load(_unreadable_code_file(tmp_path, "hint.dat", app="SQUID AC"))
    assert m.measurement_type == "unknown"
    assert m.summary()["measurement_type"] == "unknown"


def test_a_blank_transport_action_cell_falls_back_to_the_columns(tmp_path):
    """A blank code states nothing, so it can neither decide nor disagree. It
    gets the same treatment as a file with no such column at all, per row."""
    m = cio.load(
        simulate.write_mpms3_file(
            tmp_path / "blank_ta.dat",
            temperature_K=[5.0, 6.0],
            field_oe=[0.0, 100.0],
            vsm_moment_emu=[1.0e-3, 2.0e-3],
            transport_action=[1.0, np.nan],
        )
    )
    assert list(m.row_families) == ["VSM", "VSM"]
    assert m.measurement_types == ("VSM",)


def test_an_aggregated_measurement_can_still_be_split(tmp_path):
    """of_type derives through restrict, so the subclass and its provenance
    survive with no code of its own."""
    m = cio.load(_combined_run(tmp_path, "agg.dat"))
    aggregated = aggregate_repeats(m)
    vsm = aggregated.of_type("VSM")
    assert isinstance(vsm, AggregatedMeasurement)
    assert vsm.measurement_type == "VSM"
    assert vsm.grouped_by == aggregated.grouped_by


# ---------------------------------------------------------------------------
# fmt= override
# ---------------------------------------------------------------------------


def test_load_fmt_override_matches_autodetect(mpms3_dc_file):
    auto = cio.load(mpms3_dc_file.path)
    forced = cio.load(mpms3_dc_file.path, fmt="MPMS3")
    assert forced.data_format == auto.data_format == "MPMS3"
    np.testing.assert_allclose(forced.moment_emu, auto.moment_emu)


def test_load_fmt_override_unknown_format_raises(mpms3_dc_file):
    with pytest.raises(ValueError):
        cio.load(mpms3_dc_file.path, fmt="NotAFormat")


def test_load_fmt_override_unknown_format_raises_unknown_format_error(mpms3_dc_file):
    """The bad-fmt error is the typed one, and names the formats that exist."""
    with pytest.raises(UnknownFormatError) as excinfo:
        cio.load(mpms3_dc_file.path, fmt="NotAFormat")
    message = str(excinfo.value)
    assert "NotAFormat" in message
    assert "MPMS3" in message


def test_load_fmt_override_does_not_read_the_file(tmp_path):
    """An unregistered fmt= is a caller error decided before any I/O -- so it
    raises UnknownFormatError even for a path that does not exist, rather
    than a misleading FileNotFoundError."""
    with pytest.raises(UnknownFormatError):
        cio.load(tmp_path / "does_not_exist.dat", fmt="NotAFormat")


# ---------------------------------------------------------------------------
# IO-5: load() is strict -- no best-effort 'unknown' parse
# ---------------------------------------------------------------------------


def _write_unrecognised(tmp_path: Path, name: str = "not_mpms.dat") -> Path:
    """A readable, plausible-looking .dat that is not an MPMS3 measurement export."""
    path = tmp_path / name
    path.write_text("Time,Value\n0,1\n1,2\n", encoding="utf-8")
    return path


def test_load_unrecognised_file_raises(tmp_path):
    """The headline IO-5 behaviour: no loader claims it, so load() refuses
    rather than returning a half-parsed Measurement."""
    path = _write_unrecognised(tmp_path)
    with pytest.raises(UnsupportedFileError):
        cio.load(path)


def test_load_unrecognised_file_error_is_actionable(tmp_path):
    """The message must name the file, quote what it actually found, and list
    the formats that are registered -- enough to diagnose without reopening
    the file."""
    path = _write_unrecognised(tmp_path)
    with pytest.raises(UnsupportedFileError) as excinfo:
        cio.load(path)
    message = str(excinfo.value)
    assert str(path) in message
    assert "Time,Value" in message  # the offending first line, quoted back
    assert "MPMS3" in message


def test_load_mpms_log_style_file_raises(tmp_path):
    """An MPMS Log Data file has the [Header]/[Data] shape but neither
    dialect's signature -- the look-alike case content sniffing exists for."""
    path = tmp_path / "log.dat"
    path.write_text(
        "[Header]\nTITLE,MPMS Log Data\n[Data]\nTime,Temperature (K)\n0,300.0\n",
        encoding="utf-8",
    )
    with pytest.raises(UnsupportedFileError):
        cio.load(path)


def test_load_empty_file_raises_with_empty_specific_message(tmp_path):
    path = tmp_path / "empty.dat"
    path.write_text("", encoding="utf-8")
    with pytest.raises(UnsupportedFileError) as excinfo:
        cio.load(path)
    assert "empty" in str(excinfo.value)


def test_unsupported_file_error_is_catchable_as_value_and_calamari_error(tmp_path):
    """Type contract: precise enough to catch specifically, conventional
    enough that an existing `except ValueError` still works."""
    path = _write_unrecognised(tmp_path)
    assert issubclass(UnsupportedFileError, CalamariError)
    assert issubclass(UnsupportedFileError, ValueError)
    with pytest.raises(CalamariError):
        cio.load(path)
    with pytest.raises(ValueError):
        cio.load(path)


def test_load_missing_file_raises_oserror_not_unsupported(tmp_path):
    """"Cannot read this file" and "this is not a magnetometry file" are
    different problems with different fixes, so they must not collapse into
    one exception -- a typo'd path must never be reported as a bad format."""
    missing = tmp_path / "no_such_file.dat"
    with pytest.raises(FileNotFoundError) as excinfo:
        cio.load(missing)
    assert not isinstance(excinfo.value, UnsupportedFileError)
    assert not isinstance(excinfo.value, CalamariError)


def test_load_directory_raises_oserror_not_unsupported(tmp_path):
    directory = tmp_path / "a_directory.dat"
    directory.mkdir()
    with pytest.raises(OSError) as excinfo:
        cio.load(directory)
    assert not isinstance(excinfo.value, UnsupportedFileError)


def test_probes_stay_non_raising_where_load_raises(tmp_path):
    """is_supported_file/sniff_format are the non-raising counterparts, and
    must keep working on exactly the inputs load() now rejects -- that is what
    lets find_files walk a tree of mixed content."""
    unrecognised = _write_unrecognised(tmp_path)
    missing = tmp_path / "no_such_file.dat"

    assert cio.is_supported_file(unrecognised) is False
    assert cio.sniff_format(unrecognised) is None
    assert cio.is_supported_file(missing) is False
    assert cio.sniff_format(missing) is None


def test_is_supported_file_agrees_with_load(tmp_path, mpms3_dc_file):
    """The two must not drift: is_supported_file is true exactly when load
    succeeds."""
    unrecognised = _write_unrecognised(tmp_path)

    assert cio.is_supported_file(mpms3_dc_file.path)
    assert cio.load(mpms3_dc_file.path).data_format == "MPMS3"

    assert not cio.is_supported_file(unrecognised)
    with pytest.raises(UnsupportedFileError):
        cio.load(unrecognised)


def test_load_never_produces_unknown_data_format(tmp_path, mpms3_dc_file):
    """'unknown' is gone as a data_format outcome: every Measurement load()
    returns names the real loader that produced it."""
    for synthetic in (mpms3_dc_file,):
        assert cio.load(synthetic.path).data_format == "MPMS3"


# ---------------------------------------------------------------------------
# file_role classification
# ---------------------------------------------------------------------------


def test_mpms3_file_role_is_summary(mpms3_dc_file):
    m = cio.load(mpms3_dc_file.path)
    assert m.file_role == "summary"


@pytest.mark.parametrize("filename", ["run.rw.dat", "RUN.RW.DAT"])
def test_mpms3_rw_dat_is_raw(tmp_path, filename):
    """MPMS3's raw-data file is 'BaseName.rw.dat'. Its
    suffix is '.dat', the same as a summary's, so it used to be classified
    'summary' and sit in a catalog among the real sweeps."""
    path = simulate.write_mpms3_file(
        tmp_path / filename,
        temperature_K=[5.0, 5.0],
        field_oe=[0.0, 0.0],
        dc_moment_emu=[1e-3, 1e-3],
    )
    assert cio.load(path).file_role == "raw"


def _raw_scan(tmp_path, filename):
    position = np.linspace(-15.0, 15.0, 5)
    return simulate.write_mpms3_raw_file(
        tmp_path / filename, raw_position_mm=position, raw_voltage_v=0.1 * position
    )


def test_mpms3_raw_header_token_identifies_raw_whatever_the_name(tmp_path):
    """Real MPMS3 raw files state their role in-file (BYAPP,MPMS3 DC RAW DATA),
    which outranks the name: a raw
    file renamed or copied under a summary-looking name is still raw."""
    m = cio.load(_raw_scan(tmp_path, "scan.dat"))
    assert m.data_format == "MPMS3"
    assert m.file_role == "raw"
    assert "Raw Voltage (V)" in m.data.columns


def test_mpms3_raw_header_shape_matches_observed(tmp_path):
    """The observed raw header has no '; MPMS3 Data File' comment -- the
    dialect is recognised from the BYAPP prefix alone -- and the name
    fallback still agrees on a real-style .rw.dat name."""
    lines = simulate.mpms3_raw_lines(raw_position_mm=[0.0], raw_voltage_v=[0.0])
    assert not any("MPMS3 DATA FILE" in ln.upper() for ln in lines)
    assert "BYAPP,MPMS3 DC RAW DATA,1.0,1.1" in lines
    assert cio.load(_raw_scan(tmp_path, "run.rw.dat")).file_role == "raw"


def test_mpms3_rw_needs_the_whole_extension(tmp_path):
    """Only the '.rw.dat' chain marks a raw file: 'rw' elsewhere in the name
    (a base name that merely ends in it) is still a summary."""
    path = simulate.write_mpms3_file(
        tmp_path / "sample_rw.dat",
        temperature_K=[5.0, 5.0],
        field_oe=[0.0, 0.0],
        dc_moment_emu=[1e-3, 1e-3],
    )
    assert cio.load(path).file_role == "summary"


# ---------------------------------------------------------------------------
# Column resolution: exact base matching + unit conversion
#
# Quantum Design records the operator's unit selection in the column-name
# suffix, so an accessor has to read the suffix to know what it is returning
# and base names must match exactly because sibling quantities have similar names.
# ---------------------------------------------------------------------------


def _mpms3_am2_file(tmp_path, **kwargs):
    """An MPMS3 file whose columns are labelled in the A m^2 unit system."""
    return simulate.write_mpms3_file(
        tmp_path / "am2.dat", unit_system="Am2", **kwargs
    )


def test_moment_in_am2_is_converted_to_emu(tmp_path):
    """1 A m^2 = 1000 emu. The accessor promises emu, so it must deliver emu
    whichever unit the operator selected."""
    path = _mpms3_am2_file(
        tmp_path,
        app="SQUID DC",
        temperature_K=[5.0, 5.0, 5.0],
        field_oe=[0.0, 0.1, 0.2],
        dc_moment_emu=[1.0e-6, 2.0e-6, 3.0e-6],  # written as (Am2)
    )
    m = cio.load(path)
    assert "DC Moment Free Ctr (Am2)" in m.data.columns
    np.testing.assert_allclose(m.moment_emu, [1.0e-3, 2.0e-3, 3.0e-3])


def test_field_in_tesla_is_converted_to_oe(tmp_path):
    path = _mpms3_am2_file(
        tmp_path,
        app="SQUID DC",
        temperature_K=[5.0, 5.0],
        field_oe=[0.5, 1.0],  # written as (T)
        dc_moment_emu=[1.0e-6, 2.0e-6],
    )
    m = cio.load(path)
    assert "Magnetic Field (T)" in m.data.columns
    np.testing.assert_allclose(m.field_oe, [5.0e3, 1.0e4])
    # field_tesla derives from field_oe, so it must come back to the original.
    np.testing.assert_allclose(m.field_tesla, [0.5, 1.0])


def test_susceptibility_in_am2_per_tesla_is_converted(tmp_path):
    """1 A m^2 / T = 1e3 emu / 1e4 Oe = 0.1 emu/Oe."""
    path = _mpms3_am2_file(
        tmp_path,
        app="SQUID AC",
        temperature_K=[2.0, 3.0],
        field_oe=[0.0, 0.0],
        chi_real=[1.0, 2.0],
        chi_imag=[0.5, 1.0],
        ac_drive_oe=[1.0e-4, 1.0e-4],  # written as (T)
    )
    m = cio.load(path)
    np.testing.assert_allclose(m.chi_real, [0.1, 0.2])
    np.testing.assert_allclose(m.chi_imag, [0.05, 0.1])
    # AC drive shares the field unit table: 1e-4 T = 1 Oe.
    np.testing.assert_allclose(m.ac_drive_oe, [1.0, 1.0])


def test_unrecognised_unit_raises_rather_than_guessing(tmp_path):
    """Finding the column but not understanding its unit must never return
    the numbers unscaled -- that is the silent unit error DM-2 exists to
    prevent -- nor None, which is indistinguishable from 'column absent'."""
    path = simulate.write_mpms3_file(
        tmp_path / "odd_units.dat",
        app="SQUID DC",
        temperature_K=[5.0, 5.0],
        field_oe=[0.0, 100.0],
        extra_columns={"Moment (furlongs)": [1.0, 2.0]},
    )
    m = cio.load(path)
    with pytest.raises(UnknownUnitError) as excinfo:
        m.moment_emu
    message = str(excinfo.value)
    assert "Moment (furlongs)" in message
    assert "furlongs" in message
    assert "emu" in message  # lists what it does understand


def test_unknown_unit_error_is_catchable_as_calamari_and_value_error():
    assert issubclass(UnknownUnitError, CalamariError)
    assert issubclass(UnknownUnitError, ValueError)


def test_base_matching_is_exact_not_substring(tmp_path):
    """'Temperature (K)' is a substring of the documented 'Min. Temperature
    (K)', and 'Field (Oe)' of 'Max. Field (Oe)'. A file carrying only the
    diagnostic siblings must yield None, never the wrong column."""
    path = simulate.write_mpms3_file(
        tmp_path / "siblings.dat",
        app="SQUID DC",
        temperature_K=[5.0, 5.0],
        field_oe=[0.0, 100.0],
        vsm_moment_emu=[1.0e-3, 2.0e-3],
        extra_columns={
            "Min. Temperature (K)": [4.9, 4.9],
            "Max. Field (Oe)": [0.1, 100.1],
        },
    )
    m = cio.load(path)
    # The real columns still win.
    np.testing.assert_allclose(m.temperature, [5.0, 5.0])
    np.testing.assert_allclose(m.field_oe, [0.0, 100.0])

    # With the real columns renamed away, the siblings must NOT be picked up.
    stripped = m.data.drop(columns=["Temperature (K)", "Magnetic Field (Oe)"])
    bare = Measurement(
        path=Path("siblings.dat"),
        data_format="MPMS3",
        measurement_type="DC",
        file_role="summary",
        title=None,
        header={},
        data=stripped,
    )
    assert bare.temperature is None
    assert bare.field_oe is None


def test_ac_sibling_columns_do_not_shadow_chi(tmp_path):
    """Component and error columns must not shadow each other by substring."""
    path = simulate.write_mpms3_file(
        tmp_path / "ac_siblings.dat",
        app="SQUID AC",
        temperature_K=[2.0, 3.0],
        field_oe=[0.0, 0.0],
        chi_real=[1.0, 2.0],
        chi_imag=[0.1, 0.2],
        extra_columns={
            "AC X' Std Err. (emu/Oe)": [0.01, 0.02],
            "AC X'' Std Err. (emu/Oe)": [0.001, 0.002],
        },
    )
    m = cio.load(path)
    np.testing.assert_allclose(m.chi_real, [1.0, 2.0])
    np.testing.assert_allclose(m.chi_imag, [0.1, 0.2])


def test_ac_frequency_ignores_vsm_oscillation_frequency(tmp_path):
    """Mechanical VSM frequency and AC drive frequency remain distinct."""
    path = simulate.write_mpms3_file(
        tmp_path / "freqs.dat",
        app="SQUID AC",
        temperature_K=[2.0, 3.0],
        field_oe=[0.0, 0.0],
        chi_real=[1.0, 2.0],
        ac_frequency_hz=[10.0, 10.0],
        extra_columns={"Frequency (Hz)": [13.7, 13.7]},
    )
    m = cio.load(path)
    np.testing.assert_allclose(m.ac_frequency, [10.0, 10.0])


def test_ac_frequency_is_none_when_only_vsm_frequency_present(tmp_path):
    path = simulate.write_mpms3_file(
        tmp_path / "vsm_only.dat",
        app="SQUID DC",
        temperature_K=[2.0, 3.0],
        field_oe=[0.0, 0.0],
        vsm_moment_emu=[1.0e-3, 2.0e-3],
        extra_columns={"Frequency (Hz)": [13.7, 13.7]},
    )
    m = cio.load(path)
    assert m.ac_frequency is None


# ---------------------------------------------------------------------------
# Shapes observed in real MPMS3 export files.
#
# Each test below reproduces, synthetically, a header or column layout that a
# real instrument writes and that calamari previously mis-parsed. They are
# grouped here because they share a cause: an instrument writes the full
# column and INFO set for every application it has installed, so a file is
# routinely full of empty columns, blank properties, and keys for hardware
# the analysis does not care about.
# ---------------------------------------------------------------------------


def test_unused_mode_column_does_not_shadow_the_populated_one(tmp_path):
    """An AC/DC run carries an empty VSM-mode 'Moment (emu)' alongside a
    populated 'DC Moment Free Ctr (emu)'. Both match the moment bases; taking
    the first name match returns all-NaN and discards every real value."""
    n = 6
    path = simulate.write_mpms3_file(
        tmp_path / "ac_dc.dat",
        app="MPMS3 w/ AC,1.0,1.1",
        temperature_K=np.linspace(2.0, 12.0, n),
        field_oe=np.full(n, 1000.0),
        vsm_moment_emu=np.full(n, np.nan),  # the unused mode's empty column
        dc_moment_emu=np.linspace(1e-4, 6e-4, n),
    )
    m = cio.load(path)
    assert "Moment (emu)" in m.data.columns  # present but empty
    assert m.moment_emu.notna().sum() == n
    assert m.moment_emu.name == "DC Moment Free Ctr (emu)"


def test_a_present_but_empty_column_reads_as_absent(tmp_path):
    """`None` is the documented way to ask "does this file have X?", so a
    column that exists but carries nothing must answer no.

    An earlier version returned the empty column instead, reasoning that
    "exists but is empty" and "no such column" are different states. They
    are -- but no accessor is the right place to express that difference,
    and answering "yes" with a NaN-filled series breaks the check the guide
    tells people to use. `data.columns` still distinguishes the two."""
    n = 4
    path = simulate.write_mpms3_file(
        tmp_path / "empty_moment.dat",
        app="SQUID DC",
        temperature_K=np.full(n, 5.0),
        field_oe=np.linspace(0.0, 100.0, n),
        vsm_moment_emu=np.full(n, np.nan),
    )
    m = cio.load(path)
    assert "Moment (emu)" in m.data.columns  # the column is there
    assert m.moment_emu is None  # ...and holds nothing


def test_dc_file_on_an_ac_equipped_instrument_has_no_chi(tmp_path):
    """Empty AC columns in a DC export must not supply a susceptibility."""
    n = 6
    path = simulate.write_mpms3_file(
        tmp_path / "dc_on_ac_rig.dat",
        app="MPMS3 w/ AC,1.0,1.1",
        temperature_K=np.linspace(2.0, 12.0, n),
        field_oe=np.full(n, 1000.0),
        dc_moment_emu=np.full(n, 1.0e-3),
        chi_real=np.full(n, np.nan),
        chi_imag=np.full(n, np.nan),
        ac_drive_oe=np.full(n, np.nan),
    )
    m = cio.load(path)
    assert "AC X' (emu/Oe)" in m.data.columns
    assert m.chi_real is None
    assert m.chi_imag is None
    # ...and consistent with the error accessors, which already did this
    assert m.chi_real_err is None
    # the DC data it does have is unaffected
    assert m.moment_emu.notna().sum() == n


def test_moment_columns_is_in_priority_order_not_file_order(tmp_path):
    """The docstring promises priority order; a file may write its columns
    in any order at all."""
    n = 4
    path = simulate.write_mpms3_file(
        tmp_path / "reordered.dat",
        app="SQUID DC",
        temperature_K=np.full(n, 5.0),
        field_oe=np.linspace(0.0, 100.0, n),
        extra_columns={
            # deliberately written in the opposite order to _MOMENT_BASES
            "DC Moment Free Ctr (emu)": np.full(n, 2.0e-3),
            "Moment (emu)": np.full(n, 1.0e-3),
        },
    )
    m = cio.load(path)
    assert [c for c in m.data.columns if "Moment" in c] == [
        "DC Moment Free Ctr (emu)",
        "Moment (emu)",
    ]
    assert m.moment_columns == ("Moment (emu)", "DC Moment Free Ctr (emu)")


def test_blank_info_value_becomes_none_not_empty_string(tmp_path):
    """Real files write a line for every sample property whether or not the
    operator filled it in ('INFO,,SAMPLE_MASS'). An empty string propagates
    into arithmetic as a str and fails deep inside a unit conversion."""
    path = simulate.write_mpms3_file(
        tmp_path / "blank_props.dat",
        app="SQUID DC",
        sample_material=None,
        sample_mass_mg=None,
        temperature_K=[5.0, 5.0],
        field_oe=[0.0, 100.0],
        vsm_moment_emu=[1e-4, 2e-4],
        extra_info={"SAMPLE_MASS": "", "SAMPLE_MATERIAL": "", "SAMPLE_COMMENT": ""},
    )
    m = cio.load(path)
    assert m.sample_mass_mg is None
    assert m.sample_name is None
    assert m.comment is None

    from calamari.sample import Sample

    sample = Sample.from_measurement(m)
    assert sample.mass_mg is None
    assert sample.mass_source is None
    # ...and the failure is the clear one, not a TypeError from ''/1000.0
    with pytest.raises(ValueError, match="has no mass_mg set"):
        sample.mass_g


def test_unknown_mpms3_info_keys_are_not_reversed(tmp_path):
    """Real MPMS3 headers carry many instrument keys calamari cannot
    enumerate (module names, hardware/software versions, serial numbers).
    Inferring INFO ordering from a known-key set reversed every one of them,
    storing the key as the value and vice versa."""
    path = simulate.write_mpms3_file(
        tmp_path / "instrument_keys.dat",
        app="MPMS3 w/ AC,1.0,1.1",
        temperature_K=[5.0, 5.0],
        field_oe=[0.0, 100.0],
        vsm_moment_emu=[1e-4, 2e-4],
        extra_info={
            "MOTOR_SERIAL_NUMBER": "MMC1376",
            "SQUID_MODULE_NAME": "Quantum Design Squid Module",
            "AC_SOFTWARE_VERSION": "00.95.07",
        },
    )
    info = cio.load(path).header["info"]
    assert info["motor_serial_number"] == "MMC1376"
    assert info["squid_module_name"] == "Quantum Design Squid Module"
    assert info["ac_software_version"] == "00.95.07"


def test_known_sample_keys_still_parse_alongside_unknown_ones(tmp_path):
    """The dialect-driven ordering must not regress the mapped fields."""
    path = simulate.write_mpms3_file(
        tmp_path / "mixed_keys.dat",
        app="SQUID DC",
        sample_material="SyntheticSample",
        sample_mass_mg=12.5,
        temperature_K=[5.0, 5.0],
        field_oe=[0.0, 100.0],
        vsm_moment_emu=[1e-4, 2e-4],
        extra_info={"COIL_SERIAL_NUMBER": "TCI394"},
    )
    m = cio.load(path)
    assert m.sample_name == "SyntheticSample"
    assert m.sample_mass_mg == pytest.approx(12.5)
    assert m.header["info"]["coil_serial_number"] == "TCI394"


def test_mpms3_byapp_naming_the_instrument_not_the_squid_mode(tmp_path):
    """BYAPP is undocumented and not always 'SQUID <mode>'; a real
    AC-equipped instrument writes 'MPMS3 w/ AC,1.0,1.1'. Detection and type
    inference must both survive it."""
    path = simulate.write_mpms3_file(
        tmp_path / "byapp_variant.dat",
        app="MPMS3 w/ AC,1.0,1.1",
        temperature_K=[2.0, 3.0],
        field_oe=[0.0, 0.0],
        chi_real=[1e-5, 2e-5],
        chi_imag=[1e-7, 2e-7],
        ac_drive_oe=[1.0, 1.0],
    )
    assert cio.sniff_format(path) == "MPMS3"
    m = cio.load(path)
    assert m.data_format == "MPMS3"
    assert m.measurement_type == "AC"


def test_interleaved_ac_and_dc_rows_both_readable(tmp_path):
    """A single file can alternate AC and DC measurements row by row, each
    populating only its own columns. Every accessor must return its own
    quantity's rows without being confused by the other's blanks."""
    n = 8
    ac_rows = np.arange(n) % 2 == 0
    chi = np.where(ac_rows, 3.0e-5, np.nan)
    drive = np.where(ac_rows, 1.0, np.nan)
    dc = np.where(ac_rows, np.nan, 5.0e-4)

    path = simulate.write_mpms3_file(
        tmp_path / "interleaved.dat",
        app="MPMS3 w/ AC,1.0,1.1",
        temperature_K=np.linspace(2.0, 16.0, n),
        field_oe=np.full(n, 1000.0),
        vsm_moment_emu=np.full(n, np.nan),
        chi_real=chi,
        ac_drive_oe=drive,
        dc_moment_emu=dc,
        transport_action=np.where(ac_rows, 5.0, 6.0),
    )
    m = cio.load(path)
    assert m.temperature.notna().sum() == n
    assert m.chi_real.notna().sum() == n // 2
    assert m.moment_emu.notna().sum() == n // 2
    assert m.moment_emu.name == "DC Moment Free Ctr (emu)"


def test_measurement_type_inference_is_unit_aware(tmp_path):
    """The data-driven AC/DC cross-check must survive the instrument's other
    unit selection. Matching whole spellings including "(emu)" made it
    unit-blind, so an A m^2 export matched nothing and the check silently
    degraded to trusting BYAPP -- which IO-4 exists because we cannot."""
    common = dict(
        app="SQUID AC",  # BYAPP lies: this is DC data
        temperature_K=[5.0, 5.0],
        field_oe=[0.0, 100.0],
        dc_moment_emu=[1.0e-3, 2.0e-3],
    )
    emu = simulate.write_mpms3_file(tmp_path / "dc_emu.dat", **common)
    si = simulate.write_mpms3_file(
        tmp_path / "dc_si.dat", unit_system="Am2", **common
    )
    assert cio.load(emu).measurement_type == "DC"
    assert cio.load(si).measurement_type == "DC"


def test_ac_type_inference_is_unit_aware(tmp_path):
    """The same for a genuinely AC file recorded in Am2/T."""
    common = dict(
        app="SQUID DC",  # BYAPP lies the other way
        temperature_K=[2.0, 3.0],
        field_oe=[0.0, 0.0],
        chi_real=[1.0e-5, 2.0e-5],
        ac_drive_oe=[1.0, 1.0],
    )
    emu = simulate.write_mpms3_file(tmp_path / "ac_emu.dat", **common)
    si = simulate.write_mpms3_file(tmp_path / "ac_si.dat", unit_system="Am2", **common)
    assert cio.load(emu).measurement_type == "AC"
    assert cio.load(si).measurement_type == "AC"


def test_colon_in_an_mpms3_info_value_is_not_swallowed(tmp_path):
    """Punctuation inside INFO values must not change their key/value interpretation."""
    path = simulate.write_mpms3_file(
        tmp_path / "colon.dat",
        app="SQUID DC",
        sample_comment="cooled at 1 K/min: run 2",
        temperature_K=[5.0, 5.0],
        field_oe=[0.0, 100.0],
        vsm_moment_emu=[1.0e-3, 2.0e-3],
    )
    m = cio.load(path)
    assert m.comment == "cooled at 1 K/min: run 2"
    assert "cooled at 1 k/min" not in m.header["info"]


# ---------------------------------------------------------------------------
# Choosing between several moment measurements
#
# A sequence can switch measurement mode partway through, or alternate row by
# row, leaving two moment columns each populated on a different subset. Taking
# either by priority returns half the file and leaves the rest NaN -- which is
# indistinguishable from points the instrument never measured, so the loss is
# invisible. The caller chooses instead.
# ---------------------------------------------------------------------------


def _two_moment_file(tmp_path, name, vsm, dc):
    return simulate.write_mpms3_file(
        tmp_path / name,
        app="MPMS3 w/ AC,1.0,1.1",
        temperature_K=np.linspace(2.0, 24.0, len(vsm)),
        field_oe=np.full(len(vsm), 1000.0),
        vsm_moment_emu=vsm,
        dc_moment_emu=dc,
    )


@pytest.mark.parametrize(
    "label, vsm, dc",
    [
        (
            "alternating",
            np.where(np.arange(12) % 2 == 0, 1e-3, np.nan),
            np.where(np.arange(12) % 2 == 0, np.nan, 2e-3),
        ),
        (
            "switches partway",
            np.where(np.arange(12) < 6, 1e-3, np.nan),
            np.where(np.arange(12) < 6, np.nan, 2e-3),
        ),
        ("both fully populated", np.full(12, 1e-3), np.full(12, 2e-3)),
    ],
)
def test_two_populated_moment_columns_refuse_to_guess(tmp_path, label, vsm, dc):
    m = cio.load(_two_moment_file(tmp_path, f"{label.split()[0]}.dat", vsm, dc))
    with pytest.raises(AmbiguousColumnError) as excinfo:
        m.moment_emu
    message = str(excinfo.value)
    assert "Moment (emu)" in message
    assert "DC Moment Free Ctr (emu)" in message
    assert "with_moment" in message  # the message shows the fix


def test_with_moment_selects_either_measurement(tmp_path):
    """One parse, both views -- and neither loses rows to the other."""
    vsm = np.where(np.arange(12) % 2 == 0, 1e-3, np.nan)
    dc = np.where(np.arange(12) % 2 == 0, np.nan, 2e-3)
    m = cio.load(_two_moment_file(tmp_path, "alt.dat", vsm, dc))

    assert m.moment_columns == ("Moment (emu)", "DC Moment Free Ctr (emu)")
    np.testing.assert_allclose(m.with_moment("Moment").moment_emu, vsm)
    np.testing.assert_allclose(
        m.with_moment("DC Moment Free Ctr").moment_emu, dc
    )


def test_with_moment_accepts_the_unit_suffixed_spelling(tmp_path):
    """A name copied from moment_columns or data.columns must work as-is."""
    vsm = np.full(4, 1e-3)
    dc = np.full(4, 2e-3)
    m = cio.load(_two_moment_file(tmp_path, "both.dat", vsm, dc))
    bare = m.with_moment("DC Moment Free Ctr").moment_emu
    suffixed = m.with_moment("DC Moment Free Ctr (emu)").moment_emu
    np.testing.assert_allclose(bare, suffixed)


def test_with_moment_still_converts_units(tmp_path):
    """Choosing a measurement must not opt out of the unit safety layer."""
    path = simulate.write_mpms3_file(
        tmp_path / "si.dat",
        app="SQUID DC",
        unit_system="Am2",
        temperature_K=[5.0, 5.0],
        field_oe=[0.0, 100.0],
        vsm_moment_emu=[1.0e-6, 2.0e-6],
    )
    m = cio.load(path).with_moment("Moment")
    np.testing.assert_allclose(m.moment_emu, [1.0e-3, 2.0e-3])


def test_with_moment_returns_a_new_object(tmp_path):
    vsm = np.full(4, 1e-3)
    m = cio.load(_two_moment_file(tmp_path, "new.dat", vsm, np.full(4, 2e-3)))
    chosen = m.with_moment("Moment")
    assert chosen is not m
    assert m.moment_base is None
    assert chosen.moment_base == "Moment"


def test_moment_choice_survives_restrict(tmp_path):
    vsm = np.full(12, 1e-3)
    m = cio.load(_two_moment_file(tmp_path, "restrict.dat", vsm, np.full(12, 2e-3)))
    narrowed = m.with_moment("DC Moment Free Ctr").restrict(t_range=(2.0, 10.0))
    assert narrowed.moment_base == "DC Moment Free Ctr"
    np.testing.assert_allclose(narrowed.moment_emu, 2e-3)


def test_with_moment_rejects_a_column_the_file_lacks(tmp_path, mpms3_dc_file):
    """Silently falling back to the default priority would hide the typo."""
    m = cio.load(mpms3_dc_file.path)
    with pytest.raises(ValueError, match="no column named"):
        m.with_moment("DC Moment Fixed Ctr")


def test_single_moment_column_needs_no_choice(mpms3_dc_file):
    """The overwhelmingly common case stays a plain attribute access."""
    m = cio.load(mpms3_dc_file.path)
    assert len(m.moment_columns) == 1
    assert m.moment_emu is not None


# ---------------------------------------------------------------------------
# repr, summary, and per-point errors
# ---------------------------------------------------------------------------


def test_repr_is_a_one_line_summary(mpms3_ac_file):
    """The dataclass default printed the whole header dict and DataFrame --
    thousands of characters as the first thing anyone sees in a notebook."""
    m = cio.load(mpms3_ac_file.path)
    r = repr(m)
    assert len(r) < 200
    assert "\n" not in r
    for expected in ("MPMS3", "SyntheticSample", "pts", "T-scan"):
        assert expected in r


def test_repr_survives_a_file_it_cannot_fully_read(tmp_path):
    """A repr that raised on exactly the odd file you were inspecting would
    be worse than useless."""
    path = simulate.write_mpms3_file(
        tmp_path / "odd.dat",
        app="SQUID DC",
        temperature_K=[5.0, 5.0],
        field_oe=[0.0, 100.0],
        extra_columns={"Moment (furlongs)": [1.0, 2.0]},
    )
    m = cio.load(path)
    with pytest.raises(UnknownUnitError):
        m.moment_emu  # the accessor still refuses
    assert repr(m).startswith("<Measurement ")  # ...but repr does not


def test_repr_reports_the_ac_frequency(mpms3_ac_file):
    """The frequency is part of *where* an AC run was measured, so the repr
    answers it without a trip to summary() -- and placed
    with the setpoints (after H, before the classification fields)."""
    for fixture in (mpms3_ac_file,):
        r = repr(cio.load(fixture.path))
        assert "10 Hz" in r
        assert r.index("10 Hz") < r.index("T-scan")


def test_repr_lists_every_distinct_frequency(tmp_path):
    """A two-frequency interleave shows both frequencies -- an average would
    describe neither measurement -- formatted exactly as summary() formats
    them, since both read the same helper."""
    path = simulate.write_mpms3_file(
        tmp_path / "interleave.dat",
        app="SQUID AC",
        temperature_K=[2.0, 2.0, 3.0, 3.0],
        field_oe=[100.0] * 4,
        chi_real=[1.0, 1.1, 2.0, 2.1],
        ac_frequency_hz=[97.0, 997.0, 97.0, 997.0],
        ac_drive_oe=[3.0] * 4,
    )
    m = cio.load(path)
    assert "97; 997 Hz" in repr(m)
    assert m.summary()["ac_frequencies_Hz"] == "97; 997"


def test_repr_of_a_file_without_ac_frequency_is_unchanged(mpms3_dc_file):
    """The bit is omitted, not blank: a DC file's repr gains nothing."""
    assert "Hz" not in repr(cio.load(mpms3_dc_file.path))


def test_frequency_summary_excludes_housekeeping_rows(tmp_path):
    """A locate scan's row can carry a leftover value in the frequency
    column; a frequency nothing was measured at must not reach the repr,
    summary or logbook -- the same exclusion the T/H spans apply."""
    path = simulate.write_mpms3_file(
        tmp_path / "hk_freq.dat",
        app="SQUID AC",
        temperature_K=[2.0, 3.0, 4.0, 8.5],
        field_oe=[100.0, 100.0, 100.0, 0.0],
        chi_real=[1.0, 1.1, 1.2, np.nan],
        ac_frequency_hz=[997.0, 997.0, 997.0, 7.0],  # 7 Hz on the locate row
        ac_drive_oe=[3.0, 3.0, 3.0, np.nan],
        transport_action=[5, 5, 5, 3],
    )
    m = cio.load(path)
    assert "997 Hz" in repr(m)
    assert "7;" not in repr(m)
    assert m.summary()["ac_frequencies_Hz"] == "997"
    # The raw accessor still reads every row -- only descriptions exclude.
    assert m.ac_frequency.tolist() == [997.0, 997.0, 997.0, 7.0]


def test_summary_needs_no_reparse_and_matches_the_catalog(mpms3_dc_file):
    """summarise() is the same information plus path-derived fields, so the
    two cannot drift."""
    from calamari import catalog

    m = cio.load(mpms3_dc_file.path)
    summary = m.summary()
    row = catalog.summarise(mpms3_dc_file.path)
    for key, value in summary.items():
        assert row[key] == value or (value is None and row[key] is None)


def test_moment_error_pairs_with_the_moment_in_use(tmp_path):
    """Each moment column names its own error column, so a with_moment()
    choice carries over and the two can never come from different
    measurements."""
    n = 4
    path = simulate.write_mpms3_file(
        tmp_path / "errs.dat",
        app="SQUID DC",
        temperature_K=np.full(n, 5.0),
        field_oe=np.linspace(0.0, 100.0, n),
        vsm_moment_emu=np.full(n, np.nan),
        dc_moment_emu=np.full(n, 1.0e-3),
        dc_moment_err_emu=np.full(n, 2.0e-6),
        extra_columns={
            # The fixed-centre pair has no parameter: no accessor returns it
            # by default, and with_moment is how a caller reaches it.
            "DC Moment Fixed Ctr (emu)": np.full(n, 1.1e-3),
            "DC Moment Err Fixed Ctr (emu)": np.full(n, 3.0e-6),
        },
    )
    m = cio.load(path)
    np.testing.assert_allclose(m.moment_err_emu, 2.0e-6)
    np.testing.assert_allclose(
        m.with_moment("DC Moment Fixed Ctr").moment_err_emu, 3.0e-6
    )


def test_moment_error_converts_with_the_same_unit_factor(tmp_path):
    """The error must scale exactly as the value does, or the fractional
    uncertainty changes with the operator's menu setting."""
    path = simulate.write_mpms3_file(
        tmp_path / "si_err.dat",
        app="SQUID DC",
        unit_system="Am2",
        temperature_K=[5.0, 5.0],
        field_oe=[0.0, 100.0],
        vsm_moment_emu=[1.0e-6, 2.0e-6],
        vsm_moment_err_emu=[1.0e-8, 2.0e-8],
    )
    m = cio.load(path)
    np.testing.assert_allclose(m.moment_emu, [1.0e-3, 2.0e-3])
    np.testing.assert_allclose(m.moment_err_emu, [1.0e-5, 2.0e-5])
    np.testing.assert_allclose(m.moment_err_emu / m.moment_emu, 0.01)


def test_chi_errors_preserve_fractional_uncertainty(tmp_path):
    """Recorded susceptibility and error retain the expected fractional uncertainty."""
    n = 3
    drive = 0.5
    mpms3 = simulate.write_mpms3_file(
        tmp_path / "m3.dat",
        app="SQUID AC",
        temperature_K=np.linspace(2.0, 6.0, n),
        field_oe=np.zeros(n),
        chi_real=np.full(n, 1.0e-4),
        ac_drive_oe=np.full(n, drive),
        extra_columns={"AC X' Std Err. (emu/Oe)": np.full(n, 1.0e-6)},
    )
    a = cio.load(mpms3)
    np.testing.assert_allclose(a.chi_real_err, 1.0e-6)
    # fractional error is the dialect-independent quantity
    assert float((a.chi_real_err / a.chi_real).iloc[0]) == pytest.approx(0.01)


def test_error_accessors_are_none_when_the_file_records_none(mpms3_dc_file):
    m = cio.load(mpms3_dc_file.path)
    assert m.moment_err_emu is None
    assert m.chi_real_err is None


# ---------------------------------------------------------------------------
# Measurement.restrict()
# ---------------------------------------------------------------------------


def test_restrict_t_range_keeps_only_matching_rows(tmp_path):
    path = simulate.write_mpms3_file(
        tmp_path / "tsweep.dat",
        temperature_K=[2.0, 5.0, 10.0, 20.0, 30.0],
        field_oe=[100.0] * 5,
        vsm_moment_emu=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    m = cio.load(path)
    restricted = m.restrict(t_range=(4.0, 15.0))
    np.testing.assert_allclose(restricted.temperature, [5.0, 10.0])
    np.testing.assert_allclose(restricted.moment_emu, [2.0, 3.0])


def test_restrict_h_range_keeps_only_matching_rows(tmp_path):
    path = simulate.write_mpms3_file(
        tmp_path / "hsweep.dat",
        temperature_K=[5.0] * 5,
        field_oe=[-100.0, -50.0, 0.0, 50.0, 100.0],
        vsm_moment_emu=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    m = cio.load(path)
    restricted = m.restrict(h_range=(-50.0, 50.0))
    np.testing.assert_allclose(restricted.field_oe, [-50.0, 0.0, 50.0])
    np.testing.assert_allclose(restricted.moment_emu, [2.0, 3.0, 4.0])


def test_restrict_mask_keeps_only_matching_rows(tmp_path):
    path = simulate.write_mpms3_file(
        tmp_path / "mask.dat",
        temperature_K=[2.0, 5.0, 10.0],
        field_oe=[100.0, 100.0, 100.0],
        vsm_moment_emu=[1.0, 2.0, 3.0],
    )
    m = cio.load(path)
    restricted = m.restrict(mask=[True, False, True])
    np.testing.assert_allclose(restricted.temperature, [2.0, 10.0])


def test_restrict_combines_filters_with_and(tmp_path):
    path = simulate.write_mpms3_file(
        tmp_path / "combo.dat",
        temperature_K=[2.0, 5.0, 10.0, 20.0],
        field_oe=[0.0, 100.0, 100.0, 100.0],
        vsm_moment_emu=[1.0, 2.0, 3.0, 4.0],
    )
    m = cio.load(path)
    # t_range alone would keep rows 1-3; h_range alone would keep rows 1-3
    # too (row 0's H=0 fails); AND of both must still keep exactly 1-3.
    restricted = m.restrict(t_range=(4.0, 30.0), h_range=(50.0, 150.0))
    np.testing.assert_allclose(restricted.temperature, [5.0, 10.0, 20.0])


def test_restrict_returns_new_object_original_untouched(tmp_path):
    path = simulate.write_mpms3_file(
        tmp_path / "tsweep.dat",
        temperature_K=[2.0, 5.0, 10.0],
        field_oe=[100.0, 100.0, 100.0],
        vsm_moment_emu=[1.0, 2.0, 3.0],
    )
    m = cio.load(path)
    restricted = m.restrict(t_range=(0.0, 6.0))

    assert restricted is not m
    assert len(restricted.data) == 2
    assert len(m.data) == 3
    np.testing.assert_allclose(m.temperature, [2.0, 5.0, 10.0])
    # header must be a distinct dict object, not shared with the original.
    assert restricted.header is not m.header


def test_restrict_no_filters_returns_full_copy(tmp_path):
    path = simulate.write_mpms3_file(
        tmp_path / "tsweep.dat",
        temperature_K=[2.0, 5.0, 10.0],
        field_oe=[100.0, 100.0, 100.0],
        vsm_moment_emu=[1.0, 2.0, 3.0],
    )
    m = cio.load(path)
    restricted = m.restrict()
    np.testing.assert_allclose(restricted.temperature, m.temperature)
    assert restricted is not m


def test_restrict_empty_result_keeps_columns(tmp_path):
    path = simulate.write_mpms3_file(
        tmp_path / "tsweep.dat",
        temperature_K=[2.0, 5.0, 10.0],
        field_oe=[100.0, 100.0, 100.0],
        vsm_moment_emu=[1.0, 2.0, 3.0],
    )
    m = cio.load(path)
    restricted = m.restrict(t_range=(100.0, 200.0))
    assert len(restricted.data) == 0
    assert list(restricted.data.columns) == list(m.data.columns)


def test_restrict_t_range_without_temperature_column_raises():
    df = pd.DataFrame({"Magnetic Field (Oe)": [0.0, 1.0]})
    m = Measurement(
        path=Path("synthetic.dat"),
        data_format="MPMS3",
        measurement_type="DC",
        file_role="summary",
        title=None,
        header={},
        data=df,
    )
    with pytest.raises(ValueError):
        m.restrict(t_range=(0.0, 10.0))


def test_restrict_nan_dropped_only_by_referencing_filter():
    """A NaN in the temperature column fails t_range regardless of whether
    the same row's field is fine, and vice versa for h_range/field -- each
    filter only judges the column it reads."""
    df = pd.DataFrame(
        {
            "Temperature (K)": [2.0, np.nan, 10.0, 20.0],
            "Magnetic Field (Oe)": [100.0, 100.0, np.nan, 100.0],
            "Moment (emu)": [1.0, 2.0, 3.0, 4.0],
        }
    )
    m = Measurement(
        path=Path("synthetic.dat"),
        data_format="MPMS3",
        measurement_type="DC",
        file_role="summary",
        title=None,
        header={},
        data=df,
    )

    # t_range excludes the NaN-temperature row (index 1); the NaN-field row
    # (index 2) has a fine temperature and is kept.
    by_t = m.restrict(t_range=(0.0, 30.0))
    np.testing.assert_allclose(by_t.temperature, [2.0, 10.0, 20.0])

    # h_range excludes the NaN-field row (index 2); the NaN-temperature row
    # (index 1) has a fine field and is kept.
    by_h = m.restrict(h_range=(0.0, 200.0))
    np.testing.assert_allclose(by_h.field_oe, [100.0, 100.0, 100.0])


# ---------------------------------------------------------------------------
# Identity, equality and hashing
#
# The generated dataclass __eq__ compared the fields as tuples, and comparing
# two DataFrames with == is elementwise, so asking for the result's truth
# value raised. Tuple comparison checks identity first, which hid the bug
# whenever two objects happened to share one frame -- but every derived
# Measurement has a frame of its own.
# ---------------------------------------------------------------------------


def test_comparing_two_derived_measurements_does_not_raise(mpms3_dc_file):
    """restrict() builds a new DataFrame, so two restricted objects held
    distinct-but-equal frames and comparing them raised ValueError -- a
    comparison that could not be performed at all, not one that answered
    wrongly."""
    m = cio.load(mpms3_dc_file.path)
    a, b = m.restrict(), m.restrict()

    assert (a == b) is False  # distinct objects, and no exception
    assert (a != b) is True


def test_membership_testing_does_not_raise(mpms3_dc_file):
    """`in` is implemented with ==, so it inherited the same failure."""
    m = cio.load(mpms3_dc_file.path)
    others = [m.restrict(), m.restrict()]

    assert m not in others
    assert m in [*others, m]


def test_a_measurement_is_its_own_equal(mpms3_dc_file):
    """Identity comparison still has to satisfy reflexivity."""
    m = cio.load(mpms3_dc_file.path)
    assert m == m
    assert m.restrict() != m  # a derivation is a different object


def test_a_measurement_is_hashable(mpms3_dc_file):
    """A dataclass with eq=True and frozen=False has __hash__ set to None,
    so a Measurement could not go in a set or serve as a dict key."""
    m = cio.load(mpms3_dc_file.path)
    derived = m.restrict()

    assert hash(m) == hash(m)
    assert len({m, derived, m}) == 2
    assert {m: "original", derived: "restricted"}[m] == "original"


def test_legacy_format_is_unsupported(tmp_path):
    path = tmp_path / "legacy.dat"
    path.write_text(
        "[Header]\nTITLE,MPMS DC Measurement\n"
        "BYAPP,MPMS Measurement,1.0,Summary\n[Data]\n"
        "Time,Temperature (K),Field (Oe),Long Moment (emu)\n0,12,250,0.003\n"
    )
    assert cio.sniff_format(path) is None
    assert not cio.is_supported_file(path)
    with pytest.raises(UnsupportedFileError):
        cio.load(path)
    with pytest.raises(UnknownFormatError):
        cio.load(path, fmt="MultiVu")
