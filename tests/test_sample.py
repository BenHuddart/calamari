"""Tests for calamari.sample: Sample and its unit conversions.

Exercised entirely against synthetic values and the synthetic-file fixtures
from tests/conftest.py -- no real instrument data or project-specific
constants (mass, molar mass, ion count) are used; every numeric fact below
is invented for the test and independent of any real compound.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from calamari import io as cio
from calamari import simulate
from calamari.constants import AVOGADRO, K_B_ERG_PER_K, MU_B_EMU
from calamari.sample import (
    Sample,
    curie_constant_from_inverse_slope,
    mu_eff_from_curie_constant,
)

# ---------------------------------------------------------------------------
# A fully-specified sample used across the exact-value conversion tests.
# Numbers are arbitrary (not any real compound's mass/molar mass/ion count).
# ---------------------------------------------------------------------------
_MASS_MG = 10.0
_MOLAR_MASS = 50.0
_IONS_PER_FU = 2

_MASS_G = _MASS_MG / 1000.0
_MOLES = _MASS_G / _MOLAR_MASS
_FORMULA_UNITS = _MOLES * AVOGADRO
_MOLES_IONS = _MOLES * _IONS_PER_FU


def _full_sample(**overrides) -> Sample:
    fields = dict(
        name="test-sample",
        mass_mg=_MASS_MG,
        molar_mass_g_per_mol=_MOLAR_MASS,
        magnetic_ions_per_fu=_IONS_PER_FU,
    )
    fields.update(overrides)
    return Sample(**fields)


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_sample_is_frozen():
    sample = _full_sample()
    with pytest.raises(dataclasses.FrozenInstanceError):
        sample.mass_mg = 99.0


# ---------------------------------------------------------------------------
# Derived properties: exact values and missing-field ValueErrors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("property_name, expected", [
    ("mass_g", _MASS_G), ("moles", _MOLES),
    ("formula_units", _FORMULA_UNITS), ("moles_magnetic_ions", _MOLES_IONS),
])
def test_derived_property(property_name, expected):
    assert getattr(_full_sample(), property_name) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("property_name, facts, missing", [
    ("mass_g", {}, "mass_mg"),
    ("moles", {"mass_mg": _MASS_MG}, "molar_mass_g_per_mol"),
    ("moles", {"molar_mass_g_per_mol": _MOLAR_MASS}, "mass_mg"),
    ("formula_units", {"mass_mg": _MASS_MG}, "molar_mass_g_per_mol"),
    ("moles_magnetic_ions", {"mass_mg": _MASS_MG, "molar_mass_g_per_mol": _MOLAR_MASS},
     "magnetic_ions_per_fu"),
])
def test_derived_property_missing_fact(property_name, facts, missing):
    sample = Sample(name="synthetic", **facts)
    with pytest.raises(ValueError, match=missing):
        getattr(sample, property_name)


# ---------------------------------------------------------------------------
# Conversion methods: exact values, independently computed
# ---------------------------------------------------------------------------


def test_moment_to_bohr_magnetons_per_formula_unit():
    sample = _full_sample()
    moment_emu = 2.5e-4
    expected = moment_emu / MU_B_EMU / _FORMULA_UNITS
    result = sample.moment_emu_to_muB_per_fu(moment_emu)
    assert result == pytest.approx(expected, rel=1e-6)


def test_moment_to_bohr_magnetons_per_formula_unit_array():
    sample = _full_sample()
    moment_emu = np.array([1.0e-4, 2.5e-4, -3.0e-4])
    expected = moment_emu / MU_B_EMU / _FORMULA_UNITS
    result = sample.moment_emu_to_muB_per_fu(moment_emu)
    np.testing.assert_allclose(result, expected, rtol=1e-6)
    assert np.shape(result) == np.shape(moment_emu)


def test_moment_to_bohr_magnetons_per_formula_unit_missing_field_raises():
    sample = Sample(name="no-molar-mass", mass_mg=_MASS_MG)
    with pytest.raises(ValueError, match="molar_mass_g_per_mol"):
        sample.moment_emu_to_muB_per_fu(1.0e-4)


def test_moment_to_bohr_magnetons_per_ion():
    sample = _full_sample()
    moment_emu = 2.5e-4
    expected_per_fu = moment_emu / MU_B_EMU / _FORMULA_UNITS
    expected_per_ion = expected_per_fu / _IONS_PER_FU
    result = sample.moment_emu_to_muB_per_ion(moment_emu)
    assert result == pytest.approx(expected_per_ion, rel=1e-6)


def test_moment_to_bohr_magnetons_per_ion_missing_ions_raises():
    sample = Sample(name="no-ions", mass_mg=_MASS_MG, molar_mass_g_per_mol=_MOLAR_MASS)
    with pytest.raises(ValueError, match="magnetic_ions_per_fu"):
        sample.moment_emu_to_muB_per_ion(1.0e-4)


def test_chi_molar():
    sample = _full_sample()
    moment_emu = 2.5e-4
    field_oe = 1000.0
    expected = (moment_emu / field_oe) / _MOLES
    result = sample.moment_to_chi_molar(moment_emu, field_oe)
    assert result == pytest.approx(expected, rel=1e-6)


def test_chi_molar_missing_molar_mass_raises():
    sample = Sample(name="no-molar-mass", mass_mg=_MASS_MG)
    with pytest.raises(ValueError, match="molar_mass_g_per_mol"):
        sample.moment_to_chi_molar(1.0e-4, 1000.0)


def test_chi_molar_per_ion():
    sample = _full_sample()
    moment_emu = 2.5e-4
    field_oe = 1000.0
    expected = (moment_emu / field_oe) / _MOLES_IONS
    result = sample.moment_to_chi_molar_per_ion(moment_emu, field_oe)
    assert result == pytest.approx(expected, rel=1e-6)


def test_chi_molar_per_ion_missing_ions_raises():
    sample = Sample(name="no-ions", mass_mg=_MASS_MG, molar_mass_g_per_mol=_MOLAR_MASS)
    with pytest.raises(ValueError, match="magnetic_ions_per_fu"):
        sample.moment_to_chi_molar_per_ion(1.0e-4, 1000.0)


def test_chi_mass():
    sample = _full_sample()
    moment_emu = 2.5e-4
    field_oe = 1000.0
    expected = (moment_emu / field_oe) / _MASS_G
    result = sample.chi_mass(moment_emu, field_oe)
    assert result == pytest.approx(expected, rel=1e-6)


def test_chi_mass_missing_mass_raises():
    sample = Sample(name="no-mass", molar_mass_g_per_mol=_MOLAR_MASS)
    with pytest.raises(ValueError, match="mass_mg"):
        sample.chi_mass(1.0e-4, 1000.0)


# ---------------------------------------------------------------------------
# from_measurement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("app, signal", [("SQUID DC", "dc_moment_emu"), ("SQUID AC", "chi_real")])
@pytest.mark.parametrize("mass", [None, 10.0])
def test_from_measurement_retains_file_mass_provenance(tmp_path, app, signal, mass):
    path = simulate.write_mpms3_file(
        tmp_path / "sample.dat", app=app, sample_material="SyntheticSample",
        sample_mass_mg=mass, temperature_K=[5.0, 6.0], field_oe=[100.0, 100.0],
        **{signal: [0.001, 0.002]},
    )
    sample = Sample.from_measurement(cio.load(path))
    assert sample.name == "SyntheticSample"
    assert sample.mass_mg == mass
    assert sample.mass_source == ("file" if mass is not None else None)


def test_from_measurement_uses_mpms3_sample_mass(mpms3_dc_file):
    m = cio.load(mpms3_dc_file.path)
    sample = Sample.from_measurement(m)
    assert sample.mass_mg == pytest.approx(10.0)
    assert sample.mass_source == "file"


def test_caller_mass_beats_the_file(mpms3_dc_file, tmp_path):
    """The whole point of the precedence: a curated logbook value must never
    lose silently to whatever was typed into the instrument's wizard."""
    m = cio.load(mpms3_dc_file.path)
    assert m.sample_mass_mg == pytest.approx(10.0)

    sample = Sample.from_measurement(m, mass_mg=4.25)
    assert sample.mass_mg == pytest.approx(4.25)
    assert sample.mass_source == "user"


def test_none_mass_mg_means_use_the_file_not_no_mass(mpms3_dc_file):
    """`mass_mg=logbook.get(name)` must behave sensibly for a sample with no
    logbook entry -- None means "fall through", not "assert there is none"."""
    m = cio.load(mpms3_dc_file.path)
    sample = Sample.from_measurement(m, mass_mg=None)
    assert sample.mass_mg == pytest.approx(10.0)
    assert sample.mass_source == "file"


def test_from_measurement_no_mass_anywhere(tmp_path):
    path = simulate.write_mpms3_file(
        tmp_path / "mpms3_no_mass.dat",
        app="SQUID DC",
        sample_material="NoMassSample",
        sample_mass_mg=None,
        temperature_K=[5.0, 5.0],
        field_oe=[0.0, 100.0],
        vsm_moment_emu=[0.0, 1.0e-4],
    )
    m = cio.load(path)
    assert m.sample_mass_mg is None

    sample = Sample.from_measurement(m)
    assert sample.mass_mg is None
    assert sample.mass_source is None

    supplied = Sample.from_measurement(m, mass_mg=7.5)
    assert supplied.mass_mg == pytest.approx(7.5)
    assert supplied.mass_source == "user"


def test_directly_constructed_sample_records_no_provenance():
    """`None` means "not recorded", never "trust me" -- a hand-built Sample
    does not get to claim its mass was user-verified."""
    sample = Sample(name="hand-built", mass_mg=5.0)
    assert sample.mass_source is None


@pytest.mark.parametrize("keyword", ["unknown_field"])
def test_from_measurement_rejects_unknown_overrides(mpms3_dc_file, keyword):
    m = cio.load(mpms3_dc_file.path)
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{keyword}'"):
        Sample.from_measurement(m, **{keyword: 1.0})


def test_from_measurement_overrides(mpms3_dc_file):
    m = cio.load(mpms3_dc_file.path)
    sample = Sample.from_measurement(
        m,
        name="Overridden",
        mass_mg=123.0,
        molar_mass_g_per_mol=99.0,
        magnetic_ions_per_fu=3,
        shape="plate",
        orientation="H // c",
        comment="an override",
    )
    assert sample.name == "Overridden"
    assert sample.mass_mg == pytest.approx(123.0)
    assert sample.molar_mass_g_per_mol == pytest.approx(99.0)
    assert sample.magnetic_ions_per_fu == 3
    assert sample.shape == "plate"
    assert sample.orientation == "H // c"
    assert sample.comment == "an override"


def test_from_measurement_name_defaults_to_unknown_when_absent(tmp_path):
    path = simulate.write_mpms3_file(
        tmp_path / "mpms3_no_name.dat",
        app="SQUID DC",
        sample_material=None,
        sample_mass_mg=5.0,
        temperature_K=[5.0, 5.0],
        field_oe=[0.0, 100.0],
        vsm_moment_emu=[0.0, 1.0e-4],
    )
    m = cio.load(path)
    assert m.sample_name is None
    sample = Sample.from_measurement(m)
    assert sample.name == "unknown"


# ---------------------------------------------------------------------------
# mu_eff_from_curie_constant
# ---------------------------------------------------------------------------


def test_mu_eff_from_curie_constant_matches_independent_formula():
    curie_constant = 0.42
    expected = np.sqrt(3.0 * K_B_ERG_PER_K * curie_constant / AVOGADRO) / MU_B_EMU
    result = mu_eff_from_curie_constant(curie_constant)
    assert result == pytest.approx(expected, rel=1e-12)


def test_mu_eff_from_curie_constant_matches_legacy_rounded_coefficient():
    # The conventional rounded coefficient is 2.827; the exact CODATA
    # factors used here evaluate to approximately 2.8279 (see the
    # docstring of mu_eff_from_curie_constant), differing only in the
    # rounding precision -- not a different convention.
    result = mu_eff_from_curie_constant(1.0)
    assert result == pytest.approx(2.8279, abs=2e-4)


# ---------------------------------------------------------------------------
# Round-trip: recover a known per-ion effective moment from a saturating M(H)
# ---------------------------------------------------------------------------


def test_recover_bohr_magnetons_per_ion_from_saturating_mh_loop():
    sample = _full_sample()
    target_mu_b_per_ion = 7.0

    # Invert moment_emu_to_muB_per_ion to get the saturation moment (emu)
    # that should read back as target_mu_b_per_ion at full saturation.
    m_sat_emu = target_mu_b_per_ion * _IONS_PER_FU * _FORMULA_UNITS * MU_B_EMU

    field_oe = np.linspace(-10_000.0, 10_000.0, 41)
    rng = np.random.default_rng(20260722)
    moment_emu = simulate.synthetic_mh_curve(
        field_oe,
        m_sat_emu=m_sat_emu,
        h_knee_oe=2000.0,
        noise=0.01 * m_sat_emu,
        rng=rng,
    )

    # Near the field extremum the tanh curve is within tanh(5) ~ 0.9999 of
    # saturation, so the recovered moment should sit close to the target
    # despite the added noise.
    near_saturation = moment_emu[np.argmax(field_oe)]
    recovered = sample.moment_emu_to_muB_per_ion(near_saturation)
    assert recovered == pytest.approx(target_mu_b_per_ion, abs=0.5)


# ---------------------------------------------------------------------------
# Curie-Weiss: the inverse-susceptibility slope conversion
#
# Until v0.3 mu_eff_from_curie_constant's docstring told callers to convert
# an inverse-susceptibility slope with C = 1/(3a). The factor of 3 belongs to
# C = N mu_eff^2 / 3 k_B and is already applied inside that function, so
# following the instruction returned every effective moment low by sqrt(3),
# silently, at a plausible-looking value.
# ---------------------------------------------------------------------------


def test_curie_constant_from_inverse_slope_recovers_mu_eff():
    """End-to-end: plant a known mu_eff, fit 1/chi vs T, get it back."""
    mu_eff_true = 5.92  # muB; a spin-only S=5/2 ion
    C_true = (mu_eff_true * MU_B_EMU) ** 2 * AVOGADRO / (3.0 * K_B_ERG_PER_K)
    theta = -10.0
    T = np.linspace(50.0, 300.0, 100)
    chi = C_true / (T - theta)

    slope, intercept = np.polyfit(T, 1.0 / chi, 1)
    C = curie_constant_from_inverse_slope(slope)

    assert C == pytest.approx(C_true, rel=1e-9)
    assert mu_eff_from_curie_constant(C) == pytest.approx(mu_eff_true, rel=1e-9)
    # x-intercept of the 1/chi line is theta
    assert -intercept / slope == pytest.approx(theta, rel=1e-9)


def test_inverse_slope_conversion_carries_no_factor_of_three():
    """The retired C = 1/(3a) advice is low by exactly sqrt(3) in mu_eff."""
    slope = 0.22818
    correct = mu_eff_from_curie_constant(curie_constant_from_inverse_slope(slope))
    old_advice = mu_eff_from_curie_constant(1.0 / (3.0 * slope))
    assert correct / old_advice == pytest.approx(np.sqrt(3.0), rel=1e-12)


@pytest.mark.parametrize("bad", [0.0, np.nan, np.inf])
def test_inverse_slope_rejects_degenerate_fits(bad):
    """Returning inf would propagate into mu_eff as a plausible number."""
    with pytest.raises(ValueError, match="finite and non-zero"):
        curie_constant_from_inverse_slope(bad)


def test_sample_has_no_demag_factor():
    """Removed in v0.4: it was stored but unappliable, since Sample carries no
    volume from which a magnetisation could be formed."""
    assert not hasattr(Sample(name="x"), "demag_factor")
    with pytest.raises(TypeError):
        Sample(name="x", demag_factor=0.3)


def test_header_molar_mass_is_exposed_but_not_adopted(tmp_path):
    """A molar mass is a property of the compound, not the specimen -- the
    caller always has a better source, so it is offered rather than used."""
    path = simulate.write_mpms3_file(
        tmp_path / "mw.dat",
        app="SQUID DC",
        sample_material="S",
        sample_mass_mg=10.0,
        temperature_K=[5.0, 5.0],
        field_oe=[0.0, 100.0],
        vsm_moment_emu=[1.0e-3, 2.0e-3],
        extra_info={"SAMPLE_MOLECULAR_WEIGHT": "200.5"},
    )
    m = cio.load(path)
    assert m.sample_molar_mass_g_per_mol == pytest.approx(200.5)  # numeric, not str

    assert Sample.from_measurement(m).molar_mass_g_per_mol is None  # not adopted
    opted_in = Sample.from_measurement(
        m, molar_mass_g_per_mol=m.sample_molar_mass_g_per_mol
    )
    assert opted_in.molar_mass_g_per_mol == pytest.approx(200.5)


def test_header_molar_mass_absent_is_none(mpms3_dc_file):
    assert cio.load(mpms3_dc_file.path).sample_molar_mass_g_per_mol is None


# ---------------------------------------------------------------------------
# chi -> chi conversions (the AC route: the file already hands us emu/Oe)
# ---------------------------------------------------------------------------


def test_chi_to_chi_molar():
    sample = _full_sample()
    chi = 5.0e-7
    assert sample.chi_to_chi_molar(chi) == pytest.approx(chi / _MOLES, rel=1e-6)


def test_chi_to_chi_molar_per_ion():
    sample = _full_sample()
    chi = 5.0e-7
    assert sample.chi_to_chi_molar_per_ion(chi) == pytest.approx(
        chi / _MOLES_IONS, rel=1e-6
    )


def test_chi_to_chi_mass():
    sample = _full_sample()
    chi = 5.0e-7
    assert sample.chi_to_chi_mass(chi) == pytest.approx(chi / _MASS_G, rel=1e-6)


def test_chi_routes_agree():
    """chi_to_chi_*(M/H) must equal moment_to_chi_*(M, H): the AC and DC
    routes to the same physical quantity cannot be allowed to drift."""
    sample = _full_sample()
    moment_emu, field_oe = 2.5e-4, 1000.0
    assert sample.chi_to_chi_molar(moment_emu / field_oe) == pytest.approx(
        sample.moment_to_chi_molar(moment_emu, field_oe), rel=1e-12
    )
    assert sample.chi_to_chi_molar_per_ion(moment_emu / field_oe) == pytest.approx(
        sample.moment_to_chi_molar_per_ion(moment_emu, field_oe), rel=1e-12
    )
    assert sample.chi_to_chi_mass(moment_emu / field_oe) == pytest.approx(
        sample.chi_mass(moment_emu, field_oe), rel=1e-12
    )


def test_chi_to_chi_missing_facts_raise():
    no_molar = Sample(name="no-molar-mass", mass_mg=_MASS_MG)
    with pytest.raises(ValueError, match="molar_mass_g_per_mol"):
        no_molar.chi_to_chi_molar(1.0e-6)
    no_ions = Sample(name="no-ions", mass_mg=_MASS_MG, molar_mass_g_per_mol=_MOLAR_MASS)
    with pytest.raises(ValueError, match="magnetic_ions_per_fu"):
        no_ions.chi_to_chi_molar_per_ion(1.0e-6)
    no_mass = Sample(name="no-mass", molar_mass_g_per_mol=_MOLAR_MASS)
    with pytest.raises(ValueError, match="mass_mg"):
        no_mass.chi_to_chi_mass(1.0e-6)


@pytest.mark.parametrize('field', ['mass_mg', 'molar_mass_g_per_mol', 'magnetic_ions_per_fu'])
@pytest.mark.parametrize('value', [0, -1, np.nan, np.inf, -np.inf, True, np.bool_(True), '2', [2], np.array(2)])
def test_invalid_normalisation_fact(field, value):
    from calamari import CalamariError, SampleValidationError

    with pytest.raises(SampleValidationError, match=field) as caught:
        Sample('synthetic', **{field: value})
    assert isinstance(caught.value, (CalamariError, ValueError))


def test_fractional_occupancy_conversion():
    sample = Sample('synthetic', mass_mg=np.float64(10), molar_mass_g_per_mol=200,
                    magnetic_ions_per_fu=0.8)
    # 0.01 g / (200 g/mol), with 0.8 ions per formula unit.
    assert sample.moles_magnetic_ions == pytest.approx(4e-5)
    from calamari.constants import AVOGADRO, MU_B_EMU

    expected = 1e-3 / (4e-5 * AVOGADRO * MU_B_EMU)
    assert sample.moment_emu_to_muB_per_ion(1e-3) == pytest.approx(expected)


def test_missing_sample_fact_uses_common_error():
    from calamari import SampleValidationError

    sample = Sample('synthetic')
    with pytest.raises(SampleValidationError, match='mass_mg'):
        _ = sample.mass_g


def test_real_scalar_types_are_valid():
    from fractions import Fraction

    for value in (Fraction(1, 2), 10**1000, np.float32(0.8), np.longdouble(0.8), np.int64(2)):
        assert Sample('synthetic', magnetic_ions_per_fu=value).magnetic_ions_per_fu == value
