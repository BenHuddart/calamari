"""Shared synthetic-file fixtures for the calamari test suite.

Per the project's no-real-data-files policy (see :mod:`calamari.simulate`),
every fixture here writes a freshly generated file into pytest's ``tmp_path``
and returns a :class:`SyntheticFile` bundling the written path together with
the ground-truth arrays used to build it, so a test can assert accessors
against known values with ``numpy.testing.assert_allclose`` rather than
eyeballing the synthetic file's content.

The MPMS3 DC and AC fixtures provide independently known arrays for accessor,
analysis and plotting checks. They are generated afresh in each temporary directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from calamari import simulate

#: Points per synthetic sweep -- large enough that scan-kind classification
#: and interpolation-free accessor checks are unambiguous, small enough to
#: keep the test suite fast.
_N_POINTS = 21

#: Constant AC drive-field amplitude (Oe) used by the AC fixtures below, so
#: Measurement.ac_drive_oe round-trips through synthetic MPMS3 exports. An
#: arbitrary but physically modest value (typical AC susceptometry drive
#: amplitudes are sub-Oe to a few Oe); no significance beyond that.
_AC_DRIVE_OE = 0.3


@dataclass
class SyntheticFile:
    """A synthetic file's path plus the ground-truth arrays used to build it.

    Attributes
    ----------
    path : Path
        Written file location (inside the fixture's ``tmp_path``).
    temperature_K, field_oe : numpy.ndarray
        Always populated -- every synthetic file carries both columns.
    moment_emu, chi_real, chi_imag, frequency_hz, ac_drive_oe : numpy.ndarray or None
        Populated only for the fixtures that generate them (DC fixtures
        carry ``moment_emu``; AC fixtures carry the other four, including
        ``ac_drive_oe``).
    """

    path: Path
    temperature_K: np.ndarray
    field_oe: np.ndarray
    moment_emu: np.ndarray | None = None
    chi_real: np.ndarray | None = None
    chi_imag: np.ndarray | None = None
    frequency_hz: np.ndarray | None = None
    ac_drive_oe: np.ndarray | None = None


def _mh_ground_truth() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed T, swept H, saturating moment -- shared by both DC fixtures."""
    field_oe = np.linspace(-10_000.0, 10_000.0, _N_POINTS)
    temperature_K = np.full(_N_POINTS, 5.0)
    moment_emu = simulate.synthetic_mh_curve(field_oe, m_sat_emu=1.0e-3, h_knee_oe=2000.0)
    return temperature_K, field_oe, moment_emu


def _chi_ground_truth() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Fixed H, swept T, Curie-Weiss chi -- shared by both AC fixtures."""
    temperature_K = np.linspace(2.0, 300.0, _N_POINTS)
    field_oe = np.full(_N_POINTS, 5.0)
    chi_real = simulate.curie_weiss_chi(temperature_K, C=0.5, theta_K=-10.0)
    chi_imag = 0.05 * chi_real  # small out-of-phase component, no physical claim
    frequency_hz = np.full(_N_POINTS, 10.0)
    ac_drive_oe = np.full(_N_POINTS, _AC_DRIVE_OE)
    return temperature_K, field_oe, chi_real, chi_imag, frequency_hz, ac_drive_oe


@pytest.fixture
def mpms3_dc_file(tmp_path: Path) -> SyntheticFile:
    """A synthetic MPMS3 DC file: M(H) loop at fixed T."""
    temperature_K, field_oe, moment_emu = _mh_ground_truth()
    path = simulate.write_mpms3_file(
        tmp_path / "mpms3_dc.dat",
        app="SQUID DC",
        sample_material="SyntheticSample",
        sample_mass_mg=10.0,
        temperature_K=temperature_K,
        field_oe=field_oe,
        dc_moment_emu=moment_emu,
    )
    return SyntheticFile(
        path=path, temperature_K=temperature_K, field_oe=field_oe, moment_emu=moment_emu
    )


@pytest.fixture
def mpms3_ac_file(tmp_path: Path) -> SyntheticFile:
    """A synthetic MPMS3 AC file: chi'(T)/chi''(T) at fixed drive field."""
    temperature_K, field_oe, chi_real, chi_imag, frequency_hz, ac_drive_oe = (
        _chi_ground_truth()
    )
    path = simulate.write_mpms3_file(
        tmp_path / "mpms3_ac.dat",
        app="SQUID AC",
        sample_material="SyntheticSample",
        sample_mass_mg=10.0,
        temperature_K=temperature_K,
        field_oe=field_oe,
        chi_real=chi_real,
        chi_imag=chi_imag,
        ac_frequency_hz=frequency_hz,
        ac_drive_oe=ac_drive_oe,
    )
    return SyntheticFile(
        path=path,
        temperature_K=temperature_K,
        field_oe=field_oe,
        chi_real=chi_real,
        chi_imag=chi_imag,
        frequency_hz=frequency_hz,
        ac_drive_oe=ac_drive_oe,
    )
