"""Fixtures shared across these behavioural tests."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; must precede the pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pytest

from calamari.measurement import Measurement
from calamari.sample import Sample
from calamari.simulate import curie_weiss_chi, synthetic_mh_curve

from ._helpers import (
    _N,
    _measurement,
)


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


@pytest.fixture
def mh_measurement() -> Measurement:
    field_oe = np.linspace(-5000.0, 5000.0, _N)
    moment_emu = synthetic_mh_curve(field_oe, m_sat_emu=2.0e-3, h_knee_oe=1500.0)
    return _measurement(**{"Magnetic Field (Oe)": field_oe, "Moment (emu)": moment_emu})


@pytest.fixture
def mt_measurement() -> Measurement:
    """An M(T) at a fixed 1000 Oe field -- also the DC chi = M/H shape."""
    temperature_K = np.linspace(2.0, 300.0, _N)
    moment_emu = 1.0e-4 * curie_weiss_chi(temperature_K, C=0.5, theta_K=-5.0)
    return _measurement(
        **{
            "Temperature (K)": temperature_K,
            "Magnetic Field (Oe)": np.full(_N, 1000.0),
            "Moment (emu)": moment_emu,
        }
    )


@pytest.fixture
def chi_measurement() -> Measurement:
    temperature_K = np.linspace(2.0, 300.0, _N)
    chi_real = curie_weiss_chi(temperature_K, C=0.5, theta_K=-5.0)
    chi_imag = 0.05 * chi_real
    return _measurement(
        **{
            "Temperature (K)": temperature_K,
            "AC X' (emu/Oe)": chi_real,
            "AC X'' (emu/Oe)": chi_imag,
        }
    )


@pytest.fixture
def sample() -> Sample:
    return Sample(
        name="synthetic",
        mass_mg=10.0,
        molar_mass_g_per_mol=200.0,
        magnetic_ions_per_fu=1,
    )
