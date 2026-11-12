"""Shared synthetic builders and expected-value helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from calamari import plot
from calamari.measurement import Measurement


def _line_xy(line):
    return np.asarray(line.get_xdata(), dtype=float), np.asarray(
        line.get_ydata(), dtype=float
    )


def _measurement(**columns) -> Measurement:
    """Build synthetic plotting inputs without exercising file parsing."""
    return Measurement(
        path=Path("synthetic.dat"),
        data_format="MPMS3",
        measurement_type="DC",
        file_role="summary",
        title="synthetic",
        header={},
        data=pd.DataFrame(columns),
    )

_N = 15

# field_unit=: 1-D field axes in Oe (default) or Tesla

_TESLA_LABEL = plot.axis_labels("field", "tesla")


def _chi_vs_h_measurement() -> Measurement:
    h = np.array([0.0, 2500.0, 5000.0, 7500.0, 10000.0])
    return _measurement(
        **{
            "Temperature (K)": np.full(5, 2.0),
            "Magnetic Field (Oe)": h,
            "AC X' (emu/Oe)": np.array([1.0, 2.0, 3.0, 2.0, 1.0]),
        }
    )

# Transition overlays.


def _ac_tscan_measurements(tmp_path, specs):
    """Synthetic chi'(T) sweeps, one per (fixed field Oe, cusp T)."""
    from calamari import simulate
    from calamari.io import load as _load

    rng = np.random.default_rng(9)
    t = np.linspace(4.0, 30.0, 121)
    loaded = []
    for i, (h_oe, t_n) in enumerate(specs):
        chi = simulate.synthetic_chi_peak(
            t, centre=t_n, width=1.0, height=2.0, background=0.1,
            noise=0.02, rng=rng,
        )
        path = simulate.write_mpms3_file(
            tmp_path / f"ov_{i}.dat", app="SQUID AC",
            temperature_K=t, field_oe=np.full_like(t, h_oe),
            chi_real=chi, ac_frequency_hz=np.full_like(t, 511.0),
        )
        loaded.append(_load(path))
    return loaded


def _detected(tmp_path):
    from calamari.transitions import detect_transitions_all

    ms = _ac_tscan_measurements(
        tmp_path, [(100.0, 14.0), (2000.0, 13.0), (10000.0, 11.0)]
    )
    return ms, detect_transitions_all(ms).robust()


def _field_candidates() -> pd.DataFrame:
    """One H-swept candidate at mu0 H = 0.3 T and one T-swept at 5 K."""
    return pd.DataFrame(
        {
            "method": ["peak", "peak"],
            "swept_axis": ["H", "T"],
            "T_K": [2.0, 5.0],
            "H_T": [0.3, 0.1],
        }
    )

# fisher_series / plot_fisher: Fisher's d(chi T)/dT in physical units


def _toy_afm_chi(t_k, t_n=10.0, curie=1.0e-3, theta=10.0):
    """Curie-Weiss above ``t_n``; below, a quadratic fall to 2/3 chi(T_N).

    A caricature of an antiferromagnet's powder chi(T) (invented numbers),
    enough to give d(chi T)/dT a maximum just below the cusp.
    """
    chi_n = curie / (t_n + theta)
    return np.where(t_k >= t_n, curie / (t_k + theta), chi_n * (2 / 3 + (t_k / t_n) ** 2 / 3))


def _toy_afm_file(tmp_path, *, name="afm.dat", field_oe=1000.0, dc=False, frequencies=(997.0,)):
    from calamari import simulate
    from calamari.io import load as _load

    t = np.linspace(2.0, 30.0, 141)
    chi = _toy_afm_chi(t) + np.random.default_rng(11).normal(scale=5.0e-8, size=t.shape)
    reps = len(frequencies)
    if dc:
        columns = {"dc_moment_emu": chi * field_oe}
        app = "SQUID DC"
    else:
        columns = {
            "chi_real": np.repeat(chi, reps),
            "ac_frequency_hz": np.tile(np.asarray(frequencies, dtype=float), len(t)),
        }
        app = "SQUID AC"
    path = simulate.write_mpms3_file(
        tmp_path / name,
        app=app,
        temperature_K=np.repeat(t, 1 if dc else reps),
        field_oe=np.full(len(t) * (1 if dc else reps), field_oe),
        **columns,
    )
    return _load(path)
