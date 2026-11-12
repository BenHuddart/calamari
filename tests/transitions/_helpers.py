"""Shared synthetic builders and expected-value helpers."""

from __future__ import annotations

import numpy as np

from calamari import simulate

# Array-level detection.


def _cusp(x, centre=12.5, width=1.0, noise=0.02, *, rng):
    return simulate.synthetic_chi_peak(
        x, centre=centre, width=width, height=1.0, background=0.2,
        noise=noise, rng=rng,
    )

# Measurement-level detection.


def _ac_tscan_file(path, *, fixed_h_oe=100.0, t_n=14.0, frequencies=(511.0,), rng):
    """One chi'(T) cusp at ``t_n``, optionally interleaving frequencies.

    """
    t = np.linspace(4.0, 30.0, 131)
    chi = simulate.synthetic_chi_peak(
        t, centre=t_n, width=1.0, height=2.0, background=0.1, noise=0.02,
        rng=rng,
    )
    reps = len(frequencies)
    return simulate.write_mpms3_file(
        path,
        app="SQUID AC",
        temperature_K=np.repeat(t, reps),
        field_oe=np.full(reps * len(t), fixed_h_oe),
        chi_real=np.repeat(chi, reps),
        ac_frequency_hz=np.tile(np.asarray(frequencies, dtype=float), len(t)),
    )

# Smoothed curves, one per sweep: TransitionResult.curve()


def _curve_rng():
    """Repeatable noise for curve-lookup comparisons."""
    return np.random.default_rng(71)

# Kink criterion (Phase C)


def _slope_change_above(x, x0, slope, width=0.15):
    """A background whose slope changes by ``slope`` at ``x0``.

    The corner is rounded over ``width`` by a softplus -- smooth to all
    orders, so its curvature is one clean single-signed bump, and the
    curve stays exactly linear at both ends of the sweep. (Rounding by
    convolution instead needs the array padded, and edge-padding a ramp
    plants a second, unwanted corner at the sweep's own start: that
    fixture artefact used to put a spurious curvature candidate between
    the two, on half of all noise draws.)
    """
    return slope * width * np.logaddexp(0.0, (x - x0) / width)


def _kinked(x, x0=12.5, slope=0.10, slope_change=-0.40, smooth_pts=2.0, noise=0.001, *, rng):
    """A planted change of slope: ``y = slope*x + slope_change*max(x-x0, 0)``
    rounded off over a couple of grid steps (no real sweep carries a
    mathematically sharp corner) plus a little noise."""
    dx = float(x[1] - x[0])
    y = slope * x + _slope_change_above(x, x0, slope_change, smooth_pts * dx)
    return y + rng.normal(scale=noise, size=x.shape)

# Fisher's d(chi T)/dT criterion

#: The toy antiferromagnet's planted ordering temperature, Curie constant
#: (emu K/Oe) and Weiss temperature (K). Invented numbers, sized so chi is
#: ~1e-5 emu/Oe, the scale of a real sample's chi'.
FISHER_T_N = 10.0

FISHER_CURIE = 1.0e-3

FISHER_THETA = 10.0

#: Per-point scatter of the toy chi (emu/Oe). Small enough that the Fisher
#: maximum clears robust()'s default floor of 5 on 141 points.
FISHER_NOISE = 5.0e-8


def _toy_afm_chi(t_k, t_n=FISHER_T_N):
    """A caricature of an antiferromagnet's powder chi(T), not a calculation.

    Curie-Weiss, C/(T + theta), above ``t_n``. Below it, chi falls
    quadratically to 2/3 of chi(T_N) at T = 0, the powder average of a
    parallel component that vanishes and perpendicular ones that stay
    constant. So chi has a cusp at T_N, and d(chi T)/dT jumps down there:
    its smoothed maximum is Fisher's criterion, and it sits below the cusp.
    """
    chi_n = FISHER_CURIE / (t_n + FISHER_THETA)
    below = chi_n * (2.0 / 3.0 + (t_k / t_n) ** 2 / 3.0)
    return np.where(t_k >= t_n, FISHER_CURIE / (t_k + FISHER_THETA), below)


def _toy_afm_t():
    return np.linspace(2.0, 30.0, 141)


def _toy_afm_ac_file(path, *, seed=11, chi_imag=False):
    """One AC chi'(T) of the toy antiferromagnet at a held 1000 Oe."""
    rng = np.random.default_rng(seed)
    t = _toy_afm_t()
    chi = _toy_afm_chi(t) + rng.normal(scale=FISHER_NOISE, size=t.shape)
    extra = {"chi_imag": 0.05 * chi} if chi_imag else {}
    return simulate.write_mpms3_file(
        path,
        app="SQUID AC",
        temperature_K=t,
        field_oe=np.full_like(t, 1000.0),
        chi_real=chi,
        ac_frequency_hz=np.full_like(t, 997.0),
        **extra,
    )


def _toy_afm_dc_file(path, *, field_oe, t_n=FISHER_T_N, seed=13):
    """The toy antiferromagnet as a DC M(T) at a held field: M = chi*H."""
    rng = np.random.default_rng(seed)
    t = _toy_afm_t()
    moment = (_toy_afm_chi(t, t_n) + rng.normal(scale=FISHER_NOISE, size=t.shape)) * field_oe
    return simulate.write_mpms3_file(
        path,
        app="SQUID DC",
        temperature_K=t,
        field_oe=np.full_like(t, field_oe),
        dc_moment_emu=moment,
    )
