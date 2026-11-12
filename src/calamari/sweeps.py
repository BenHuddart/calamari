"""Sweep segmentation and branch-folding primitives.

This module is deliberately **array-first and backend-free**: its two lower
helpers (:func:`segment_by_constant`, :func:`increasing_field_branch`) operate
on plain NumPy arrays and know nothing about :class:`~calamari.measurement.Measurement`,
so they are reusable well beyond magnetometry; the one higher helper
(:func:`split_isotherms`) is a thin :class:`~calamari.measurement.Measurement`-aware
wrapper that uses
:meth:`Measurement.restrict <calamari.measurement.Measurement.restrict>`
to carve a stacked multi-isotherm file back into
real sub-measurements.

Why these three
---------------
They exist to prepare raw DC sweep data for the transition detectors in
:mod:`calamari.transitions` (:func:`~calamari.transitions.detect_transitions`
and :func:`~calamari.transitions.detect_transitions_all`, which accepts the
``(T_K, Measurement)`` pairs :func:`split_isotherms` returns):

* A single physical data file is not always a single sweep. A "phase-diagram"
  DC run is often a **stack of isothermal M(H) sweeps** (e.g. 2 K, 3 K, ...
  each ramping 0 -> H_max) laid end to end in one file, which a naive
  scan-kind classifier tags as "mixed" because *both* T and H vary across the
  file as a whole. :func:`split_isotherms` splits it back into one
  sub-measurement per temperature plateau so each isotherm is searched
  independently; :func:`segment_by_constant` is the generic plateau-finding
  primitive underneath it.
* An M(H) file frequently ramps up **and** back down in field (a hysteresis
  loop, or a bidirectional sweep). Detecting a metamagnetic step on the full
  loop would double-count it (once per leg) and let the return leg add
  spurious high-field candidates. :func:`increasing_field_branch` folds the
  sweep to its initial increasing-\\|field\\| branch so a feature is counted
  once.

Determinism
-----------
Nothing here is random: given the same input arrays and parameters every call
returns the same segmentation/branch, so downstream boundary tables are
reproducible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from calamari.exceptions import SweepInputError

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from calamari.measurement import Measurement

__all__ = [
    "segment_by_constant",
    "split_isotherms",
    "increasing_field_branch",
]

#: Absolute tolerance (in the field array's own units) below which a
#: point-to-point *decrease* in \\|field\\| is treated as setpoint-settle
#: jitter rather than the start of a genuine down-leg, in
#: :func:`increasing_field_branch`. Small next to any real field step; ported
#: verbatim from the reference DC extractor.
_FIELD_SETTLE_TOL = 1e-6

#: Minimum length of the leading increasing-\\|field\\| run for
#: :func:`increasing_field_branch` to trust it as a real up-leg rather than a
#: sweep that started mid-loop; below this it falls back to sorting the whole
#: sweep by \\|field\\|. Three points is the smallest run that can show a
#: direction at all.
_MIN_MONOTONE_RUN = 3


def segment_by_constant(
    x: ArrayLike, *, tol: float, min_points: int = 1
) -> list[np.ndarray]:
    """Segment a 1-D signal into contiguous runs (plateaus) of ~constant value.

    Generic primitive for a signal that steps between plateaus: a stacked
    multi-setpoint measurement (temperature held at 2 K for a whole field
    sweep, stepped to 3 K for the next, ...), a staircase drive, or any trace
    that dwells at a level then jumps to the next. A new segment begins
    wherever the point-to-point change ``|x[i] - x[i-1]|`` exceeds ``tol``, so
    ``tol`` must be chosen between the within-plateau jitter (kept together)
    and the inter-plateau step size (which splits). Because the test is on the
    *consecutive* difference, slow drift within one plateau -- where each step
    stays under ``tol`` even though the plateau's total span does not -- is
    tolerated and stays one segment, which is the desired behaviour for real
    setpoint data that settles gradually.

    Parameters
    ----------
    x : array_like
        The stepped signal (e.g. a temperature column in Kelvin). Converted to
        ``float``. NaNs are kept in place and counted as ordinary points: a NaN
        neighbour makes ``|diff|`` NaN, which does not exceed ``tol``, so a NaN
        never *forces* a split (callers who want NaN-free segments should mask
        first).
    tol : float
        Maximum consecutive change tolerated *within* a plateau. Must be
        non-negative. A ``tol`` of exactly 0 splits on any change at all,
        which is the natural choice when ``x`` has already been quantised onto
        a grid (see :func:`split_isotherms`).
    min_points : int, optional
        Segments shorter than this many points are dropped from the result.
        Default 1 (keep every run, including singletons).

    Returns
    -------
    list of numpy.ndarray
        One integer index array per kept segment, in order, each holding the
        positions into ``x`` belonging to that plateau (contiguous and
        ascending). Index arrays -- rather than slices -- are returned so a
        caller can build a boolean row-mask directly (``mask[seg] = True``)
        for :meth:`~calamari.measurement.Measurement.restrict`.

    Raises
    ------
    SweepInputError
        If ``tol`` is negative or ``min_points`` is less than 1.
    """
    if tol < 0:
        raise SweepInputError(f"tol must be non-negative, got {tol}")
    if min_points < 1:
        raise SweepInputError(f"min_points must be >= 1, got {min_points}")

    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n == 0:
        return []

    # Boundary just before index i whenever the step from i-1 to i is a
    # genuine jump (> tol). NaN diffs compare False, so they never split.
    boundaries = [0]
    for i in range(1, n):
        if abs(x[i] - x[i - 1]) > tol:
            boundaries.append(i)
    boundaries.append(n)

    segments = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        if stop - start >= min_points:
            segments.append(np.arange(start, stop))
    return segments


def split_isotherms(
    measurement: Measurement, *, t_round: float = 0.5, min_points: int = 3
) -> list[tuple[float, Measurement]]:
    """Split a stacked multi-isotherm M(H) file into per-temperature sweeps.

    An ordinary single-temperature M(H) file yields one ``(T_K, Measurement)``
    pair. A multi-isotherm "phase-diagram" file -- where each temperature
    setpoint is a full field ramp laid end to end -- yields one pair per
    temperature plateau, each a real :class:`~calamari.measurement.Measurement`
    (built via
    :meth:`Measurement.restrict <calamari.measurement.Measurement.restrict>`
    with a row mask, so it carries the full row
    set of that isotherm and every accessor keeps working) rather than a bare
    array triple.

    Grouping is by temperature quantised to a ``t_round`` grid: each row's
    temperature is snapped to the nearest multiple of ``t_round`` and
    consecutive rows sharing a snapped value form one isotherm (via
    :func:`segment_by_constant` on the snapped column with ``tol=0``). Snapping
    first -- rather than segmenting the raw temperature with a jitter tolerance
    -- reproduces the reference extractor's behaviour exactly and is robust to
    a setpoint that drifts within its own ``t_round`` cell. As a guard, a file
    whose whole temperature span is within one ``t_round`` step is treated as a
    single isotherm outright (its mean temperature), so ordinary M(H) files
    with a little thermometry jitter are never spuriously split across a grid
    boundary.

    Parameters
    ----------
    measurement : Measurement
        A DC field-sweep measurement (single isotherm or a stack). Must expose
        a temperature column; a :class:`ValueError` is raised otherwise, since
        splitting by temperature is meaningless without one.
    t_round : float, optional
        Temperature grid (K) rows are snapped to before grouping. Default
        0.5 K -- comfortably finer than the ~1 K setpoint steps of a typical
        stacked run yet coarse enough to absorb per-isotherm jitter. Tune to
        the setpoint spacing of the file at hand.
    min_points : int, optional
        Isotherms with fewer than this many rows are dropped (a handful of
        points is too coarse to trust). Default 3; note
        :func:`calamari.transitions.find_transitions` itself needs at least
        5 finite points, so raise this for isotherms headed there.

    Returns
    -------
    list of (float, Measurement)
        ``(T_K, sub_measurement)`` pairs in file order, where ``T_K`` is the
        mean of that isotherm's *raw* (unsnapped) temperature column.

    Raises
    ------
    SweepInputError
        If ``measurement`` has no temperature column.

    Examples
    --------
    A stacked multi-isotherm file needs no dedicated simulate writer -- the
    existing per-column writers already express it once the caller
    concatenates the isotherms, because a stacked file is literally its
    constituent sweeps laid end to end::

        import numpy as np
        from calamari import simulate
        from calamari.io import load
        from calamari.sweeps import split_isotherms

        field_oe = np.linspace(0.0, 25_000.0, 40)            # one 0 -> 2.5 T ramp
        temps_k = [2.0, 3.0, 4.0]
        temperature_K = np.concatenate([np.full_like(field_oe, t) for t in temps_k])
        field_stacked = np.tile(field_oe, len(temps_k))
        moment_emu = np.concatenate([                        # a step that moves down in T
            simulate.synthetic_mh_curve(field_oe, m_sat_emu=1e-3, h_knee_oe=knee)
            for knee in (12_000.0, 9_000.0, 6_000.0)
        ])
        path = simulate.write_mpms3_file(
            "stacked.dat", app="SQUID DC",
            temperature_K=temperature_K, field_oe=field_stacked, dc_moment_emu=moment_emu,
        )
        isotherms = split_isotherms(load(path))              # -> 3 (T_K, Measurement) pairs
    """
    temperature = measurement.temperature
    if temperature is None:
        raise SweepInputError(
            "split_isotherms requires a temperature column, but this "
            "Measurement has none"
        )

    t = temperature.to_numpy(dtype=float)
    finite = t[np.isfinite(t)]
    t_span = float(finite.max() - finite.min()) if finite.size else 0.0

    if t_span <= t_round:
        # Single isotherm: whole file is one block at its mean temperature.
        return [(float(np.nanmean(t)), measurement.restrict())]

    t_grid = np.round(t / t_round) * t_round
    n_rows = len(measurement.data)

    pairs: list[tuple[float, Measurement]] = []
    for seg in segment_by_constant(t_grid, tol=0.0, min_points=min_points):
        mask = np.zeros(n_rows, dtype=bool)
        mask[seg] = True
        t_k = float(np.nanmean(t[seg]))
        pairs.append((t_k, measurement.restrict(mask=mask)))
    return pairs


def increasing_field_branch(
    field: ArrayLike, moment: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Fold an M(H) sweep to its initial increasing-\\|field\\| branch.

    Many M(H) files ramp ``0 -> +H_max -> 0`` (and sometimes on into negative
    field): a hysteresis loop or a bidirectional sweep. Running a step/kink
    detector on the *whole* trace would (a) report a metamagnetic feature
    twice, once from the up-leg and once from the down-leg, and (b) let the
    return leg contribute spurious high-field candidates. This helper returns
    the leading monotonically-increasing run of ``|field|`` from the start of
    the sweep, paired with the corresponding moment values, so a feature is
    counted once on a clean monotone field axis.

    The function is **unit-agnostic**: it operates purely on ``|field|`` and
    monotonicity, so the returned field array is in whatever unit ``field`` was
    supplied in (Oe or Tesla). The parameter is therefore named plainly
    ``field`` rather than carrying a unit suffix -- attaching one would be an
    active lie whenever the other unit is passed. Callers that need a specific
    output unit should pass that unit in.

    Parameters
    ----------
    field : array_like
        Applied-field column, in any single consistent unit. May be signed;
        only its magnitude is used.
    moment : array_like
        Moment column aligned to ``field`` (same length).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(field_increasing, moment)`` -- ``|field|`` over the leading
        increasing run and the matching moments. If the sweep starts on a
        decreasing leg (leading run shorter than three points, e.g. a
        ``+H_max -> 0 -> -H_max`` sweep), falls back to the *whole* sweep
        sorted by ``|field|`` so a usable monotone axis is still returned
        rather than nothing.

    Raises
    ------
    SweepInputError
        If ``field`` and ``moment`` have different lengths.
    """
    field = np.asarray(field, dtype=float)
    moment = np.asarray(moment, dtype=float)
    if field.shape != moment.shape:
        raise SweepInputError(
            f"field and moment must have the same shape, got "
            f"{field.shape} vs {moment.shape}"
        )

    abs_h = np.abs(field)
    n = len(abs_h)
    end = n
    for i in range(1, n):
        # A genuine decrease (beyond settle jitter) ends the leading up-leg.
        if abs_h[i] < abs_h[i - 1] - _FIELD_SETTLE_TOL:
            end = i
            break

    if end < _MIN_MONOTONE_RUN:
        order = np.argsort(abs_h)
        return abs_h[order], moment[order]
    return abs_h[:end], moment[:end]
