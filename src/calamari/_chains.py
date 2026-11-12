"""Deterministic candidate-chain walk shared by linking and smoothing scans."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


def _chain_walk(
    cands: pd.DataFrame,
    by: tuple[str, ...],
    *,
    geometry: dict[str, tuple[str, str]],
    tolerances: dict[str, float | None],
    family_values: dict[str, Sequence[float]] | None = None,
) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    """Deterministic nearest-unclaimed walk for candidate chains.

    Used by :meth:`calamari.transitions.TransitionResult.link` and smoothing scans.

    Linking follows adjacent measurements; smoothing scans follow window
    indices. Both use the same grouping, seed order, tie-breaks and first-gap
    stopping rules, with explicit geometry and tolerances.

    Parameters
    ----------
    cands : DataFrame
        Rows to walk. Must carry ``swept_axis``, ``significance``, every
        ``by`` column, and every column ``geometry`` names.
    by : tuple of str
        Columns that must match within a chain, on top of ``swept_axis``.
    geometry : dict
        Per swept axis, ``(family column, location column)``: which column
        orders the family and which one places the feature within a member.
        Rows on an axis not in ``geometry`` are unplaceable.
    tolerances : dict
        Per swept axis, the largest location step between adjacent family
        members. Needed only for axes that have placeable rows.
    family_values : dict, optional
        Per swept axis, the complete ordered family. ``None`` (``link()``'s
        behaviour) derives it as the sorted distinct family values among
        *every* placeable row on that axis. Pass it when the family is
        known independently of which members fired, so a member with no
        rows at all still counts as a gap.

    Returns
    -------
    chains : list of list of int
        Row positions, one list per chain, each sorted by family value,
        location, then position. Unplaceable rows come first, one chain
        each; the caller orders chains as it needs.
    fam, loc : ndarray
        The per-row family value and location the walk used.
    """
    n_rows = len(cands)
    if n_rows == 0:
        return [], np.array([], dtype=float), np.array([], dtype=float)

    axis_col = cands["swept_axis"].to_numpy()
    sig = cands["significance"].to_numpy(dtype=float)

    fam = np.full(n_rows, np.nan)
    loc = np.full(n_rows, np.nan)
    for axis, (fam_col, loc_col) in geometry.items():
        on_axis = axis_col == axis
        fam[on_axis] = cands[fam_col].to_numpy(dtype=float)[on_axis]
        loc[on_axis] = cands[loc_col].to_numpy(dtype=float)[on_axis]

    # The sweep ordering is per swept axis and spans EVERY candidate on it,
    # not just the group being linked: a sweep that fired detections but
    # none in this group is a gap in the evidence, and a chain must end
    # there rather than step over it as if the sweep had not been taken.
    sweep_index = np.full(n_rows, -1)
    n_sweeps = {}
    for axis in geometry:
        on_axis = (axis_col == axis) & np.isfinite(fam) & np.isfinite(loc)
        if family_values is not None and axis in family_values:
            values = sorted({float(v) for v in family_values[axis]})
        else:
            values = sorted({float(v) for v in fam[on_axis]})
        n_sweeps[axis] = len(values)
        order = {v: i for i, v in enumerate(values)}
        sweep_index[on_axis] = [order[float(v)] for v in fam[on_axis]]

    if by:
        by_columns = [cands[c].astype(str).to_numpy() for c in by]
        group_keys = [tuple(col[i] for col in by_columns) for i in range(n_rows)]
    else:
        group_keys = [() for _ in range(n_rows)]

    groups: dict[tuple, list[int]] = {}
    unplaceable: list[int] = []
    for i in range(n_rows):
        if sweep_index[i] < 0:  # no usable (family, location) pair
            unplaceable.append(i)
            continue
        groups.setdefault((*group_keys[i], str(axis_col[i])), []).append(i)

    def seed_key(i: int) -> tuple:
        # -significance sorts descending; a NaN significance seeds last
        # rather than crashing the sort.
        s = sig[i]
        return (math.inf if math.isnan(s) else -s, fam[i], loc[i], i)

    chains: list[list[int]] = [[i] for i in unplaceable]
    for key, members in sorted(groups.items(), key=lambda kv: kv[0]):
        axis = key[-1]
        tol = float(tolerances[axis])
        by_sweep: dict[int, list[int]] = {}
        for i in members:
            by_sweep.setdefault(int(sweep_index[i]), []).append(i)
        claimed: set[int] = set()
        for seed in sorted(members, key=seed_key):
            if seed in claimed:
                continue
            claimed.add(seed)
            chain = [seed]
            for step in (1, -1):
                ref, fi = loc[seed], int(sweep_index[seed])
                while True:
                    fi += step
                    if not (0 <= fi < n_sweeps[axis]):
                        break
                    pool = [i for i in by_sweep.get(fi, ()) if i not in claimed]
                    if not pool:
                        break
                    best = min(pool, key=lambda i: (abs(loc[i] - ref), loc[i], i))
                    if abs(loc[best] - ref) > tol:
                        break
                    claimed.add(best)
                    chain.append(best)
                    ref = loc[best]
            chains.append(sorted(chain, key=lambda i: (fam[i], loc[i], i)))
    return chains, fam, loc

