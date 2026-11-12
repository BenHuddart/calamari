"""How detected features behave across a declared set of smoothing windows.

Every transition candidate is found in a *smoothed* representation of a
sweep, and the smoothing window is an analysis choice. It matters most
where the feature is weakest: the curvature a ``'kink'`` is found in is
far noisier than the signal a ``'peak'`` is found in, so a slope change's
significance, and sometimes its sign, can move a long way between two
reasonable windows, while a strong cusp barely moves at all. Fisher's
d(yT)/dT maximum shifts with the window too (see ``docs/DETECTION.md``).
The tempting response -- try windows until one gives a clean answer, then
quote that one -- is a forking path: the window was chosen after looking
at the result, and the quoted significance no longer means what it says.

:func:`scan_smoothing` closes that path by making the choice explicit and
complete. The caller declares a set of windows up front; every one of them
is run through :func:`~calamari.transitions.detect_transitions_all`, every
result is kept, and each sweep's features are *tracked* across the set
with the same deterministic nearest-unclaimed walk that
:meth:`~calamari.transitions.TransitionResult.link` uses across sweeps,
the window index playing the part of the held field. The ``tracks`` table
then says, per feature, in how many of the windows it appeared, how far
it moved, whether its sign held, and the range of its per-window
significance.

This is the scale-space picture: follow each extremum of a signal and of
its derivatives as the smoothing scale changes, and prefer the
description that is stable over a wide range of scales (A. P. Witkin,
"Scale-space filtering", *Proc. 8th Int. Joint Conf. Artificial
Intelligence* (IJCAI-83), vol. 2, pp. 1019-1022 (1983)). Its statistical
counterpart, which asks where a curve's slope is significantly non-zero
at each bandwidth of a family and shows the whole family at once, is
SiZer (P. Chaudhuri and J. S. Marron, "SiZer for exploration of
structures in curves", *J. Am. Stat. Assoc.* **94**, 807-823 (1999)).

What persistence is, and is not
-------------------------------
Persistence is robustness to an *analysis choice*. It is not evidence.
Every window sees the same data, so the per-window significances are
strongly correlated: a noise excursion that scores 4 at one window scores
about 4 at its neighbour, and a feature found in all four windows has not
been seen four times. So:

* the per-window significances are **never combined** -- no Stouffer sum,
  no mean used as a score. :meth:`~calamari.transitions.TransitionResult.link`
  may combine across sweeps because each sweep is a separate measurement;
  windows are not. ``tracks`` therefore carries the range of the
  significance (min, median, max) and no combined statistic;
* there is no ``best_window``. Quoting whichever window happened to
  maximise the significance is exactly what the scan exists to prevent.
  Report the whole declared set;
* the scan gates nothing. It is a recorded description, like ``link()``;
  ``scan.reference(window)`` attaches diagnostics to a declared result,
  whose optional ``robust()`` sensitivity thresholds record the decision.

A feature that persists is one whose existence does not hinge on the
window you picked. A feature that does not persist may still be real
(too sharp for the widest window, too weak for the narrowest); a feature
that persists may still be an artefact the data share at every scale (a
glitch, a background step). Persistence answers "would another
reasonable window have told me something different?", nothing more.

This module never imports :mod:`calamari.plot` or matplotlib (PK-1).
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from calamari._chains import _chain_walk
from calamari._provenance import calamari_version as _calamari_version
from calamari.exceptions import SmoothingSensitivityError
from calamari.measurement import Measurement
from calamari.transitions import (
    TransitionResult,
    detect_transitions_all,
)
from calamari.transitions._results import _filtered_result

__all__ = [
    "MIN_WINDOWS",
    "TRACK_BY",
    "MEMBER_COLUMNS",
    "TRACK_COLUMNS",
    "SmoothingScanResult",
    "scan_smoothing",
]

#: The fewest windows a scan accepts. Two windows cannot show a trend in a
#: feature's location or significance, only a difference, and a "scan" of
#: two invites choosing the better of the pair after looking. Three is the
#: least that says anything about how a feature behaves across the set.
MIN_WINDOWS = 3

#: Columns that must match along a track, on top of ``swept_axis`` (which
#: the walk always adds). ``sign`` is deliberately absent: a kink that
#: turns from 'down' to 'up' as the window changes is one of the things the
#: scan exists to expose, so it stays in one track and ``sign_consistency``
#: reports it, rather than splitting into two tracks that each look stable.
TRACK_BY = ("sweep_id", "signal", "method")

#: Column order of :attr:`SmoothingScanResult.members`: one row per
#: (track, window) hit. The candidate's own coordinates and its error bar
#: keep :data:`~calamari.transitions.CANDIDATE_COLUMNS`' names (DM-2), and
#: ``candidate_id`` addresses the row in that window's result. A linked
#: scan appends :data:`_MEMBER_CHAIN_COLUMNS`.
MEMBER_COLUMNS = [
    "track_id",
    "window",
    "smooth_points",
    "sweep_id",
    "candidate_id",
    "T_K",
    "H_T",
    "T_err_K",
    "H_err_T",
    "sign",
    "significance",
    "edge",
]

#: What a linked scan adds to each member: its chain in that window's result.
_MEMBER_CHAIN_COLUMNS = ["chain_id", "chain_significance"]

#: Column order of :attr:`SmoothingScanResult.tracks`: one row per track.
#: ``T_K`` / ``H_T`` place the *sweep* -- the held coordinate is filled and
#: the swept one is NaN -- because a track has no single location: how far
#: it moved across the windows is the point, so the location is summarised
#: in the ``T_K_*`` columns (T-swept tracks) or the ``H_T_*`` columns
#: (H-swept), each NaN on the other axis. ``*_spread`` is max minus min.
#: There is deliberately no combined significance and no best window (see
#: the module docstring). A linked scan appends
#: :data:`_TRACK_CHAIN_COLUMNS`.
TRACK_COLUMNS = [
    "track_id",
    "sweep_id",
    "swept_axis",
    "signal",
    "method",
    "T_K",
    "H_T",
    "n_windows",
    "n_present",
    "persistence",
    "T_K_median",
    "T_K_min",
    "T_K_max",
    "T_K_spread",
    "H_T_median",
    "H_T_min",
    "H_T_max",
    "H_T_spread",
    "sign_mode",
    "sign_consistency",
    "significance_min",
    "significance_median",
    "significance_max",
    "edge_any",
]

#: What a linked scan adds to each track: the range of its members'
#: cross-sweep chain statistic. The range, never a combination -- each
#: window's chain statistic is computed on the same data as the others.
_TRACK_CHAIN_COLUMNS = ["chain_significance_min", "chain_significance_median"]

#: Per swept axis: the tracking-tolerance keyword and its unit, the location
#: column, and the held-coordinate column.
_TRACK_AXES = {
    "T": ("track_tol_k", "K", "T_K", "H_T"),
    "H": ("track_tol_t", "T", "H_T", "T_K"),
}

#: The walk's family coordinate: a member's position in the sorted windows.
_WINDOW_INDEX = "_window_index"


@dataclass(frozen=True, eq=False)
class SmoothingScanResult:
    """One detection per declared smoothing window, and each feature tracked across them.

    A *description*, like :meth:`TransitionResult.link
    <calamari.transitions.TransitionResult.link>`: nothing is dropped, and
    nothing here is a verdict. Read the module docstring before quoting
    anything from it -- persistence is robustness to an analysis choice,
    not evidence, and the per-window significances must never be combined.

    Attributes
    ----------
    results : dict of float to calamari.transitions.TransitionResult
        Every window's full result, keyed by the window (in the swept
        axis's units), in ascending order. Each carries its own
        ``params``, so any window can be inspected, filtered or plotted
        exactly as if it had been run alone. :meth:`at` is the checked
        lookup.
    members : pandas.DataFrame
        One row per (track, window) hit (:data:`MEMBER_COLUMNS`, plus
        ``chain_id`` and ``chain_significance`` when the scan was linked).
        ``candidate_id`` addresses the row in ``results[window]``.
    tracks : pandas.DataFrame
        One row per track (:data:`TRACK_COLUMNS`, plus
        ``chain_significance_min`` and ``chain_significance_median`` when
        linked). ``n_windows`` counts the *distinct* point counts the
        declared windows resolved to on that sweep and signal, and
        ``n_present`` the distinct ones the track appears in, so
        ``persistence = n_present / n_windows`` is not inflated by two
        windows that clamped to one analysis. The location, sign and
        significance summaries are taken over those distinct windows too.
    params : dict
        The recipe: ``windows`` (as given) and ``windows_sorted``, the two
        tracking tolerances, ``track_by``, ``detect_kwargs``, ``link``,
        ``smooth_points`` (per sweep and signal, the point count each
        window resolved to, and ``n_windows``), ``resolved_together`` (the
        windows that clamped to one count, as warned), and
        ``calamari_version``.
    """

    results: dict
    members: pd.DataFrame
    tracks: pd.DataFrame
    params: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # DataFrames make the default repr unusable
        windows = ", ".join(f"{w:g}" for w in self.params.get("windows_sorted", ()))
        linked = ", linked" if self.params.get("link") is not None else ""
        together = self.params.get("resolved_together", [])
        clamp = f", {len(together)} sweep(s) with windows resolved together" if together else ""
        return (
            f"SmoothingScanResult(windows [{windows}], {len(self.tracks)} tracks, "
            f"{len(self.members)} members{linked}{clamp})"
        )

    def at(self, window: float) -> TransitionResult:
        """The full detection at one declared window.

        An exact lookup, not a nearest match: a window that was not
        declared was not run, and handing back a neighbour would pass off
        one analysis as another.

        Parameters
        ----------
        window : float
            One of the declared windows, in the swept axis's units.

        Returns
        -------
        calamari.transitions.TransitionResult
            That window's result, as
            :func:`~calamari.transitions.detect_transitions_all`
            (and ``link()``, if the scan was linked) returned it.

        Raises
        ------
        SmoothingSensitivityError
            If ``window`` is not one of the declared windows, naming them.
        """
        try:
            key = float(window)
        except (TypeError, ValueError):
            key = None
        if key is None or key not in self.results:
            raise SmoothingSensitivityError(
                f"at(): no result for window={window!r}. This scan ran windows "
                f"{list(self.results)}; pass one of those exactly. A window that was "
                "not declared was not run -- add it to windows= and scan again."
            )
        return self.results[key]

    def reference(
        self, window: float, *, min_member_significance: float | None = None
    ) -> TransitionResult:
        """Attach sensitivity to native reference candidates without moving them.

        Only native ``(window, candidate_id)`` membership associates a track.
        A missing reference detection is never manufactured. Both unqualified
        and significance-qualified summaries count distinct resolved point counts
        once. No new filtering occurs; any recorded per-window filters are retained.

        Parameters
        ----------
        window : float
            An exactly declared reference window, as for :meth:`at`.
        min_member_significance : float, optional
            Non-negative finite floor for members used by sensitivity gates.
            ``None`` includes every member. This is an analysis choice, not a
            combination of the correlated per-window significances.

        Returns
        -------
        calamari.transitions.TransitionResult
            Full reference detection with ``smoothing_*`` diagnostics, a
            ``sensitivity_audit`` table, and ``params['smoothing']`` containing
            the scan recipe, all members, tracks and per-window detections.
            ``all()`` retains this description; ``link()`` remains independent.
            Sweep/signals searched at fewer than three distinct representations
            are explicitly unassessed, as are missing tracks and empty qualified
            member sets. ``smoothing_n_windows`` is the searched count;
            ``smoothing_n_present`` and ``smoothing_persistence`` describe support.
            ``smoothing_sign_consistency``, ``smoothing_T_K_spread`` /
            ``smoothing_H_T_spread``, ``smoothing_interval_overlap`` and
            ``smoothing_edge_any`` describe qualified members. Their ``raw_``
            counterparts describe all members, e.g. ``smoothing_raw_T_K_spread``.
        """
        try:
            result = self.at(window)
        except ValueError as exc:
            raise SmoothingSensitivityError(str(exc).replace("at():", "reference():")) from exc
        floor = min_member_significance
        if floor is not None and (
            isinstance(floor, bool) or not isinstance(floor, (int, float, np.integer, np.floating))
            or not np.isfinite(floor) or floor < 0
        ):
            raise SmoothingSensitivityError(
                f"reference(): min_member_significance={floor!r}; supply a finite "
                "non-negative number or None."
            )
        cands, dets = result._full()
        diagnostics = []
        at_reference = self.members[self.members["window"] == float(window)]
        for _, cand in cands.iterrows():
            hits = at_reference[at_reference["candidate_id"] == cand["candidate_id"]]
            record = {"smoothing_track_id": None, "smoothing_status": "missing_track"}
            if len(hits):
                track_id = int(hits.iloc[0]["track_id"])
                track = self.tracks[self.tracks["track_id"] == track_id].iloc[0]
                members = self.members[self.members["track_id"] == track_id]
                qualified = members if floor is None else members[members["significance"] >= floor]
                record.update(smoothing_track_id=track_id, smoothing_n_windows=int(track.n_windows))
                for prefix, subset in (("raw_", members), ("", qualified)):
                    summary = _sensitivity_summary(subset, cand.swept_axis, int(track.n_windows))
                    record.update({f"smoothing_{prefix}{k}": v for k, v in summary.items()})
                record["smoothing_status"] = (
                    "insufficient_windows" if track.n_windows < MIN_WINDOWS else
                    "no_qualified_members" if not len(qualified) else "assessed"
                )
            diagnostics.append(record)
        # Fixed columns also describe an empty reference result and missing tracks.
        columns = ["smoothing_track_id", "smoothing_status", "smoothing_n_windows"] + [
            f"smoothing_{prefix}{key}" for prefix in ("raw_", "") for key in
            ("n_present", "persistence", "sign_consistency", "T_K_spread", "H_T_spread",
             "interval_overlap", "edge_any")
        ]
        augmented = cands.copy()
        frame = pd.DataFrame(diagnostics, columns=columns, index=cands.index)
        for col in columns:
            augmented[col] = frame[col]
        recipe = deepcopy(self.params)
        recipe.update({
            "reference_window": float(window), "min_member_significance": floor,
            "members": self.members.to_dict("records"), "tracks": self.tracks.to_dict("records"),
            "results": {w: {"params": deepcopy(r.params),
                            "candidates": r._full()[0].to_dict("records"),
                            "detections": r._full()[1].to_dict("records"),
                            "skipped": r.skipped}
                        for w, r in self.results.items()},
        })
        return _filtered_result(augmented, dets, result._full_chains(),
                                {**result.params, "smoothing": recipe},
                                result.skipped, result.curves)


def _check_windows(windows) -> tuple[float, ...]:
    """The declared windows as floats, in the order given; refused by name."""
    if isinstance(windows, (str, bytes)) or not isinstance(windows, Iterable):
        raise SmoothingSensitivityError(
            f"scan_smoothing(): windows must be a sequence of at least {MIN_WINDOWS} "
            f"smoothing windows in the swept axis's units, got {windows!r}. Pass e.g. "
            "windows=(1.0, 1.5, 2.0)."
        )
    values = list(windows)
    out = []
    for w in values:
        if isinstance(w, bool) or not isinstance(w, (int, float, np.integer, np.floating)):
            raise SmoothingSensitivityError(
                f"scan_smoothing(): every window must be a number, got {w!r} in "
                f"windows={values!r}."
            )
        w = float(w)
        if not math.isfinite(w) or w <= 0:
            raise SmoothingSensitivityError(
                f"scan_smoothing(): every window must be positive and finite, got {w!r} "
                f"in windows={values!r}. A window is a Savitzky-Golay width in the swept "
                "axis's units (K for a T-scan, T for an H-scan)."
            )
        out.append(w)
    if len(out) < MIN_WINDOWS:
        raise SmoothingSensitivityError(
            f"scan_smoothing(): got {len(out)} window(s) {out!r}; a scan needs at least "
            f"{MIN_WINDOWS}. Two windows show a difference, not how a feature behaves "
            "across a range. Declare the whole range you consider reasonable, e.g. "
            "windows=(1.0, 1.5, 2.0), before looking at the result."
        )
    repeated = sorted({w for w in out if out.count(w) > 1})
    if repeated:
        raise SmoothingSensitivityError(
            f"scan_smoothing(): windows={values!r} repeats {repeated!r}. A repeated window "
            "is the same analysis run twice and would count twice towards persistence; "
            "pass each window once."
        )
    return tuple(out)


def _check_tolerance(name: str, value) -> float | None:
    """A given tracking tolerance must be a positive, finite number."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise SmoothingSensitivityError(
            f"scan_smoothing(): {name} must be a number, got {value!r}."
        )
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise SmoothingSensitivityError(
            f"scan_smoothing(): {name} must be positive and finite, got {value!r}."
        )
    return value


def _resolved_points(results: dict[float, TransitionResult]) -> list[dict]:
    """Per (sweep, signal): the point count each window resolved to.

    Read from each result's ``params['per_sweep'][i]['per_signal'][signal]
    ['smooth_points']``. A sweep skipped at some window, or a signal not
    searched there, is recorded as ``None`` for that window.
    """
    entries: dict[tuple[int, str], dict] = {}
    for window, result in results.items():
        for sweep in result.params["per_sweep"]:
            for signal, record in sweep["per_signal"].items():
                key = (int(sweep["sweep_id"]), signal)
                entry = entries.setdefault(
                    key,
                    {
                        "sweep_id": key[0],
                        "measurement_file": sweep["measurement_file"],
                        "signal": signal,
                        "smooth_points": {w: None for w in results},
                    },
                )
                entry["smooth_points"][window] = int(record["smooth_points"])
    out = []
    for key in sorted(entries):
        entry = entries[key]
        counts = {p for p in entry["smooth_points"].values() if p is not None}
        entry["n_windows"] = len(counts)
        out.append(entry)
    return out


def _resolved_together(points: list[dict]) -> list[dict]:
    """Groups of two or more windows that resolved to one point count."""
    together = []
    for entry in points:
        by_count: dict[int, list[float]] = {}
        for window, pts in entry["smooth_points"].items():
            if pts is not None:
                by_count.setdefault(pts, []).append(window)
        for pts, windows in sorted(by_count.items()):
            if len(windows) > 1:
                together.append(
                    {
                        "sweep_id": entry["sweep_id"],
                        "measurement_file": entry["measurement_file"],
                        "signal": entry["signal"],
                        "windows": tuple(sorted(windows)),
                        "smooth_points": pts,
                    }
                )
    return together


def _warn_resolved_together(together: list[dict]) -> None:
    """Warn once per affected sweep, naming the windows and the count."""
    by_sweep: dict[int, list[dict]] = {}
    for group in together:
        by_sweep.setdefault(group["sweep_id"], []).append(group)
    for sweep_id, groups in sorted(by_sweep.items()):
        parts = "; ".join(
            f"windows {list(g['windows'])} all resolve to {g['smooth_points']} points "
            f"for {g['signal']}"
            for g in groups
        )
        warnings.warn(
            f"scan_smoothing(): on sweep_id={sweep_id} ({groups[0]['measurement_file']}), "
            f"{parts}. The window is clamped to at least 5 points and at most the "
            "sweep, and rounded to an odd count, so those windows are one analysis run "
            "more than once; they are counted once in that sweep's n_windows and "
            "n_present, so persistence is not inflated. To scan this sweep over more "
            "distinct windows, declare windows that differ by more than its grid step "
            "and fit within its length.",
            UserWarning,
            stacklevel=3,
        )


def _mode_and_consistency(signs: list[str]) -> tuple[str | None, float]:
    """The modal sign and the fraction of windows that carry it.

    A tie is broken alphabetically, so it is deterministic; a tied track
    reports ``sign_consistency`` 0.5 or less, which is the number to read.
    """
    present = [s for s in signs if isinstance(s, str)]
    if not present:
        return None, float("nan")
    counts = Counter(present)
    best = max(counts.values())
    mode = sorted(s for s, c in counts.items() if c == best)[0]
    return mode, best / len(present)


def _sensitivity_summary(members: pd.DataFrame, axis: str, n_windows: int) -> dict:
    """Describe distinct representations, without treating an empty set as agreement.

    Windows resolving to the same point count reuse exactly the same smoothed
    curve. Keep the smallest declared window once, as in the scan's track
    summaries. Interval overlap is descriptive: these are correlated estimates,
    not independent confidence intervals.
    """
    distinct = members.sort_values("window").drop_duplicates("smooth_points")
    coordinate, error = ("T_K", "T_err_K") if axis == "T" else ("H_T", "H_err_T")
    locations = distinct[coordinate].to_numpy(dtype=float)
    errors = distinct[error].to_numpy(dtype=float)
    _, consistency = _mode_and_consistency(distinct["sign"].tolist())
    finite_locations = bool(len(locations) and np.isfinite(locations).all())
    intervals_known = finite_locations and bool(np.isfinite(errors).all() and (errors >= 0).all())
    return {
        "n_present": len(distinct),
        "persistence": len(distinct) / n_windows if n_windows else float("nan"),
        "sign_consistency": consistency,
        "T_K_spread": float(np.ptp(locations)) if finite_locations and axis == "T" else np.nan,
        "H_T_spread": float(np.ptp(locations)) if finite_locations and axis == "H" else np.nan,
        "interval_overlap": (
            bool(np.max(locations - errors) <= np.min(locations + errors))
            if intervals_known else None
        ),
        "edge_any": bool(distinct["edge"].any()) if len(distinct) else None,
    }


def scan_smoothing(
    measurements: Measurement | Iterable,
    *,
    windows: Sequence[float],
    track_tol_k: float | None = None,
    track_tol_t: float | None = None,
    link: dict | None = None,
    **detect_kwargs,
) -> SmoothingScanResult:
    """Run detection at every declared smoothing window and track each feature across them.

    Why this exists: the smoothing window is an analysis choice, and a
    weak feature's significance -- a kink's especially, and sometimes its
    sign -- can depend strongly on it. Trying windows until one looks good
    and quoting that one is a forking path. This runs a set of windows the
    caller declares *before* looking, keeps every result, and describes
    how each feature behaves across the whole set. It picks nothing.

    Persistence is robustness to an analysis choice, **not evidence**:
    every window sees the same data, so the per-window significances are
    strongly correlated, and they must never be combined across windows
    (no Stouffer sum, no mean used as a score). That is why ``tracks``
    holds the range of the significance and no combined statistic, and
    why there is no ``best_window``: report the whole declared set. The
    window dependence of Fisher's offset (``docs/DETECTION.md``) is one
    thing a scan makes visible. See the module docstring for what a
    persistent or a transient feature does and does not tell you.

    How it works, deterministically (TR-1):

    1. For each window ``w``, ascending, run
       ``detect_transitions_all(measurements, smooth_window=w, **detect_kwargs)``,
       then ``.link(**link)`` if ``link`` was given. Every result is kept.
    2. Within each ``(sweep_id, signal, method, swept_axis)``, walk the
       windows exactly as :meth:`~calamari.transitions.TransitionResult.link`
       walks a family of sweeps, with the window's position in the sorted
       set as the family coordinate and the swept location as the
       position: seeds in descending significance (ties to the smaller
       window, then the lower location), each taking the nearest unclaimed
       candidate at the adjacent window within the tolerance and stopping
       at the first window that offers none. Every declared window is in
       the ordering, so a window where the feature is absent ends the
       track instead of being stepped over. ``sign`` is not a matching key
       (see :data:`TRACK_BY`). Every candidate is in exactly one track; a
       candidate that matches nothing is a track of one.
    3. Summarise each track over its members.

    Parameters
    ----------
    measurements : Measurement or iterable
        What :func:`~calamari.transitions.detect_transitions_all` accepts:
        Measurements and/or ``split_isotherms`` ``(T_K, Measurement)``
        pairs. A single Measurement is taken as a batch of one. The input
        is read once, so a generator is fine; ``sweep_id`` is the input
        position, the same in every window.
    windows : sequence of float
        The smoothing windows to run, in the swept axis's units (K for a
        T-scan, T for an H-scan). Required, at least :data:`MIN_WINDOWS`,
        each positive, finite and distinct. There is no default set: which
        range of windows is reasonable depends on the sweep's sampling and
        the feature's width, which only you know. Recorded as given and
        sorted.
    track_tol_k : float, optional
        How far a T-swept feature may move between adjacent windows and
        still be the same feature, in K. Required when any T-swept
        candidate is found; there is no default, as for ``link()``.
    track_tol_t : float, optional
        The same for H-swept features, in Tesla.
    link : dict, optional
        Keyword arguments for :meth:`TransitionResult.link
        <calamari.transitions.TransitionResult.link>`, applied to every
        window's result. The chain columns are then carried per member,
        and each track reports the range of its chain statistic.
    **detect_kwargs
        Forwarded to :func:`~calamari.transitions.detect_transitions_all`
        unchanged at every window, and recorded. ``criterion='kink'`` is
        the natural companion when the question is a slope change. A
        ``robust=`` given here filters every window's result before the
        tracking, so the tracks then describe the filtered candidates.

    Returns
    -------
    SmoothingScanResult
        ``results`` per window, the ``members`` and ``tracks`` tables, and
        the recipe in ``params``.

    Raises
    ------
    SmoothingSensitivityError
        If ``smooth_window`` is passed (the scan sets it); if ``windows``
        has fewer than three entries, or one that is non-numeric,
        non-positive, non-finite or repeated; if a given tolerance is not
        a positive, finite number; or if candidates on a swept axis were
        found and that axis's tolerance was not given. Each message names
        the fix.

    Warns
    -----
    UserWarning
        Once per sweep on which two or more declared windows resolved to
        the same point count. A window is clamped to at least 5 points and
        at most the sweep, and rounded to an odd count, so on a short or
        coarse sweep two windows can be one analysis. They are counted
        once, and the groups are recorded in ``params['resolved_together']``.

    Examples
    --------
    >>> from calamari.scales import scan_smoothing
    >>> scan = scan_smoothing(sweeps, windows=(1.0, 1.5, 2.0, 3.0),  # doctest: +SKIP
    ...                       track_tol_k=0.5, criterion="kink",
    ...                       methods=("peak", "derivative", "kink"))
    >>> scan.tracks.query("method == 'kink'")  # doctest: +SKIP
    """
    if "smooth_window" in detect_kwargs:
        raise SmoothingSensitivityError(
            "scan_smoothing(): smooth_window is set by the scan, one run per entry of "
            f"windows=, so it cannot also be passed (got smooth_window="
            f"{detect_kwargs['smooth_window']!r}). Put the windows you want in windows=; "
            "to detect at one window, call detect_transitions_all(smooth_window=...)."
        )
    given = _check_windows(windows)
    tolerances = {
        "T": _check_tolerance("track_tol_k", track_tol_k),
        "H": _check_tolerance("track_tol_t", track_tol_t),
    }
    if link is not None and not isinstance(link, dict):
        raise SmoothingSensitivityError(
            f"scan_smoothing(): link must be a dict of TransitionResult.link() keyword "
            f"arguments, e.g. link={{'t_tol_k': 0.8}}, got {link!r}."
        )
    ordered = tuple(sorted(given))

    # Read the input once: a generator would be exhausted by the first
    # window, and every window must see the same sweeps with the same ids.
    batch = [measurements] if isinstance(measurements, Measurement) else list(measurements)

    results: dict[float, TransitionResult] = {}
    for w in ordered:
        result = detect_transitions_all(batch, smooth_window=w, **detect_kwargs)
        if link is not None:
            result = result.link(**link)
        results[w] = result

    frames = []
    for index, (w, result) in enumerate(results.items()):
        # The view the caller asked each window for: filtered, if a robust=
        # was forwarded, so the tracks describe what that filter kept.
        view = result.candidates
        if not len(view):
            continue
        frame = view.copy()
        frame["window"] = w
        frame[_WINDOW_INDEX] = float(index)
        frames.append(frame)
    cands = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    axes = set(cands["swept_axis"].dropna()) if len(cands) else set()
    for axis in sorted(axes):
        name, unit, _, _ = _TRACK_AXES[axis]
        if tolerances[axis] is None:
            raise SmoothingSensitivityError(
                f"scan_smoothing(): candidates swept along {axis} were found but {name} "
                f"is None. How far a feature may move between adjacent windows and still "
                f"be the same feature is a judgement only you can make, so there is no "
                f"default; pass {name}= in {unit} (e.g. about the location error you "
                "would accept between two reasonable windows)."
            )

    points = _resolved_points(results)
    together = _resolved_together(points)
    if together:
        _warn_resolved_together(together)
    point_lookup = {(e["sweep_id"], e["signal"]): e for e in points}

    linked = link is not None
    member_columns = MEMBER_COLUMNS + (_MEMBER_CHAIN_COLUMNS if linked else [])
    track_columns = TRACK_COLUMNS + (_TRACK_CHAIN_COLUMNS if linked else [])

    if len(cands):
        family = {axis: [float(i) for i in range(len(ordered))] for axis in _TRACK_AXES}
        geometry = {axis: (_WINDOW_INDEX, spec[2]) for axis, spec in _TRACK_AXES.items()}
        chains, _, _ = _chain_walk(
            cands, TRACK_BY, geometry=geometry, tolerances=tolerances, family_values=family
        )
    else:
        chains = []

    member_rows, track_rows = [], []
    summaries = []
    for rows in chains:
        first = cands.iloc[rows[0]]
        axis = first["swept_axis"]
        sweep_id, signal = int(first["sweep_id"]), first["signal"]
        entry = point_lookup[(sweep_id, signal)]
        _, _, loc_col, held_col = _TRACK_AXES[axis]
        hits = []
        for i in sorted(rows, key=lambda i: cands.at[i, _WINDOW_INDEX]):
            row = cands.iloc[i]
            hits.append((row, entry["smooth_points"][row["window"]]))
        # Windows that resolved to one point count are one analysis: keep
        # the first (smallest) of each, so no summary counts it twice.
        distinct, seen = [], set()
        for row, pts in hits:
            if pts not in seen:
                seen.add(pts)
                distinct.append(row)
        locations = np.array([r[loc_col] for r in distinct], dtype=float)
        sigs = np.array([r["significance"] for r in distinct], dtype=float)
        mode, consistency = _mode_and_consistency([r["sign"] for r in distinct])
        n_windows = entry["n_windows"]
        summary = {
            "sweep_id": sweep_id,
            "swept_axis": axis,
            "signal": signal,
            "method": first["method"],
            "T_K": np.nan,
            "H_T": np.nan,
            "n_windows": n_windows,
            "n_present": len(distinct),
            "persistence": len(distinct) / n_windows if n_windows else float("nan"),
            "sign_mode": mode,
            "sign_consistency": consistency,
            "significance_min": _nan_reduce(sigs, np.min),
            "significance_median": _nan_reduce(sigs, np.median),
            "significance_max": _nan_reduce(sigs, np.max),
            "edge_any": bool(any(bool(r["edge"]) for r in distinct)),
        }
        summary[held_col] = float(first[held_col])
        for col in ("T_K", "H_T"):
            for stat in ("median", "min", "max", "spread"):
                summary[f"{col}_{stat}"] = np.nan
        summary[f"{loc_col}_median"] = _nan_reduce(locations, np.median)
        summary[f"{loc_col}_min"] = _nan_reduce(locations, np.min)
        summary[f"{loc_col}_max"] = _nan_reduce(locations, np.max)
        summary[f"{loc_col}_spread"] = summary[f"{loc_col}_max"] - summary[f"{loc_col}_min"]
        if linked:
            chain_sig = np.array([r["chain_significance"] for r in distinct], dtype=float)
            summary["chain_significance_min"] = _nan_reduce(chain_sig, np.min)
            summary["chain_significance_median"] = _nan_reduce(chain_sig, np.median)
        first_window = float(hits[0][0][_WINDOW_INDEX])
        order_key = (
            sweep_id,
            axis,
            str(signal),
            str(first["method"]),
            _finite_or_last(summary[f"{loc_col}_median"]),
            first_window,
            min(rows),
        )
        summaries.append((order_key, summary, hits))

    # Track ids read along each sweep: by sweep, axis, signal and method, then
    # by where the feature sits. Deliberately not by significance -- ordering
    # by any one window's score would single that window out.
    summaries.sort(key=lambda item: item[0])
    for track_id, (_, summary, hits) in enumerate(summaries):
        track_rows.append({"track_id": track_id, **summary})
        for row, pts in hits:
            member = {
                "track_id": track_id,
                "window": float(row["window"]),
                "smooth_points": pts,
                "sweep_id": int(row["sweep_id"]),
                "candidate_id": int(row["candidate_id"]),
                "T_K": row["T_K"],
                "H_T": row["H_T"],
                "T_err_K": row["T_err_K"],
                "H_err_T": row["H_err_T"],
                "sign": row["sign"],
                "significance": row["significance"],
                "edge": bool(row["edge"]),
            }
            if linked:
                member["chain_id"] = int(row["chain_id"])
                member["chain_significance"] = float(row["chain_significance"])
            member_rows.append(member)

    members = pd.DataFrame(member_rows, columns=member_columns)
    tracks = pd.DataFrame(track_rows, columns=track_columns)

    params = {
        "windows": given,
        "windows_sorted": ordered,
        "track_tol_k": tolerances["T"],
        "track_tol_t": tolerances["H"],
        "track_by": TRACK_BY,
        "detect_kwargs": dict(detect_kwargs),
        "link": dict(link) if link is not None else None,
        "smooth_points": points,
        "resolved_together": together,
        "calamari_version": _calamari_version(),
    }
    return SmoothingScanResult(results=results, members=members, tracks=tracks, params=params)


def _nan_reduce(values: np.ndarray, how) -> float:
    """``how`` over the non-NaN entries; NaN, without a RuntimeWarning, if none."""
    present = values[~np.isnan(values)]
    return float(how(present)) if present.size else float("nan")


def _finite_or_last(value: float) -> float:
    """NaN sorts last rather than comparing false against everything."""
    return math.inf if math.isnan(value) else value
