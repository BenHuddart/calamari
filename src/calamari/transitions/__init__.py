"""Phase-transition candidate detection for 1-D magnetometry sweeps.

Search DC moment or AC susceptibility on a single T- or H-scan with
:func:`detect_transitions`, or process a campaign with :func:`detect_transitions_all`.
:func:`find_transitions` accepts arrays directly. Candidates describe signal features;
identifying a phase transition requires the caller's scientific interpretation.

Methods and defaults
--------------------
The :data:`METHODS` registry contains criterion definitions and citations:

* ``'peak'``: maxima and minima of the smoothed signal.
* ``'derivative'``: extrema of the signed first derivative.
* ``'kink'``: extrema of the second derivative, for changes of slope.
* ``'onset'``: intersection of a baseline and the steepest-slope tangent.
* ``'fisher'``: extrema of d(yT)/dT, subject to the susceptibility assumptions
  documented by :func:`find_transitions`. Fisher candidates remain separate
  from other criteria when grouping detections.

H-scans default to ``'peak'`` for susceptibility and ``'derivative'`` for moment.
T-scans run both. Kink, onset and Fisher searches are opt-in because they assume
particular feature shapes or response interpretations. Smoothing affects each
criterion's location; see :func:`find_transitions` for grouping and error conventions.

Noise and significance
----------------------
Detection uses a permissive prominence floor. ``significance`` normalises prominence
by the noise of the smoothed representation, with a sweep-length correction. It is
not a calibrated z-score or false-alarm probability.

``noise='auto'`` prefers usable instrument errors, otherwise a second-difference
scatter estimate. It warns when instrument errors exceed the scatter estimate by
:data:`NOISE_RATIO_WARN`. Explicit ``'instrument'``, ``'scatter'`` or positive sigma
choices are available. Both estimates and the resolved choice are recorded.

Result transformations
----------------------
All operations are deterministic for their inputs and settings.
:meth:`TransitionResult.robust` filters by declared diagnostic thresholds.
:meth:`TransitionResult.link` describes chains across neighbouring sweeps; combined
chain scores remain diagnostics. :meth:`TransitionResult.select` records an explicit
criterion, sign and swept-axis window, retaining at most one candidate per sweep.
:meth:`TransitionResult.reconcile` compares selected T- and H-scan boundaries by
interpolation, without extrapolation or implicit concordance tolerances.
Linking and reconciliation describe results without dropping points. See
:class:`TransitionResult` for transformation order and recorded settings.
"""

from calamari.transitions import _metadata
from calamari.transitions._measurement import detect_transitions, detect_transitions_all
from calamari.transitions._metadata import Citation, MethodInfo
from calamari.transitions._numerics import find_transitions, smooth_derivative
from calamari.transitions._results import TransitionResult
from calamari.transitions._results import (
    pd as pd,  # Resolve the public result class's postponed DataFrame annotations.
)

#: Detection identities, display markers and scientific/statistical citations.
METHODS = _metadata.METHODS

#: Reporting priority within a grouped feature; Fisher candidates remain separate.
CRITERION_PRIORITY = _metadata.CRITERION_PRIORITY

#: Permissive prominence floor, as a fraction of the signal's peak-to-peak range.
DEFAULT_PROMINENCE_FRACTION = _metadata.DEFAULT_PROMINENCE_FRACTION

#: Uncalibrated diagnostic threshold used by :meth:`TransitionResult.robust`.
DEFAULT_MIN_SIGNIFICANCE = _metadata.DEFAULT_MIN_SIGNIFICANCE

#: Instrument/scatter noise ratio above which auto noise selection warns.
NOISE_RATIO_WARN = _metadata.NOISE_RATIO_WARN

#: Canonical measurement-level candidate columns, including stable sweep identity.
CANDIDATE_COLUMNS = _metadata.CANDIDATE_COLUMNS

#: Canonical constituent-detection columns, linked to their candidate by ID.
DETECTION_COLUMNS = _metadata.DETECTION_COLUMNS

#: Chain summaries; their significance statistic is not a calibrated probability.
CHAIN_COLUMNS = _metadata.CHAIN_COLUMNS

#: Candidate fields that must match within a linked chain, besides the swept axis.
DEFAULT_LINK_BY = _metadata.DEFAULT_LINK_BY

__all__ = [
    "Citation",
    "MethodInfo",
    "METHODS",
    "TransitionResult",
    "CANDIDATE_COLUMNS",
    "DETECTION_COLUMNS",
    "NOISE_RATIO_WARN",
    "CHAIN_COLUMNS",
    "find_transitions",
    "detect_transitions",
    "detect_transitions_all",
    "smooth_derivative",
]

# Preserve public import/pickle paths after moving implementations internally.
for _public in (
    Citation, MethodInfo, TransitionResult, find_transitions, smooth_derivative,
    detect_transitions, detect_transitions_all,
):
    _public.__module__ = __name__
del _public
