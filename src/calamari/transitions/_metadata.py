"""Detection method identities, scientific defaults and result schemas."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    """One literature citation backing a detection method.

    ``kind`` says which side of the method it supports (``'statistical'``
    for the numerical machinery, ``'magnetometry'`` for the domain
    convention, or a neighbouring probe's domain -- ``'transport'`` --
    where the convention is borrowed from measurements of a different
    quantity; naming the other probe rather than relabelling it
    magnetometry keeps visible that the exemplar is rho(T), not chi(T));
    ``role`` says how strong the backing is (``'origin'`` --
    the paper that introduced it; ``'review'``; ``'exemplar'`` -- a
    representative use of a convention that has no origin paper;
    ``'implementation'`` -- the software that implements it). The split
    exists so the output never overstates its own bibliography.
    """

    text: str
    kind: str
    role: str


@dataclass(frozen=True)
class MethodInfo:
    """Identity card for one detection method.

    ``marker`` is the canonical matplotlib marker *string* for this method
    (data, not a plotting import -- PK-1 is untouched), so the same
    criterion gets the same shape on every calamari figure.
    """

    key: str
    name: str
    criterion: str
    citations: tuple[Citation, ...]
    marker: str


#: The detection methods this module implements, keyed by the ``method``
#: value that appears in every output row. Citations live here, once --
#: rows carry only the key; :meth:`TransitionResult.provenance` joins them.
METHODS: dict[str, MethodInfo] = {
    "peak": MethodInfo(
        key="peak",
        name="signal extremum (cusp criterion)",
        criterion="prominence-filtered local extrema (maxima and minima) of the smoothed signal",
        citations=(
            Citation(
                "A. Savitzky and M. J. E. Golay, Anal. Chem. 36, 1627 (1964)",
                kind="statistical",
                role="origin",
            ),
            Citation(
                "P. Virtanen et al. (SciPy), Nat. Methods 17, 261 (2020) -- "
                "prominence-based peak selection has no canonical origin paper",
                kind="statistical",
                role="implementation",
            ),
            Citation(
                "M. Balanda, Acta Phys. Pol. A 124, 964 (2013)",
                kind="magnetometry",
                role="review",
            ),
            Citation(
                "V. Cannella and J. A. Mydosh, Phys. Rev. B 6, 4220 (1972)",
                kind="magnetometry",
                role="origin",
            ),
        ),
        marker="o",
    ),
    "derivative": MethodInfo(
        key="derivative",
        name="derivative extremum",
        criterion="local extrema of the signed first derivative dy/dx "
        "(Savitzky-Golay smoothed differentiation)",
        citations=(
            Citation(
                "A. Savitzky and M. J. E. Golay, Anal. Chem. 36, 1627 (1964)",
                kind="statistical",
                role="origin",
            ),
            Citation(
                "E. Stryjewski and N. Giordano, Adv. Phys. 26, 487 (1977) -- dM/dH "
                "at metamagnetic critical fields",
                kind="magnetometry",
                role="review",
            ),
            # The M(T) inflection-point convention has no origin paper, and it
            # is NOT an approximation to Fisher's d(chi T)/dT criterion (Philos.
            # Mag. 7, 1731 (1962)) -- the two mark systematically different
            # temperatures, offset by a material-dependent fraction of the
            # feature width. Fisher is cited only where d(chi T)/dT itself is
            # implemented: the separate, opt-in 'fisher' entry below.
        ),
        marker="^",
    ),
    "kink": MethodInfo(
        key="kink",
        name="curvature extremum (kink criterion)",
        criterion="local extrema of the second derivative d²y/dx² "
        "(Savitzky-Golay smoothed differentiation): the point of largest curvature "
        "of a slope change",
        citations=(
            Citation(
                "A. Savitzky and M. J. E. Golay, Anal. Chem. 36, 1627 (1964)",
                kind="statistical",
                role="origin",
            ),
            Citation(
                "P. Virtanen et al. (SciPy), Nat. Methods 17, 261 (2020) -- "
                "prominence-based peak selection on the second derivative",
                kind="statistical",
                role="implementation",
            ),
            Citation(
                "G. Asti and S. Rinaldi, Phys. Rev. Lett. 28, 1584 (1972) -- "
                "singular-point detection: a singular point of a magnetisation curve "
                "located at the extremum of d²M/dH²",
                kind="magnetometry",
                role="origin",
            ),
            Citation(
                "K. Semeniuk et al., Phys. Rev. B 107, L220504 (2023) -- transition "
                "temperature defined as the point of largest curvature of rho(T), the "
                "second derivative taken by Savitzky-Golay smoothing (transport; the "
                "same criterion)",
                kind="transport",
                role="exemplar",
            ),
        ),
        marker="D",
    ),
    "onset": MethodInfo(
        key="onset",
        name="two-tangent onset",
        criterion="intersection of a linear baseline fit with the steepest-slope tangent",
        citations=(
            Citation(
                "ordinary least squares; the intersection uncertainty propagates "
                "from the two fits' covariances",
                kind="statistical",
                role="implementation",
            ),
            Citation(
                "standard practice for superconducting T_c from chi(T)/M(T); "
                "exemplar: S. L. Bud'ko et al., arXiv:cond-mat/0104088 -- no origin paper",
                kind="magnetometry",
                role="exemplar",
            ),
        ),
        marker="s",
    ),
    # Opt-in and never grouped with the other methods (see find_transitions):
    # it marks a systematically different temperature from the chi cusp.
    # Citation checked against the paper itself (DOI 10.1080/14786436208213705;
    # also the Crossref record: volume 7, issue 82, pages 1731-1743, 1962).
    # The relation is its Eq. (1), C_M(T) ~ A d[T chi_par(T)]/dT, with
    # chi_par the ZERO-FIELD susceptibility parallel to the ordering axis and
    # A slowly varying; section 2 extends the argument to chi_perp in the
    # paramagnetic and critical region. T_c sits at the specific-heat
    # singularity, i.e. the d(chi T)/dT anomaly, and the paper shows the chi
    # maximum lying above it (5-8% for MnO and MnF2, section 3) -- the offset
    # this method exists to keep visible.
    "fisher": MethodInfo(
        key="fisher",
        name="Fisher specific-heat criterion",
        criterion="extrema of d(yT)/dT, the magnetic specific-heat proxy "
        "(Savitzky-Golay smoothed differentiation of y times T)",
        citations=(
            Citation(
                "A. Savitzky and M. J. E. Golay, Anal. Chem. 36, 1627 (1964)",
                kind="statistical",
                role="origin",
            ),
            Citation(
                "M. E. Fisher, Philos. Mag. 7, 1731 (1962), Eq. (1) -- the magnetic "
                "specific heat of a simple antiferromagnet varies like d(T chi)/dT, "
                "chi the zero-field (parallel) susceptibility, so its anomaly marks "
                "the ordering temperature",
                kind="magnetometry",
                role="origin",
            ),
        ),
        marker="v",
    ),
}

#: Which member of a grouped feature supplies the reported coordinate, in
#: priority order. Fixed rather than "highest confidence" on purpose: the
#: methods' confidence measures are not commensurable, and letting the
#: winner vary per row would silently interleave systematically different
#: criteria along one phase boundary. The cusp outranks the derivative
#: because it is a directly measured feature of the response, while the
#: derivative extremum's position inherits smoothing dependence. The kink
#: comes last, below the derivative, because of where the two can meet:
#: the only groups holding both are rounded steps, whose centre IS the
#: derivative extremum and whose curvature extrema are its two shoulders,
#: about half a smoothing window either side. Ranking the kink first there
#: would move the reported coordinate onto whichever shoulder happened to
#: be stronger -- the very flank defect the kink criterion cures, in
#: reverse. A kink that is a slope change in its own right has no
#: derivative extremum in its group to lose to, so it reports itself.
#: Overridable per call via ``criterion=``. Promoting ``'kink'`` does not
#: let just any curvature extremum report: only a group's own slope change
#: may (a lone kink, or a cusp's central curvature extremum between its
#: derivative flanks), never a cusp's flank companion or a step's shoulder,
#: so a group without one reports what it would under this default order.
#: See :func:`_group_detections`. ``'fisher'`` is absent on
#: purpose, and for the reason this ordering exists: its d(yT)/dT extremum
#: sits systematically below the chi cusp, by an amount that depends on the
#: material and on the smoothing window, so it is never grouped with the other methods at all -- ranked into a
#: group, it would either hide that offset behind the cusp or report a
#: different temperature under the cusp's row.
CRITERION_PRIORITY = ("peak", "onset", "derivative", "kink")

#: Detection floor: a feature must stand at least this fraction of the
#: searched signal's own peak-to-peak range above its surroundings. A
#: permissive well-posedness floor (it bounds the output to features
#: rather than every noise wiggle), NOT a robustness judgement -- that is
#: :meth:`TransitionResult.robust`'s job.
DEFAULT_PROMINENCE_FRACTION = 0.05

#: Default significance threshold for :meth:`TransitionResult.robust`: the
#: feature's prominence must exceed 5x the extreme-value-corrected noise
#: level of its own representation. This is an uncalibrated diagnostic
#: screening convention; record noise, smoothing and threshold choices.
DEFAULT_MIN_SIGNIFICANCE = 5.0


#: median|second difference| -> sigma for uncorrelated Gaussian noise:
#: Var(d2y) = 6 sigma^2 and median|N(0,s)| = 0.6745 s, so the estimator
#: reads 0.6745 * sqrt(6) = 1.652 sigma and must be divided by this to
#: report sigma itself.
_SECOND_DIFF_TO_SIGMA = 0.6745 * math.sqrt(6.0)

#: Warn when reported instrument error exceeds the scatter estimate by this
#: ratio, while retaining the explicitly selected noise policy. Ten is an
#: empirical diagnostic default, not a calibrated test that either estimate
#: is incorrect: correlations, drift and systematic terms can differ.
NOISE_RATIO_WARN = 10.0

#: Accepted string values of noise= (a positive float is also accepted;
#: see find_transitions).
_VALID_NOISE_STRINGS = ("auto", "instrument", "scatter")

#: Measurement channels this module can search: signal accessor ->
#: (error accessor, probe tag). The probe tag feeds the batch table.
_SIGNALS = {
    "chi_real": ("chi_real_err", "AC"),
    "chi_imag": ("chi_imag_err", "AC"),
    "moment_emu": ("moment_err_emu", "DC"),
}

#: Canonical column order of the measurement-level candidates table.
#: ``sweep_id`` identifies the sweep a
#: row came from -- ``0`` from :func:`detect_transitions`, the input position
#: from :func:`detect_transitions_all` -- because ``measurement_file`` cannot:
#: the isotherms :func:`calamari.sweeps.split_isotherms` cuts from one file
#: all share that file's path.
CANDIDATE_COLUMNS = [
    "candidate_id",
    "T_K",
    "H_T",
    "T_err_K",
    "H_err_T",
    "T_span_K",
    "H_span_T",
    "swept_axis",
    "signal",
    "frequency_hz",
    "method",
    "n_methods",
    "feature",
    "sign",
    "significance",
    "sweep_range_over_noise",
    "edge",
    "measurement_file",
    "sweep_id",
]

#: Canonical column order of the measurement-level detections table (one
#: row per raw detector hit; ``candidate_id`` links to the grouped row).
#: Ends with ``sweep_id`` for the same reason as :data:`CANDIDATE_COLUMNS`.
DETECTION_COLUMNS = [
    "candidate_id",
    "method",
    "signal",
    "T_K",
    "H_T",
    "raw_strength",
    "raw_strength_unit",
    "significance",
    "feature",
    "sign",
    "edge",
    "measurement_file",
    "sweep_id",
]

#: Canonical column order of the chain table produced by
#: :meth:`TransitionResult.link` -- one row per linked chain of
#: candidates across a family of sweeps.
CHAIN_COLUMNS = [
    "chain_id",
    "n",
    "signal",
    "method",
    "sign",
    "swept_axis",
    "T_K_min",
    "T_K_max",
    "H_T_min",
    "H_T_max",
    "significance_median",
    "significance_max",
    "chain_significance",
]

#: Columns :meth:`TransitionResult.link` adds to the candidates table.
_CHAIN_CANDIDATE_COLUMNS = ["chain_id", "chain_n", "chain_significance"]

#: Default grouping keys for :meth:`TransitionResult.link`: a chain is a
#: recurrence of the SAME feature seen the same way. Mixing signals would
#: chain a chi' cusp to the chi'' shoulder below it; mixing methods would
#: chain a cusp to a derivative flank; mixing signs would chain a maximum
#: to the minimum next door. Each is a different measurement of (possibly)
#: different physics, so none of them is evidence that the other recurred.
DEFAULT_LINK_BY = ("signal", "method", "sign")
