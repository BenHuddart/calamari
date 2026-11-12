# Calamari — Architecture

Calamari helps researchers analyse magnetometry measurements through repeatable
workflows and present the results in publication-quality figures. These binding
requirements protect quantities, uncertainties,
explicit scientific choices and composable presentation. Their IDs are stable references
for code review and issues.

Requirements describe the intended contract. An implementation gap is a defect to report,
not permission to weaken the requirement. Resolve conflicts using evidence and history;
agree changes to scientific meaning or public API before implementation. A passing test
establishes behaviour, not scientific correctness.

## Responsibilities and documentation

| Home | Responsibility |
|---|---|
| `AGENTS.md` | Working practices, scientific-code rules, research-data policy and verification |
| This document | Current design contracts and module boundaries |
| User guide and [API reference](api.md) | Detailed behaviour, units, schemas, scientific assumptions and runnable workflows |
| [GitHub issues](https://github.com/BenHuddart/calamari/issues) | Plans, decisions, investigations and unfinished work |

A figure recipe retains its input selection, sample facts, analysis settings, software
identity, environment and export settings. Result provenance supports that recipe;
external inputs and preprocessing remain the caller's responsibility.

| Modules | Responsibility and collaboration |
|---|---|
| `io`, `measurement` | Recognise and parse files; expose recorded quantities and row families |
| `sample`, `constants` | Explicit specimen facts, normalisations and shared physical constants |
| `catalog`, `sweeps`, `repeats` | Inventory, select, split and explicitly aggregate measurements |
| `quality`, `scales` | Point quality and smoothing sensitivity |
| `transitions/` | Public detection API, backed by the internal responsibilities below |
| `fitting`, `boundary` | Fit explicit models to observations or selected candidate points |
| `_plot_data` | Prepare series/grids without a plotting backend; reuse measurement/analysis numerics |
| `plot/`, `style` | Render results; apply publication styling, geometry and panel lettering |
| `simulate` | Generate synthetic inputs for tests |
| `_sweep_rules`, `_provenance` | Shared sweep arithmetic and software-version identity |
| `_chains` | Candidate-chain walk shared by linking and smoothing scans |

Within `transitions/`, `_metadata` owns method identities, defaults and schemas;
`_numerics` owns array detection; `_results` owns linking, filtering, selection and
reconciliation; `_measurement` owns measurement preparation and batch orchestration.
The public facade retains import and pickle paths. Smoothing scans share the chain
walk in `_chains` and rebuild result views through `_results`.

This map describes responsibilities; PK-1 defines the plotting import restriction.
Source module paths below are relative to `src/calamari/`.
The package root re-exports public entry points. Shared arithmetic has one production
implementation; callers retain their documented preparation and selection policies.

## IO-1 — content-sniffing loader registry

Identify a format by content, never filename or extension. `load`, `sniff_format` and
`is_supported_file` use the same registered signatures (`BaseLoader.sniff`, through
`LoaderRegistry.detect_lines`, over the first approximately 40 lines). `load` preserves
read failures; the discovery probes can report an unsupported input without raising.
Register a loader to support a new format.

**Rationale.** File extensions do not distinguish the supported dialects or file roles.

**Reference.** `io/base.py`, `io/__init__.py`;
[adding a format](EXTENDING.md).

## IO-2 — evidenced MPMS3 exports

Support MPMS3 DC, VSM and AC measurements, including mixed-family files. Preserve
value-before-key INFO ordering, known header variants and documented column quantities.
Read each file's column titles dynamically and resolve units explicitly. Legacy
MPMS/MPMS-XL exports are outside the supported scope.

**Rationale.** Supported formats must have inspectable evidence and maintained regression
coverage; the retired legacy dialect lacks a current empirical validation corpus.

**Reference.** `io/quantum_design.py`, `measurement.py`;
[supported exports](USER_GUIDE.md#install-and-load).

## IO-3 — epoch trap

Expose the recorded numeric timestamp as `date_epoch` without converting it to a calendar
date. Inspected native MPMS 3 exports empirically support seconds from 1899-12-30;
timezone, clock handling and all-version applicability remain unspecified.
Derive `date_iso` from the human-readable M/D/YYYY token. Do not infer an instrument
timezone that the file does not record.

**Rationale.** Assuming an epoch or timezone can silently change the date. The earlier
MPMS3 1904/LabVIEW interpretation was refuted; the parser does not depend on it.

**Reference.** `io/quantum_design.py`, `Measurement.date_iso` / `date_epoch`;
[date handling](METHODS.md#loading-and-preparation).

## IO-4 — untrusted metadata: cross-check it, or record where it came from

Cross-check metadata against populated data where possible. Classify measurement families
per row using `Transport Action` where present and populated quantities otherwise,
checking them against one another. Consult the header only where rows contain no
measurement. A file can hold several families: expose all of them, with `of_type` /
`split_by_type` for explicit selection.

Mass has no in-file corroboration. `Sample.from_measurement` accepts the recorded
`SAMPLE_MASS` through `Measurement.sample_mass_mg`, lets caller input take
precedence, and records `mass_source` as `file`, `user` or `None`. An unknown source or
missing value must not be presented as independently verified.

**Rationale.** Header application labels and operator-entered sample facts can be stale.
Cross-checkable information and uncheckable specimen facts need different handling.

**Reference.** `measurement.py`, `sample.py`;
[measurement families](USER_GUIDE.md#install-and-load),
[sample mass](USER_GUIDE.md#supply-sample-facts-and-choose-units) and
[mass provenance](METHODS.md).

## IO-5 — strict load, distinct failure modes

`load` returns a fully parsed `Measurement` with a recognised `data_format`; there is no
best-effort `unknown` format. Keep these failures distinct:

| Condition | Outcome |
|---|---|
| Unreadable input | The filesystem's own `OSError` subtype |
| Readable, unrecognised content | `UnsupportedFileError` |
| Unregistered explicit `fmt=` | `UnknownFormatError`, before reading the file |
| Recognised but malformed structure | `FileParseError` |

The three Calamari exceptions also derive from `CalamariError` and `ValueError`.
Valid zero-row exports retain ordered column titles. `sniff_format` is the cheap signature
probe; `is_supported_file` parses and validates without raising and is true exactly when
`load` succeeds.

The common error contract applies at all public boundaries: deliberate Calamari errors
subclass `CalamariError`, retaining appropriate `ValueError`, `TypeError` or `KeyError`
compatibility. Propagated filesystem/third-party exceptions retain their types. Remaining
implementation gaps must be reported; an ordinary unsuccessful fit remains a result (FT-1).

**Rationale.** Each failure requires a different remedy; incomplete data must not look loaded.

**Reference.** `io/__init__.py`, `exceptions.py`; [loading failures](USER_GUIDE.md#install-and-load).

## DM-1 — non-mutating Measurement operations

Library operations derive new measurements or samples rather than mutating their inputs
(`restrict`, `with_moment`). `Sample` is frozen. Exposed DataFrames and nested dictionaries
are not guaranteed to be deeply immutable; callers should treat shared data as read-only.

**Rationale.** A transformation must not silently alter another holder's measurement.

**Reference.** `measurement.py`, `sample.py`;
[restricting a sweep](USER_GUIDE.md#prepare-the-selected-measurements).

## DM-2 — explicit unit accessors

One accessor returns one quantity in one documented unit and normalisation, independent
of dialect. New accessors/conversions name units explicitly; there is no ambiguous bare
`.moment` or `.field`. Preserve established names: `temperature` is K and `chi_real` /
`chi_imag` are emu/Oe; `Sample` conversions document their mass/mole/ion basis. Where
different measurements could satisfy an accessor, require a caller choice.

Resolve column base names exactly and interpret the unit suffix before conversion.
An unrecognised unit raises `UnknownUnitError`, rather than unscaled values or an absent
column. AC susceptibilities are read directly from the recorded MPMS3 components;
no legacy AC moment-to-susceptibility derivation is performed. Resolve from available
columns, not an application label.

`Measurement` may reconcile encodings using only file-supplied values. `Sample` owns
normalisations requiring specimen facts the file cannot establish.

**Rationale.** A plausible-looking value in the wrong quantity, unit or normalisation can
invalidate every later scientific result.

**Reference.** `measurement.py`, `sample.py`; [units](METHODS.md#units-and-specimen-facts),
[moment selection](USER_GUIDE.md#install-and-load),
[AC susceptibility](USER_GUIDE.md#install-and-load).

## SA-1 — typed Sample, no hidden defaults

Use a typed, frozen `Sample` with explicit specimen facts. Fields default to `None` except
`name`; do not assume a mass, molar mass or ion count. Supplied normalisation facts must
satisfy their documented finite-positive scalar contract, including fractional average
ion occupancy. A conversion needing a missing fact raises `SampleValidationError` (also
`ValueError`), naming the field, rather than substituting a guessed value.

**Rationale.** Sample-dependent normalisation must be inspectable at the call site.

**Reference.** `sample.py`;
[constructing a sample](USER_GUIDE.md#supply-sample-facts-and-choose-units), [units](METHODS.md#units-and-specimen-facts).

## TR-1 — deterministic transition detection; robustness filtered, not gated

Detection and result transformations are deterministic for their inputs and parameters.
Numerical choices (including detection floor, smoothing window and noise estimate) and
caller-dependent choices are explicit and documented; record resolved settings per sweep
and signal. Screening selects input points before detection and records that selection.

Detection produces candidates before robustness filtering. `robust()` owns the robustness,
chain and sensitivity gates and records them. `select()` keeps at most one candidate per
sweep for a caller-named criterion/sign and an explicit swept-axis window; significance
ties go to the lower location. Use `sweep_id` for identity: zero for a single sweep, input
position including skipped entries for a batch. File paths cannot identify split sweeps.

`link()` describes chains and `reconcile()` describes agreement across selected swept
axes; neither drops points. Reconciliation uses interpolation without extrapolation or
default tolerances. Record their settings, resolved selection windows and empty sweeps.
On rebuild, robustness precedes selection and reconciliation is reapplied last. `all()`
restores filtered candidates, retains links and smoothing-reference descriptions, and
removes selections and dependent reconciliation.

`scan_smoothing` retains every declared-window result and tracks features with the linking
walk, using window index as the family coordinate and explicit tolerance. It combines no
per-window significance scores. `reference()` describes a caller-declared native window
without moving its coordinates/errors. Reference sensitivity gates are opt-in, applied
before selection, and recoverable when loosened or through `all()`; retain the full audit,
per-window recipe and explicit unassessed statuses.

Significance is a normalised diagnostic, not a calibrated probability or Gaussian z-score.
Record the default `min_significance=5` and overrides; probability claims need calibration
for the actual sampling, noise and processing. Other physics-dependent choices require
caller input. A transition candidate is not automatically a phase boundary.

**Rationale.** Detection, descriptions and scientific selection must remain separately
inspectable and repeatable; filtering cannot hide the evidence needed for grouping.

**Reference.** `transitions/`, `scales.py`;
[detection](METHODS.md#transition-detection), [selection](METHODS.md#transition-detection),
[reconciliation](METHODS.md#transition-detection),
[smoothing sensitivity](METHODS.md#transition-detection),
[external provenance](USER_GUIDE.md#plot-save-and-record-the-analysis).

Explicit AC-frequency requests must be finite positive real scalars and match exactly
one recorded frequency cluster within 0.1% of its recorded representative, with no
absolute tolerance. Missing or ambiguous matches raise; available frequencies are listed.
Detection selects per measurement; maps pool the campaign and require each sweep to
contribute at the selected frequency. Clustering and row-selection rules are unchanged.

## FT-1 — iminuit/HESSE canonical

Fit through `iminuit.Minuit` / `iminuit.cost.LeastSquares`, run MIGRAD then explicit HESSE,
and report the canonical symmetric parameter uncertainties in `FitResult.uncertainties`.
Optional MINOS bands are separate `minos_errors`, used for display. Document supplied-error
versus residual-scaled weighting and retain `yerr_assumed` and covariance information.

Validate finite aligned one-dimensional x/y and finite positive scalar/aligned yerr before
minimisation (`FitInputError`). Valid inputs that cannot support a numerical fit return
`success=False`, including `n_points <= n_free`, rather than raising. Report the true
`dof = n_points - n_free`, including non-positive values on failure. Keep invalid caller
configuration distinct from an unsuccessful numerical fit.

Unexpected model/backend exceptions propagate, including during optional MINOS.
ArithmeticError and numpy.linalg.LinAlgError are numerical failures: during minimisation
they return `success=False`; during MINOS they leave the HESSE result available with
`minos_errors=None`.

**Rationale.** Uncertainty meanings and unsuccessful outcomes must be predictable across models.

**Reference.** `fitting/engine.py`;
[fit-error conventions](METHODS.md#fitting),
[fitting API](api/calamari.fitting.rst).

## PL-1 — matplotlib is the sole plotting backend

Measurement drawing uses matplotlib, taking the measurement/family first and optional
`ax=` for composition; create a figure only when axes are omitted. Leave layout,
colourbars and `legend()` to callers and pass `**style` to the underlying drawing call.
Separate measurement/analysis choices from presentation; drawing a result must not silently
repeat detection or change the data.

`plot.save` saves the whole figure from axes or a figure, writes sibling formats from one
stem and validates all formats before writing. It adds no metadata. Publication defaults
(vector format, 600 dpi, TrueType fonts) belong to `style`'s rcParams, and figure suitability
requires inspection of the saved result.

**Rationale.** One backend with composable axes keeps scientific analysis independent of
presentation and supports repeatable multi-panel figures.

**Reference.** `plot/`, `style.py`; [plotting](USER_GUIDE.md#plot-save-and-record-the-analysis), [styles](api/calamari.style.rst),
[saving](USER_GUIDE.md#plot-save-and-record-the-analysis). Earlier GLE workflows require `calamari<0.5`;
migration details remain in the changelog and Git history.

## RP-1 — repeats are detected silently, combined only on request

Expose instrument-counter repeats through `Measurement.max_repeats`, repr and summary;
combine only on an explicit `aggregate_repeats` call. Group consecutively within each
measurement family from `row_families`, never across families or separated sweep branches.
Plan each family independently (tolerances, counter/setpoint choice and separation check)
and record the mechanisms in `grouped_by_family`.

Verify every proposed group against its setpoint columns (`RepeatGroupError` on mismatch).
For tolerance grouping, the smallest consecutive step between groups must be at least ten
times the largest within-group step, in tolerance-normalised units with a maximum across
keys. Compare boundaries within each family, including interleaved rows. Do not mistake a
slow ramp or misnumbered counter for repeated setpoints.

The default combined uncertainty is Birge-scaled, `sigma_int * max(1, sqrt(chi2/nu))`;
`internal` and `external` are explicit alternatives. Document the upward bias at small
repeat counts. Mixed zero/non-zero standard errors cannot be inverse-variance weighted
(`RepeatWeightError`); callers may explicitly request external scatter. Do not silently
reject repeat members.

**Rationale.** Aggregation changes observations and uncertainties, and must be a recorded
scientific choice rather than an implicit prerequisite for plotting or fitting.

**Reference.** `repeats.py`, `Measurement.max_repeats`;
[repeat aggregation](METHODS.md#loading-and-preparation), [uncertainties](METHODS.md#loading-and-preparation),
[grouping guards](METHODS.md#loading-and-preparation), [API](api/calamari.repeats.rst).

## PK-1 — plotting confined to the presentation modules

Direct imports of plotting libraries are confined to `calamari.plot` and its submodules,
plus `calamari.style`. Matplotlib is the sole backend. Analysis modules import neither
presentation module and contain no drawing logic. Package-root re-exports are allowed;
tests/examples may use matplotlib.
`_plot_data` prepares numerical series/grids without importing a plotting backend and
retains its public exports through `calamari.plot`. `style` handles styling, geometry and
panel lettering without measurement analysis. Matplotlib remains a required dependency.

**Rationale.** Headless numerical workflows must remain independent of rendering choices.

**Reference.** `tests/test_plot_architecture.py`, `pyproject.toml`;
[plotting](USER_GUIDE.md#plot-save-and-record-the-analysis).

## PK-2 — synthetic-data-only tests

Commit no real instrument data. Generate test files with `simulate` in temporary
directories; use invented array values and explicit seeded generators where needed.
Authorised real data remains external, inspected locally in place for evidence that
synthetic tests cannot establish. Keep identifying information and local research paths
out of maintained content and shared diagnostics.

**Rationale.** A checkout must be runnable without proprietary inputs while preserving the
scientific distinction between real evidence and synthetic implementation checks.

**Reference.** `simulate.py`, `tests/conftest.py`; the research-data policy in `AGENTS.md`.

## PK-3 — SCM-derived software identity

Build versions with setuptools-scm from complete Git history and tags. Import
`calamari.__version__` from the generated, Git-ignored `calamari._version` module;
installed distribution metadata supplies the same identity to provenance and Sphinx.
No runtime SCM dependency or hard-coded release fallback is permitted.

Untagged and dirty checkouts report development versions. Built source distributions
preserve their version without Git; tagged Git archives use substituted archival metadata.
Reject shallow history and raw copies without identifying metadata. Reinstall after
changing revisions to refresh generated and installed versions.

`scripts/check-versioning.py` exercises these states in disposable repositories without
changing real tags. Real tags and publication require explicit maintainer approval.

**Rationale.** Development and built artifacts must report an honest, consistent identity.

**Reference.** `pyproject.toml`, `calamari.__init__`;
[release verification](RELEASING.md).
