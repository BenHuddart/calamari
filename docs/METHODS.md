# Quantities and method conventions

This reference explains the quantities, assumptions and uncertainty conventions used by
Calamari. The API pages describe accepted inputs, parameters and result fields. Choose
methods that suit your measurements and the question you want to answer.

## Units and specimen facts

Calamari's canonical quantities are temperature in K, field in Oe, moment in emu and
susceptibility in emu/Oe. Named accessors resolve documented column units: for example,
{math}`1\,\mathrm{A\,m^2} = 10^3\,\mathrm{emu}` and
{math}`1\,\mathrm{T} = 10^4\,\mathrm{Oe}` for the applied-field representation.
`field_tesla` represents the applied {math}`\mu_0 H`. Calamari does not apply a
demagnetisation correction or calculate the internal field or flux density.

Recorded susceptibility refers to the whole specimen. `Sample` conversions use
explicit mass, molar mass and, where applicable, magnetic ions per formula unit:

| Output | Basis | Units |
| --- | --- | --- |
| `chi_to_chi_mass` | Per gram | {math}`\mathrm{emu\,g^{-1}\,Oe^{-1}}` |
| `chi_to_chi_molar` | Per mole of formula units | {math}`\mathrm{emu\,mol^{-1}\,Oe^{-1}}` |
| `chi_to_chi_molar_per_ion` | Per mole of magnetic ions | {math}`\mathrm{emu\,mol^{-1}\,Oe^{-1}}` |
| `moment_emu_to_muB_per_fu` | Per formula unit | {math}`\mu_{\mathrm{B}}`/f.u. |
| `moment_emu_to_muB_per_ion` | Per magnetic ion | {math}`\mu_{\mathrm{B}}`/ion |

Required normalisation facts are finite positive real scalars. Fractional
`magnetic_ions_per_fu` can describe average occupancy; the caller defines the counted ions.
Calamari does not separate magnetic species. A missing fact may remain `None` until needed.
`Sample` is frozen. A directly constructed sample has no recorded mass source;
`from_measurement` records `file` or `user`, without independently validating the mass.
Molar mass is not adopted automatically from a header.

CGS molar susceptibility in {math}`\mathrm{emu\,mol^{-1}\,Oe^{-1}}` converts to
SI {math}`\mathrm{m^3\,mol^{-1}}` by multiplying by {math}`4\pi\times10^{-6}`.
Calamari's named conversions do not implicitly introduce SI volume normalisation.
Constants and their provenance are documented in {py:mod}`calamari.constants`.

{py:func}`calamari.sample.mu_eff_from_curie_constant` returns the number of Bohr magnetons
from a molar CGS Curie constant. Its defining relation is

```{math}
\frac{\mu_{\mathrm{eff}}}{\mu_{\mathrm{B}}}
= \frac{1}{\mu_{\mathrm{B}}}\sqrt{\frac{3 k_{\mathrm{B}} C}{N_{\mathrm{A}}}}.
```

The mole basis follows that of {math}`C`. A fitted inverse-susceptibility slope is
{math}`1/C`;
`curie_constant_from_inverse_slope` performs this inversion. Effective and saturation
moments are distinct quantities.

## Loading and preparation

The public loader supports MPMS 3 DC, VSM and AC exports, including mixed-family files.
Column titles and populated quantities determine available channels. Recorded AC components
are read as susceptibility, without deriving them from legacy AC moment columns.
Transport actions and populated data identify measurement rows; a header application label
can describe a different family. Ambiguous moment choices require explicit selection.
Rows without measurements and missing values are not interchangeable with zero signals.

`date_epoch` preserves the numeric header value. Inspected native exports empirically
support seconds from 1899-12-30; timezone, clock handling and applicability to all versions
remain unspecified. `date_iso` uses the readable creation-date token, which need not date
all subsequently appended measurements. No timezone is inferred.

Raw status codes remain in `Measurement.data`. Non-zero or unusual values are not automatic
rejection rules. Quality/continuity scores and working thresholds are diagnostics, not
calibrated instrument errors or probabilities. Status departures compare against the modal
code per measurement family, pooling AC frequencies. Complete SQUID/motor status decoders
are not established. See {py:mod}`calamari.quality` for flags and policies.

`aggregate_repeats` combines observations only on request, within consecutive groups of a
single measurement family. Default errors are Birge-scaled:

```{math}
\sigma_{\mathrm{combined}} = \sigma_{\mathrm{internal}}
\max\left(1,\sqrt{\chi^2/\nu}\right).
```

Small groups can have upward-biased scaled errors. `internal` and `external` are explicit
alternatives; mixed zero/non-zero errors cannot be inverse-variance weighted. Grouping
validates setpoint agreement and separation; see {py:mod}`calamari.repeats` for refusal rules.
Caller-supplied screening, branch selection and aggregation remain recorded analysis choices.

## Fitting

The six built-in models are available by registry name or through a `ModelDefinition`.
The model functions document parameter units and domains:

| Model | Definition / interpretation |
| --- | --- |
| `linear` | {math}`y=ax+b` |
| `curie_weiss` | {math}`\chi_0+C/(T-\theta)` |
| `inverse_curie_weiss` | {math}`(T-\theta)/C`; no constant susceptibility background |
| `brillouin` | {math}`M_{\mathrm{s}}B_J(z)` |
| `segmented_linear` | Continuous intersection of two straight arms |
| `bent_cable` | Straight arms joined by a symmetric parabolic bend |

For Curie–Weiss models, {math}`T` is in kelvin and {math}`C` in susceptibility × kelvin.
The Brillouin argument is

```{math}
z = \frac{gJ\mu_{\mathrm{B}}H}{k_{\mathrm{B}}T},
```

with {math}`H` in oersted and {math}`T` in kelvin, using CGS constants.

For Curie–Weiss fits choose a justified paramagnetic range away from the pole/ordering
region. A fitted constant {math}`\chi_0` is an empirical background, not automatically a diamagnetic
correction. Transforming {math}`\chi` to {math}`1/\chi` changes residuals and weighting; the two
models are not
interchangeable fit recipes. First-order error propagation gives
{math}`\sigma_{1/\chi}\simeq\sigma_{\chi}/\chi^2` and can
fail near zero. `mu_eff_from_curie_constant` requires C in the documented molar CGS basis.

The Brillouin model assumes ideal independent local moments. Supply justified T/g/J
constraints; one curve cannot generally identify them independently. {math}`M_{\mathrm{s}}` has the
units of
the fitted y values and is not automatically a per-ion moment. It approaches
{math}`gJ\mu_{\mathrm{B}}` per ion
only under the corresponding physical and normalisation assumptions. Fit convergence
does not identify magnetic order or establish model adequacy.

{py:func}`calamari.fitting.engine.fit` validates finite aligned 1D x/y inputs and positive
finite scalar/aligned `yerr`, then uses iminuit MIGRAD and HESSE. Supplied standard errors
are treated as absolute. Without `yerr`, constant unknown scatter is estimated from the
residuals and parameter uncertainties are scaled accordingly (`yerr_assumed=True`);
`chi_squared` then reports the raw residual sum of squares. Systematics, x errors and
correlated noise are not automatically included.

`FitResult.uncertainties` contains symmetric HESSE errors. Optional MINOS intervals are
separate `minos_errors`. `derive` uses the full covariance in first-order propagation;
strong nonlinearities can invalidate that approximation. Sample-fact uncertainty is separate
from fit covariance. Invalid inputs/configuration raise; a valid but unsuccessful or
underdetermined fit returns `success=False`. Do not use its parameters as an estimate.

`fit_windows`, `fit_breakpoint` and `fit_boundary` document their selection, weighting,
sensitivity and unsuccessful-result records. Overlapping windows are correlated; stability
across them is a diagnostic, not a coverage guarantee. A fit through selected candidate
points inherits the uncertainty and criterion assumptions of those points.

Unexpected model/backend exceptions propagate, including during optional MINOS.
ArithmeticError and numpy.linalg.LinAlgError are numerical failures: during minimisation
they return `success=False`; during MINOS they leave the HESSE result available with
`minos_errors=None`.

## Transition detection

`find_transitions` works on arrays, `detect_transitions` on one measurement and
`detect_transitions_all` on a campaign. Measurement APIs resolve the swept/held coordinates
and record settings. Split mixed/static or stacked/branched inputs appropriately; choose
AC frequency and moment source explicitly when ambiguous. Shared invalid settings raise
before batch processing; genuine per-sweep failures remain inspectable in `skipped`.

| Method | Located feature | Principal limitation |
| --- | --- | --- |
| `peak` | Local maximum/minimum of the smoothed signal | A response extremum need not locate a transition |
| `derivative` | Local extremum of the first derivative | Often locates a step midpoint; depends on smoothing |
| `kink` | Local extremum of curvature | Rounding and smoothing affect a slope-change coordinate |
| `onset` | Intersection of a baseline and local tangent | Requires the assumed baseline/step geometry |
| `fisher` | Extremum of {math}`\mathrm{d}(yT)/\mathrm{d}T` | Requires an appropriate susceptibility interpretation |

Fisher's relation concerns simple-antiferromagnet susceptibility under zero-field and
orientation assumptions {footcite:p}`fisher1962`. Finite-field, powder and finite-frequency
applications
need their own justification. The method requires positive absolute temperatures, refuses
field scans and imaginary susceptibility, and permits moment searches only away from the
zero-held-field tolerance, where the caller must justify linear response. A negative held
field reverses derivative sign. Fisher hits are kept separate from the other grouped
criteria. Recorded AC {math}`\chi^\prime`,
{math}`\chi^{\prime\prime}` and DC {math}`M/H` describe different responses.

Smoothing/differentiation uses local polynomial processing; window widths are in the swept
axis units. Sampling, smoothing and noise choices affect locations and scores. See the
function defaults and resolved `params`; no single window or threshold is universal.
Smoothing uses Savitzky–Golay filtering {footcite:p}`savitzky1964`.
`TransitionResult.provenance()` lists the references for each detection criterion.

`significance` is a normalised diagnostic score, not a calibrated probability or Gaussian
z-score. `robust()` defaults to `min_significance=5`; larger scores do not establish a phase
transition.

Extremum location errors propagate the representation's noise through a local quadratic
fit. Onset errors propagate the fitted baseline/tangent intersection. Neither is reported below the
grid spacing. These estimates are conditional on the chosen smoothing, noise and
local model; they do not fully account for smoothing-induced correlations, systematics
or criterion-choice uncertainty. Inspect the processed curves,
raw `detections`, grouped `candidates` and recipe before selecting points. Result tables
and curve keys are specified in the API; `sweep_id`, not a filename, identifies split sweeps.

`select()` keeps at most one candidate per sweep using an explicit criterion/sign/window;
score ties go to lower location. `link()` describes chains; `reconcile()` describes agreement
across selected sweep axes without extrapolation or implicit tolerances. Neither establishes
a physical boundary. `scan_smoothing` records each declared-window result; correlated
windows are not independent repeats or scores to combine. Reference sensitivity gates are
opt-in and preserve the declared reference coordinates. `all()` restores candidates while
retaining links/reference descriptions and removing dependent selections/reconciliation.
See {py:mod}`calamari.transitions` and {py:mod}`calamari.scales` for transformation order,
empty/failed sweeps, uncertainty meanings and retained provenance.

Explicit AC-frequency requests must be finite positive real scalars and match exactly
one recorded frequency cluster within 0.1% of its recorded representative, with no
absolute tolerance. Missing or ambiguous matches raise; available frequencies are listed.
Detection selects per measurement; maps pool the campaign and require each sweep to
contribute at the selected frequency. Clustering and row-selection rules are unchanged.

## Software and callable provenance

Result records use the installed distribution version. A missing installation is recorded
as `unknown`; unexpected metadata failures propagate. Callable settings use qualified
function/method names, callable-type names or `partial(function)` descriptions. These
stable labels do not serialise bound arguments, instance state or source code and do not
uniquely identify configured instances. Retain callable source/configuration with your
recipe, alongside the resolved numerical settings recorded by each operation.

## References

```{footbibliography}
```
