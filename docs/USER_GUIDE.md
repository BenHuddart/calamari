# User guide

Calamari provides Python tools for analysing SQUID magnetometry data. It supports loading
Quantum Design MPMS 3 SQUID magnetometer exports containing DC, VSM and AC measurements.
Legacy MPMS/MPMS-XL exports are unsupported.

The examples below use your own measurement files. Replace filenames, selection ranges
and specimen facts with those for your analysis. The [API reference](api.md) documents each
operation; [method conventions](METHODS.md) define quantities, assumptions and uncertainties.

## Install and load

Calamari is being prepared for its first stable release. Install the development version
from a complete Git checkout with Python 3.10–3.13:

```bash
git clone https://github.com/BenHuddart/calamari.git
cd calamari
python -m pip install .
```

Include the checkout's Git history and tags so the build can determine its version.
Untagged checkouts report a development version. Install again after changing revisions
to refresh the recorded version. GitHub source downloads are supported only for release
tags.

Load an export and inspect its contents before choosing a signal:

```python
import calamari

measurement = calamari.load("measurement.dat")
print(measurement.summary())
print(measurement.measurement_types)
print(measurement.moment_columns)
```

`load` recognises content rather than extensions. An unsupported file raises
`UnsupportedFileError`; an unregistered `fmt=` raises `UnknownFormatError`. Recognised
malformed files raise `FileParseError`, while filesystem failures retain their `OSError`.
The original table and metadata remain accessible as `measurement.data` and `.header`.
Absent quantities return `None`; missing entries in a recorded column can remain `NaN`.

A combined export can hold several measurement families. `of_type("DC")` selects DC rows;
`split_by_type()` returns the available families. Select a moment explicitly when several
columns are populated. This DC example uses the free-centre moment:

```python
measurement = measurement.of_type("DC").with_moment("DC Moment Free Ctr")
```

Use a column reported by `moment_columns` for your file. AC data exposes `chi_real` and
`chi_imag`; choose the frequency explicitly for operations on multi-frequency exports.
An application label alone does not establish which measurements a file contains.

## Catalogue and select runs

For a directory, build a catalogue and inspect its rows before loading a selection:

```python
catalogue = calamari.build_catalog(".")
print(catalogue.df[["relative_path", "measurement_type", "scan_kind"]])
dc_runs = catalogue.of_type("DC")
print(dc_runs.paths)
```

A catalogue selects whole files. Selecting an AC-containing file does not remove its DC
rows; select row families after loading when needed. Inspect `parse_errors` as well as the
successful rows. `path_filter=` can restrict discovery before files are read.

`Catalog.at` selects held temperature/field setpoints; `select_representative` chooses one
file per setpoint under explicit rules. Content deduplication, selecting a representative
run and aggregating repeated measurements are separate operations. Inspect each choice
rather than treating repeated filenames or similar ranges as duplicates.

## Supply sample facts and choose units

Use your weighing record and composition to supply normalisation facts. The numerical
values below illustrate the arguments; replace all three with your own values:

```python
sample = calamari.Sample.from_measurement(
    measurement, mass_mg=8.0, molar_mass_g_per_mol=180.0, magnetic_ions_per_fu=2.0,
)
print(sample.mass_source)
moment_per_ion = sample.moment_emu_to_muB_per_ion(measurement.moment_emu)
```

An explicit `mass_mg` overrides a file-supplied mass and records `mass_source="user"`.
Without an override, `from_measurement` can adopt the header mass; check it before use.
Molar mass and ion count must be supplied explicitly for conversions that need them.
Missing required facts raise `SampleValidationError` rather than introducing defaults.

| Quantity | Accessor or conversion | Unit/basis |
| --- | --- | --- |
| Temperature | `measurement.temperature` | K |
| Applied field | `field_oe`, `field_tesla` | Oe, tesla (applied {math}`\mu_0 H`) |
| Moment | `moment_emu` | emu, before sample normalisation |
| AC susceptibility | `chi_real`, `chi_imag` | emu/Oe for the whole specimen |
| Moment per ion | `sample.moment_emu_to_muB_per_ion` | Bohr magnetons per magnetic ion |
| Molar susceptibility | `sample.chi_to_chi_molar` | emu/(mol Oe), per mole of formula units |

Choose the normalisation that matches your comparison: per formula unit, per magnetic ion
or per unit mass. See
[units and specimen facts](METHODS.md#units-and-specimen-facts) for their conventions.

## Prepare the selected measurements

Restrict a sweep to a declared range and inspect its quality report:

```python
selected = measurement.restrict(t_range=(15.0, 45.0))
quality = selected.quality()
print(quality)
```

These ranges are analysis choices. `restrict` and `with_moment` return new objects;
Calamari operations do not mutate the input measurement. Exposed DataFrames are not deeply
immutable. A quality flag is a diagnostic to inspect; it does not justify automatic rejection.

For instrument-counter repeats, inspect `max_repeats` and use `aggregate_repeats` only when
combining them is appropriate. Its default uncertainty is Birge-scaled; available weighting
and refusal rules are documented in the [API](api/calamari.repeats.rst). Record the grouping
and error convention. Do not combine rows across measurement families or sweep branches.
Use `split_isotherms` for stacked isotherms and explicit branch selection where needed.

## Fit or assess transition candidates

### Fit a model

`fit` accepts aligned x/y arrays and optional standard errors. This call illustrates a
linear model over the selected interval; choose a model and range appropriate to your data:

```python
fit = calamari.fit(
    "linear", selected.temperature, selected.moment_emu, yerr=selected.moment_err_emu,
)
print(fit.summary())
```

Inspect `success` before using `parameters`, `uncertainties` or `covariance`. Supply
`initial=`, `fixed=` and `limits=` by parameter name when needed. `fit_windows` compares
specified fitting ranges, and `derive` propagates the full fit covariance into a scalar
quantity. Keep supplied-error and residual-scaled uncertainties distinct. The
[model and error reference](METHODS.md#fitting) lists the built-in models and conventions.

### Detect and inspect candidates

Choose the searched quantity, criterion, range and smoothing scale. This separate file is
assumed to contain a DC temperature sweep; the example requests signal maxima/minima:

```python
transition = calamari.load("transition.dat").of_type("DC")
transition = transition.with_moment("DC Moment Free Ctr")
candidates = calamari.detect_transitions(
    transition, signal="moment_emu", methods="peak", smooth_window=2.0, noise="scatter",
)
print(candidates.candidates)
print(candidates.params)
```

Here `smooth_window` is in kelvin. Calamari uses Savitzky–Golay smoothing
{footcite:p}`savitzky1964`; `noise="scatter"` estimates noise from the data.
Choose both from the sampling, feature width and error information for your analysis. Inspect
warnings, the processed curve and raw detections alongside the candidates. The
[detection reference](METHODS.md#transition-detection) distinguishes `peak`, `derivative`,
`kink`, `onset` and `fisher`; they locate different features.

`robust()` applies recorded diagnostic filters. `select()` requires an explicit criterion,
sign and swept-axis window when selecting a point per sweep. Batch detection records
per-sweep refusals in `skipped`; invalid shared settings raise. Retain these failures in
your analysis record. Smoothing scans, linking, reconciliation, response maps and fitting
selected boundaries are available through the API. Choose their settings for your analysis and
assess whether the selected points represent
a physical phase boundary.

## Plot, save and record the analysis

Plotting presents the selected results and accepts an existing matplotlib `ax=` for
composition. Styling is optional; this example sets a journal-sized single-column figure:

```python
from calamari import style

with style.context("jpcm", width="single"):
    ax = calamari.plot.plot_mt(selected, units="emu", label="Selected measurement")
    ax.legend()
    calamari.plot.save(ax, "analysis", formats=("pdf", "png"))
```

Inspect the saved figure at its intended size: labels, units, normalisation, legends and
uncertainty displays must be readable and unclipped. Plotting does not choose an
interpretation or rerun detection. `plot_fit` and `plot_detection_curve` display existing
analysis results; family and map functions document their selection requirements in the API.

Keep the input selection, chosen columns, sample facts, exclusions, aggregation, model,
weighting, windows and thresholds with the script. Save resolved result settings and your
Python/dependency environment, and record `calamari.__version__` or the source revision.
Result provenance supports this record but does not replace the external inputs and recipe.
Cite the archived release actually used; see the repository's `CITATION.cff`.

## References

```{footbibliography}
```
