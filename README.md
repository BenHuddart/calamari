<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://benhuddart.github.io/calamari/_static/calamari-logo-white.svg">
    <img src="https://benhuddart.github.io/calamari/_static/favicon.svg" alt="Calamari logo" width="96">
  </picture>
</p>

# Calamari

**Python tools for analysing SQUID magnetometry data.**

[![Zenodo DOI][zenodo-badge]][release-doi]

Calamari loads DC, VSM and AC measurements exported by Quantum Design MPMS 3 SQUID
magnetometers. It supports cataloguing and selecting measurements, unit conversion and
normalisation, repeat aggregation, model fitting and transition-candidate detection.
Plotting tools present measurements and analysis results using matplotlib, with composable
figures and journal styles.

[Documentation](https://benhuddart.github.io/calamari/) ·
[Releases](https://github.com/BenHuddart/calamari/releases) ·
[Issues](https://github.com/BenHuddart/calamari/issues)

## What you can do

- Load measurements and catalogue a directory of runs, selecting the files for your analysis.
- Convert moments and susceptibilities using explicit specimen facts and units.
- Fit Curie–Weiss and magnetisation models, inspect uncertainties and compare fitting ranges.
- Assess transition candidates and assemble selected points into H–T phase diagrams.
- Combine repeat measurements and export figures with labelled quantities and journal styles.

## Installation

Calamari is tested with Python 3.10–3.13. Install from PyPI:

```bash
python -m pip install calamari-magnetometry
```

For development, use a complete Git checkout with its tags; see
[Contributing](https://github.com/BenHuddart/calamari/blob/main/CONTRIBUTING.md).
GitHub source downloads are supported only for release tags.

## Getting started

Load an MPMS 3 export and inspect its measurement types, ranges and recorded metadata:

```python
import calamari

measurement = calamari.load("measurement.dat")
print(measurement.summary())
```

Follow the [user guide](https://benhuddart.github.io/calamari/USER_GUIDE.html)
for loading, selection, sample facts, analysis and plotting. The
[API reference](https://benhuddart.github.io/calamari/api.html) documents
individual operations and their units, assumptions and uncertainty conventions.

## Development disclosure

Generative AI tools, including OpenAI Codex and Anthropic's Claude Code, were used
extensively to generate and revise code, tests, documentation and development tooling,
and to assist with code review. Benjamin M. Huddart directed the scientific requirements
and development decisions and retains responsibility for the released software.
Validation includes automated tests, independent numerical reference checks and
continuous integration.

## Citation and support

If you use Calamari 0.5.0 in research, cite [the archived release][release-doi] and record
the software version or commit with your analysis. Citation metadata are available in
[CITATION.cff](https://github.com/BenHuddart/calamari/blob/main/CITATION.cff).
For other versions, cite the corresponding archive.

[Report a problem or request a feature](https://github.com/BenHuddart/calamari/issues).
Calamari is available under the [MIT licence](https://github.com/BenHuddart/calamari/blob/main/LICENSE).

Contributions are welcome; see [Contributing](https://github.com/BenHuddart/calamari/blob/main/CONTRIBUTING.md).

Calamari is an independent project and is not affiliated with or endorsed by Quantum Design.

[zenodo-badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.24170118.svg
[release-doi]: https://doi.org/10.5281/zenodo.24170118
