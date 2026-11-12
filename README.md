<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://benhuddart.github.io/calamari/_static/calamari-logo-white.svg">
    <img src="https://benhuddart.github.io/calamari/_static/favicon.svg" alt="Calamari logo" width="96">
  </picture>
</p>

# Calamari

**Python tools for analysing SQUID magnetometry data.**

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

Calamari is being prepared for its first public release and is tested with Python 3.10–3.13.
The source repository is currently private. If you have access, install from a checkout:

```bash
git clone https://github.com/BenHuddart/calamari.git
cd calamari
python -m pip install .
```

Use a complete Git checkout with its tags. Builds derive the version from Git; untagged
checkouts report a development version. Install again after changing revisions to refresh
the recorded version. Use GitHub source downloads only for release tags.

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

If you use Calamari in research, use the citation metadata in [CITATION.cff](https://github.com/BenHuddart/calamari/blob/main/CITATION.cff)
and record the software version or commit with your analysis.

[Report a problem or request a feature](https://github.com/BenHuddart/calamari/issues).
Calamari is available under the [MIT licence](https://github.com/BenHuddart/calamari/blob/main/LICENSE).

Contributions are welcome; see [Contributing](https://github.com/BenHuddart/calamari/blob/main/CONTRIBUTING.md).

Calamari is an independent project and is not affiliated with or endorsed by Quantum Design.
