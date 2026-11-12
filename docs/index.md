# Calamari

**Python tools for analysing SQUID magnetometry data.**

Calamari loads DC, VSM and AC measurements exported by Quantum Design MPMS 3 SQUID
magnetometers. It supports cataloguing and selecting measurements, unit conversion and
normalisation, repeat aggregation, model fitting and transition-candidate detection.
Plotting tools present measurements and analysis results using matplotlib, with composable
figures and journal styles.

Start with the [user guide](USER_GUIDE.md). The [API reference](api.md) documents the
available operations and their [method conventions](METHODS.md).

[Release 0.5.0](https://github.com/BenHuddart/calamari/releases/tag/v0.5.0) is available
on [PyPI](https://pypi.org/project/calamari-magnetometry/) and archived on
[Zenodo](https://doi.org/10.5281/zenodo.24170118).

```{toctree}
:maxdepth: 1
:caption: Using Calamari

USER_GUIDE
api
```

```{toctree}
:maxdepth: 1
:caption: Maintainer documentation

contributing
EXTENDING
ARCHITECTURE
RELEASING
```
