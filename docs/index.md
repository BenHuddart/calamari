# Calamari

**Python tools for analysing SQUID magnetometry data.**

```{note}
This is pre-release documentation. The source repository is currently private;
public installation instructions will accompany the first public release.
```

Calamari loads DC, VSM and AC measurements exported by Quantum Design MPMS 3 SQUID
magnetometers. It supports cataloguing and selecting measurements, unit conversion and
normalisation, repeat aggregation, model fitting and transition-candidate detection.
Plotting tools present measurements and analysis results using matplotlib, with composable
figures and journal styles.

Start with the [user guide](USER_GUIDE.md). The [API reference](api.md) documents the
available operations and their [method conventions](METHODS.md).

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
