# API reference

Use the task table below to find functions, result types and their input requirements.
Each entry documents accepted inputs, units, assumptions and return values. Common
entry points are also available from the top-level `calamari` namespace.

```{toctree}
:maxdepth: 1

METHODS
```

| Task | Modules |
| --- | --- |
| Load and inspect | {py:mod}`io <calamari.io>`, {py:mod}`measurement <calamari.measurement>` |
| Supply facts and convert units | {py:mod}`sample <calamari.sample>`, {py:mod}`constants <calamari.constants>` |
| Catalogue, select and prepare | {py:mod}`catalog <calamari.catalog>`, {py:mod}`sweeps <calamari.sweeps>`, {py:mod}`quality <calamari.quality>`, {py:mod}`repeats <calamari.repeats>` |
| Fit models and selected boundaries | {py:mod}`fitting <calamari.fitting>`, {py:mod}`boundary <calamari.boundary>` |
| Detect and assess candidates | {py:mod}`transitions <calamari.transitions>`, {py:mod}`scales <calamari.scales>` |
| Plot and export | {py:mod}`plot <calamari.plot>`, {py:mod}`style <calamari.style>` |

```{eval-rst}
.. autosummary::
   :toctree: api
   :recursive:

   calamari
```

To add a custom input format, see [Registering a loader](EXTENDING.md)
in the contributor documentation.
