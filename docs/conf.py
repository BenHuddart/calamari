"""Sphinx configuration for the calamari documentation site.

The build is deliberately strict: ``nitpicky`` here plus ``-W`` in CI
means every unresolvable cross-reference or malformed docstring fails
the build rather than degrading the site silently. Loosen a check only
by adding a targeted entry to ``nitpick_ignore_regex`` with a comment
saying why the reference cannot resolve.
"""

import sys
import xml.etree.ElementTree as ET
from importlib.metadata import version as _dist_version
from pathlib import Path

project = "Calamari"
author = "Ben Huddart"
copyright = "%Y, Ben Huddart"  # noqa: A001 -- Sphinx expects this name
# Read the built SCM version from installed metadata (PK-3).
release = _dist_version("calamari-magnetometry")
version = release

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "aps"
bibtex_reference_style = "label"

extensions = [
    "sphinxcontrib.bibtex",
    "aps_references",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

# Maintained sources are site content; build products are excluded.
exclude_patterns = ["_build"]

# -- Docstrings ---------------------------------------------------------
# House style is NumPy-format docstrings only (AGENTS.md); Google-style
# parsing is disabled so a malformed section is a visible failure, not a
# silently different rendering.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# Shorthand used in the type position of Parameters/Returns sections,
# expanded to the importable name so the reference links. The calamari
# names map to their defining module because that is where autosummary
# documents them -- the top-level re-exports have no pages of their own.
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "ndarray": "numpy.ndarray",
    "Path": "pathlib.Path",
    "path": "pathlib.Path",
    "sequence": "collections.abc.Sequence",
    "mapping": "collections.abc.Mapping",
    "iterable": "collections.abc.Iterable",
    "callable": "collections.abc.Callable",
    "Measurement": "calamari.measurement.Measurement",
    "Sample": "calamari.sample.Sample",
    "Catalog": "calamari.catalog.Catalog",
    "FitResult": "calamari.fitting.engine.FitResult",
    "FitWindowsResult": "calamari.fitting.windows.FitWindowsResult",
    "ModelDefinition": "calamari.fitting.models.ModelDefinition",
}

autosummary_generate = True
autodoc_member_order = "bysource"
# No autodoc_typehints setting: sphinx_autodoc_typehints supplies the
# description-style rendering itself, and setting autodoc's own option
# bypasses the extension (and with it qualname_overrides below).

# Annotations resolve to runtime qualnames (pandas.core.frame.DataFrame),
# but the pandas inventory registers the public aliases; map to those.
qualname_overrides = {
    "pandas.core.frame.DataFrame": "pandas.DataFrame",
    "pandas.core.series.Series": "pandas.Series",
}

# -- Cross-references ---------------------------------------------------
nitpicky = True
nitpick_ignore_regex = [
    # Docstrings reference private helpers and module constants where the
    # rationale lives (house style); private objects get no pages, so the
    # names render as plain literals rather than links.
    (r"py:.*", r"(.*\.)?_\w+"),
    # Literal-choice sets in the type position ({'emu', 'muB_per_ion'}),
    # prose type descriptions ("dict of str to float"), tuple notation
    # ("(float, float)") and generics ("list[str]") are vocabulary, not
    # importable names. Real targets never contain quotes, spaces,
    # parentheses or brackets.
    (r"py:.*", r".*'.*"),
    (r"py:.*", r".* .*"),
    (r"py:.*", r".*[()\[\]].*"),
    # Informal type vocabulary with no importable target.
    ("py:class", r"array[-_]like"),
    ("py:class", r"path-like"),
    ("py:class", r"optional"),
    # numpy registers its scalar types as attributes, not classes, so the
    # py:class references generated from annotations cannot match them.
    ("py:class", r"numpy\.(float64|int64)"),
    # Private numpy typing internals leaked by ArrayLike expansion.
    (r"py:.*", r"numpy\._typing\..*"),
    # Older sphinx/typehints resolutions (a Python 3.10 environment pins
    # sphinx 8.1) render property and dataclass-attribute annotations as
    # the source's ``pd.`` shorthand or the runtime qualname, out of
    # qualname_overrides' reach; current versions (the docs CI job, RTD)
    # link them properly, so these only mask the old-version fallback.
    ("py:class", r"pd\.(Series|DataFrame)"),
    ("py:class", r"pandas\.core\..*"),
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "iminuit": ("https://scikit-hep.org/iminuit/", None),
}

# -- MyST ---------------------------------------------------------------
# The user guides deep-link to their own section headings; auto-generate
# anchors so those links survive the move onto the site.
myst_heading_anchors = 3

# -- HTML ---------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "Calamari"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["branding.css"]
html_logo = "_static/calamari-logo.svg"
html_theme_options = {
    "logo": {"text": "Calamari", "alt_text": "Calamari home"},
    "navbar_center": [],
    "navbar_end": ["theme-switcher"],
    "navbar_persistent": ["search-button"],
}
html_sidebars = {"**": ["global-nav.html"]}

# Keep the supplied SVG untouched. A generated white-backed favicon remains
# visible on either browser-tab theme; it is build output, not another source asset.
logo = ET.parse(Path(__file__).parent / html_logo)
svg = "http://www.w3.org/2000/svg"
ET.register_namespace("", svg)
backdrop = dict(zip(("x", "y", "width", "height"), logo.getroot().attrib["viewBox"].split()))
logo.getroot().insert(0, ET.Element(f"{{{svg}}}rect", {**backdrop, "fill": "white"}))
favicon = Path(__file__).parent / "_build" / "branding" / "favicon.svg"
favicon.parent.mkdir(parents=True, exist_ok=True)
logo.write(favicon, encoding="utf-8", xml_declaration=True)
html_favicon = str(favicon)
