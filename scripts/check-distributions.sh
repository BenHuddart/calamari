#!/usr/bin/env bash
# Exercise the shipped sources against a wheel rebuilt from that source archive.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
uv build --sdist --out-dir "$work/dist" "$root"
tar -xzf "$work"/dist/*.tar.gz -C "$work"
source_dir=$(find "$work" -maxdepth 1 -type d -name 'calamari_magnetometry-*')
uv build --wheel --out-dir "$work/wheels" "$source_dir"
uv tool run --from 'twine>=6' twine check --strict "$work"/dist/*.tar.gz "$work"/wheels/*.whl
uv venv "$work/env"
uv pip install --python "$work/env/bin/python" "$work"/wheels/*.whl \
    --group "$source_dir/pyproject.toml:dev" --group "$source_dir/pyproject.toml:docs"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$work/matplotlib"
unset PYTHONPATH
cd "$source_dir"
package_dir=$("$work/env/bin/python" -I - <<'PY'
import sys
from importlib.metadata import version
from pathlib import Path

import calamari

package = Path(calamari.__file__).parent
assert package.is_relative_to(Path(sys.prefix)), package
assert calamari.__version__ == version("calamari-magnetometry")
from calamari._provenance import calamari_version
assert calamari_version() == calamari.__version__
print(package)
PY
)
"$work/env/bin/python" -m pytest tests/
"$work/env/bin/python" -m pytest --doctest-plus --doctest-modules "$package_dir"
"$work/env/bin/sphinx-build" -E -a \
    -n -W --keep-going -b html docs "$work/html"
