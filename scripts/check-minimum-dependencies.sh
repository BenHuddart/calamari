#!/usr/bin/env bash
# Resolve each runtime dependency at its declared floor, with current compatible tooling.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
uv venv --python 3.10 "$work/env"
uv pip install --python "$work/env/bin/python" "$root" --group "$root/pyproject.toml:dev"
"$work/env/bin/python" - <<'PY' > "$work/minimum.txt"
from importlib.metadata import requires

from packaging.requirements import Requirement

for value in requires("calamari-magnetometry"):
    requirement = Requirement(value)
    # More complex bounds need an explicit decision about the tested minimum.
    bounds = list(requirement.specifier)
    assert len(bounds) == 1 and bounds[0].operator == ">=", value
    assert not requirement.marker and not requirement.extras, value
    print(f"{requirement.name}=={bounds[0].version}")
PY
uv pip install --python "$work/env/bin/python" -r "$work/minimum.txt"
uv pip check --python "$work/env/bin/python"
uv pip freeze --python "$work/env/bin/python"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$work/matplotlib"
unset PYTHONPATH
cd "$root"
"$work/env/bin/python" -m pytest tests/
package_dir=$("$work/env/bin/python" -I -c 'import calamari; print(calamari.__path__[0])')
"$work/env/bin/python" -m pytest --doctest-plus --doctest-modules "$package_dir"
