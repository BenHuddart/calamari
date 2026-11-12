"""Run the maintained entry-point example as a user would from an empty directory."""

import os
import subprocess
import sys
from pathlib import Path

from _guide_runner import parse_fences


def test_readme_getting_started(tmp_path):
    readme = (Path(__file__).resolve().parents[1] / 'README.md').read_text()
    section = readme.split('## Getting started\n', 1)[1].split('\n## ', 1)[0]
    blocks = [f.body for f in parse_fences(section) if f.is_python]
    assert blocks, 'Getting started has no runnable Python'
    script = tmp_path / 'getting_started.py'
    script.write_text('''
from calamari import simulate

simulate.write_mpms3_file(
    "measurement.dat", app="SQUID DC", sample_material="SyntheticSpecimen",
    temperature_K=[8.0, 8.0, 8.0], field_oe=[-250.0, 0.0, 250.0],
    dc_moment_emu=[-0.003, 0.0, 0.003],
)
''' + '\n'.join(blocks) + '''
summary = measurement.summary()
assert summary["measurement_type"] == "DC"
assert summary["sample_name"] == "SyntheticSpecimen"
assert summary["n_measurements"] == 3
assert summary["T_min"] == summary["T_max"] == 8.0
assert summary["H_min_Oe"] == -250.0
assert summary["H_max_Oe"] == 250.0
''')
    env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(tmp_path / 'matplotlib'))
    env.pop('PYTHONPATH', None)
    subprocess.run([sys.executable, '-I', str(script)], cwd=tmp_path,
                   env=env, check=True, timeout=120)
