"""Small installed-package workflow and file fixtures, independent of teaching examples."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import calamari


def write_inputs(root):
    """Write invented DC inputs consumed by the reader-facing snippets."""
    root.mkdir(parents=True, exist_ok=True)
    temperature = np.linspace(10.0, 50.0, 81)
    for filename, moment in (
        ('measurement.dat', 1e-4 * temperature + 2e-3),
        ('transition.dat', 0.01 * np.exp(-((temperature - 30.0) / 3.0)**2)),
    ):
        calamari.simulate.write_mpms3_file(
            root / filename, app='SQUID DC', temperature_K=temperature,
            field_oe=np.full_like(temperature, 500.0), dc_moment_emu=moment,
            dc_moment_err_emu=np.full_like(temperature, 2e-5),
        )


def run_workflow(root):
    """Load, normalise, fit, detect and export through public entry points."""
    write_inputs(root)
    measurement = calamari.load(root / 'measurement.dat').with_moment('DC Moment Free Ctr')
    sample = calamari.Sample.from_measurement(
        measurement, mass_mg=8.0, molar_mass_g_per_mol=180.0, magnetic_ions_per_fu=2.0,
    )
    fit = calamari.fit('linear', measurement.temperature, measurement.moment_emu,
                       yerr=measurement.moment_err_emu)
    result = calamari.detect_transitions(
        calamari.load(root / 'transition.dat'), signal='moment_emu', methods='peak',
        smooth_window=2.0, noise=2e-5, outliers='ignore',
    )
    ax = calamari.plot.plot_mt(measurement, units='emu')
    calamari.plot.save(ax, root / 'measurement', formats=('pdf', 'png'))
    plt.close(ax.figure)
    return {
        'version': calamari.__version__,
        'provenance_version': result.params['calamari_version'],
        'mass_source': sample.mass_source,
        'normalised_moment': sample.moment_emu_to_muB_per_ion(0.003),
        'success': fit.success,
        'parameters': fit.parameters,
        'uncertainties': fit.uncertainties,
        'yerr_assumed': fit.yerr_assumed,
        'candidate_temperature_K': result.candidates['T_K'].tolist(),
    }


if __name__ == '__main__':
    print(json.dumps(run_workflow(Path(sys.argv[1]))))
