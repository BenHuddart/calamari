"""Physical constants and unit-conversion factors used across calamari.

All values are CODATA 2018 where applicable, each with a comment giving
its provenance. Magnetometry in this library follows the CGS-Gaussian
conventions native to Quantum Design instruments (moments in emu,
fields in Oe); conversions to SI or to "physics" units (Bohr magnetons,
Tesla) go through the factors defined here and nowhere else.
"""

from __future__ import annotations

#: Oe per Tesla (exact; 1 T = 1e4 Oe in CGS-Gaussian field units).
OE_PER_TESLA = 1.0e4

#: Bohr magneton in erg/G, which is numerically the magneton expressed
#: in emu (CODATA 2018: mu_B = 9.274 010 0783 e-21 erg/G).
MU_B_EMU = 9.2740100783e-21

#: Boltzmann constant in erg/K (exact since the 2019 SI redefinition:
#: k_B = 1.380 649 e-16 erg/K).
K_B_ERG_PER_K = 1.380649e-16

#: Avogadro constant in 1/mol (exact since the 2019 SI redefinition).
AVOGADRO = 6.02214076e23

#: emu -> Bohr magneton conversion factor (reciprocal of MU_B_EMU).
#: Older analyses often round this to 1.07828e20; the exact form here
#: differs from that only in the 5th significant figure.
MU_B_PER_EMU = 1.0 / MU_B_EMU

#: emu per A m^2 (exact; 1 emu = 1e-3 A m^2 for a magnetic moment, so
#: 1 A m^2 = 1e3 emu). Needed because Quantum Design instruments let the
#: operator record moments in either unit. The data-column suffix identifies
#: the recorded unit; accessors convert it to the canonical unit.
EMU_PER_A_M2 = 1.0e3

#: Susceptibility conversion: (emu/Oe) per (A m^2 / T). Derived from the
#: two factors above rather than written as a literal 0.1, so the
#: relationship stays visible: 1 A m^2 / T = 1e3 emu / 1e4 Oe = 0.1 emu/Oe.
EMU_PER_OE_PER_A_M2_PER_TESLA = EMU_PER_A_M2 / OE_PER_TESLA
