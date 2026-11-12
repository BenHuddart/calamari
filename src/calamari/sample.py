"""Sample metadata and unit conversion between raw SQUID readings and physics units.

A :class:`Sample` bundles the physical-sample facts (mass, molar mass,
magnetic-ion count per formula unit, shape, orientation, ...) that a raw
moment reading (emu) needs to become a physically meaningful quantity
(Bohr magnetons per formula unit / per ion, molar or mass susceptibility).
Almost none of those facts can be inferred from a data file: the mass is
the sole exception, and only because the operator may have typed one into
the header. This module therefore never invents a molar mass, ion count,
or mass default of its own. Every conversion either has the fact it needs
or raises a :class:`ValueError` naming exactly which :class:`Sample` field
is missing.

Because the mass *can* come from two places -- the file or the caller --
:class:`Sample` records which, on :attr:`Sample.mass_source`. Normalised moments, susceptibilities and Curie constants scale inversely
with the supplied mass; inferred effective moments scale with its inverse square root, so
"where did this mass come from" is not bookkeeping: it is the difference
between a result you can defend and one you cannot.

This mirrors the asymmetry toolkit's "typed policy object, not dict"
lesson: bundling the sample facts into one frozen, named, documented
dataclass (rather than passing loose ``mass``/``molar_mass``/... keyword
arguments around, or hiding them in an untyped dict) makes every call site
self-describing and catches missing/misspelled fields at construction
time rather than deep inside a conversion.

Conventions
-----------
See :mod:`calamari.constants` for the underlying CODATA factors and
:func:`mu_eff_from_curie_constant` below for the effective-moment formula.
All arithmetic is CGS-Gaussian (moments in emu, fields in Oe, masses in
grams once converted from the header's milligrams), matching the rest of
calamari.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Rational, Real
from typing import TYPE_CHECKING

import numpy as np

from calamari.constants import AVOGADRO, K_B_ERG_PER_K, MU_B_EMU, MU_B_PER_EMU
from calamari.exceptions import SampleValidationError

if TYPE_CHECKING:
    from calamari.measurement import Measurement


@dataclass(frozen=True)
class Sample:
    """The physical-sample facts needed to convert raw readings to physics units.

    Frozen (immutable) per the house rule that nothing mutates a
    ``Measurement`` or ``Sample`` in place -- derive a new ``Sample`` (e.g.
    via :func:`dataclasses.replace`) rather than editing one.

    Every field defaults to ``None`` except ``name`` -- there are no
    project-specific numeric defaults (no assumed mass, no assumed molar
    mass) because those facts are specific to whatever sample and compound
    the caller is actually measuring. Conversions that need a missing
    field raise :class:`ValueError` naming it, rather than silently
    propagating ``None`` into arithmetic.

    Attributes
    ----------
    name : str
        Sample identifier (material name, crystal label, ...).
    mass_mg : float or None
        Sample mass in milligrams. See :meth:`from_measurement` for how it
        is derived from a parsed file, and :attr:`mass_source` for how to
        tell afterwards where it came from.
    mass_source : str or None
        Provenance of :attr:`mass_mg`: ``'user'`` when the caller supplied
        it, ``'file'`` when it was read from the measurement header, or
        ``None`` when there is no mass (or when the ``Sample`` was
        constructed directly rather than via :meth:`from_measurement`,
        which is the only thing that records provenance -- a directly
        built ``Sample`` leaves this ``None`` rather than inferring
        ``'user'``, so ``None`` means "not recorded", never "trust me").

        This exists because a mass that silently came from the wrong place
        is the most damaging thing that can happen to a magnetometry
        conversion -- every ``muB``-per-ion and molar-susceptibility number
        scales inversely with it. Recording the provenance turns a silent
        substitution into a checkable one::

            if sample.mass_source == "file" and name in logbook:
                ratio = sample.mass_mg / logbook[name]
                if not 0.5 < ratio < 2.0:
                    warn(f"header mass is {ratio:.0f}x the logbook value")
    molar_mass_g_per_mol : float or None
        Molar mass of the compound, g/mol.
    magnetic_ions_per_fu : float or None
        Number, or average number, of magnetic ions per formula unit (e.g. the number of
        rare-earth/transition-metal ions carrying the moment). Used by the
        "per ion" conversions and properties. Fractional occupancy is supported;
        the caller defines the counted species. Contributions from different
        magnetic species are not separated automatically.
    shape : str or None
        Free-text sample shape/geometry note (e.g. ``'plate'``, ``'sphere'``).
    orientation : str or None
        Free-text field/crystal-axis orientation note (e.g. ``'H // c'``).
    comment : str or None
        Free-text note, e.g. carried over from the measurement header.
    """

    name: str
    mass_mg: float | None = None
    mass_source: str | None = None
    molar_mass_g_per_mol: float | None = None
    magnetic_ions_per_fu: float | None = None
    shape: str | None = None
    orientation: str | None = None
    comment: str | None = None

    def __post_init__(self) -> None:
        """Reject invalid normalisation facts before they enter a calculation."""
        for field in ("mass_mg", "molar_mass_g_per_mol", "magnetic_ions_per_fu"):
            value = getattr(self, field)
            if value is not None and (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not (
                    isinstance(value, Rational)
                    or (np.isfinite(value) if isinstance(value, np.floating) else isfinite(value))
                )
                or value <= 0
            ):
                raise SampleValidationError(
                    f"Sample {self.name!r}: {field}={value!r}; expected a finite positive "
                    f"real scalar or None. Supply a valid {field}."
                )

    @property
    def mass_g(self) -> float:
        """Sample mass in grams.

        Raises
        ------
        ValueError
            If :attr:`mass_mg` is None.
        """
        if self.mass_mg is None:
            raise SampleValidationError(
                f"Sample {self.name!r} has no mass_mg set; supply it directly "
                "as Sample(mass_mg=...) or Sample.from_measurement(m, mass_mg=...). "
                "A Sample built from a measurement has no mass only when the file "
                "recorded none -- check Measurement.sample_mass_mg."
            )
        return self.mass_mg / 1000.0

    @property
    def moles(self) -> float:
        """Amount of substance, in moles of formula units.

        Raises
        ------
        ValueError
            If :attr:`mass_mg` or :attr:`molar_mass_g_per_mol` is None.
        """
        if self.molar_mass_g_per_mol is None:
            raise SampleValidationError(
                f"Sample {self.name!r} has no molar_mass_g_per_mol set; "
                "supply it to compute moles/formula_units"
            )
        return self.mass_g / self.molar_mass_g_per_mol

    @property
    def formula_units(self) -> float:
        """Number of formula units in the sample (``moles * N_A``).

        Raises
        ------
        ValueError
            If :attr:`mass_mg` or :attr:`molar_mass_g_per_mol` is None
            (propagated from :attr:`moles`).
        """
        return self.moles * AVOGADRO

    @property
    def moles_magnetic_ions(self) -> float:
        """Amount of substance, in moles of magnetic ions.

        Raises
        ------
        ValueError
            If :attr:`mass_mg`, :attr:`molar_mass_g_per_mol`, or
            :attr:`magnetic_ions_per_fu` is None.
        """
        if self.magnetic_ions_per_fu is None:
            raise SampleValidationError(
                f"Sample {self.name!r} has no magnetic_ions_per_fu set; "
                "supply it to compute moles_magnetic_ions / *_per_ion conversions"
            )
        return self.moles * self.magnetic_ions_per_fu

    def moment_emu_to_muB_per_fu(self, moment_emu):
        """Convert a measured moment (emu) to Bohr magnetons per formula unit.

        The number of formula units in the sample is ``FU = moles * N_A``
        (:attr:`formula_units`), and ``muB_per_fu = moment_emu *
        MU_B_PER_EMU / FU``, using the exact emu<->Bohr-magneton factor
        from :mod:`calamari.constants`. Older analyses often round that
        factor to 1.07828e20, which differs only in the 5th significant
        figure.

        Parameters
        ----------
        moment_emu : float, ndarray, or pandas.Series
            Measured moment in emu. Returned in the same shape/type.

        Returns
        -------
        float, ndarray, or pandas.Series
            Moment in Bohr magnetons per formula unit.

        Raises
        ------
        ValueError
            If :attr:`mass_mg` or :attr:`molar_mass_g_per_mol` is None
            (propagated from :attr:`formula_units`).
        """
        return moment_emu * MU_B_PER_EMU / self.formula_units

    def moment_emu_to_muB_per_ion(self, moment_emu):
        """Convert a measured moment (emu) to Bohr magnetons per magnetic ion.

        Equivalent to :meth:`moment_emu_to_muB_per_fu` divided by
        :attr:`magnetic_ions_per_fu`. The ion species is caller-defined via
        :attr:`magnetic_ions_per_fu`, never assumed by this library.

        Parameters
        ----------
        moment_emu : float, ndarray, or pandas.Series
            Measured moment in emu. Returned in the same shape/type.

        Returns
        -------
        float, ndarray, or pandas.Series
            Moment in Bohr magnetons per magnetic ion.

        Raises
        ------
        ValueError
            If :attr:`mass_mg`, :attr:`molar_mass_g_per_mol`, or
            :attr:`magnetic_ions_per_fu` is None.
        """
        if self.magnetic_ions_per_fu is None:
            raise SampleValidationError(
                f"Sample {self.name!r} has no magnetic_ions_per_fu set; "
                "supply it to compute moment_emu_to_muB_per_ion"
            )
        return self.moment_emu_to_muB_per_fu(moment_emu) / self.magnetic_ions_per_fu

    def moment_to_chi_molar(self, moment_emu, field_oe):
        """Molar susceptibility, emu/(mol Oe), per mole of formula units.

        ``chi_molar = (moment_emu / field_oe) / moles``, i.e. the moment
        per unit applied field normalised by the amount of substance
        (:attr:`moles`, formula-unit basis). Field-dependent (non-linear)
        moments should be evaluated in a regime where ``moment/field`` is
        a meaningful susceptibility (e.g. the low-field limit) -- this
        method performs no linearity check.

        Parameters
        ----------
        moment_emu : float, ndarray, or pandas.Series
            Measured moment in emu.
        field_oe : float, ndarray, or pandas.Series
            Applied field in Oe, same shape as ``moment_emu``.

        Returns
        -------
        float, ndarray, or pandas.Series
            Molar susceptibility, emu/(mol Oe), per mole of formula units.

        Raises
        ------
        ValueError
            If :attr:`mass_mg` or :attr:`molar_mass_g_per_mol` is None.
        """
        return (moment_emu / field_oe) / self.moles

    def moment_to_chi_molar_per_ion(self, moment_emu, field_oe):
        """Molar susceptibility, emu/(mol Oe), per mole of magnetic ions.

        As :meth:`moment_to_chi_molar` but normalised by
        :attr:`moles_magnetic_ions` instead of :attr:`moles`, i.e. per
        mole of magnetic ion rather than per mole of formula unit --
        the natural basis for comparing a Curie constant to a
        single-ion effective moment (see
        :func:`mu_eff_from_curie_constant`).

        Parameters
        ----------
        moment_emu : float, ndarray, or pandas.Series
            Measured moment in emu.
        field_oe : float, ndarray, or pandas.Series
            Applied field in Oe, same shape as ``moment_emu``.

        Returns
        -------
        float, ndarray, or pandas.Series
            Molar susceptibility, emu/(mol Oe), per mole of magnetic ions.

        Raises
        ------
        ValueError
            If :attr:`mass_mg`, :attr:`molar_mass_g_per_mol`, or
            :attr:`magnetic_ions_per_fu` is None.
        """
        return (moment_emu / field_oe) / self.moles_magnetic_ions

    def chi_mass(self, moment_emu, field_oe):
        """Mass susceptibility, emu/(g Oe).

        ``chi_mass = (moment_emu / field_oe) / mass_g``. Useful when the
        molar mass is not known (e.g. an unidentified phase) but the
        sample mass is.

        Parameters
        ----------
        moment_emu : float, ndarray, or pandas.Series
            Measured moment in emu.
        field_oe : float, ndarray, or pandas.Series
            Applied field in Oe, same shape as ``moment_emu``.

        Returns
        -------
        float, ndarray, or pandas.Series
            Mass susceptibility, emu/(g Oe).

        Raises
        ------
        ValueError
            If :attr:`mass_mg` is None.
        """
        return (moment_emu / field_oe) / self.mass_g

    def chi_to_chi_molar(self, chi_emu_per_oe):
        """Molar susceptibility, emu/(mol Oe), from a susceptibility in emu/Oe.

        The AC-side counterpart of :meth:`moment_to_chi_molar`:
        :attr:`~calamari.measurement.Measurement.chi_real` exposes susceptibility
        in emu/Oe, so
        normalising it needs no field -- just the amount of substance.
        ``chi_molar = chi_emu_per_oe / moles``.

        Parameters
        ----------
        chi_emu_per_oe : float, ndarray, or pandas.Series
            Susceptibility in emu/Oe (the extensive, whole-sample
            quantity the instrument reports).

        Returns
        -------
        float, ndarray, or pandas.Series
            Molar susceptibility, emu/(mol Oe), per mole of formula units.

        Raises
        ------
        ValueError
            If :attr:`mass_mg` or :attr:`molar_mass_g_per_mol` is None.
        """
        return chi_emu_per_oe / self.moles

    def chi_to_chi_molar_per_ion(self, chi_emu_per_oe):
        """Molar susceptibility, emu/(mol Oe), per mole of magnetic ions.

        As :meth:`chi_to_chi_molar` but normalised by
        :attr:`moles_magnetic_ions` -- the AC-side counterpart of
        :meth:`moment_to_chi_molar_per_ion`.

        Raises
        ------
        ValueError
            If :attr:`mass_mg`, :attr:`molar_mass_g_per_mol`, or
            :attr:`magnetic_ions_per_fu` is None.
        """
        return chi_emu_per_oe / self.moles_magnetic_ions

    def chi_to_chi_mass(self, chi_emu_per_oe):
        """Mass susceptibility, emu/(g Oe), from a susceptibility in emu/Oe.

        As :meth:`chi_to_chi_molar` but normalised by :attr:`mass_g` --
        the AC-side counterpart of :meth:`chi_mass`, for when the molar
        mass is not known but the sample mass is.

        Raises
        ------
        ValueError
            If :attr:`mass_mg` is None.
        """
        return chi_emu_per_oe / self.mass_g

    @classmethod
    def from_measurement(
        cls,
        m: Measurement,
        *,
        mass_mg: float | None = None,
        **overrides,
    ) -> Sample:
        """Build a :class:`Sample` from a parsed file's header metadata.

        The mass is resolved by a single rule -- **the caller wins, then
        the file** -- and where it came from is recorded on
        :attr:`mass_source`:

        ===================  ==============  ===============================
        ``mass_mg`` given?   File records a  Result
                             mass?
        ===================  ==============  ===============================
        yes                  either          caller's value, ``'user'``
        no                   yes             file's value, ``'file'``
        no                   no              ``None``, ``None``
        ===================  ==============  ===============================

        The file mass comes from SAMPLE_MASS through
        :attr:`~calamari.measurement.Measurement.sample_mass_mg`. This is an
        operator-entered value, not an independently verified specimen fact.
        Check it against the weighing record before using mass normalisation.

        The sample name defaults to ``m.sample_name`` (falling back to
        ``'unknown'`` if the header carries none); every other
        :class:`Sample` field (``molar_mass_g_per_mol``,
        ``magnetic_ions_per_fu``, ``shape``,
        ``orientation``, ``comment``) has no file-derived source and must
        be supplied via ``**overrides`` if needed.

        Parameters
        ----------
        m : Measurement
            Parsed measurement (see :mod:`calamari.io`).
        mass_mg : float, optional
            Sample mass in mg, overriding whatever the file records.
            Passing ``None`` (the default) is not an assertion that there
            is no mass -- it simply means "use the file's, if it has one",
            so ``mass_mg=logbook.get(name)`` behaves sensibly for a sample
            you have no logbook entry for. Has no project-specific default;
            callers own it.
        **overrides
            Additional :class:`Sample` fields, forwarded to the
            constructor. These are applied last, so passing ``mass_source``
            here overrides the derived provenance -- don't, unless you are
            deliberately re-labelling it.

        Returns
        -------
        Sample
        """
        derived: dict = {}
        derived["name"] = m.sample_name if m.sample_name is not None else "unknown"

        file_mass_mg = m.sample_mass_mg if mass_mg is None else None

        if mass_mg is not None:
            derived["mass_mg"] = mass_mg
            derived["mass_source"] = "user"
        elif file_mass_mg is not None:
            derived["mass_mg"] = file_mass_mg
            derived["mass_source"] = "file"
        else:
            derived["mass_mg"] = None
            derived["mass_source"] = None

        derived.update(overrides)
        return cls(**derived)


def mu_eff_from_curie_constant(curie_constant: float) -> float:
    """Estimate the Curie-Weiss effective moment from a Curie constant.

    ``mu_eff / mu_B = sqrt(3 * k_B * C / N_A) / mu_B``, computed from the
    CODATA factors in :mod:`calamari.constants`. This evaluates to
    ``mu_eff / mu_B = 2.8279 * sqrt(C)`` to the shown precision,
    conventionally rounded to ``2.828 * sqrt(C)``, with ``C`` in
    emu K/(mol Oe).

    ``C`` must be in the ``chi = C / (T - theta)`` convention, on whatever
    mole basis the caller normalised to (typically per mole of magnetic
    ion, e.g. via :meth:`Sample.moment_to_chi_molar_per_ion`, so the
    returned moment reads as mu_eff per ion). If you fitted the *inverse*
    susceptibility instead, convert with
    :func:`curie_constant_from_inverse_slope` rather than by hand.

    The factor of 3 here is the one from ``C = N mu_eff^2 / 3 k_B``, and
    it is applied exactly once, in this function.

    Parameters
    ----------
    curie_constant : float
        Curie constant C, in emu K / (Oe mol), on whatever mole basis
        (formula unit or magnetic ion) the caller wants ``mu_eff``
        expressed per.

    Returns
    -------
    float
        Effective moment mu_eff, in Bohr magnetons, on the same mole
        basis as ``curie_constant``.
    """
    return np.sqrt(3.0 * K_B_ERG_PER_K * curie_constant / AVOGADRO) / MU_B_EMU


def curie_constant_from_inverse_slope(slope: float) -> float:
    """Curie constant ``C`` from a straight-line fit to ``1/chi`` vs ``T``.

    The standard Curie-Weiss analysis plots inverse susceptibility against
    temperature, because ``chi = C/(T - theta)`` rearranges to the straight
    line ``1/chi = (T - theta)/C``. That line has **slope ``1/C``**, so::

        C = 1 / slope

    and its x-intercept is ``theta``.

    Parameters
    ----------
    slope : float
        Slope of ``1/chi`` against ``T``, in the reciprocal units of
        whatever ``chi`` was fitted, e.g. Oe mol / (emu K) for a molar
        susceptibility in emu/(Oe mol).

    Returns
    -------
    float
        Curie constant ``C``, on the same mole basis as the fitted
        susceptibility, ready for :func:`mu_eff_from_curie_constant`.

    Raises
    ------
    ValueError
        If ``slope`` is zero or non-finite -- a flat or degenerate fit
        carries no Curie constant, and silently returning ``inf`` would
        propagate into ``mu_eff`` as a plausible-looking number.

    Examples
    --------
    >>> C = curie_constant_from_inverse_slope(0.22818)
    >>> round(float(mu_eff_from_curie_constant(C)), 2)
    5.92
    """
    if not np.isfinite(slope) or slope == 0.0:
        raise SampleValidationError(
            f"inverse-susceptibility slope must be finite and non-zero, got "
            f"{slope!r}; a flat or degenerate 1/chi fit has no Curie constant"
        )
    return 1.0 / slope
