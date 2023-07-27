import math
from pint import UnitRegistry

side_length = 30  # angstroms

ureg = UnitRegistry()
verbose = True

#  avogadro's number
n_avo = 6.02214076e23 / ureg.mol
n_avo.ito_base_units()

# density
dcm_density = 1.326 * ureg.g / (ureg.cm ** 3)
dcm_density.ito_base_units()
if verbose:
    print(f"dcm density: {dcm_density}")

# molecular weight
dcm_MW = 84.93 * ureg.g / ureg.mol
dcm_MW.ito_base_units()
if verbose:
    print(f"dcm MW: {dcm_MW}")

# molar volume
dcm_molar_volume = dcm_MW / dcm_density
dcm_molar_volume.ito(ureg.cm ** 3 / ureg.mol)
dcm_molar_volume.ito_base_units()
if verbose:
    print(f"dcm molar volume: {dcm_molar_volume}")

# number of molecules in a (eg.) 30 A^3 box
box_volume = side_length**3 * (ureg.angstrom ** 3)
# box_volume.ito(ureg.cm ** 3)
box_volume.ito_base_units()

if verbose:
    print(f"box volume: {box_volume}")

n_molecules = (box_volume / dcm_molar_volume) * n_avo
n_molecules.ito_base_units()

Nmols = math.ceil(n_molecules.magnitude)

if verbose:
    print(f"number of molecules: {Nmols}")
