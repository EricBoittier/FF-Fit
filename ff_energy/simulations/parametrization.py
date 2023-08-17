import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pint import UnitRegistry
import warnings

# ignore the casting errors for units
warnings.simplefilter("ignore")
ureg = UnitRegistry()

#  matplotlib styles
plt.style.use(["science", "no-latex", "ieee"])

#  constants
Avogadro_const = 6.02214129 * 10**23 * ureg("1/mol")  # % mol-1
Boltzmann_const = 1.38064852 * 10 ** (-23) * ureg("J/K")  # % J K-1
Gas_const = 8.3144621 * ureg("J/(mol*K)")  # % JK^−1mol^−1
Cal2Joule = 4.184  # % J
J2kcal = 0.000239006
J2kcm = 1.43932643164436e20  # ;
atm2Pa = 101325  # ;
ang2m = 10 ** (-10)  # ;
Pa2atm = 1 / atm2Pa  # ;
angsqr_to_msqr = 1e-27
Jm2kcm = 0.000239005736
k_B_kcal_mol_K = 1.987204259 * 10 ** (-3) * ureg("kcal/(mol*K)")


@dataclass
class PropertyRun:
    """Class for keeping track of all the data from a simulation run"""

    directory: str
    MW: float
    Nmol: int
    energies: np.ndarray[int, np.float64]  # kcal/mol
    potentials: np.ndarray[int, np.float64]  # kcal/mol
    temperatures: np.ndarray[int, np.float64]
    avgEgas: float  # kcal/mol
    avgUgas: float  # kcal/mol
    avgTgas: float  # kcal/mol
    volumes: np.ndarray[int, np.float64]  # m^3
    pressures: np.ndarray[int, np.float64]  # atm
    times: np.ndarray[int, np.float64]  # ps
    msds: np.ndarray[int, np.float64]
    msd_times: np.ndarray[int, np.float64]

    #  save the original raw data
    energies_save: np.ndarray[int, np.float64] = field(init=False)
    temperatures_save: np.ndarray[int, np.float64] = field(init=False)
    volumes_save: np.ndarray[int, np.float64] = field(init=False)
    times_save: np.ndarray[int, np.float64] = field(init=False)
    pressures_save: np.ndarray[int, np.float64] = field(init=False)

    #  properties to calculate on initialization
    D: float = field(init=False)
    D_c: float = field(init=False)
    densities: np.ndarray[int, np.float64] = field(init=False)
    density: np.float64 = field(init=False)
    density_SD: np.ndarray[int, np.float64] = field(init=False)
    avgTemp: np.ndarray[int, np.float64] = field(init=False)
    average_volume: np.ndarray[int, np.float64] = field(init=False)
    average_energy: np.ndarray[int, np.float64] = field(init=False)
    average_potential: np.ndarray[int, np.float64] = field(init=False)
    average_pressure: np.ndarray[int, np.float64] = field(init=False)
    sd_pressure: np.ndarray[int, np.float64] = field(init=False)
    totalTime: np.float64 = field(init=False)
    kappa: np.float64 = field(init=False)
    H: np.ndarray[int, np.float64] = field(init=False)
    potential_per_molecule: np.ndarray[int, np.float64] = field(init=False)
    avgH: np.ndarray[int, np.float64] = field(init=False)
    avg_H_2: np.ndarray[int, np.float64] = field(init=False)
    avgV2: np.ndarray[int, np.float64] = field(init=False)
    # ln_density: np.float64 = field(init=False)
    dHvap: np.ndarray[int, np.float64] = field(init=False)
    Cp_SF: np.ndarray[int, np.float64] = field(init=False)

    def __post_init__(self):
        self.totalTime = self.times[-1]
        self.D, self.D_c = get_D(self.msd_times, self.msds)
        self.energies_save = self.energies
        self.volumes_save = self.volumes
        self.potentials_save = self.potentials
        self.temperatures_save = self.temperatures
        self.times_save = self.times
        self.pressures_save = self.pressures
        self.init_properties()

    def slice_data(self, start, stop, stride):
        """trim and take strides of the data to account for equilibration"""
        self.energies = self.energies_save[start:stop:stride]
        self.temperatures = self.temperatures_save[start:stop:stride]
        self.volumes = self.volumes_save[start:stop:stride]
        self.times = self.times_save[start:stop:stride]
        self.pressures = self.pressures_save[start:stop:stride]
        self.potentials = self.potentials_save[start:stop:stride]
        self.init_properties()

    def init_properties(self):
        self.average_energy = np.mean(self.energies)
        self.average_potential = np.mean(self.potentials)
        self.potential_per_molecule = self.average_potential / self.Nmol
        self.average_volume = np.mean(self.volumes, dtype=np.float64).to(
            "angstrom cubed"
        )
        self.average_pressure = np.mean(self.pressures)
        self.sd_pressure = np.std(self.pressures)
        self.densities = get_density(self.volumes, self.Nmol, self.MW)
        self.density = get_density(self.average_volume, self.Nmol, self.MW).to("kg/m^3")
        self.density_SD = np.std(self.densities, dtype=np.float64)
        self.avgTemp = np.mean(self.temperatures).to("kelvin")
        self.H = get_enthalpy(self.potential_per_molecule, self.average_volume)
        self.avgH = np.mean(self.H)
        self.avg_H_2 = np.mean(self.H * self.H)
        self.avgV2 = np.mean(self.volumes * self.volumes)
        self.kappa = get_kappa(self.average_volume, self.avgV2, self.avgTemp)
        self.Cp_SF = get_C_p_STANDARD_FLUCTUATIONS(
            self.avgH, self.avg_H_2, self.Nmol, self.avgTemp
        )
        self.dHvap = (
            self.avgUgas
            - self.potential_per_molecule
            + self.avgTemp * Gas_const.to("kcal/(mol*K)")
        )


@dataclass
class MultiPropertyRun:
    """group simulations with similar conditions"""

    directory: str
    multirun: list[PropertyRun] = field(default_factory=list)
    multirun_df: pd.DataFrame = field(init=False)
    temperatures: np.ndarray[int, np.float64] = field(init=False)
    enthalpies: np.ndarray[int, np.float64] = field(init=False)
    densities: np.ndarray[int, np.float64] = field(init=False)
    kappas: np.ndarray[int, np.float64] = field(init=False)
    dHvaps: np.ndarray[int, np.float64] = field(init=False)
    C_ps: np.ndarray[int, np.float64] = field(init=False)
    C_ps_SF: np.ndarray[int, np.float64] = field(init=False)
    alphas: np.ndarray[int, np.float64] = field(init=False)

    def __post_init__(self):
        self.init_properties()

    def init_properties(self):
        self.multirun_df = pd.DataFrame(self.multirun)
        self.temperatures = self.multirun_df["avgTemp"]
        self.enthalpies = self.multirun_df["avgH"]
        self.densities = self.multirun_df["density"]
        self.kappas = self.multirun_df["kappa"]
        self.dHvaps = self.multirun_df["dHvap"]
        self.alphas = get_alpha(self.densities, self.temperatures)
        self.C_ps = get_C_p(self.enthalpies, self.temperatures)
        self.C_ps_SF = self.multirun_df["Cp_SF"]
        # fill the dataframe with the extra values
        self.multirun_df["alpha"] = self.alphas
        self.multirun_df["C_p"] = self.C_ps

    def slice_data(self, start, stop, stride):
        for i, run in enumerate(self.multirun):
            self.multirun[i].slice_data(start, stop, stride)
        self.init_properties()


def get_volumes(path):
    """read grepped output from CHARMM log file and return list of volumes
    Reads volumes in Angstrom^3"""
    with open(path) as f:
        _ = f.readlines()
    return np.array([float(x.split()[-1]) for x in _])


def get_pressures(path):
    """get pressures from charmm simulation
    :param path:
    :return:
    """
    with open(path) as f:
        _ = f.readlines()
    return np.array([float(x.split()[-2]) for x in _])


def get_density(volume, N_res, MW):
    """
    get the density of the simulation for a given volume, number of molecules and
    molecular weight :param volume: m^3 :param N_res: number :param MW: g/mol
    :return: density in g/m^3
    """
    MW = MW * ureg("g/mol")

    return N_res * MW / (Avogadro_const * volume)


def get_D(time, msd, skip=5):
    # msd = msd.to_value(ureg.dimensionless_unscaled)
    """Get self-diffusion coefficient by fitting a linear line to the MSD vs Time
    graph"""
    coef = np.polyfit(time[skip:], msd[skip:], 1)
    return float(coef[0]), float(coef[1])


def get_C_p(enthalpies, temperatures):
    differences = differences_symmetric(enthalpies, temperatures)
    Cp = [_.to("cal/mol/K") for _ in differences]
    return Cp


def get_alpha(densities, temperatures):
    differences = differences_symmetric(densities, temperatures, ln=True)
    #  conversion
    return [-1 * x for x in differences]


def get_kappa(Vavg, V2avg, T):
    """Calculate isothermal compressibility
    k_B in j K^-1
    :param V2avg: m^3
    :param Vavg: m^3
    :param T: K
    :return: kappa in Pa (j/m^3) converted to 1/atm
    """
    Vavg2 = Vavg * Vavg
    kappa = (V2avg - Vavg2) / (T * Boltzmann_const * Vavg)
    return kappa.to("1/atm")


def get_enthalpy(energy, volume, pressure=1.0 * ureg("atm")):
    """return the enthalpy in kcal/mol"""
    enthalpy = energy + (volume * pressure.to("Pa")) * Avogadro_const
    return enthalpy


def differences_symmetric(A, B, ln=False):
    """
    Calculate symmetric difference, with left right difference at each endpoint
    :param A: numerator
    :param B: denominator
    :return:
    """
    n_vals = len(A) - 1
    # left difference
    if not ln:
        _ = (A[1] - A[0]) / (B[1] - B[0])
    else:
        _ = np.log(A[1] / A[0]) / (B[1] - B[0])
    OUT = [_]
    # symmetric difference
    for i in range(1, n_vals):
        indx1 = i + 1
        indx2 = i - 1
        if not ln:
            OUT.append((A[indx1] - A[indx2]) / (B[indx1] - B[indx2]))
        else:
            OUT.append(np.log(A[indx1] / A[indx2]) / (B[indx1] - B[indx2]))
    # right difference
    if not ln:
        OUT.append((A[n_vals - 1] - A[n_vals]) / (B[n_vals - 1] - B[n_vals]))
    else:
        OUT.append(np.log(A[n_vals - 1] / A[n_vals]) / (B[n_vals - 1] - B[n_vals]))

    return OUT


def get_C_p_STANDARD_FLUCTUATIONS(avgH, avg_H2_, N, T):
    """
    Calculates the heat capacity (C_p)

    k_B in j K^-1
    :param avgH: average enthalpy
    :param avg_H2_: averaged squared enthalpy -     kcal/mol
    :param N: number of molecules
    :param T: temperature                           K

    :return: heat capacity
    """
    #  convert units
    avgH2 = avgH * avgH
    C_p = (avg_H2_ - avgH2) / (N * k_B_kcal_mol_K * (T**2)) + 3 * Gas_const
    C_p = C_p.to("cal/(mol*K)")
    return C_p


#  Helper functions
def fill_property_dataclass(path, MW=34.02, nMol=202):
    """starting from the directory, get and fill all the required data for the
    dataclass"""
    #  paths
    volumes_path = os.path.join(path, "pressure.raw")
    dyna_path = os.path.join(path, "dyna.raw")
    gas_dyna_path = os.path.join(path, "gas")  # , "job.out")
    msds_path = os.path.join(path, "anal", "methanol.msd")
    #  values
    time, energy, potential, temp = process_dyna(dyna_path)

    # Process gas data and take average over multiple runs
    gas_energies = []
    gas_pots = []
    gas_temps = []
    outfiles = [
        x
        for x in os.listdir(gas_dyna_path)
        if x.startswith("job") and x.endswith("out")
    ]

    for file in outfiles:
        _ = os.path.join(gas_dyna_path, file)
        gas_energy, gas_pot, gas_temp = get_gas_phase_results(_)
        gas_energies.append(gas_energy)
        gas_pots.append(gas_pot)
        gas_temps.append(gas_temp)

    gas_pot = np.mean(gas_pots)
    gas_temp = np.mean(gas_temps)
    gas_energy = np.mean(gas_energies)

    volumes = get_volumes(volumes_path)
    pressures = get_pressures(volumes_path)
    msd_times, msds = get_MSDs(msds_path)

    time = time * ureg("picoseconds")
    energy = energy * ureg("kcal/mol")
    potential = potential * ureg("kcal/mol")
    temp = temp * ureg("kelvin")
    gas_energy = gas_energy * ureg("kcal/mol")
    gas_pot = gas_pot * ureg("kcal/mol")
    gas_temp = gas_temp * ureg("kelvin")
    volumes = volumes * ureg("angstrom^3")
    pressures = pressures * ureg("atm")
    msds = msds

    #  create dataclass object
    prop_run = PropertyRun(
        path,
        MW,
        nMol,
        energy,
        potential,
        temp,
        gas_energy,
        gas_pot,
        gas_temp,
        volumes,
        pressures,
        time,
        msds,
        msd_times,
    )
    return prop_run


def fill_multirun_prob_dataclass(path):
    """fill the dataclass from a single run"""
    pruns = fill_multiprop_run(path)
    multirun_prob_run = MultiPropertyRun(path, pruns)
    return multirun_prob_run


def fill_multiprop_run(path):
    """fill the dataclass from runs at multiple temperatures"""
    subdirs = [_ for _ in os.listdir(path) if os.path.isdir(os.path.join(path, _))]
    pruns = []
    for sd in subdirs:
        _ = os.path.join(path, sd)
        prun = fill_property_dataclass(_)
        pruns.append(prun)
    return pruns


def get_MSDs(path):
    """Read the mean squared displacement data"""
    data = open(path).read()
    msd = np.array([float(_.split()[0]) for _ in data.split("\n")[:-1]])
    time = np.array([float(_.split()[1]) for _ in data.split("\n")[:-1]])
    return msd, time


def process_dyna(path):
    """read energy temperature and time from the grep'd log file"""
    with open(path) as f:
        lines = f.readlines()
        time = np.array([float(x.split()[2]) for x in lines])
        energy = np.array([float(x.split()[3]) for x in lines])
        pot = np.array([float(x.split()[5]) for x in lines])
        temp = np.array([float(x.split()[-1]) for x in lines])
    return time, energy, pot, temp


def get_gas_phase_results(path):
    """
    Read output file from gasphase simulation and return average energy and
    temperature as calculated by CHARMM... :param path: :return: average energy,
    average time
    """
    with open(path) as f:
        lines = [line for line in f.readlines() if line.startswith("AVER>")]
        # 3 vs 5
    avgE = np.array([float(_.split()[3]) for _ in lines])
    avgU = np.array([float(_.split()[5]) for _ in lines])
    avgT = np.array([float(_.split()[-1]) for _ in lines])
    return np.mean(avgE), np.mean(avgU), np.mean(avgT)
