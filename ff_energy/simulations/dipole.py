import MDAnalysis as mda

# Statistics
from statsmodels.tsa.stattools import acovf
# Miscellaneous
import ase.units as units

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

angstrom_charge_to_debye = 0.2081943

xs = []
ys = []
with open("/home/boittier/Documents/phd/67-56-1-IR.jdx") as f:
    lines = f.readlines()[37:-2]
    for line in lines:
        x, y = [float(_) for _ in line.split()][:2]
        xs.append(x)
        ys.append(1 - y)
methanol_exp_spectrum = [xs, ys]


def fit_morse_potential(energies, distances):
    """Fit a morse potential to a set of energies and distances

    energies: np.array of energies in kcal/mol
    distances: np.array of distances in angstrom

    """
    from scipy.optimize import curve_fit

    def morse(r, a, b, c):
        return a * (1 - np.exp(-b * (r - c))) ** 2

    popt, pcov = curve_fit(morse, distances, energies)
    return popt


def calculate_IR_spectrum(dipoles, timestep=1.0):
    """Calculate the IR spectrum from a list of dipole moments

    dipoles: list of dipole moments in Debye
    timestep: timestep in fs

    """
    dipoles = np.array(dipoles)
    freqs = np.fft.fftfreq(len(dipoles), d=timestep)
    spectrum = np.fft.fft(dipoles)
    return freqs, spectrum


def get_mag_dipole(dipole):
    """Get the magnitude of a dipole moment
    calculates the magnitude of a dipole moment from a
    vector in units of angstrom and atomic mass units
    and returns the magnitude in Debye
    """
    dipole_mag = np.sqrt(np.sum(dipole ** 2))
    converted_dipole = dipole_mag / angstrom_charge_to_debye
    return converted_dipole


def calculate_dipole(xyz, masses, charges):
    """Calculate the dipole moment of a molecule

    xyz: np.array of shape (n_atoms, 3) in angstrom
    masses: np.array of shape (n_atoms,) in atomic mass units
    charges: np.array of shape (n_atoms,) in elementry charge

    """
    dipole = np.zeros(3)
    com = np.average(xyz, weights=masses, axis=0)
    for i, atom in enumerate(xyz):
        dipole += charges[i] * (atom - com)
    return dipole, get_mag_dipole(dipole)


def load_from_dcd(psf_path, dcd_path):
    u = mda.Universe(psf_path, dcd_path)
    print(psf_path)
    print(dcd_path)
    print(u)
    print(u.trajectory)

    # select the first residue
    atom_group = u.select_atoms('resid 1')

    total_dipole_vectors = []
    Ds = []

    for _ in u.trajectory:
        atoms = []
        charges = []
        mass = []
        for atom in atom_group:
            atoms.append(atom.position)
            charges.append(atom.charge)
            mass.append(atom.mass)
        vector, D = calculate_dipole(
            np.array(atoms), np.array(mass), np.array(charges))
        Ds.append(D)
        total_dipole_vectors.append(vector)
    return total_dipole_vectors, Ds


def load_from_dcm(file):
    total_dipole_vectors = []
    Ds = []
    with open(file, "r") as f:
        for i, line in enumerate(f.readlines()):
            vector = np.array([float(line.split()[0]),
                               float(line.split()[1]),
                               float(line.split()[2])])
            total_dipole_vectors.append(vector)
            Ds.append(get_mag_dipole(vector))
    return total_dipole_vectors, Ds


def plot_dipole_timeseries(Ds, NSKIP=0, filename=None, effective_timestep=1.0):
    Ds = Ds[NSKIP::]
    print("Average dipole: ", np.mean(Ds))

    time = np.arange(len(Ds)) * effective_timestep
    plt.plot(time, Ds)
    plt.axhline(np.mean(Ds), color="k", linestyle="--")
    plt.title("$\hat{\mu} =$" + " {:.1f}".format(np.mean(Ds)) + " Debye",
              fontsize=20, fontweight="bold")
    plt.xlabel("Time (ps)", fontsize=20, fontweight="bold")
    plt.ylabel("Dipole moment (Debye)", fontsize=20, fontweight="bold")

    if filename is not None:
        filename = filename + "_dipole" + ".pdf"
        plt.savefig(filename)

    plt.show()
    plt.clf()


def calc_spectra(total_dipole_vectors, NSKIP=0, filename=None, effective_timestep=1.0):
    total_dipole_vectors = total_dipole_vectors[NSKIP::]

    # Time for speed of light in vacuum to travel 1cm (0.01) in 1fs (1e15)
    jiffy = 0.01 / units._c * 1e12

    dipo_list = np.array(total_dipole_vectors)
    dtime = effective_timestep
    # Frequency range in cm^-1
    Nframes = dipo_list.shape[0]

    print("Nframes: ", Nframes)

    Nfreq = int(Nframes / 2) + 1
    freq = np.arange(Nfreq) / float(Nframes) / dtime * jiffy
    # temperature in K
    Ti = 298.15
    # Quantum correction factor qcf
    Kb = 3.1668114e-6  # Boltzmann constant in atomic units (Hartree/K)
    beta = 1.0 / Kb / float(Ti)  # atomic units
    hbar = 1.0  # atomic units
    const = beta * hbar  # atomic units
    cminvtoau = 1.0 / 2.1947e7  # inv cm to atomic units

    qcf = np.tanh(const * freq * cminvtoau / 2.)

    # Dipole-Dipole autocorrelation function
    acvx = acovf(dipo_list[:, 0], fft=True)
    acvy = acovf(dipo_list[:, 1], fft=True)
    acvz = acovf(dipo_list[:, 2], fft=True)
    acv = acvx + acvy + acvz

    acv = acv * np.blackman(Nframes)
    spectra = np.abs(np.fft.rfftn(acv)) * qcf

    # save to csv
    if filename is not None:
        filename = filename + ".csv"
        df = pd.DataFrame({"freq": freq, "spectra": spectra})
        df.to_csv(filename, index=False)

    return freq, spectra


def plot_spectra(freq, spectra, filename=None, csv=False,
                 ax = None,
                 Nsmooth=10, xlim=None, ylim=None, plot_methanol=False, plot_water=False, axvlines=False):
    if csv:
        df = pd.read_csv(csv)
        df = df[df["freq"] > 800]
        freq = df["freq"].values
        spectra = df["spectra"].values  # - df["spectra"].min()
    # else:
    # spectra = spectra - spectra.min()

    # create a moving average
    window = np.ones(Nsmooth) / Nsmooth
    spectra = np.convolve(spectra, window, 'same')
    spectra = spectra / spectra.max()
    
    if ax is None:
        ax = plt.gca()
    
    if plot_methanol:
        ax.plot(methanol_exp_spectrum[0], methanol_exp_spectrum[1],
                 linewidth=1, color="k", label="Exp. (Methanol)")
    if plot_water:
        ax.plot(freq, spectra, linewidth=1)
        
        ax.set_xlabel("Frequency (cm$^{-1}$)", fontsize=20, fontweight="bold")
        
    ax.plot(freq, spectra, linewidth=1)
    ax.set_xlabel("Frequency (cm$^{-1}$)", fontsize=20, fontweight="bold")
        
    if xlim is None:
        ax.set_xlim(500, 4000)
    else:
        ax.set_xlim(xlim[0], xlim[1])

    if ylim is None:
        ax.set_ylim(0, np.max(spectra))
        ax.set_ylim(0, 1)
    else:
        plt.ylim(ylim[0], ylim[1])

    # reverse the y axis
    # plt.gca().invert_yaxis()
    ax.set_ylabel("Intensity (a.u.)", fontsize=20, fontweight="bold")
    # reverse the x axis
    # plt.gca().invert_xaxis()

    if axvlines:
        for x in axvlines:
            ax.axvline(x, color="k", linestyle="--")

    # tight layout
    plt.tight_layout()
    if filename is not None:
        filename = filename + "_spectra" + ".pdf"
        plt.savefig(filename, bbox_inches="tight")

    #plt.show()
    return ax

def dipole_dcm(filename, dcm, NSKIP=0, effective_timestep=1.0, nsmooth=10):
    total_dipole_vectors, Ds = load_from_dcm(dcm)
    plot_dipole_timeseries(Ds, NSKIP=NSKIP, filename=filename,
                           effective_timestep=effective_timestep)
    freq, spectra = calc_spectra(total_dipole_vectors, NSKIP=NSKIP,
                                 filename=filename,
                                 effective_timestep=effective_timestep)
    plot_spectra(freq, spectra, filename=filename, csv=False, Nsmooth=nsmooth)


def dipole_dcd(filename, psf_path, dcd_path, NSKIP=0, effective_timestep=1.0,
               nsmooth=10):
    total_dipole_vectors, Ds = load_from_dcd(psf_path, dcd_path)
    plot_dipole_timeseries(Ds, NSKIP=NSKIP, filename=filename,
                           effective_timestep=effective_timestep)
    freq, spectra = calc_spectra(total_dipole_vectors, NSKIP=NSKIP, filename=filename,
                                 effective_timestep=effective_timestep)
    plot_spectra(freq, spectra, filename=filename, csv=False, Nsmooth=nsmooth)

# psf_path = "/home/boittier/pcbach/param/methanol/charmm/r0.0.0/t298.15/gas2/min/int
# .psf" dcd_path = "/home/boittier/pcbach/param/methanol/charmm/r0.0.0/t298.15/gas2
# /eq/md_eq1.dcd" timestep = 0.0001 save_freq = 2 effective_timestep = timestep *
# save_freq

# psf_path = "/home/boittier/pcbach/param/methanol/charmm/r1.0.0/t298.15/min/int.psf"
# dcd_path = "/home/boittier/pcbach/param/methanol/charmm/r1.0.0/t298.15/eq/md_eq2.dcd"
# timestep = 0.0001
# save_freq = 5
# effective_timestep = timestep * save_freq

# file = "/home/boittier/pcbach/param/methanol/kern/k.ffc8.0.1/t298.15/dump.xyz"
# timestep = 0.0001
# save_freq = 2
# effective_timestep = timestep * save_freq

# file = "/home/boittier/pcbach/param/methanol/kern/k.ffc8.0.1/t298.15/gas2/dump.xyz"
# timestep = 0.0001
# save_freq = 2
# effective_timestep = timestep * save_freq
