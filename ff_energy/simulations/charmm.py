import os
import pandas as pd

DYNASTART = "CHARMM>    DYNA"
DYNAEXTERN = "DYNA EXTERN>"
DYNA = "DYNA>"
DYNAPRESS = "DYNA PRESS>"

def read_charmm_log(path, title=None):
    """Read a charmm log file and return the lines"""
    with open(path, "r") as f:
        lines = f.readlines()
    df = read_charmm_lines(lines)
    df["path"] = path

    if title:
        df["title"] = title
    else:
        df["title"] = path.replace("/", "_")

    return df

def read_pressures(pressures):
    x = pressures
    volume = float(x[67:])
    pressi = float(x[55:68])
    presse = float(x[40:53])
    return volume, presse, pressi

def read_energies(energies):
    x = energies
    T = float(x[70:])
    TOTE = float(x[27:40])
    E = float(x[55:70])
    t = float(x[16:27])
    return t, T, TOTE, E

#DYNA EXTERN>     1526.65445  -9553.35670      0.00000      0.00000      0.00000

def read_extern(externs):
    x = externs
    vdw = float(x[13:27])
    elec = float(x[27:41])
    user = float(x[66:])
    return vdw, elec, user



def read_charmm_lines(lines):
    """Read a list of lines and return the complexation energy"""
    dynamics = []
    pressures = []
    energies = []
    externs = []

    dcds = []

    starts = 0

    for line in lines:
        if DYNASTART in line.upper():
            dyna_name = " ".join(line.split()[1:4])
            dynamics.append("{}: ".format(starts) + dyna_name)
            starts += 1
        if DYNAEXTERN in line:
            externs.append([*read_extern(line), dynamics[-1], dcds[-1]])
        if DYNA in line:
            energies.append([*read_energies(line)])
        if DYNAPRESS in line:
            pressures.append([*read_pressures(line)])

        if line.startswith(" OPNLGU>") and ".dcd" in line:
            dcds.append(os.path.basename(line.split()[-1]))

    df = pd.concat([pd.DataFrame(externs,
                                 columns=["vdw", "elec", "user", "dyna", "dcd"]),
            pd.DataFrame(energies, columns=["time", "temp", "tot", "energy"]),
            pd.DataFrame(pressures, columns=["volume", "pressi", "presse"])], axis=1)

    return df
