import os
import pandas as pd

DYNASTART = "CHARMM>    DYNA"
DYNAEXTERN = "DYNA EXTERN>"
DYNA = "DYNA>"
DYNAPRESS = "A PRESS>"


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
    try:
        x = pressures
        volume = float(x[67:73])
        pressi = float(x[55:68])
        presse = float(x[40:53])
        return volume, pressi, presse
    except ValueError:
        return None, None, None


def read_energies(energies):
    try:
        x = energies
        T = float(x[70:])
        TOTE = float(x[27:40])
        E = float(x[55:70])
        t = float(x[16:27])
        return t, T, TOTE, E
    except ValueError:
        return None, None, None, None


# DYNA EXTERN>     1526.65445  -9553.35670      0.00000      0.00000      0.00000

def read_extern(externs):
    try:
        x = externs
        vdw = float(x[13:27])
        elec = float(x[27:41])
        user = float(x[41:55])
        return vdw, elec, user
    except ValueError:
        return None, None, None


def read_charmm_lines(lines):
    """Read a list of lines and return the complexation energy"""
    dynamics = []
    pressures = []
    energies = []
    externs = []
    #  dcd files
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
        #  record the name of the dcd file
        if line.startswith(" CHARMM>    OPEN WRITE") and ".dcd" in line:
            dcdfilename = [_ for _ in line.split() if _.endswith(".dcd")]
            assert len(dcdfilename) == 1

            dcdfilename = os.path.basename(dcdfilename[0])
            #  if the dcd file is already in the list, remove it
            if dcdfilename in dcds:
                pass
            else:
                print(dcdfilename)
                dcds.append(dcdfilename)

    df = pd.concat([pd.DataFrame(externs,
                                 columns=["vdw", "elec", "user", "dyna", "dcd"]),
                    pd.DataFrame(energies, columns=["time", "temp", "tot", "energy"]),
                    pd.DataFrame(pressures, columns=["volume", "pressi", "presse"])],
                   axis=1)

    return df
