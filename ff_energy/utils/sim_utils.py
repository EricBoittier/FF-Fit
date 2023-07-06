import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from MDAnalysis.analysis import rdf
from scipy.signal import find_peaks
import numpy as np


def angle(p):
    ba = p[0] - p[1]
    bc = p[2] - p[1]
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    a3 = np.degrees(np.arccos(cosine_angle))
    return a3


def get_dyna(path):
    lines = open(path).readlines()
    T = [float(x[70:]) for x in lines]
    TOTE = [float(x[27:40]) for x in lines]
    E = [float(x[55:70]) for x in lines]
    t = [float(x[16:27]) for x in lines]
    return t, T, TOTE, E


def md_analysis(psf, dcd):
    return mda.Universe(psf, dcd)


def get_angles(u, n_atoms=3, step = 1, start=0, stop=-1):
    n_res = len(u.residues)
    n_frames = len(u.trajectory[start:stop:step])
    angles = np.zeros((n_res, n_frames))
    angle_key = [1, 0, 2]

    # loop frames
    for t, ts in enumerate(u.trajectory[start:stop:step]):
        # loop residues
        for i in range(n_res):
            # save angles
            a = u.atoms[i * n_atoms + angle_key[0]].position
            b = u.atoms[i * n_atoms + angle_key[1]].position
            c = u.atoms[i * n_atoms + angle_key[2]].position
            _ = angle([a, b, c])
            angles[i][t] = _
    return angles


def hb_analysis(u, start=0, stop=-1, step=1):
    hbonds = HBA(
        universe=u,
        donors_sel=None,
        hydrogens_sel="name H1 H2",
        acceptors_sel="name OH2",
        d_a_cutoff=3.0,
        d_h_a_angle_cutoff=150,
        update_selections=False
    )
    hbonds.run(start=start, stop=stop, step=step)
    return hbonds


def g_rdf(u, sel='resname TIP3 and type OT', step=1, start=0, stop=-1):
    selection = u.select_atoms(sel)
    print(selection)

    irdf = rdf.InterRDF(selection, selection,
                        nbins=int(10//0.1),  # default
                        range=(0.0, 10.0),  # distance in angstroms
                        exclusion_block=(1, 1),
                        )
    irdf.run(step=step, start=start, stop=stop)
    return irdf


def get_rdf_peaks(irdf):
    peaks, _ = find_peaks(irdf.results.rdf, height=1)
    return peaks
