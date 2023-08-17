# Basics
import os
import numpy as np
import pickle
import pandas as pd
import argparse

# Load FDCM modules

# Optimization
from scipy.optimize import minimize

bohr_to_a = 0.529177


mdcm_cxyz = "/home/unibas/boittier/fdcm_project/mdcms/methanol/10charges.xyz"
mdcm_clcl = (
    "/home/unibas/boittier/MDCM/examples/multi-conformer/5-charmm-files/"
    "10charges.dcm"
)


def get_clcl(local_pos, charges):
    NCHARGES = len(charges)
    _clcl_ = np.zeros(NCHARGES)
    for i in range(NCHARGES):
        if (i + 1) % 4 == 0:
            _clcl_[i] = charges[i]
        else:
            _clcl_[i] = local_pos[i - ((i) // 4)]
    return _clcl_


def set_bounds(local_pos, change=0.1):
    bounds = []
    for i, x in enumerate(local_pos):
        bounds.append((x - abs(x) * change, x + abs(x) * change))
    return tuple(bounds)


def mdcm_set_up(
    scan_fesp,
    scan_fdns,
    mdcm_cxyz="/home/unibas/boittier/fdcm_project/" "mdcms/methanol/10charges.xyz",
    mdcm_clcl="/home/unibas/boittier/MDCM/examples/multi-conformer/"
    "5-charmm-files/10charges.dcm",
    local_pos=None,
):
    mdcm_.dealloc_all()

    Nfiles = len(scan_fesp)
    Nchars = int(
        np.max(
            [
                len(filename)
                for filelist in [scan_fesp, scan_fdns]
                for filename in filelist
            ]
        )
    )

    esplist = np.empty([Nfiles, Nchars], dtype="c")
    dnslist = np.empty([Nfiles, Nchars], dtype="c")

    for ifle in range(Nfiles):
        esplist[ifle] = "{0:{1}s}".format(scan_fesp[ifle], Nchars)
        dnslist[ifle] = "{0:{1}s}".format(scan_fdns[ifle], Nchars)

    # Load cube files, read MDCM global and local files
    mdcm_.load_cube_files(Nfiles, Nchars, esplist.T, dnslist.T)
    mdcm_.load_clcl_file(mdcm_clcl)
    mdcm_.load_cxyz_file(mdcm_cxyz)

    # Get and set local MDCM array (to check if manipulation is possible)
    clcl = mdcm_.mdcm_clcl
    mdcm_.set_clcl(clcl)

    if local_pos is not None:
        mdcm_.set_clcl(local_pos)

    # Get and set global MDCM array (to check if manipulation is possible)
    cxyz = mdcm_.mdcm_cxyz
    mdcm_.set_cxyz(cxyz)

    # Write MDCM global from local and Fitted ESP cube files
    mdcm_.write_cxyz_files()
    mdcm_.write_mdcm_cube_files()

    return mdcm_


def optimize_mdcm(mdcm, clcl, outdir, outname, l2=100.0):
    # Get RMSE, averaged or weighted over ESP files, or per ESP file each
    rmse = mdcm.get_rmse()
    print(rmse)

    #  save an array containing original charges
    charges = clcl.copy()
    local_pos = clcl[np.mod(np.arange(clcl.size) + 1, 4) != 0]
    local_ref = local_pos.copy()

    def mdcm_rmse(local_pos, local_ref=local_ref, l2=l2):
        """Minimization routine"""
        _clcl_ = get_clcl(local_pos, charges)
        mdcm.set_clcl(_clcl_)
        rmse = mdcm.get_rmse()
        if local_ref is not None:
            l2diff = l2 * np.sum((local_pos - local_ref) ** 2) / local_pos.shape[0]
            # print(rmse, l2diff)
            rmse += l2diff
        return rmse

    # Apply simple minimization without any feasibility check (!)
    # Leads to high amplitudes of MDCM charges and local positions
    res = minimize(
        mdcm_rmse,
        local_pos,
        method="L-BFGS-B",
        options={
            "disp": None,
            "maxls": 20,
            "iprint": -1,
            "gtol": 1e-06,
            "eps": 1e-09,
            "maxiter": 15000,
            "ftol": 1e-8,
            "maxcor": 10,
            "maxfun": 15000,
        },
    )
    print(res)
    # Recompute final RMSE each
    rmse = mdcm.get_rmse()
    print(rmse)
    mdcm.write_cxyz_files()
    #  get the local charges array after optimization
    clcl_out = get_clcl(res.x, charges)
    difference = np.sum((res.x - local_ref) ** 2) / local_pos.shape[0]
    print("charge RMSD:", difference)

    obj_name = os.path.join(outdir, f"pickles/{outname}_clcl.obj")
    #  save as pickle
    filehandler = open(obj_name, "wb")
    pickle.dump(clcl_out, filehandler)
    # Not necessary but who knows when it become important to deallocate all
    # global arrays
    mdcm.dealloc_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan and average for fMDCM")

    parser.add_argument("-n", "--nodes_to_avg", help="", required=True, type=int)

    parser.add_argument("-l", "--local_pos", help="", default=None, type=str)
    parser.add_argument(
        "-l2", "--l2", help="lambda coef. for l2 reg.", default=100.0, type=float
    )
    parser.add_argument("-o", "--outdir", help="", default=None, type=str)
    parser.add_argument("-opt", "--opt", help="", default=False, type=bool)
    parser.add_argument(
        "-esp", "--esp", help="format string for esp files", default=None, type=str
    )
    parser.add_argument(
        "-dens",
        "--dens",
        help="format string for density files",
        default=None,
        type=str,
    )

    args = parser.parse_args()
    print(" ".join(f"{k}={v}\n" for k, v in vars(args).items()))

    if args.local_pos is not None:
        local = pd.read_pickle(args.local_pos)
    else:
        local = None

    # print(local)
    i = args.nodes_to_avg

    mdcm_clcl = "/home/unibas/boittier/water_kern/water.dcm"
    mdcm_xyz = "/home/unibas/boittier/water_kern/refined.xyz"

    if args.esp is not None:
        esp = args.esp
    else:
        esp = "/data/unibas/boittier/data/gaussian_{}_water_pbe0_dz_esp.cube"

    if args.dens is not None:
        dens = args.dens
    else:
        dens = "/data/unibas/boittier/data/gaussian_{}_water_pbe0_dz_dens.cube"

    mdcm_ = mdcm_set_up(
        [esp.format(i)],
        [dens.format(i)],
        mdcm_cxyz=mdcm_xyz,
        mdcm_clcl=mdcm_clcl,
        local_pos=local,
    )

    clcl = mdcm_.mdcm_clcl

    if args.opt:
        print("Optimizing")
        outname = esp.format(i).split("/")[-1]
        optimize_mdcm(mdcm_, clcl, args.outdir, outname, l2=args.l2)
    else:
        rmse = mdcm_.get_rmse()
        print("RMSE:", i, rmse)
