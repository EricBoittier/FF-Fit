from ff_energy.pydcm.dcm_fortran import dcm_fortran
import numpy as np
# Basics
import os
import subprocess
import time
from pathlib import Path, PosixPath
import pickle
import pandas as pd
import argparse
# Optimization
from scipy.optimize import minimize

bohr_to_a = 0.529177

DCM_PY_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "dcm.py"
HOME_PATH = Path(os.path.expanduser("~"))
FFE_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parents[1]
print("DCMPY:", DCM_PY_PATH)
print("HOME:", HOME_PATH)
print("FFE:", FFE_PATH)

mdcm = dcm_fortran

espform = FFE_PATH / "cubes/dcm/nms/" \
          "test_nms_0_0.xyz_esp.cube"
densform = FFE_PATH / "cubes/dcm/nms/" \
           "test_nms_0_0.xyz_dens.cube"

scan_fesp = [espform]
scan_fdns = [densform]

mdcm_cxyz = FFE_PATH / "ff_energy/pydcm/sources/" \
    "dcm8.xyz"
mdcm_clcl = FFE_PATH / "ff_energy/pydcm/sources/" \
            "dcm.mdcm"

local_pos = None


def mdcm_set_up(scan_fesp, scan_fdns,
                mdcm_cxyz=None,
                mdcm_clcl=None,
                local_pos=None):
    # convert PosixPath to string
    if isinstance(scan_fesp, PosixPath):
        scan_fesp = str(scan_fesp)
    if isinstance(scan_fdns, PosixPath):
        scan_fdns = str(scan_fdns)
    if isinstance(mdcm_cxyz, PosixPath):
        mdcm_cxyz = str(mdcm_cxyz)
    if isinstance(mdcm_clcl, PosixPath):
        mdcm_clcl = str(mdcm_clcl)

    # Load MDCM global and local files
    mdcm.dealloc_all()
    Nfiles = len(scan_fesp)
    Nchars = int(np.max([
        len(str(filename)) for filelist in [scan_fesp, scan_fdns]
        for filename in filelist]))
    esplist = np.empty([Nfiles, Nchars], dtype='c')
    dnslist = np.empty([Nfiles, Nchars], dtype='c')
    for ifle in range(Nfiles):
        esplist[ifle] = "{0:{1}s}".format(str(scan_fesp[ifle]), Nchars)
        dnslist[ifle] = "{0:{1}s}".format(str(scan_fdns[ifle]), Nchars)
    # Load cube files, read MDCM global and local files
    mdcm.load_cube_files(Nfiles, Nchars, esplist.T, dnslist.T)
    if mdcm_clcl is not None:
        mdcm.load_clcl_file(mdcm_clcl)
    if mdcm_cxyz is not None:
        mdcm.load_cxyz_file(mdcm_cxyz)
    if local_pos is not None:
        mdcm.set_clcl(local_pos)
    # Get and set global MDCM array (to check if manipulation is possible)
    cxyz = mdcm.mdcm_cxyz
    mdcm.set_cxyz(cxyz)
    # Write MDCM global from local and Fitted ESP cube files
    mdcm.write_cxyz_files()
    mdcm.write_mdcm_cube_files()
    return mdcm



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


def optimize_mdcm(mdcm, clcl, outdir, outname, l2=100.0):
    # Get RMSE, averaged or weighted over ESP files,
    # or per ESP file each
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
            l2diff = l2 * np.sum((local_pos - local_ref) ** 2) \
                     / local_pos.shape[0]
            # print(rmse, l2diff)
            rmse += l2diff
        return rmse

    # Apply simple minimization without any feasibility check (!)
    # Leads to high amplitudes of MDCM charges and local positions
    res = minimize(
        mdcm_rmse, local_pos,
        method="L-BFGS-B",
        options={'disp': None, 'maxls': 20, 'iprint': -1, 'gtol': 1e-06,
                 'eps': 1e-09, 'maxiter': 15000,
                 'ftol': 1e-8, 'maxcor': 10, 'maxfun': 15000})
    print(res)
    # Recompute final RMSE each
    rmse = mdcm.get_rmse()
    print(rmse)
    mdcm.write_cxyz_files()
    #  get the local charges array after optimization
    clcl_out = get_clcl(res.x, charges)
    difference = np.sum((res.x - local_ref) ** 2) \
                 / local_pos.shape[0]
    print("charge RMSD:", difference)
    outname = f"{outname}_l2_{l2:.1e}_rmse_{rmse:.4f}_rmsd_{difference:.4f}"
    obj_name = f"{FFE_PATH}/" \
               f"cubes/clcl/{l2}/{outname}_clcl.obj"
    #  save as pickle
    with open(obj_name, 'wb') as filehandler:
        pickle.dump(clcl_out, filehandler)

    # Not necessary but who knows when it becomes important to deallocate all
    # global arrays
    mdcm.dealloc_all()


def eval_kernel(clcls, esp_path, dens_path,
                load_pkl=False, opt=False, l2=100.0, verbose=False):
    rmses = []
    commands = []
    N = len(clcls)
    path__ = f"{FFE_PATH}/ff_energy/cubes/clcl/{l2}"
    print("path:", path__)
    for i in range(N):
        ESP_PATH = esp_path[i]
        DENS_PATH = dens_path[i]
        job_command = f'python {DCM_PY_PATH} -esp {ESP_PATH} -dens {DENS_PATH}' \
                        f' -mdcm_clcl ' \
                      f'{FFE_PATH}/ff_energy/pydcm/sources/' \
                      f'dcm.mdcm -mdcm_xyz {FFE_PATH}/' \
                      f'ff_energy/pydcm/sources/dcm8.xyz '
        if load_pkl:
            job_command += f' -l {clcls[i]}'
        if opt:
            job_command += f' -opt True -l2 {l2} ' \
                           f'-o {FFE_PATH}/cubes/clcl/{l2}'
            Path(path__
                 ).mkdir(parents=True, exist_ok=True)
        # print(job_command)
        commands.append(job_command)

    procs = []
    for command in commands:
        p = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        procs.append(p)
        time.sleep(0.3)

    for p in procs:
        p.wait()
        result = p.stdout.readlines()
        if verbose:
            for _ in result:
                print(_)
        if not opt:
            rmse_out = float(result[-1].split()[-1])
        else:
            rmse_out = float(result[-2].split()[-1])
        rmses.append(rmse_out)

    return rmses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scan and average for fMDCM')
    parser.add_argument('-n', '--nodes_to_avg', help='', required=False,
                        type=int)
    parser.add_argument('-l', '--local_pos', help='', default=None, type=str)
    parser.add_argument('-l2', '--l2', help='lambda coef. for l2 reg.',
                        default=100.0, type=float)
    parser.add_argument('-o', '--outdir', help='', default=None, type=str)
    parser.add_argument('-opt', '--opt', help='', default=False, type=bool)
    parser.add_argument('-esp', '--esp', help='format string for esp files',
                        default=None, type=str)
    parser.add_argument('-dens', '--dens', help='format string for density files',
                        default=None, type=str)
    parser.add_argument('-mdcm_clcl', '--mdcm_clcl', help='mdcm clcl file',
                        default=None, type=str)
    parser.add_argument('-mdcm_xyz', '--mdcm_xyz', help='mdcm xyz file',
                        default=None, type=str)

    args = parser.parse_args()
    print(' '.join(f'{k}={v}\n' for k, v in vars(args).items()))

    if args.local_pos is not None:
        local = pd.read_pickle(args.local_pos)
    else:
        print("WARNING: No local positions specified")
        local = None

    if args.mdcm_clcl is not None:
        mdcm_clcl = args.mdcm_clcl
    else:
        print("WARNING: No MDCM clcl file specified")

    if args.mdcm_xyz is not None:
        mdcm_xyz = args.mdcm_xyz
    else:
        mdcm_xyz = None
        print("WARNING: No MDCM xyz file specified")
    if args.esp is not None:
        esp = args.esp
    else:
        raise ValueError("No ESP file specified")

    if args.dens is not None:
        dens = args.dens
    else:
        raise ValueError("No density file specified")

    if args.outdir is not None:
        outdir = args.outdir
    else:
        print("WARNING: No output directory specified")

    if args.nodes_to_avg is not None:
        i = args.nodes_to_avg
    else:
        i = None
        print("WARNING: No nodes to average specified")

    if args.nodes_to_avg is not None:
        ESPF = [esp.format(i)]
        DENSF = [dens.format(i)]
    else:
        ESPF = [esp]
        DENSF = [dens]

    #print(ESPF)
    #print(DENSF)
    mdcm = mdcm_set_up(ESPF, DENSF,
                       mdcm_cxyz=mdcm_xyz,
                       mdcm_clcl=mdcm_clcl,
                       local_pos=local)

    clcl = mdcm.mdcm_clcl

    if args.opt:
        print("Optimizing")
        outname = esp.format(i).split("/")[-1]
        optimize_mdcm(mdcm, clcl, args.outdir, outname, l2=args.l2)
    else:
        rmse = mdcm.get_rmse()
        print("RMSE:", i, rmse)

