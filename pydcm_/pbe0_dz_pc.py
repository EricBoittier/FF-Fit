#!/usr/bin/python

# Basics
import os
import numpy as np

# Load FDCM modules
from pydcm import Scan, DCM, dcm_fortran

from itertools import product
import pandas as pd

# Write dictionary for scan class
config_scan = {
    "system_label"              : "water_pbe0_adz",
    "system_file_coord"         : "source/water.xyz",
    "system_file_type"          : "xyz",
    "system_total_charge"       : 0,
    "system_spin_multiplicity"  : 1,
    "scan_dofs"                 : [
        [0, 1],
        [0, 2],
        [1, 0, 2]
        ],
    "scan_steps"                : [
        [0.909, 0.959, 1.009],
        [0.909, 0.959, 1.009],
        np.append(np.arange(84.45,104.45,2.0), np.arange(104.45,124.45,2.0))
        ],
    "scan_qm_program"           : "Gaussian",
    "scan_qm_method"            : "PBE1PBE",
    "scan_qm_basis_set"         : "aug-cc-pVDZ",
    "scan_constrained_opt"      : False,
    "scan_parallel_tasks"       : 10,
    "scan_cpus_per_task"        : 4,
    "scan_memory_per_task"      : 1600,
    "scan_overwrite"            : False,
    "scan_time_check_tasks"     : 10 # check if the job has completed every 10s
    }

file_config_scan = "config_scan_scn.txt"

# Initialize scan class either by:
scan = Scan()
scan.initialize_scan(config_scan)
# or direct:
#scan = Scan(config=config_scan)

# After initialization, a config file is written that can be used to read config
#scan.initialize_scan(file_config_scan)

# Print documentation of config dictionary
scan.print_doc()

# Optional: Prepare all input files for scan to check for correct creation
scan.prepare_scan()

# Execute scan (prepare_scan, submit jobs, evaluate_scan) according to 
# config_scan definition
scan.execute_scan()

# Optional: Evaluate results, already done at the end of execute_scan
scan.evaluate_scan()

# Get final list of ESP and density cube files
dnslist = scan.get_files_cube_dens()
esplist = scan.get_files_cube_esp()
potlist = scan.get_potential()

# Identify energetically lowest scan step
scan_smin = np.nanargmin(scan.get_potential())

#exit()


#------------------

# MDCM:
#-------

#file_mdcm_poly = "source/water_poly_fixc.dcm"

file_mdcm_clcl = "source/water_pc.dcm"


# Prepare some cube file list
scan_fesp = [
    "data/gaussian_{:d}_{:s}_esp.cube".format(ii, config_scan["system_label"]) for ii in range(180)]
scan_fdns = [
    "data/gaussian_{:d}_{:s}_dens.cube".format(ii, config_scan["system_label"]) for ii in range(180)]

Nfiles = len(scan_fesp)
Nchars = int(np.max([
    len(filename) for filelist in [scan_fesp, scan_fdns] 
    for filename in filelist]))

esplist = np.empty([Nfiles, Nchars], dtype='c')
dnslist = np.empty([Nfiles, Nchars], dtype='c')

for ifle in range(Nfiles):
    esplist[ifle] = "{0:{1}s}".format(scan_fesp[ifle], Nchars)
    dnslist[ifle] = "{0:{1}s}".format(scan_fdns[ifle], Nchars)

# Load cube files, read MDCM global and local files
dcm_fortran.load_cube_files(Nfiles, Nchars, esplist.T, dnslist.T)
dcm_fortran.load_clcl_file(file_mdcm_clcl)
#dcm_fortran.load_cxyz_file(file_mdcm_cxyz)
#dcm_fortran.load_poly_file(file_mdcm_poly)

# Write MDCM global from local and Fitted ESP cube files
dcm_fortran.write_cxyz_files()
dcm_fortran.write_mdcm_cube_files()

## Get and set local MDCM array (to check if manipulation is possible)
#clcl = dcm_fortran.mdcm_clcl
#dcm_fortran.set_clcl(clcl)
#clcl = dcm_fortran.mdcm_clcl
#poly = dcm_fortran.mdcm_poly
#dcm_fortran.set_poly(poly)
poly = dcm_fortran.mdcm_poly

# Get and set global MDCM array (to check if manipulation is possible)
#cxyz = dcm_fortran.mdcm_cxyz
#dcm_fortran.set_cxyz(cxyz)
#cxyz = dcm_fortran.mdcm_cxyz

# Get RMSE, averaged or weighted over ESP files, or per ESP file each
rmse = dcm_fortran.get_rmse()
print(rmse)
wrmse = dcm_fortran.get_rmse_weighted(Nfiles, [1.]*Nfiles)
print(wrmse)
srmse = dcm_fortran.get_rmse_each(Nfiles)
print(srmse)


for i, scanstep in enumerate(scan.scan_grid):
   print(i, scanstep, srmse[i])


df = pd.DataFrame({"idx": list(range(len(scan.scan_grid))),
                   "r1": [_[0] for _ in scan.scan_grid],
                   "r2": [_[1] for _ in scan.scan_grid],
                   "a2": [_[2] for _ in scan.scan_grid],
                   "E": potlist*627.503,
                   "rmse": srmse })

df["Erel"] = df["E"] - df["E"].min()

df.to_csv(f"{scan.syst_ltag}_PC.csv", index=False)

#exit()


# Define simple charge constrained function returning RMSE for given local MDCM
# configuration
# Number of polynomial coefficients
#Npoly = dcm_fortran.npoly

# RMSE weighting
potlist = np.array(potlist)
weight_list = 1.0 + ((potlist - potlist[scan_smin])*27.211385)**4

#fix_q = []
#for i in range(3*Npoly, len(poly), 3*Npoly + 1):
#    fix_q.append(poly[i])
#fix_q = None

# Fixed charges
#resolution = np.array([
     #0,  1,  2,  3,
     #0,  4,  5,  6,
     #7,  8,  9, 10,
     #0,
     #0, 11, 12, 13,
     #0, 14, 15, 16,
    #17, 18, 19, 20,
     #0,
     #0, 21, 22, 23,
     #0, 24, 25, 26,
    #27, 28, 29, 30,
     #0,
     #0, 21, 22, 23,
     #0,-24,-25,-26,
    #27, 28, 29, 30,
     #0,
     #0, -1, -2, -3,
     #0,  4,  5,  6,
     #7,  8,  9, 10,
     #0,
     #0,-11,-12,-13,
     #0, 14, 15, 16,
    #17, 18, 19, 20,
     #0,
    #], dtype=int)


# Flexible charges
resolution = np.array([
     0,  1,  2,  3,
     0,  4,  5,  6,
     7,  8,  9, 10,
    11,
     0, 12, 13, 14,
     0, 15, 16, 17,
    18, 19, 20, 21,
    22,
     0, 23, 24, 25,
     0, 26, 27, 28,
    29, 30, 31, 32,
    33,
     0, 23, 24, 25,
     0,-26,-27,-28,
    29, 30, 31, 32,
    33,
     0, -1, -2, -3,
     0,  4,  5,  6,
     7,  8,  9, 10,
    11,
     0,-12,-13,-14,
     0, 15, 16, 17,
    18, 19, 20, 21,
    22,
    ], dtype=int)


#poly_red = np.array(poly)[resolution!=0]

a02A = 0.52917721067
x01 = np.arange(0.0, 1.01, 0.1)
x01_2 = x01**2
x01_3 = x01_2*x01

local_pos = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
charg_array = np.array([0,0,0,-0.834, 0,0,0,0.417, 0,0,0,0.417])

def get_clcl(local_pos, charges):
    NCHARGES = len(charges)
    _clcl_ = np.zeros(NCHARGES)
    for i in range(NCHARGES):
        if (i + 1) % 4 == 0:
            _clcl_[i] = charges[i]
        else:
            _clcl_[i] = local_pos[i - ((i) // 4)]
    return _clcl_

def mdcm_rmse(charges,
    qtot=config_scan["system_total_charge"], qmax=[-1.0, +1.0], qfor=1.0e4,
    lrng=[-1.0, +1.0], lfor=1.0e4):
    
    clcl = [0,0,0,charges[1],0,0,0,charges[0],0,0,0,charges[1]] 
    dcm_fortran.set_clcl(clcl)
    rmse = dcm_fortran.get_rmse()
    wrmse = dcm_fortran.get_rmse_weighted(Nfiles, list(weight_list))
    srmse = dcm_fortran.get_rmse_each(Nfiles)
    
    
    print(wrmse, rmse)
    print(srmse[::10], np.min(srmse), np.max(srmse))
    
    return wrmse
    
# Apply simple minimization without any feasibility check (!)
# Leads to high amplitudes of MDCM charges and local positions
from scipy.optimize import minimize

def qtot0(x):
    return x[0] + x[1] + x[1]

cons = {"type": "eq", "fun": qtot0}

res = minimize(mdcm_rmse, [-0.84, 0.417], constraints=cons, tol=1e-5)
print(res)

# Recompute final RMSE each
srmse = dcm_fortran.get_rmse_each(Nfiles)
print(srmse)

print(dcm_fortran.mdcm_clcl)

dcm_fortran.write_cxyz_files()
dcm_fortran.write_mdcm_cube_files()

# Not necessary but who knows when it become important to deallocate all 
# global arrays
dcm_fortran.dealloc_all()

