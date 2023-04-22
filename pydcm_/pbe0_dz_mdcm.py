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
    "system_label"              : "water_pbe0_dz",
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

file_mdcm_poly = "source/water_poly_fixc.dcm"

file_mdcm_clcl = "source/water.dcm"


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

df.to_csv(f"{scan.syst_ltag}.csv", index=False)

#exit()


# Define simple charge constrained function returning RMSE for given local MDCM
# configuration
# Number of polynomial coefficients
Npoly = dcm_fortran.npoly

# RMSE weighting
potlist = np.array(potlist)
weight_list = 1.0 + ((potlist - potlist[scan_smin])*27.211385)**4

a02A = 0.52917721067
x01 = np.arange(0.0, 1.01, 0.1)
x01_2 = x01**2
x01_3 = x01_2*x01

def mdcm_rmse(
    clcl,
    qtot=config_scan["system_total_charge"], qmax=[-1.0, +1.0], qfor=1.0e4,
    lrng=[-1.0, +1.0], lfor=1.0e4):
    clcl_m = [clcl[0], 0.0, clcl[1], clcl[2], \
          clcl[3], 0.0, clcl[4], clcl[5],
          clcl[6], clcl[7], clcl[8], clcl[9], \
          clcl[6], -clcl[7], clcl[8], clcl[9], \
          -clcl[0], 0.0, clcl[1], clcl[2], \
          -clcl[3], 0.0, clcl[4], clcl[5]          
          ]
    dcm_fortran.set_clcl(clcl_m)
    rmse = dcm_fortran.get_rmse()
    wrmse = dcm_fortran.get_rmse_weighted(Nfiles, list(weight_list))
    srmse = dcm_fortran.get_rmse_each(Nfiles)
    print(clcl) 
    print(wrmse, rmse)
    print(srmse[::10], np.min(srmse), np.max(srmse))
    
    return wrmse
    
# Apply simple minimization without any feasibility check (!)
# Leads to high amplitudes of MDCM charges and local positions
from scipy.optimize import minimize

clcl = dcm_fortran.mdcm_clcl

clcl = [ 0.1341137118,             -0.3768455083,     0.5225510325,
     -0.0439006885,                 0.3975768770,     0.0948963129,
      0.0144935335,  0.3455139784, -0.0111132466,     -0.6174473453]

xyz_max = 0.55
xyz_min = -1 * xyz_max

bounds = [(xyz_min,xyz_max),(xyz_min,xyz_max),(-1,1),(xyz_min,xyz_max), \
          (xyz_min,xyz_max),(-1,1),(xyz_min,xyz_max),(xyz_min,xyz_max), \
          (xyz_min,xyz_max),(-1,1)]

def qtot0(x):
    return x[2] + x[5] + x[9]

cons = {"type": "eq", "fun": qtot0}

print("clcl", clcl)

res = minimize(mdcm_rmse, clcl, constraints=cons, bounds=bounds, tol=1e-5)
print(res)

# Recompute final RMSE each
srmse = dcm_fortran.get_rmse_each(Nfiles)
print(srmse)

#print(dcm_fortran.mdcm_clcl)
poly = dcm_fortran.mdcm_poly
#for i in range(0, len(poly), 3*Npoly + 1):
    #print(poly[i:(i + Npoly)]*0.52917721067)
    #print(poly[(i + Npoly):(i + 2*Npoly)]*0.52917721067)
    #print(poly[(i + 2*Npoly):(i + 3*Npoly)]*0.52917721067)
    #print(poly[(i + 3*Npoly)])

dcm_fortran.write_cxyz_files()
dcm_fortran.write_mdcm_cube_files()

clcl = res.x

clcl_m = [clcl[0]*a02A, 0.0, clcl[1]*a02A, clcl[2], \
          clcl[3]*a02A, 0.0, clcl[4]*a02A, clcl[5],
          clcl[6]*a02A, clcl[7]*a02A, clcl[8]*a02A, clcl[9], \
          clcl[6]*a02A, -clcl[7]*a02A, clcl[8]*a02A, clcl[9], \
          -clcl[0]*a02A, 0.0, clcl[1]*a02A, clcl[2], \
          -clcl[3]*a02A, 0.0, clcl[4]*a02A, clcl[5]          
          ]


#clcl_m = np.array(clcl_m) * a02A


dcm_str = """1 1          ! no. residue types defined here

LIG        ! residue name
1          ! no. axis system frames
 2  1  3 BO  ! atom indices involved in frame  1
2  0          ! no. chgs and polarizable sites for atom  2
     {:.6f}  {:.6f}  {:.6f} {:.6f}
     {:.6f}  {:.6f}  {:.6f} {:.6f}
2  0          ! no. chgs and polarizable sites for atom  1
     {:.6f}  {:.6f}  {:.6f} {:.6f}
     {:.6f}  {:.6f}  {:.6f} {:.6f}
2  0          ! no. chgs and polarizable sites for atom  3
     {:.6f}  {:.6f}  {:.6f} {:.6f}
     {:.6f}  {:.6f}  {:.6f} {:.6f}
"""




print(dcm_str.format(*clcl_m))

# Not necessary but who knows when it become important to deallocate all 
# global arrays
dcm_fortran.dealloc_all()

