from pydcm import Scan, DCM, dcm_fortran

import numpy as np

file_mdcm_clcl = "/home/boittier/Documents/phd/ff_energy/pydcm_/source/water_pc.dcm"

# Write dictionary for scan class
config_scan = {
    "system_label"              : "water_ccsdt_atz",
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
    "scan_qm_method"            : "CCSD(T)",
    "scan_qm_basis_set"         : "aug-cc-pVTZ",
    "scan_constrained_opt"      : False,
    "scan_parallel_tasks"       : 10,
    "scan_cpus_per_task"        : 4,
    "scan_memory_per_task"      : 1600,
    "scan_overwrite"            : False,
    "scan_time_check_tasks"     : 10 # check if the job has completed every 10s
    }

# file_config_scan = "config_scan_scn.txt"

# # Initialize scan class either by:
# scan = Scan()

# Prepare some cube file list
scan_fesp = [
    "/home/boittier/Documents/phd/ff_energy/pydcm_/data/gaussian_{:d}_{:s}_esp.cube".format(ii, config_scan["system_label"]) for ii in range(180)]
scan_fdns = [
    "/home/boittier/Documents/phd/ff_energy/pydcm_/data/gaussian_{:d}_{:s}_dens.cube".format(ii, config_scan["system_label"]) for ii in range(180)]

Nfiles = 180 #len(scan_fesp)
Nchars = int(np.max([
    len(filename) for filelist in [scan_fesp, scan_fdns] 
    for filename in filelist]))

esplist = np.empty([Nfiles, Nchars], dtype='c')
dnslist = np.empty([Nfiles, Nchars], dtype='c')

for ifle in range(1):
    esplist[ifle] = "{0:{1}s}".format(scan_fesp[ifle], Nchars)
    dnslist[ifle] = "{0:{1}s}".format(scan_fdns[ifle], Nchars)

# Load cube files, read MDCM global and local files
print("loading cube files")
dcm_fortran.load_cube_files(Nfiles, Nchars, esplist.T, dnslist.T)
print("loading clcl")
dcm_fortran.load_clcl_file(file_mdcm_clcl)


