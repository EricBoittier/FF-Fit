import sys
sys.path.append("/home/boittier/Documents/phd/ff_energy")

from ff_energy.structure import Structure
from ff_energy.job import Job
from ff_energy.jobmaker import get_structures_pdbs, JobMaker
from ff_energy.plot import plot_energy_MSE
from ff_energy.configmaker import *

from pathlib import Path
import pandas as pd

# s = Structure("/home/boittier/charmm/mix3/jobs/pdbs/mix0.pdb")

atom_types = {
              ("TIP3", "OH2"): "OT",
              ("TIP3", "H1"): "HT",
              ("TIP3", "H2"): "HT",
              }

def MakeJob(name, ConfigMaker, atom_types=atom_types, system_name=None):
    structures, pdbs = get_structures_pdbs(
        Path(ConfigMaker.pdbs),
        atom_types=atom_types,
        system_name=system_name
    ) 
    return JobMaker(name, pdbs, structures, ConfigMaker.make().__dict__)

def load_config_maker(theory, system, elec):
    cm = ConfigMaker(theory, system, elec)
    return [cm]

def load_all_theory_and_elec():
    CMS = []
    for system in system_names:
        for theory in THEORY.keys():
            for elec in ["pc", "mdcm"]:
                # print(system, theory, elec)
                cm = ConfigMaker(theory, system, elec)
                CMS.append(cm)
    return CMS

def load_all_theory():
    CMS = []
    for system in system_names:
        for theory in THEORY.keys():
            cm = ConfigMaker(theory, system, "pc")
            CMS.append(cm)
            
    return CMS

def charmm_jobs(CMS):
    jobmakers = []
    for cms in CMS:
        print(cms.elec)
        jm = MakeJob(f"{cms.system_name}/{cms.theory_name}_{cms.elec}", cms, 
                     atom_types=cms.atom_types, system_name=cms.system_name)
        HOMEDIR = f"/home/boittier/homeb/"
        PCBACH = f"/home/boittier/pcbach/{cms.system_name}/{cms.theory_name}"
        # jm.gather_data(HOMEDIR, PCBACH, PCBACH)
        jobmakers.append(jm)
    return jobmakers
        
def cluster_submit(cluster,jm,max_jobs=120,Check=True):
    
#     cluster=('ssh', 'boittier@pc-bach')

    from ff_energy.slurm import SlurmJobHandler

    shj = SlurmJobHandler(max_jobs=max_jobs,cluster=cluster)
    print("Running jobs: ", shj.get_running_jobs())

    for jm in jobmakers:
        # for js in jm.get_charmm_jobs(HOMEDIR):
        for js in jm.get_monomer_jobs("/home/boittier/pcbach/"):
            shj.add_job(js)
        for js in jm.get_cluster_jobs("/home/boittier/pcbach/"):
            shj.add_job(js)

    print("Jobs: ", len(shj.jobs))
    # print(shj.jobs)
    print(len(shj.jobs))

    shj.submit_jobs(Check=Check)

def molpro_jobs(CMS):
    jobmakers = []
    for cms in CMS:
        print(cms)
        jm = MakeJob(f"{cms.system_name}/{cms.theory_name}", cms, 
                     atom_types=cms.atom_types,
                    system_name=cms.system_name)
        HOMEDIR = f"/home/boittier/homeb/"
        PCBACH = f"/home/boittier/pcbach/{cms.system_name}/{cms.theory_name}"
        jobmakers.append(jm)
        #print(jm.data)
    return jobmakers
    

def data_jobs(CMS):
    jobmakers = []
    for cms in CMS:
        print(cms)
        jm = MakeJob(f"{cms.system_name}/{cms.theory_name}_{cms.elec}", cms, 
                     atom_types=cms.atom_types,
                    system_name=cms.system_name)
        HOMEDIR = f"/home/boittier/homeb/"
        PCBACH = f"/home/boittier/pcbach/{cms.system_name}/{cms.theory_name}"
        COLOUMB = f"/home/boittier/homeb/{cms.system_name}/{cms.theory_name}"
        CHM = f"/home/boittier/homeb/{cms.system_name}/{cms.theory_name}_{cms.elec}"
        jm.gather_data(HOMEDIR, 
                       PCBACH, # cluster
                       PCBACH, # monomers
                       PCBACH, # pairs
                       COLOUMB, # coloumn
                       CHM) # charmm
        jobmakers.append(jm)
    return jobmakers

def coloumb_jobs(CMS):
    jobmakers = []
    for cms in CMS:
        print(cms)
        jm = MakeJob(f"{cms.system_name}/{cms.theory_name}", cms, 
                     atom_types=cms.atom_types,
                    system_name=cms.system_name)
        HOMEDIR = f"/home/boittier/homeb/"
        PCBACH = f"/home/boittier/pcbach/{cms.system_name}/{cms.theory_name}"
        jm.make_coloumb(HOMEDIR,
                        f"/home/boittier/pcbach/{cms.system_name}/{cms.theory_name}/""{}/monomers")
        jobmakers.append(jm)
        #print(jm.data)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        description='What the program does',
                        epilog='Text at the bottom of help')
    print("----")
    
    # parser.add_argument('filename')           # positional argument
    parser.add_argument('-d', '--data', required=False, default=False, action='store_true')      # option that takes a value
    parser.add_argument('-a', '--all', required=False, default=False, action='store_true') 
    parser.add_argument('-t', '--theory', required=False, default=None)
    parser.add_argument('-m', '--model', required=False, default=None) 
    parser.add_argument('-e', '--elec', required=False, default=None) 
    parser.add_argument('-v', '--verbose',
                        action='store_true')  # on/off flag
    
    CMS = None
    args = parser.parse_args()
    
    if args.all:
        if args.verbose:
            print("Loading all data")
        CMS = load_all_theory_and_elec()
    if args.theory and args.model and args.elec:
        CMS = load_config_maker(args.theory, args.model, args.elec)
    else:
        print("Missing one of args.theory and args.model and args.elec")
        sys.exit(1)
    
    if CMS is not None:
        if args.data:
            if args.verbose:
                print("Gathering Data")        
            jobmakers = data_jobs(CMS)
    else:
        print("No Jobs Found...")
    