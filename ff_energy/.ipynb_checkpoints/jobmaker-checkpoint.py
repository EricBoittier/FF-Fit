import os.path
import os
from pathlib import Path
import subprocess
from multiprocessing.pool import ThreadPool, Pool

# import ipywidgets as widgets
# from tqdm.notebook import tqdm
from itertools import repeat
from tqdm import tqdm
from ff_energy.job import Job
from ff_energy.structure import Structure

atom_types = {
              # ("LIG", "O"): "OG311",
              # ("LIG", "C"): "CG331",
              # ("LIG", "H1"): "HGP1",
              # ("LIG", "H2"): "HGA3",
              # ("LIG", "H3"): "HGA3",
              # ("LIG", "H4"): "HGA3",
              # ("TIP3", "OH2"): "OT",
              # ("TIP3", "H1"): "HT",
              # ("TIP3", "H2"): "HT",
              ("LIG", "O"): "OT",
              ("LIG", "H1"): "HT",
              ("LIG", "H"): "HT",
              ("LIG", "H2"): "HT",
              }

def get_structures_pdbs(PDBPATH, atom_types=atom_types,
                        system_name=None):
    structures = []
    pdbs = [_ for _ in os.listdir(PDBPATH) if _.endswith("pdb")]
    for p in pdbs:
        s_path = PDBPATH / p
        s = Structure(s_path, atom_types=atom_types, system_name=system_name)
        s.set_2body()
        structures.append(s)
        
    return structures, pdbs


class JobMaker:
    def __init__(self, jobdir, pdbs, structures, kwargs):
        self.pdbs = pdbs
        self.jobdir = jobdir
        self.structures = structures
        self.kwargs = kwargs
        self.molpro_jobs = {}
        self.charmm_jobs = {}
        self.coloumb_jobs = {}
        self.data = []

    def loop(self, func, args, **kwargs):
        # Create a thread pool
        pool = Pool()
        # Start the jobs
        pool.starmap(func,
                     tqdm(zip(self.pdbs, self.structures, repeat(args)),
                          total=len(self.pdbs)),
                    )
        # Close the pool and wait for the work to finish
        pool.close()
        pool.join()

    def get_cluster_jobs(self, homedir):
        jobs = []
        for p in self.pdbs:
            ID = p.split(".")[0]
            slurm_path = f"{homedir}/{self.jobdir}/{ID}/cluster/{ID}.sh"
            outfilename = f"{homedir}/{self.jobdir}/{ID}/cluster/{ID}.out"
            if os.path.exists(outfilename):
                if open(outfilename).read().find("Molpro calculation terminated") == -1:
                    jobs.append(slurm_path)
            else:
                jobs.append(slurm_path)
        return jobs
    
    def get_monomer_jobs(self, homedir):
        jobs = []
        for p in self.pdbs:
            ID = p.split(".")[0]
            monomers_path = f"{homedir}/{self.jobdir}/{ID}/monomers/"
            out_jobs = Path(monomers_path).glob(f"{ID}*sh")
            # print(monomers_path)
            # print(list(out_jobs))
            for outfilename in out_jobs:
                keep = True
                of = Path(str(outfilename)[:-3]+".out")
                if of.exists():
                    if open(of).read().find("terminated"):
                        # print("**", outfilename, of)
                        keep = False
                    else:
                        # print("-", outfilename, of)
                        keep = True
                else:
                    # slurm_path = outfilename[:-4]+".sh
                    # print(outfilename, of)
                    keep = True
                if keep:
                    jobs.append(outfilename)
        return jobs

    def get_pairs_jobs(self, homedir):
        jobs = []
        for p in self.pdbs:
            ID = p.split(".")[0]
            monomers_path = f"{homedir}/{self.jobdir}/{ID}/pairs/"
            out_jobs = Path(monomers_path).glob(f"{ID}*sh")
            # print(monomers_path)
            # print(list(out_jobs))
            for outfilename in out_jobs:
                keep = True
                of = Path(str(outfilename)[:-3]+".out")
                if of.exists():
                    if open(of).read().find("terminated"):
                        keep = False
                    else:
                        keep = True
                else:
                    keep = True
                if keep:
                    jobs.append(outfilename)
        return jobs
    
    def get_coloumb_jobs(self, homedir):
        jobs = []
        for p in self.pdbs:
            ID = p.split(".")[0]
            monomers_path = f"{homedir}/{self.jobdir}/{ID}/coloumb/"
            out_jobs = Path(monomers_path).glob(f"{ID}*sh")
            # print(monomers_path)
            # print(list(out_jobs))
            for outfilename in out_jobs:
                keep = True
                of = Path(str(outfilename)[:-3]+".py.out")
                if of.exists():
                    if open(of).read().find("kcal"):
                        # print("**", outfilename, of)
                        keep = False
                    else:
                        # print("-", outfilename, of)
                        keep = True
                else:
                    # slurm_path = outfilename[:-4]+".sh
                    # print(outfilename, of)
                    keep = True
                if keep:
                    jobs.append(outfilename)
        return jobs
    
    def get_charmm_jobs(self, homedir):
        jobs = []
        for p in self.pdbs:
            ID = p.split(".")[0]
            slurm_path = f"{homedir}/{self.jobdir}/{ID}/charmm/{ID}.slurm"
            jobs.append(slurm_path)
        return jobs

    def esp_view(self, homedir, chmdir):
        chmdir = f"{chmdir}/""{}/charmm/"
        self.loop(self.make_esp_view, [homedir, chmdir])
    
    def gather_data(self, homedir, monomers, clusters, pairs, coloumb, charmm):
        monomers = f"{monomers}/""{}/monomers/"
        clusters = f"{clusters}/""{}/cluster/"
        coloumb = f"{coloumb}/""{}/coloumb/"
        charmm = f"{charmm}/""{}/charmm/"
        pairs = f"{pairs}/""{}/pairs/"
        self.loop(self.make_data_job, [homedir, monomers, clusters,pairs,coloumb,charmm])

    def make_charmm(self, homedir):
        self.loop(self.make_charmm_job, homedir)

    def make_molpro(self, homedir):
        self.loop(self.make_molpro_job, homedir)
        
    def make_coloumb(self, homedir, mp):
        self.loop(self.make_coloumb_job, [homedir, mp])
        
    def make_data_job(self, p, s, args):
        homedir, mp, cp, p_p, c_p, chm_p = args
        if isinstance(homedir, tuple):
            homedir = homedir[0]
        # print(p)
        ID = p.split(".")[0]
        j = Job(ID, f"{homedir}/{self.jobdir}/{ID}", s, kwargs=self.kwargs)
        o = j.gather_data(monomers_path=Path(mp.format(ID)),
                          cluster_path=Path(cp.format(ID)),
                          pairs_path=Path(p_p.format(ID)),
                          coloumb_path=Path(c_p.format(ID)), 
                          chm_path=Path(chm_p.format(ID)))
        

    def make_charmm_job(self, p, s, homedir):
        #  check if the homedir is a tuple
        if isinstance(homedir, tuple):
            homedir = homedir[0]
        # print(p)
        ID = p.split(".")[0]
        j = Job(ID, f"{homedir}/{self.jobdir}/{ID}", s, kwargs=self.kwargs)
        j.generate_charmm()
        self.charmm_jobs[ID] = j

    def make_esp_view(self, p, s, args):
        homedir, charm_path = args
        ID = p.split(".")[0]
        j = Job(ID, f"{homedir}/{self.jobdir}/{ID}", s, kwargs=self.kwargs)
        j.generate_esp_view(charmm_path=charm_path.format(ID))
        
    def make_coloumb_job(self, p, s, args):
        homedir, mp = args
        ID = p.split(".")[0]
        j = Job(ID, f"{homedir}/{self.jobdir}/{ID}", s, kwargs=self.kwargs)
        j.generate_coloumb_interactions(monomers_path=Path(mp.format(ID)))

    def make_molpro_job(self, p, s, homedir):
        #  clear the terminal
        # subprocess.call("clear")
        # print(p)
        if isinstance(homedir, tuple):
            homedir = homedir[0]

        ID = p.split(".")[0]
        # print(f"{homedir}/{self.jobdir}/{ID}")
        j = Job(ID, f"{homedir}/{self.jobdir}/{ID}", s, kwargs=self.kwargs)
        j.generate_molpro()
        self.molpro_jobs[ID] = j
