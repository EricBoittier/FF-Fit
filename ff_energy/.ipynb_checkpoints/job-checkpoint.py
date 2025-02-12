import os
from pathlib import Path
from ff_energy import molpro_job_template, m_slurm_template, orbkit_ci_template, o_slurm_template, \
    c_slurm_template, c_job_template, PAR, molpro_pol_template, orbkit_pol_template

from ff_energy import esp_view_template, vmd_template, g_template
from shutil import copy
import pandas as pd

h2kcalmol = 627.5095

"""
#  job types
#  molpro
   -  cluster energies
   -  dimer energies
   -  monomer energies
#  charmm
   -  create DCM positions, ref charmm energies
#  orbkit
   -  ref electrostatics (2body)
   -  ref polarization energies
"""

CHM_FILES_PATH = Path("/home/boittier/Documents/phd/ff_energy/ff_energy/charmm_files")

M_to_G = {"gdirect;\n{ks,pbe0}": "PBE1PBE",
          "hf":"hf",
          "avdz": "aug-cc-pVDZ"}
# aug-cc-pVDZ
class Job:
    def __init__(self, name, path, structure, kwargs=None):
        self.name = name
        # self.type = type
        self.path = Path(path)
        # print(self.path)

        self.make_dir(self.path)

        self.structure = structure
        self.slurm_files = {"cluster": {},
                            "pairs": {},
                            "monomers": {},
                            "coloumb": {},
                            "polarization_molpro": {},
                            "polarization_orbkit": {},
                            "charmm": {}
                            }
        self.job_files = None
        self.slurm_templates = None
        self.job_templates = None
        self.charmm_path = self.path / "charmm"
        self.coloumb_path = self.path / "coloumb"
        self.polarization_path = self.path / "polarization"
        self.cluster_path = self.path / "cluster"
        self.pairs_path = self.path / "pairs"
        self.monomers_path = self.path / "monomers"
        self.esp_view_path = self.path / "esp_view"

        if kwargs is None:
            kwargs = {"m_nproc": 1, "m_memory": 480, "m_queue": "short", "m_basis": "6-31g", "m_method": "hf",
                      "chmpath": "/home/boittier/dev-release-dcm/build/cmake/charmm",
                      "modules": "module load cmake/cmake-3.23.0-gcc-11.2.0-openmpi-4.1.3",
                      "c_files": ["poly_hf.dcm"],
                      "c_dcm_command": "open unit 11 card read name poly_hf.dcm \nDCM FLUX 11 IUDCM 11 TSHIFT XYZ 15",
                      }
        self.kwargs = kwargs

    def make_dir(self, path):
        """Make a directory if it doesn't exist"""
        if not os.path.exists(path):
            os.makedirs(path)

    def generate_molpro(self, cluster=True, pairs=True, monomers=True):
        if cluster:
            self.generate_cluster()
        if pairs:
            self.generate_pairs()
        if monomers:
            self.generate_monomers()

    def generate_orbkit(self, ci=True, pol=True):
        if ci:
            self.generate_coloumb_interactions()
        if pol:
            self.generate_polarization()

    def generate_charmm(self):

        #  move pdb file
        self.make_dir(self.charmm_path)
        copy(self.structure.path, self.charmm_path)
        if self.kwargs["c_files"] is not None:
            for file in self.kwargs["c_files"]:
                copy(CHM_FILES_PATH / file, self.charmm_path)
            dcm_command = self.kwargs["c_dcm_command"]

        charmm_job = c_job_template.render(
            NAME=self.name,
            PAR=PAR, 
            PDB=self.structure.name, 
            DCM_COMMAND=dcm_command,
            PSF=self.structure.get_psf()
        )
        charmm_file = self.path / "charmm" / f"{self.name}.inp"
        with open(charmm_file, "w") as f:
            f.write(charmm_job)

        command = f"$charmm < {charmm_file.name} > {charmm_file.name}.out"
        slurm_str = c_slurm_template.render(NAME=self.name, COMMAND=command,
                                            CHARMMPATH=self.kwargs["chmpath"], MODULES=self.kwargs["modules"], )
        slurm_file = self.path / "charmm" / f"{self.name}.slurm"
        with open(slurm_file, "w") as f:
            f.write(slurm_str)
            self.slurm_files["charmm"][self.name] = slurm_file

    def generate_coloumb_interactions(self, monomers_path=None):

        self.make_dir(self.coloumb_path)
        pairs = self.structure.get_pairs()
        monomers = self.structure.get_monomers()

        if monomers_path is None:
            monomers_path = self.monomers_path

        for monomers in monomers:
            wfpath = monomers_path / f"{self.name}_{monomers}.molden"
            try:
                copy(wfpath, self.coloumb_path)
            except Exception as e:
                print(e)
                pass

        for pair in pairs:
            orbkit_str = orbkit_ci_template.render(M1=f"{self.name}_{pair[0]}.molden",
                                                   M2=f"{self.name}_{pair[1]}.molden")
            orbkit_job = self.path / "coloumb" / f"{self.name}_{pair[0]}_{pair[1]}.py"
            with open(orbkit_job, "w") as f:
                f.write(orbkit_str)
            command = f"export LIBCINTDIR=/opt/cluster/programs/libcint/lib64 \n python {orbkit_job.name} > {orbkit_job.name}.out"

            slurm_job = o_slurm_template.render(NAME=f"{self.name}_{pair[0]}_{pair[1]}",
                                                COMMAND=command)
            slurm_file = self.path / "coloumb" / f"{self.name}_{pair[0]}_{pair[1]}.sh"
            # print(slurm_file)
            with open(slurm_file, "w") as f:
                f.write(slurm_job)
                self.slurm_files["coloumb"][f"{self.name}_{pair[0]}_{pair[1]}"] = slurm_file

    def load_dcm(self, dcm_path=None):
        if dcm_path is None:
            dcm_path = self.path / "charmm" / "dcm.xyz"

        self.structure.load_dcm(dcm_path)

    def generate_esp_view(self, charmm_path=None):
        if charmm_path is None:
            charmm_path = self.charmm_path
        if type(charmm_path) is str:
            charmm_path = Path(charmm_path)

        print(self.esp_view_path)
        self.make_dir(self.esp_view_path)
        # get the cluster dcm coordinates
        self.structure.load_dcm(charmm_path / "dcm.xyz")
        dcm_charges = self.structure.dcm_charges
        xyz_s = self.structure.get_cluster_xyz()
        g_theory = M_to_G[self.kwargs["m_method"]]
        g_basis = M_to_G[self.kwargs["m_basis"]]
        # generate gaussian input
        g_com = g_template.render(XYZ_STRING=xyz_s,
                                  METHOD=g_theory,
                                  BASIS=g_basis,
                                  KEY=self.name)
        g_com_file = self.esp_view_path / f"{self.name}.com"
        with open(g_com_file, "w") as f:
            f.write(g_com)

        dcm_file = self.esp_view_path / "dcm.xyz"
        with open(dcm_file, "w") as f:
            f.write(f"{len(dcm_charges)}\n\n")
            for c in dcm_charges:
                f.write(f"X {c[0]} {c[1]} {c[2]} {c[3]}\n")


        # generate vmd script
        vmd = vmd_template.render(KEY=self.name,
                                  ERROR_CUBE="error.cube",
                                  DENS_CUBE=f"{self.name}_dens.cube",)
        vmd_file = self.esp_view_path / f"{self.name}.vmd"
        with open(vmd_file, "w") as f:
            f.write(vmd)

        # generate slurm file
        esp_view = esp_view_template.render(KEY=self.name,
                                            NCHG=len(dcm_charges))
        esp_view_file = self.esp_view_path / f"{self.name}.sh"
        with open(esp_view_file, "w") as f:
            f.write(esp_view)
            # self.slurm_files["esp_view"][self.name] = esp_view_file


    def generate_polarization(self, nch_per_monomer=(6, 6), monomers_path=None):

        self.make_dir(self.polarization_path)
        #  collect the molden file of the monomer
        #  and
        #  create a lattice file with the dcm charges of all other monomers
        self.load_dcm()
        monomers = self.structure.get_monomers()
        if monomers_path is None:
            monomers_path = self.monomers_path

        for monomer in monomers:
            wfpath = monomers_path / f"{self.name}_{monomer}.molden"
            copy(wfpath, self.polarization_path)
            #  make a lattice file
            lat_fn = f"{self.name}_{monomer}.lat"
            start_end_dcm = sum([nchg for i, nchg in enumerate(nch_per_monomer) if i < monomer - 1])
            end_start_dcm = start_end_dcm + nch_per_monomer[monomer - 1]
            dcm_charges = self.structure.dcm_charges[0:start_end_dcm]
            dcm_charges.extend(self.structure.dcm_charges[end_start_dcm:])
            dcm_str = f"HEADER\n{len(dcm_charges)}\n"
            for x, y, z, charge in dcm_charges:
                dcm_str += f"{x}, {y}, {z}, {charge}, 1\n"
            lattice_path = self.polarization_path / lat_fn
            with open(lattice_path, "w") as f:
                f.write(dcm_str)

            #  create molpro job
            molpro_job = molpro_pol_template.render(JOBNAME=f"{self.name}_{monomer}",
                                                    BASIS=self.kwargs["m_basis"],
                                                    RUN=self.kwargs["m_method"],
                                                    LAT=lat_fn,
                                                    XYZ=self.structure.get_monomer_xyz(monomer))
            molpro_path = self.polarization_path / f"{self.name}_{monomer}.inp"
            with open(molpro_path, "w") as f:
                f.write(molpro_job)

            #  create slurm job
            slurm_job = m_slurm_template.render(NAME=f"{self.name}_{monomer}", NPROC=self.kwargs["m_nproc"])
            slurm_path = self.polarization_path / f"molpro_{self.name}_{monomer}.slurm"
            with open(slurm_path, "w") as f:
                f.write(slurm_job)
                f.write(f"sbatch {self.name}_{monomer}.sh")
                self.slurm_files["polarization_molpro"][f"{self.name}_{monomer}"] = slurm_path

            #  create orbkit job
            orbkit_nof = orbkit_pol_template.render(MOLDEN=f"{self.name}_{monomer}.molden",
                                                    LAT=lat_fn)
            orbkit_path = self.polarization_path / f"{self.name}_{monomer}_NOFIELD.py"
            with open(orbkit_path, "w") as f:
                f.write(orbkit_nof)

            orbkit_qmmm = orbkit_pol_template.render(MOLDEN=f"{self.name}_{monomer}_qmmm.molden",
                                                     LAT=lat_fn)
            orbkit_path = self.polarization_path / f"{self.name}_{monomer}_QMMM.py"
            with open(orbkit_path, "w") as f:
                f.write(orbkit_qmmm)

            COMMAND = f""" export LIBCINTDIR=/opt/cluster/programs/libcint/lib64
python {self.name}_{monomer}_NOFIELD.py > {self.name}_{monomer}_NOFIELD.out
python {self.name}_{monomer}_QMMM.py > {self.name}_{monomer}_QMMM.out
            """
            #  create slurm job
            slurm_job = o_slurm_template.render(NAME=f"{self.name}_{monomer}", COMMAND=COMMAND)
            slurm_path = self.polarization_path / f"{self.name}_{monomer}.sh"
            with open(slurm_path, "w") as f:
                f.write(slurm_job)
                self.slurm_files["polarization_orbkit"][f"{self.name}_{monomer}"] = slurm_path

    def generate_cluster(self):

        self.make_dir(self.cluster_path)
        XYZSTR = self.structure.get_cluster_xyz()
        molpro_job = molpro_job_template.render(XYZ=XYZSTR,
                                                NAME=self.name,
                                                BASIS=self.kwargs["m_basis"],
                                                RUN=self.kwargs["m_method"],
                                                MEMORY=self.kwargs["m_memory"])
        with open(self.cluster_path / f"{self.name}.inp", "w") as f:
            f.write(molpro_job)
        slurm_job = m_slurm_template.render(NAME=self.name, NPROC=self.kwargs["m_nproc"])
        with open(self.cluster_path / f"{self.name}.sh", "w") as f:
            f.write(slurm_job)
            self.slurm_files["cluster"][self.name] = self.cluster_path / f"{self.name}.sh"

    def generate_pairs(self):
        pairs = self.structure.get_pairs()

        self.make_dir(self.pairs_path)
        for pair in pairs:
            XYZSTR = self.structure.get_pair_xyz(*pair)
            NAME = f"{self.name}_{pair[0]}_{pair[1]}"
            molpro_job = molpro_job_template.render(XYZ=XYZSTR,
                                                    NAME=NAME,
                                                    BASIS=self.kwargs["m_basis"],
                                                    RUN=self.kwargs["m_method"],
                                                    MEMORY=self.kwargs["m_memory"])
            with open(self.pairs_path / f"{self.name}_{pair[0]}_{pair[1]}.inp", "w") as f:
                f.write(molpro_job)
            slurm_job = m_slurm_template.render(NAME=NAME, NPROC=self.kwargs["m_nproc"])
            with open(self.pairs_path / f"{self.name}_{pair[0]}_{pair[1]}.sh", "w") as f:
                f.write(slurm_job)
                self.slurm_files["pairs"][NAME] = self.pairs_path / f"{self.name}_{pair[0]}_{pair[1]}.sh"

    def generate_monomers(self):
        monomers = self.structure.get_monomers()

        self.make_dir(self.monomers_path)
        for monomer in monomers:
            XYZSTR = self.structure.get_monomer_xyz(monomer)
            NAME = f"{self.name}_{monomer}"
            molpro_job = molpro_job_template.render(XYZ=XYZSTR,
                                                    BASIS=self.kwargs["m_basis"],
                                                    NAME=NAME,
                                                    RUN=self.kwargs["m_method"],
                                                    MEMORY=self.kwargs["m_memory"])
            with open(self.monomers_path / f"{self.name}_{monomer}.inp", "w") as f:
                f.write(molpro_job)
            slurm_job = m_slurm_template.render(NAME=NAME, NPROC=self.kwargs["m_nproc"])
            with open(self.monomers_path / f"{self.name}_{monomer}.sh", "w") as f:
                f.write(slurm_job)
                self.slurm_files["monomers"][NAME] = self.monomers_path / f"{self.name}_{monomer}.sh"

    def gather_data(self, monomers_path=None, 
                    cluster_path=None, 
                    pairs_path=None,
                    coloumb_path=None,
                    chm_path=None):
        if monomers_path is None:
            monomers_path = self.monomers_path
        if cluster_path is None:
            cluster_path = self.cluster_path
        if pairs_path is None:
            pairs_path = self.pairs_path
        if coloumb_path is None:
            coloumb_path = self.coloumb_path
        if chm_path is None:
            chm_path = self.charmm_path
        
        # print(chm_path)
        # print(coloumb_path)
        # print(monomers_path)
        # print(cluster_path)
        # print(self.charmm_path)

        #  charmm data
        charmm_output = [_ for _ in chm_path.glob("*inp.out") if _.is_file()]
        # print(charmm_output)
        # print(self.charmm_path)
        TOTAL = None
        ELEC = None
        VDW = None
        for output in charmm_output:
            TOTAL = None
            ELEC = None
            VDW = None
            with open(output, "r") as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith("ENER>"):
                    TOTAL = float(line.split()[-1])
                if line.startswith("ENER EXTERN>"):
                    ELEC = float(line.split()[3])
                    VDW = float(line.split()[2])

        charmm_data = {"TOTAL": TOTAL, "ELEC": ELEC, "VDW": VDW, "KEY": self.name}
        charmm_df = pd.DataFrame(charmm_data, index=[self.name])
        # print(charmm_df)

        #  monomers data
        monomers_output = [_ for _ in monomers_path.glob(f"{self.name}*out") if _.is_file()]
        monomers_data = {}
        monomers_df = None
        for m in monomers_output:
            with open(m, "r") as f:
                lines = f.readlines()
            k = m.stem
            try:
                monomers_data[k] = {"m_ENERGY": float(lines[-3].split()[-1]), "KEY": k}
                monomers_df = pd.DataFrame(monomers_data).T
            except Exception as e:
                print("Failed reading monomer data:", m, e)
                monomers_df = None
                
        monomers_sum_df = None 
        if monomers_df is not None and len(monomers_df) > 0:
            monomers_sum_df = pd.DataFrame({"M_ENERGY": [monomers_df["m_ENERGY"].sum()],
                                            "KEY": [self.name]},
                                           index=[self.name])
            # print(len(list(set(self.structure.resids))))
            if len(monomers_df) != len(list(set(self.structure.resids))):
                print("WARNING: number of monomers does not match number of residues")
                monomers_sum_df = None

        #  pairs data
        pairs_output = [_ for _ in pairs_path.glob(f"{self.name}*out") if _.is_file()]
        print('n pairs output',len(pairs_output),pairs_path)
        pairs_data = {}
        for p in pairs_output:
            with open(p, "r") as f:
                lines = f.readlines()
            pairs_data[p.stem] = {"p_ENERGY": float(lines[-3].split()[-1])}
        pairs_df = pd.DataFrame(pairs_data).T

        # cluster data
        cluster_output = [_ for _ in cluster_path.glob(f"{self.name}*out") if _.is_file()]
        cluster_data = {}
        for c in cluster_output:
            with open(c, "r") as f:
                lines = f.readlines()
            try:
                cluster_data[str(c.name).split(".")[0]] = {"C_ENERGY": float(lines[-3].split()[-1])}
            except Exception as e:
                print(f"{self.name}, {e}")

        cluster_df = pd.DataFrame(cluster_data).T
        
        # polarization data
        polarization_data = {}
        polarization_output = [_ for _ in self.polarization_path.glob(f"{self.name}*out")
                               if _.is_file() and "QMMM" in _.name]
        for p in polarization_output:
            with open(p, "r") as f:
                lines = f.readlines()
            k = p.name.strip("_QMMM.out")
            polarization_data[k] = {"QMMM": float(lines[-1].split()[-1]),
                                    "KEY": k}
        polarization_output = [_ for _ in self.polarization_path.glob(f"{self.name}*out")
                               if _.is_file() and "NOF" in _.name]
        for p in polarization_output:
            with open(p, "r") as f:
                lines = f.readlines()
            k = p.name.strip("_NOFIELD.out")
            polarization_data[k]["NOFIELD"] = float(lines[-1].split()[-1])

        df = pd.DataFrame(polarization_data).T
        pol_df = None
        pol_total = None
        if len(df) != 0:
            df["pol"] = (df["QMMM"] - df["NOFIELD"]) * h2kcalmol
            pol_df = df.copy()
            # print(pol_df)
            total_pol = df["pol"].sum()
            # print(total_pol)
            pol_total = pd.DataFrame({"POL": total_pol, "KEY": self.name}, index=[self.name])

        #  coloumb data
        coloumb_output = [_ for _ in coloumb_path.glob(f"{self.name}*out") if _.is_file()]
        coloumb_data = {}
        for c in coloumb_output:
            with open(c, "r") as f:
                lines = f.readlines()
            key = c.name.strip(".py.out")
            try:
                coloumb_data[key] = {"ECOL": float(lines[-1].split()[0]), "KEY": key}
            except IndexError:
                coloumb_data[key] = {"ECOL": None, "KEY": key}

        coloumb_df = pd.DataFrame(coloumb_data).T
        coloumb_total = None
        if len(coloumb_df) != 0:
            coloumb_total = coloumb_df["ECOL"].sum()
        coloumb_total = pd.DataFrame({"ECOL": coloumb_total, "KEY": self.name}, index=[self.name])
        
        if "ECOL" in coloumb_df.keys():
            if None in coloumb_df["ECOL"].values:
                coloumb_total = None
        else:
            coloumb_total = None

        output = {"charmm": charmm_df,
                  "monomers": monomers_df,
                    "monomers_sum": monomers_sum_df,
                  "pairs": pairs_df,
                  "cluster": cluster_df,
                  "polarization": pol_df,
                  "pol_total": pol_total,
                  "coloumb": coloumb_df,
                  "coloumb_total": coloumb_total}
        # print(output)
        print(pairs_df)
        self.pickle_output(output)
        return output

    def pickle_output(self,output):
        import pickle
        
        pickle_path = Path(f'pickles/{self.structure.system_name}/{self.kwargs["theory_name"]}/{self.kwargs["c_files"][0]}/{self.name}.pickle')

        pickle_path.parents[0].mkdir(parents=True, exist_ok=True)
        
        with open(pickle_path, 'wb') as handle:
            pickle.dump(output, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)