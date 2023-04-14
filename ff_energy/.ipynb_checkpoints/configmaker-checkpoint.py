from ff_energy.config import Config
from ff_energy.config import DCM_STATES #, kDCM_STATES
from pathlib import Path
system_names = ["water_cluster",
           "water_dimer", 
           "methanol_cluster"]

pdbs = ["pdbs/pdbs4/",
"pdbs/dimer3d/",
    "pdbs/pdbsclean/"]

system_types = ["water", 
                "water", 
                "methanol"]

SYSTEMS = {k:{"system_name": k, "pdbs": p, "system_type": s}
           for k,p,s in zip(system_names, pdbs, system_types)}


MODELS = {"water": {
                    "pc": DCM_STATES("pbe0_dz.pc"), 
                    "mdcm": DCM_STATES("pbe0_dz.mdcm"),
                    # "k-mdcm": kDCM_STATES(""),
                    # "f-mdcm": DCM_STATES("")
                   },
         

        "methanol": {
                    "pc": DCM_STATES("meoh_pbe0dz.pc"), 
                    "mdcm": DCM_STATES("meoh_pbe0dz.pc"),
                    # "k-mdcm": kDCM_STATES(""),
                    # "f-mdcm": DCM_STATES("")},
         }
         }
          
THEORY = {
    "hfdz": {"m_basis": "avdz", "m_method": "gdirect;\n{hf}"},
    "hftz": {"m_basis": "avtz", "m_method": "gdirect;\n{hf}"},
    "pbe0dz": {"m_basis": "avdz",
                     "m_method": "gdirect;\n{ks,pbe0}"},
    "pbe0tz": {"m_basis": "avtz",
                        "m_method": "gdirect;\n{ks,pbe0}"},
    "pno-lccsd-pvtzdf": {"m_basis": """basis={
default=aug-cc-pvtz-f12
set,jkfit,context=jkfit
default=aug-cc-pvtz-f12
set,mp2fit,context=mp2fit
default=aug-cc-pvtz-f12
}
! Set wave function properties
wf,spin=0,charge=0
! F12 parameters
explicit,ri_basis=jkfit,df_basis=mp2fit,df_basis_exch=jkfit
! density fitting parameters
cfit,basis=mp2fit
""",
         "m_method": "{df-hf,basis=jkfit}\n{df-mp2-f12,cabs_singles=-1}\n{pno-lccsd(t)-f12}",
                         "m_memory": "500",
                                },
}


ATOM_TYPES = {"water_cluster": {
              ("LIG", "O"): "OT",
              ("LIG", "H1"): "HT",
              ("LIG", "H"): "HT",
              ("LIG", "H2"): "HT",
              },
              "water_dimer": {
              ("TIP3", "OH2"): "OT",
              ("TIP3", "H1"): "HT",
              ("TIP3", "H2"): "HT",
              },
              "methanol_cluster": {
              ("LIG", "O"): "OG311",
              ("LIG", "C"): "CG331",
              ("LIG", "H1"): "HGP1",
              ("LIG", "H2"): "HGA3",
              ("LIG", "H3"): "HGA3",
              ("LIG", "H4"): "HGA3",
              }
             }
          
class ConfigMaker:
    def __init__(self, theory, system, elec):
        
        self.theory_name = theory
        self.system_tag = system
        self.elec = elec
        
        self.theory = THEORY[theory]
        
        self.pdbs = SYSTEMS[system]["pdbs"]
        
        self.system_name = SYSTEMS[system]["system_name"]
        
        self.system_type = SYSTEMS[system]["system_type"]
        self.atom_types = ATOM_TYPES[self.system_name]
        
        self.model = MODELS[self.system_type][elec]
        
        self.kwargs = {**self.theory, 
                       **self.model} 
        self.kwargs["theory_name"] = theory
        
        self.config = Config(**self.kwargs)
        
    def make(self):
        c = Config(**self.kwargs)
        return c
    
    def __repr__(self):
        return f"{self.theory_name}-{self.system_tag}-{self.elec}"

    def write_config(self):
        op = Path(f"/home/boittier/Documents/phd/ff_energy/configs/"
                  f"{self.theory_name}-{self.system_tag}-{self.elec}.config"
        )
        op.parents[0].mkdir(parents=True, exist_ok=True)
        self.config.write_config(op)
    

