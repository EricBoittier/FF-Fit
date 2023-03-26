from ff_energy.config import Config
from ff_energy.config import DCM_STATES #, kDCM_STATES

system_names = ["water_cluster",
           "water_dimer", 
           "methanol_cluster"]

pdbs = ["pdbs/pdbs4/",
"pdbs/dimer3d/",
    "pdbs/pdbsclean/"]

system_types = ["water", 
                "water", 
                "methanol"]

SYSTEMS = {k:{"system_name": k, "pdbs": p, "system_type": s} for k,p,s in zip(system_names, pdbs, system_types)}


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
          
THEORY = {"pbe0dz": {"m_basis": "avdz", 
                     "m_method": "{ks,pbe0}"}}

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
        
        self.theory = THEORY["pbe0dz"]
        
        self.pdbs = SYSTEMS[system]["pdbs"]
        
        self.system_name = SYSTEMS[system]["system_name"]
        
        self.system_type = SYSTEMS[system]["system_type"]
        self.atom_types = ATOM_TYPES[self.system_name]
        
        self.model = MODELS[self.system_type][elec]
        
        self.kwargs = {**self.theory, **self.model} 
        
        self.config = Config(**self.kwargs)
        
    def make(self):
        c = Config(**self.kwargs)
        return c
    
    def __repr__(self):
        return f"{self.theory_name}-{self.system_tag}-{self.elec}"
    
    
    
# cm = ConfigMaker("pbe0dz", "water_cluster", "pc")
# # print(cm.kwargs)
# print(cm.make())

