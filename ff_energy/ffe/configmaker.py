from ff_energy.ffe.config import Config
from ff_energy.ffe.config import DCM_STATES, kMDCM_STATES

from pathlib import Path

system_names = [
    "water_cluster",
    "water_dimer",
    "methanol_cluster",
    "water_tests",
    "lithium",
    "ions",
    "dcm",
    'dcmdimerscan',
    "ions_ext",
]

pdbs = [
    "pdbs4/",
    "dimer3d/",
    "pdbsclean/",
    "water_tests/",
    "lithium/",
    "ions/",
    "dcm/",
    'dcmdimerscan/',
    "ions_ext/",
]

system_types = [
    "water",
    "water",
    "methanol",
    "water",
    "water",
    "water",
    "dcm",
    "dcm",
    "water"
]

SYSTEMS = {
    k: {"system_name": k, "pdbs": p, "system_type": s}
    for k, p, s in zip(system_names, pdbs, system_types)
}

water_kmdcm = [
    "pbe0_dz.kmdcm",
    "water_kern/water.kern",
    "water_kern/x_fit.txt",
    *[f"water_kern/coefs{i}.txt" for i in range(18)],
]

dcm_kmdcm = [
    "dcm_pbe0dz.kmdcm",
    "dcm_kern/dcm.kern",
    "dcm_kern/x_fit.txt",
    *[f"dcm_kern/coefs{i}.txt" for i in range(18)],
]

MODELS = {
    "water": {
        "pc": DCM_STATES("pbe0_dz.pc"),
        "mdcm": DCM_STATES("pbe0_dz.mdcm"),
        "kmdcm": kMDCM_STATES(water_kmdcm),
        "tip3": DCM_STATES("tip3.pc"),
    },
    "methanol": {
        "pc": DCM_STATES("meoh_pbe0dz.pc"),
        "mdcm": DCM_STATES("meoh_pbe0dz.pc"),
    },
    "dcm": {
        "pc": DCM_STATES("dcm_pbe0dz.pc"),
        "mdcm": DCM_STATES("dcm_pbe0dz.mdcm"),
        "kmdcm": kMDCM_STATES("dcm_pbe0dz.kmdcm"),
    },
    "dcmdimerscan": {
        "pc": DCM_STATES("dcm_pbe0dz.pc"),
        "mdcm": DCM_STATES("dcm_pbe0dz.mdcm"),
        "kmdcm": kMDCM_STATES("dcm_pbe0dz.kmdcm"),
    },

    "ions_ext": {
        "pc": DCM_STATES("ions_ext_pbe0dz.pc"),
        "mdcm": DCM_STATES("ions_ext_pbe0dz.mdcm"),
        "kmdcm": kMDCM_STATES(water_kmdcm),
    },
}

THEORY = {
    "hfdz": {"m_basis": "avdz", "m_method": "gdirect;\n{hf}"},
    "hftz": {"m_basis": "avtz", "m_method": "gdirect;\n{hf}"},
    "pbe0dz": {"m_basis": "avdz", "m_method": "gdirect;\n{ks,pbe0}"},
    # disp,d4
    "pbe0dzd4": {"m_basis": "avdz", "m_method": "gdirect;\n{ks,pbe0;disp,d4}"},
    "pbe0tz": {"m_basis": "avtz", "m_method": "gdirect;\n{ks,pbe0}"},
    "pbe0tzd4": {"m_basis": "avtz", "m_method": "gdirect;\n{ks,pbe0;disp,d4}"},
    "pno-lccsd-pvtzdf": {
        "m_basis": """basis={
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
        "m_method": "{df-hf,basis=jkfit}"
                    "\n{df-mp2-f12,cabs_singles=-1}\n{pno-lccsd(t)-f12}",
        "m_memory": "950",
    },
}

ATOM_TYPES = {
    "water_cluster": {
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
    },
    "water_tests": {
        ("HOH", "O"): "OT",
        ("HOH", "H1"): "HT",
        ("HOH", "H2"): "HT",
    },
    "lithium": {
        ("UNL", "LI"): "LI",
        ("TIP3", "OH2"): "OT",
        ("TIP3", "H1"): "HT",
        ("TIP3", "H2"): "HT",
    },
    "ions": {
        ("CLA", "CLA"): "CLA",
        ("POT", "POT"): "POT",
        ("TIP3", "OH2"): "OT",
        ("TIP3", "H1"): "HT",
        ("TIP3", "H2"): "HT",
    },
    "ions_ext": {
        ("CLA", "CLA"): "CLA",
        ("CLA", "Cl"): "CLA",
        ("POT", "POT"): "POT",
        ("POT", "K"): "POT",
        ("TIP3", "OH2"): "OT",
        ("TIP3", "H1"): "HT",
        ("TIP3", "H2"): "HT",
    },
    "dcm": {
        ("DCM", "C"): "C",
        ("DCM", "CL1"): "CL",
        ("DCM", "CL2"): "H",
        ("DCM", "H1"): "H",
        ("DCM", "H2"): "H",
        ("DCM", "H"): "H",
    },
    "dcmdimerscan": {
        ("DCM", "C"): "C",
        ("DCM", "CL1"): "CL",
        ("DCM", "CL2"): "H",
        ("DCM", "H1"): "H",
        ("DCM", "H2"): "H",
        ("DCM", "H"): "H",
    },
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
        self.kwargs = {**self.theory, **self.model, "theory_name": theory}

        self.config = Config(**self.kwargs)

    def make(self):
        c = Config(**self.kwargs)
        return c

    def __repr__(self):
        return f"{self.theory_name}-{self.system_tag}-{self.elec}"

    def write_config(self):
        op = Path(
            f"/home/boittier/Documents/phd/ff_energy/configs/"
            f"{self.theory_name}-{self.system_tag}-{self.elec}.config"
        )
        op.parents[0].mkdir(parents=True, exist_ok=True)
        self.config.write_config(op)
