from pathlib import Path

#  paths
FFEPATH = Path(__file__).parent.parent.parent
CONFIG_PATH = FFEPATH / "configs"
REPORTS_PATH = FFEPATH / "latex_reports"
PKL_PATH = FFEPATH / "pickles"
PDB_PATH = FFEPATH / "pdbs"

#  cluster details
clusterBACH = ("ssh", "boittier@pc-bach")
clusterBEETHOVEN = ("ssh", "boittier@beethoven")
clusterNCCR = ("ssh", "boittier@pc-nccr-cluster")
# drive details
CLUSTER_DRIVE = {
    "boittier@pc-bach": "/home/boittier/pcbach",
    "boittier@beethoven": "/home/boittier/homeb",
    "boittier@pc-nccr-cluster": "/home/boittier/pcnccr",
}

atom_types = {
    ("TIP3", "O"): "OT",
    ("TIP3", "H"): "HT",
    ("UNL", "LI"): "LI",
    ("LIG", "O"): "OT",
    ("LIG", "H1"): "HT",
    ("LIG", "H"): "HT",
    ("LIG", "H2"): "HT",
    ("HOH", "O"): "OT",
    ("HOH", "H"): "HT",
    ("HOH", "H1"): "HT",
    ("HOH", "H2"): "HT",
    ("CLA", "CLA"): "CLA",
    ("POT", "POT"): "POT",

    ("CLA", "Cl"): "Cl",
    ("POT", "K"): "K",
    ("TIP3", "OH2"): "OT",
    ("TIP3", "H1"): "HT",
    ("TIP3", "H2"): "HT",
    ("DCM", "C"): "C",
    ("DCM", "CL1"): "CL",
    ("DCM", "CL2"): "CL",
    ("DCM", "Cl"): "CL",
    ("DCM", "H1"): "H",
    ("DCM", "H2"): "H",
}
