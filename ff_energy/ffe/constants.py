from pathlib import Path

FFEPATH = Path(__file__).parent.parent.parent


clusterBACH = ("ssh", "boittier@pc-bach")
clusterBEETHOVEN = ("ssh", "boittier@beethoven")
clusterNCCR = ("ssh", "boittier@pc-nccr-cluster")
CLUSTER_DRIVE = {
    "boittier@pc-bach": "/home/boittier/pcbach",
    "boittier@beethoven": "/home/boittier/homeb",
    "boittier@pc-nccr-cluster": "/home/boittier/pcnccr",
}

CONFIG_PATH = "/home/boittier/Documents/phd/ff_energy/configs/"
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
}
