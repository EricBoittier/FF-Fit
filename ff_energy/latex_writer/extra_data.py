from pathlib import Path
import pandas as pd

""" Extra Data
"""

extra_data_path = Path("/home/boittier/Documents/phd/ff_energy/notebooks/working/data/")

mikes_data = open(extra_data_path / "mike_elec_dcm.txt", "r").readlines()
data = [_.split("\t") for _ in mikes_data]
index = [_[0] for _ in data]
elec = [float(_[1]) for _ in data]
dcm_elec_ = pd.DataFrame({"ELEC": elec}, index=index)
dcm_elec_ = dcm_elec_.drop("100_3050_DCM_355_1200")

mikes_data = open(extra_data_path / "mike_pol_dcm_no_intern.txt", "r").readlines()
data = [_.split() for _ in mikes_data]
index = [_[0] for _ in data]
elec = [float(_[1]) for _ in data]
dcm_pol_no_intern = pd.DataFrame({"ELEC": elec}, index=index)
dcm_pol_no_intern = dcm_pol_no_intern.drop("100_3050_DCM_355_1200")