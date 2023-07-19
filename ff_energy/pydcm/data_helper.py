import numpy as np
import pandas as pd
import cclib
import ase
from ase.visualize import view

filename = "/home/boittier/Documents/phd/ff_energy/cubes/dcm/nms/test_nms_0_0.xyz.out"


def get_cclib_data(filename):
    data = cclib.io.ccread(filename)
    return data


def prepare_paired_df(ds: pd.DataFrame,
                      csv_dict: dict):
    key1s = []
    key2s = []
    alphas = []
    lambdas = []
    split = []
    rmse1 = []
    rmse2 = []

    # ds["uuid"] =

    for _ in list(set(ds["uuid"])):
        row = ds[ds["uuid"] == _]
        alphas.append(row["alpha"].mean())
        lambdas.append(row["l2"].mean())
        l = list(row["key"])

        if l[0].__contains__("kernel_"):
            key1s.append(l[0])
            key2s.append(l[1])
            TEST1 = csv_dict[l[0]][csv_dict[l[0]]["class"] == "test"]
            TEST2 = csv_dict[l[1]][csv_dict[l[1]]["class"] == "test"]
        else:
            key1s.append(l[1])
            key2s.append(l[0])
            TEST1 = csv_dict[l[1]][csv_dict[l[1]]["class"] == "test"]
            TEST2 = csv_dict[l[0]][csv_dict[l[0]]["class"] == "test"]

        assert len(TEST1) == len(TEST2)
        split.append(len(TEST1))
        rmse1.append(TEST1[TEST1["rmse"] != 0]["rmse"].median())
        rmse2.append(TEST2[TEST2["rmse"] != 0]["rmse"].median())

    ALPHA = "$\\alpha$"
    LAMBDA = "$\lambda$"
    ds_paired = pd.DataFrame({"key1": key1s,
                              "key2": key2s,
                              "split": split,
                              "rmse_kernel": rmse1,
                              "rmse_opt": rmse2,
                              ALPHA: alphas,
                              LAMBDA: lambdas})

    return ds_paired


def prepare_dataframe(csv_dict: dict):
    l2s = []
    alphas = []
    keys = []
    rmses = []
    evaluated = []
    uuids = []

    def finduuid(x):
        spl = x.split("_")
        for _ in spl:
            if len(_) == 36:
                return _

    for k, v in csv_dict.items():
        if "_standard_" not in k:
            evaluated.append(k.split("_")[0])
            l2s.append(v["l2"].mean())
            alphas.append(v["alpha"].mean())
            rmses.append(v["rmse"].median())
            keys.append(k)
            uuids.append(finduuid(k))

    ds = pd.DataFrame(
        {
            "key": keys,
            "l2": l2s,
            "alpha": alphas,
            "rmse": rmses,
            "class": evaluated,
            "uuid": uuids
        }
    )
    def name_(x):
        if "kernel" in x:
            return "kernel"
        elif "opt" in x:
            return "opt"
        else:
            return "standard?"

    ds["_"] = ds["key"].apply(lambda x: name_(x))
    return ds

def read_global_charges(filename):
    with open(filename) as f:
        lines = f.readlines()
    Nchg = int(lines[0].split()[0])
    charges = np.zeros((Nchg, 4))
    for i in range(Nchg):
        charges[i, :] = [float(x) for x in
                         lines[i + 2].split()[1:]]
    RMSE = None
    for _ in lines:
        if "RMSE" in _:
            RMSE = float(_.split()[1])

    return charges, RMSE
