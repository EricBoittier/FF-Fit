import pandas as pd
import cclib
import ase
from ase.visualize import view

filename = "/home/boittier/Documents/phd/ff_energy/cubes/dcm/nms/test_nms_0_0.xyz.out"

def get_cclib_data(filename):
    data = cclib.io.ccread(filename)
    return data

def prepare_paired_df(ds: pd.DataFrame, csv_dict: dict):
    key1s = []
    key2s = []
    alphas = []
    lambdas = []
    split = []
    rmse1 = []
    rmse2 = []

    for _ in list(set(ds["_"])):
        row = ds[ds["_"] == _]
        alphas.append(row["alpha"].mean())
        lambdas.append(row["l2"].mean())
        l = list(row["key"])

        if l[0].startswith("kernel_"):
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

    for k, v in csv_dict.items():
        if k != "standard_":
            evaluated.append(k.split("_")[0])
            l2s.append(v["l2"].mean())
            alphas.append(round(v["alpha"].mean(), 3))
            rmses.append(v["rmse"].median())
            keys.append(k)

    ds = pd.DataFrame({"key": keys,
                       "l2": l2s,
                       "alpha": alphas,
                       "rmse": rmses,
                       "class": evaluated})

    ds["_"] = ds["key"].apply(lambda x: x.split("_")[1])
    return ds
