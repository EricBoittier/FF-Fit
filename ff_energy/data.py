import pandas as pd
from pathlib import Path
import pickle
import itertools
from ff_energy.plot import plot_energy_MSE

H2KCALMOL = 627.503

def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            pass


def load_pickles(path):
    output = []
    for x in Path(path).glob("*pickle"):
        a = read_from_pickle(x)
        a = next(a)
        output.append(a)
    return output


def validate_data(_):
    if len(_) == 0:
        _ = None
    else:
        _ = pd.concat(_)
    return _


def unload_data(output):
    _ = [_["coloumb_total"] for _ in output
         if _["coloumb_total"] is not None]
    ctot = validate_data(_)


    chm_df = pd.concat([_["charmm"] for _ in output
                        if _["charmm"] is not None])

    _ = [_["monomers_sum"] for _ in output
                             if _["monomers_sum"] is not None]
    monomers_df = validate_data(_)
    _ = list(itertools.chain([_["cluster"] for _ in output
                                                 if len(_["cluster"]) > 0]))
    cluster_df = validate_data(_)
    data = pd.concat([ctot,chm_df,monomers_df,cluster_df],axis=1)

    return data, ctot, chm_df, monomers_df, cluster_df


def plot_ecol(data):
    data = data.dropna()
    data = data[data["ECOL"] < -50]
    fit = plot_energy_MSE(data, "ECOL", "ELEC", 
                    elec="ECOL", CMAP="plasma",
                   xlabel="Coulomb integral [kcal/mol]",
                   ylabel="CHM ELEC [kcal/mol]")


def plot_intE(data):
    # data = data.dropna()
    # data = data[data["ECOL"] < -50]
    fit = plot_energy_MSE(data, "intE", "nb_intE",
                    elec="ECOL", CMAP="viridis",
                   xlabel="intE [kcal/mol]",
                   ylabel="NBONDS [kcal/mol]")


class Data:
    def __init__(self,output_path):
        self.output_path = output_path
        self.output = load_pickles(output_path)
        data, ctot, chm_df, monomers_df, cluster_df = unload_data(self.output)
        self.data = data
        self.data = self.data.loc[:, ~self.data.columns.duplicated()].copy()
        if cluster_df is not None:
            self.data["intE"] = (data["C_ENERGY"] - data["M_ENERGY"])*H2KCALMOL
        if chm_df is not None:
            self.data["NBONDS"] = data["ELEC"] + data["VDW"]
            self.data["nb_intE"] = data["ELEC"] + data["VDW"]
        self.ctot = ctot
        self.chm_df = chm_df
        self.monomers_df = monomers_df
        self.cluster_df = cluster_df
        
    def data(self):
        return self.data.copy()

    def plot_intE(self):
        plot_intE(self.data)

    def plot_ecol(self):
        plot_ecol(self.data)



