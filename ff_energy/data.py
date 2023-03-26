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
# "pickles/water_cluster/pbe0_dz.pc"

def load_pickles(path):
    output = []
    for x in Path(path).glob("*pickle"):
        a = read_from_pickle(x)
        a = next(a)
        output.append(a)
    return output
        
def unload_data(output):
    ctot = pd.concat([_["coloumb_total"] for _ in output])
    chm_df = pd.concat([_["charmm"] for _ in output])
    monomers_df = pd.concat([_["monomers_sum"] for _ in output])
    cluster_df = pd.concat(list(itertools.chain([_["cluster"] for _ in output if len(_["cluster"]) > 0])))
    data = pd.concat([ctot,chm_df,monomers_df,cluster_df],axis=1)
    return data, ctot, chm_df, monomers_df, cluster_df

def plot_ecol(data):
    data = data.dropna()
    data = data[data["ECOL"] < -50]
    fit = plot_energy_MSE(data, "ECOL", "ELEC", 
                    elec="ECOL", CMAP="plasma",
                   xlabel="Coulomb integral [kcal/mol]",
                   ylabel="CHM ELEC [kcal/mol]")
    
class Data:
    def __init__(self,output_path):
        self.output_path = output_path
        self.output = load_pickles(output_path)
        data, ctot, chm_df, monomers_df, cluster_df = unload_data(self.output)
        self.data = data
        self.data["intE"] = (data["C_ENERGY"] - data["M_ENERGY"])*H2KCALMOL
        self.data["NBONDS"] = data["ELEC"] + data["VDW"]
        self.ctot = ctot
        self.chm_df = chm_df
        self.monomers_df = monomers_df
        self.cluster_df = cluster_df
        
    def data():
        return self.data.copy()

