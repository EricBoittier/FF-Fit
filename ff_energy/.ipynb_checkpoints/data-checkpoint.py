import pandas as pd
from pathlib import Path
import pickle
import itertools
from ff_energy import plot_energy_MSE

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
    print("loading pickles from ", path)
    for x in Path(path).glob("*pickle"):
        a = read_from_pickle(x)
        a = next(a)
        output.append(a)
    # print(output)
    return output


def validate_data(_):
    # print(_)
    if len(_) == 0:
        _ = None
    else:
        _ = pd.concat(_)
    return _


def unload_data(output):
    # print(output)
    _ = [_["coloumb_total"] for _ in output
         if _["coloumb_total"] is not None]
    ctot = validate_data(_)


    chm_df = validate_data([_["charmm"] for _ in output
                        if _["charmm"] is not None])

    _ = [_["monomers_sum"] for _ in output
                             if _["monomers_sum"] is not None]
    monomer_df = validate_data(_)
    _ = list(itertools.chain([_["cluster"] for _ in output
                                                 if len(_["cluster"]) > 0]))
    cluster_df = validate_data(_)
    
    _ = [_["pairs"] for _ in output
         if _["pairs"] is not None]
    pairs_df = validate_data(_)
    
    _ = [_["monomers"] for _ in output
         if _["monomers"] is not None]
    monomers_df = validate_data(_)    
    
    data = pd.DataFrame()
    for df in [ctot,chm_df,monomer_df,cluster_df]:
        # print(df)
        data = pd.concat([data,df],axis=1)
        # print(data)

    # print(data.keys())
    return data, ctot, chm_df, monomers_df, cluster_df, pairs_df, monomers_df


def plot_ecol(data):
    data = data.dropna()
    data = data[data["ECOL"] < -50]
    plot_energy_MSE(data, "ECOL", "ELEC", 
                    elec="ECOL", CMAP="plasma",
                   xlabel="Coulomb integral [kcal/mol]",
                   ylabel="CHM ELEC [kcal/mol]")



class Data:
    def __init__(self,output_path):
        self.output_path = output_path
        self.output = load_pickles(output_path)
        data, ctot, chm_df, monomers_df, cluster_df, pairs_df, monomer_df = unload_data(self.output)
        self.data = data
        # self.data = self.data.loc[:, ~self.data.columns.duplicated()].copy()
        if "M_ENERGY" in data.keys() and cluster_df is not None and monomers_df is not None:
            self.data["intE"] = (data["C_ENERGY"] - data["M_ENERGY"])*H2KCALMOL

        self.ctot = ctot
        self.chm_df = chm_df
        self.monomers_df = monomers_df
        self.monomer_df = monomer_df
        self.cluster_df = cluster_df
        self.pairs_df = pairs_df
        
        index = self.monomers_df.index
        self.monomers_df["key"] = [x.split("_")[0] for x in index]
        self.monomers_df["monomer"] = [int(x.split("_")[1]) for x in index]
        
        index = list(self.pairs_df.index)
        self.pairs_df["key"] = [x.split("_")[0] for x in index]
        self.pairs_df["pair"] = [(int(x.split("_")[1]),
                                   int(x.split("_")[2])) for x in index]
        print("pairs_df", self.pairs_df)
        try:
            self.pairs_df["p_m_ENERGY"] = self.pairs_df.apply(lambda x: self.pair_monomer_E(x), axis=1)

            self.pairs_df["p_int_ENERGY"] = self.pairs_df["p_ENERGY"] - self.pairs_df["p_m_ENERGY"]
            
            self.data["n_pairs"] = [len(self.pairs_df[self.pairs_df["key"] == x]) for x in self.data.index]
            
            self.data["p_intE"] = [self.pairs_df[self.pairs_df["key"] == x]["p_int_ENERGY"].sum()*627.5 for x in self.data.index]
            
        except Exception as e:
            print(e)
        
    def pair_monomer_E(self,x):
        pair = x.pair
        key = x.key
        a,b = pair
        _m = self.monomers_df[self.monomers_df["key"] == key]
        if len(_m) < 190:
            try:
                aE = _m[_m["monomer"] == a]["m_ENERGY"]
                bE = _m[_m["monomer"] == b]["m_ENERGY"]
                # print(aE,bE)
                return float(aE) + float(bE)
            except:
                pass
        return None
        
    def data(self):
        return self.data.copy()

    def plot_intE(self):
        if self.chm_df is not None:
            self.data["NBONDS"] = self.data["ELEC"] + self.data["VDW"]
            self.data["nb_intE"] = self.data["ELEC"] + self.data["VDW"]
            _ = self.data[self.data["ECOL"] < -40].copy()
            plot_energy_MSE(_, "intE", "nb_intE",
                    elec="ECOL", CMAP="viridis",
                   xlabel="intE [kcal/mol]",
                   ylabel="NBONDS [kcal/mol]")
        else:
            print("Data not available")

    def plot_ecol(self):
        plot_ecol(self.data)



