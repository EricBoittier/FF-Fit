import pandas as pd
from pathlib import Path

import jax.numpy as jnp

from ff_energy.utils.ffe_utils import read_from_pickle, str2int, get_structures, \
    pickle_output
from ff_energy.plotting.data_plots import plot_energy_MSE
from ff_energy.ffe.geometry import angle, dist
from ff_energy.ffe.potential import Ecoloumb, LJ, LJ_bound
from ff_energy.ffe.bonded_terms import FitBonded
from ff_energy.ffe.constants import PDB_PATH
from ff_energy.logs.logging import logger
from ff_energy.ffe.ff import FF

import numpy as np

H2KCALMOL = 627.503


def load_pickles(path):
    output = []
    logger.info("loading pickles from ", path)
    pkls = list(Path(path).glob("*.pickle"))
    logger.info("loading pickles from ", pkls)
    for x in pkls:
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
    """
    Unload data
    :param output:
    :return:
    """
    logger.info("Unloading data, output keys: {}".format(output[0].keys()))

    fields = [
        "coloumb_total",
        "coloumb",
        "charmm",
        "monomers_sum",
        "cluster",
        "pairs",
        "pairs_sum",
        "monomers",
    ]
    data = {}
    for field in fields:
        _ = [_[field] for _ in output if _[field] is not None]
        _ = validate_data(_)
        if _ is not None and len(_) > 0 and not all_none(_):
            data[field] = _
        else:
            logger.warning("No data for {}".format(field))

    select_data = {k: v for k, v in data.items() if k not in ["pairs", "monomers"]}
    data_df = concat_dataframes(select_data)

    data["n_pairs"] = [len(x["pairs"]) for x in output if x["pairs"] is not None]

    return data_df, data


def concat_dataframes(dataframes: dict) -> pd.DataFrame:
    """
    Concatenate dataframes
    :param dataframes:
    :return:
    """
    data = pd.DataFrame()
    for k, v in dataframes.items():
        if v is not None:
            v = v.reset_index(drop=True)
            print(k, v)
            print(data)
            data = pd.concat([data, v], axis=1)
    return data


def all_none(_):
    """
    Check if all elements in a list are None
    :param _:
    :return:
    """
    return all([x is None for x in _])


class Data:
    def __init__(self, output_path, system, min_m_E=None):
        self.system = system
        self.output_path = output_path
        self.output = load_pickles(output_path)
        (
            data,
            ddict,
        ) = unload_data(self.output)
        #  Unload the dictionary
        ctot = ddict["coloumb_total"] if "coloumb_total" in ddict.keys() else None
        col = ddict["coloumb"] if "coloumb" in ddict.keys() else None
        chm_df = ddict["charmm"] if "charmm" in ddict.keys() else None
        monomers_df = ddict["monomers_sum"] if "monomers_sum" in ddict.keys() else None
        cluster_df = ddict["cluster"] if "cluster" in ddict.keys() else None
        pairs_df = ddict["pairs"] if "pairs" in ddict.keys() else None
        monomer_df = ddict["monomers"] if "monomers" in ddict.keys() else None
        #  Unload the data
        if self.data is not None:
            self.data = data.drop_duplicates()
            print("data", data.drop_duplicates())
        self.coloumb = col
        if (
                "M_ENERGY" in data.keys()
                and cluster_df is not None
                and monomers_df is not None
        ):
            self.data["intE"] = (
                                        self.data["C_ENERGY"] - self.data["M_ENERGY"]
                                ) * H2KCALMOL

        self.ctot = ctot
        self.chm_df = chm_df
        self.monomers_df = monomers_df  # .dropna()
        self.monomer_df = monomer_df  # .dropna()
        self.cluster_df = cluster_df
        if pairs_df is not None:
            self.pairs_df = pairs_df.drop_duplicates()
        else:
            self.pairs_df = None

        #print(self.monomers_df)

        self.prepare_monomers(min_m_E=min_m_E)

        self.jax_data = None
        self.pair_ff_data = None

    def prepare_monomers(self, min_m_E=None):
        if min_m_E is None:
            min_m_E = -76.359922443809 # water pbe1pbe/augdz
        sum_pairs = None
        if self.monomers_df is not None:
            monomidx = list(self.monomers_df.index)
            self.monomers_df["key"] = [x.split("_")[-1] for x in monomidx]
            self.monomers_df["monomer"] = [
                monomidx.index(x) for x in self.monomers_df.index
            ]
            if self.pairs_df is not None:
                self.pairs_df["key"] = [
                    "_".join(_.split("_")[:-2]) for _ in self.pairs_df.index
                ]
                self.pairs_df["pair"] = [
                    (int(x.split("_")[-2]), int(x.split("_")[-1]))
                    for x in self.pairs_df.index
                ]

                sum_pairs = self.pairs_df.groupby("key")["p_int_ENERGY"].sum()
                print("len(sumpairs)", len(sum_pairs))
            # print(sum_pairs)
            df = self.data
            df = df.loc[:,~df.columns.duplicated()].copy()
            self.data = df
            # print(self.data["KEY"])
            if sum_pairs is not None:
                self.data["P_intE"] = [
                    sum_pairs.loc[i] * 627.5
                    if i in sum_pairs.keys()
                    else None
                    for i in self.data["KEY"]
                ]

            self.structures, self.pdbs = get_structures(self.system,
                                                        pdbpath=PDB_PATH / self.system,
                                                        )
            self.structure_key_pairs = {
                Path(p).stem: s for p, s in zip(self.pdbs, self.structures)
            }
            if min_m_E is None:
                self.min_m_E = self.monomer_df["m_ENERGY"].min()

            if self.system.__contains__("water_cluster"):
                self.add_internal_dof()
                self.bonded_fit = FitBonded(self.monomers_df, self.min_m_E)
                self.data["m_E_tot"] = self.bonded_fit.sum_monomer_df["m_E_tot"]
                self.data["p_m_E_tot"] = self.bonded_fit.sum_monomer_df["p_m_E_tot"]
            else:
                self.bonded_fit = None

            if "C_ENERGY" in self.data.keys():
                self.data["C_ENERGY_kcalmol"] = self.data["C_ENERGY"] * H2KCALMOL

    def __str__(self):
        s = f"{self.system} {self.output_path} {list(self.data.keys())}"
        return s

    def __repr__(self):
        return self.__str__()

    def pair_monomer_E(self, x):
        pair = x.pair
        key = x.key
        a, b = pair
        _m = self.monomers_df[self.monomers_df["key"] == key]
        try:
            aE = _m[_m["monomer"] == a]["m_ENERGY"]
            bE = _m[_m["monomer"] == b]["m_ENERGY"]
            return float(aE) + float(bE)
        except Exception as e:
            logger.warning(e)
            return None

    def data(self) -> pd.DataFrame:
        return self.data.copy()

    def get_internals_water(self, key, res):
        print(key, res)
        mask = self.structure_key_pairs[key].res_mask[res]
        water_mol = self.structure_key_pairs[key].xyzs[mask]
        b1 = water_mol[0] - water_mol[1]
        b2 = water_mol[0] - water_mol[2]
        a = angle(b1, b2)
        r1 = dist(b1)
        r2 = dist(b2)
        return a, r1, r2

    def get_internals_dcm(self, key, res):
        print(key, res)
        mask = self.structure_key_pairs[key].res_mask[res]
        mol = self.structure_key_pairs[key].xyzs[mask]

        # array(['C', 'CL', 'CL', 'H', 'H'], dtype='<U2')
        # bonds
        b1 = mol[0] - mol[1] # C-Cl
        b2 = mol[0] - mol[2] # C-Cl
        b3 = mol[0] - mol[3] # C-H
        b4 = mol[0] - mol[4] # C-H

        r1 = dist(b1)
        r2 = dist(b2)
        r3 = dist(b3)
        r4 = dist(b4)

        a1 = angle(b1, b2) # Cl-C-Cl
        a2 = angle(b1, b3) # Cl-C-H
        a3 = angle(b1, b4) # Cl-C-H
        a4 = angle(b2, b3) # Cl-C-H
        a5 = angle(b2, b4) # Cl-C-H
        a6 = angle(b3, b4) # H-C-H

        return r1, r2, r3, r4, a1, a2, a3, a4, a5, a6

    def add_internal_dof(self, res="water") -> None:

        if res == "water":
            a_s = []
            r1_s = []
            r2_s = []
            for r in self.monomers_df.iterrows():
                key = r[1]["KEY"].split("_")[0] + ".xyz"
                monomer = int(r[1]["key"])

                a, r1, r2 = self.get_internals_water(key, monomer)
                a_s.append(a)
                r1_s.append(r1)
                r2_s.append(r2)

            #  add to dataframe
            self.monomers_df["a"] = a_s
            self.monomers_df["r1"] = r1_s
            self.monomers_df["r2"] = r2_s

        elif res == "dcm":

            a1 = []
            a2 = []
            a3 = []
            a4 = []
            a5 = []
            a6 = []
            r1 = []
            r2 = []
            r3 = []
            r4 = []
            for r in self.monomer_df.iterrows():
                print(r)
                key = "_".join(r[1]["KEY"].split("_")[:-1])
                monomer = int(r[1]["KEY"].split("_")[-1])
                print(key, monomer)
                _r1, _r2, _r3, _r4, _a1, _a2, _a3, _a4, _a5, _a6 = \
                    self.get_internals_dcm(key, monomer)
                a1.append(_a1)
                a2.append(_a2)
                a3.append(_a3)
                a4.append(_a4)
                a5.append(_a5)
                a6.append(_a6)
                r1.append(_r1)
                r2.append(_r2)
                r3.append(_r3)
                r4.append(_r4)

            #  add to dataframe
            self.monomer_df["a1"] = a1
            self.monomer_df["a2"] = a2
            self.monomer_df["a3"] = a3
            self.monomer_df["a4"] = a4
            self.monomer_df["a5"] = a5
            self.monomer_df["a6"] = a6
            self.monomer_df["r1"] = r1
            self.monomer_df["r2"] = r2
            self.monomer_df["r3"] = r3
            self.monomer_df["r4"] = r4
        else:
            raise ValueError("res must be water or dcm")

    def plot_pair_monomer_E(self) -> None:
        _ = self.data[self.data["n_pairs"] == 190].copy()
        _ = _[_["P_intE"] < 1]
        _ = _[_["intE"] < 1]
        logger.info("n:", len(_))

        plot_energy_MSE(
            _,
            "intE",
            "P_intE",
            xlabel="intE [kcal/mol]",
            ylabel="pair_monomer_E [kcal/mol]",
            elec="intE",
            CMAP="viridis",
        )

    def plot_intE(self) -> None:
        if self.chm_df is not None:
            self.data["NBONDS"] = self.data["ELEC"] + self.data["VDW"]
            self.data["nb_intE"] = self.data["ELEC"] + self.data["VDW"]
            _ = self.data[self.data["ECOL"] < -40].copy()
            plot_energy_MSE(
                _,
                "intE",
                "nb_intE",
                xlabel="intE [kcal/mol]",
                ylabel="NBONDS [kcal/mol]",
                elec="ECOL",
                CMAP="viridis",
            )
        else:
            logger.warning("Data not available")


def name_pair():
    pass


def pairs_data(
        dataobject,
        system=None,
        name=None,
        dcm_path_=None,
        dcm_charges_per_res=None,
):
    """    """
    if system is None or name is None or dcm_path_ is None:
        raise ValueError("system, name, and DCM path must be provided")
    logger.info("Preparing pairs_data")

    # structures, pdbs = get_structures(system)
    # structure_key_pairs = {Path(p).stem: s for p, s in zip(pdbs, structures)}

    data = dataobject.pairs_df
    structures = dataobject.structures
    pdbs = dataobject.pdbs
    structure_key_pairs = dataobject.structure_key_pairs

    #  arrays for jax
    dcm_c_idx1 = []  # dcm charge idx1
    dcm_c_idx2 = []  # dcm charge idx2
    dcm_c1s = []  # dcm charges1
    dcm_c2s = []  # dcm charges1
    dcm_pairs_idx = []  # dcm distance idx
    dcm_dists = []  # dcm distances
    dcm_dists_labels = []  # dcm distances labels
    dcm_ecols = []  # dcm ecol
    cluster_labels = []  # cluster labels

    #  loop through the DF by index (slow)
    for idx, fnkey in enumerate(data.index):
        #  key - split by underscore, join
        k = "_".join(fnkey.split("_")[:-2])
        #  a b - keys
        kab = fnkey.split("_")[-2:]
        s = structure_key_pairs[k]
        dcm_path = dcm_path_.format(k)
        s.load_dcm(dcm_path,
                   dcm_charges_per_res=dcm_charges_per_res)
        dcms = np.array(s.dcm_charges)

        # calculate electrostatic energy
        E = 0
        dists = []
        pairs = [(int(kab[0]), int(kab[1]))]
        for pair in pairs:
            p1, p2 = pair
            mask1 = s.res_mask[p1]
            mask2 = s.res_mask[p2]
            if "dcm" in name:
                mask1 = s.dcm_charges_mask[p1]
                mask2 = s.dcm_charges_mask[p2]
            dcm1 = dcms[mask1]
            dcm2 = dcms[mask2]
            dcm1c = dcm1.copy()
            dcm2c = dcm2.copy()
            # dcm1 = dcm1[:, :-1]
            # dcm2 = dcm2[:, :-1]
            #  loop through all combinations of charge pairs
            for i, a in enumerate(dcm1c):
                for j, b in enumerate(dcm2c):
                    dist = np.linalg.norm(a[:-1] - b[:-1])
                    Ec = Ecoloumb(a[-1], b[-1], dist)
                    #  append the distance,c1 and c2
                    dcm_ecols.append(Ec)
                    dcm_c_idx1.append(i)
                    dcm_c_idx2.append(j)
                    dcm_c1s.append(a[-1])
                    dcm_c2s.append(b[-1])
                    dcm_dists.append(dist)
                    dcm_dists_labels.append(idx)
                    dcm_pairs_idx.append(pairs)
                    cluster_labels.append(str2int(k))
                    E += Ec
                    if i == 1 and (j == 0 or j == 2):
                        dists.append(dist)
                    if j == 1 and (i == 0 or i == 2):
                        dists.append(dist)

    #  prepare data for jax as dictionary
    jax_data = {
        "dcm_ecols": jnp.array(dcm_ecols),
        "dcm_c_idx1": jnp.array(dcm_c_idx1),
        "dcm_c_idx2": jnp.array(dcm_c_idx2),
        "dcm_c1s": jnp.array(dcm_c1s),
        "dcm_c2s": jnp.array(dcm_c2s),
        "dcm_pairs_idx": jnp.array(dcm_pairs_idx),
        "dcm_dists": jnp.array(dcm_dists),
        "dcm_dists_labels": jnp.array(dcm_dists_labels),
        "cluster_labels": jnp.array(cluster_labels),
    }

    return jax_data


import pickle


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Manager':
            from ff_energy.latex_writer.energydata.energy_data_report import \
                EnergyReport
            return EnergyReport
        return super().find_class(module, name)


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd
    from ff_energy.latex_writer.energydata.energy_data_report import EnergyReport
    from ff_energy.utils.ffe_utils import read_from_pickle, pickle_output
    from ff_energy.ffe.constants import PKL_PATH, PDB_PATH
    from ff_energy.ffe.data import pairs_data

    pkl_path = "/home/boittier/Documents/phd/ff_energy/pickles/energy_report.pkl"
    er = pd.read_pickle(pkl_path)
    eg_dcm_path_ = "/home/boittier/homeb/dcm/pbe0dz_mdcm/{}/charmm/dcm.xyz"
    data = er.data_plots[1].obj
    pairs_out = pairs_data(data,
                           system="dcm",
                           name="mdcm",
                           dcm_path_=eg_dcm_path_,
                           #  some_n_charges_per_line
                           dcm_charges_per_res=8
                           )
