import pandas as pd
from pathlib import Path
import itertools

import jax.numpy as jnp

from ff_energy.ffe.ffe_utils import read_from_pickle, str2int
from ff_energy.ffe.plot import plot_energy_MSE
from ff_energy.ffe.geometry import dihedral3, bisector, angle, dist
from ff_energy.ffe.ffe_utils import get_structures
from ff_energy.ffe.potential import Ecoloumb
from ff_energy.ffe.bonded_terms import FitBonded

import numpy as np

H2KCALMOL = 627.503


def load_pickles(path):
    output = []
    print("loading pickles from ", path)
    for x in Path(path).glob("*"):
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
    # print(output)
    _ = [_["coloumb_total"] for _ in output if _["coloumb_total"] is not None]
    ctot = validate_data(_)

    _ = [_["coloumb"] for _ in output if _["coloumb"] is not None]
    col = validate_data(_)

    chm_df = validate_data([_["charmm"] for _ in output if _["charmm"] is not None])

    _ = [_["monomers_sum"] for _ in output if _["monomers_sum"] is not None]
    monomer_df = validate_data(_)

    _ = list(itertools.chain([_["cluster"] for _ in output if len(_["cluster"]) > 0]))
    cluster_df = validate_data(_)

    _ = [_["pairs"] for _ in output if _["pairs"] is not None]
    pairs_df = validate_data(_)

    _ = [_["pairs_sum"] for _ in output if _["pairs_sum"] is not None]
    pairs_sum_df = validate_data(_)

    _ = [_["monomers"] for _ in output if _["monomers"] is not None]
    monomers_df = validate_data(_)

    data = pd.DataFrame()
    for df in [ctot, chm_df, monomer_df, pairs_sum_df, cluster_df]:
        data = pd.concat([data, df], axis=1)

    data["n_pairs"] = [len(x["pairs"]) for x in output if x["pairs"] is not None]

    return data, ctot, col, chm_df, monomers_df, cluster_df, pairs_df, monomer_df


def plot_ecol(data):
    data = data.dropna()
    data = data[data["ECOL"] < -50]
    plot_energy_MSE(data, "ECOL", "ELEC", xlabel="Coulomb integral [kcal/mol]",
                    ylabel="CHM ELEC [kcal/mol]", elec="ECOL", CMAP="plasma")


def plot_intE(data):
    data["nb_intE"] = data["ELEC"] + data["VDW"]
    plot_energy_MSE(data, "intE", "nb_intE", xlabel="intE [kcal/mol]",
                    ylabel="NBONDS [kcal/mol]", elec="ELEC", CMAP="viridis")


def plot_LJintE(data, ax=None, elec="ELEC"):
    data["nb_intE"] = data[elec] + data["LJ"]
    data = data.dropna()
    ax = plot_energy_MSE(data, "intE", "nb_intE", xlabel="intE [kcal/mol]",
                         ylabel="NBONDS [kcal/mol]", elec=elec, CMAP="viridis", ax=ax)
    return ax


class Data:
    def __init__(self, output_path, system="water_cluster", min_m_E=None):
        self.system = system
        self.output_path = output_path
        # print("output_path", output_path)
        self.output = load_pickles(output_path)
        # print("output", self.output)
        (
            data,
            ctot,
            col,
            chm_df,
            monomers_df,
            cluster_df,
            pairs_df,
            monomer_df,
        ) = unload_data(self.output)
        self.data = data
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
        self.monomers_df = monomers_df
        self.monomer_df = monomer_df
        self.cluster_df = cluster_df
        self.pairs_df = pairs_df

        if self.monomers_df is not None:
            index = self.monomers_df.index
            self.monomers_df["key"] = [x.split("_")[0] for x in index]
            self.monomers_df["monomer"] = [int(x.split("_")[1]) for x in index]

            index = list(self.pairs_df.index)
            self.pairs_df["key"] = [x.split("_")[0] for x in index]
            self.pairs_df["pair"] = [
                (int(x.split("_")[1]), int(x.split("_")[2])) for x in index
            ]

            sum_pairs = self.pairs_df.groupby("key")["p_int_ENERGY"].sum()
            self.data["P_intE"] = [sum_pairs[i] * 627.5 for i in self.data.index]

            self.structures, self.pdbs = get_structures(self.system)
            self.structure_key_pairs = {
                p.split(".")[0]: s for p, s in zip(self.pdbs, self.structures)
            }

            if min_m_E is None:
                self.min_m_E = self.monomers_df["m_ENERGY"].min()

            self.add_internal_dof()
            self.bonded_fit = FitBonded(self.monomers_df, self.min_m_E)
            self.data["m_E_tot"] = self.bonded_fit.sum_monomer_df["m_E_tot"]
            self.data["p_m_E_tot"] = self.bonded_fit.sum_monomer_df["p_m_E_tot"]
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
        if len(_m) < 190:
            try:
                aE = _m[_m["monomer"] == a]["m_ENERGY"]
                bE = _m[_m["monomer"] == b]["m_ENERGY"]
                # print(aE,bE)
                return float(aE) + float(bE)
            except Exception as e:
                print(e)
                return None
        return None

    def data(self) -> pd.DataFrame:
        return self.data.copy()

    def get_internals_water(self, key, res):
        mask = self.structure_key_pairs[key].res_mask[res]
        water_mol = self.structure_key_pairs[key].xyzs[mask]
        b1 = water_mol[0] - water_mol[1]
        b2 = water_mol[0] - water_mol[2]
        a = angle(b1, b2)
        r1 = dist(b1)
        r2 = dist(b2)
        return a, r1, r2

    def add_internal_dof(self) -> None:
        a_s = []
        r1_s = []
        r2_s = []
        for r in self.monomers_df.iterrows():
            key = r[1]["key"]
            monomer = r[1]["monomer"]
            a, r1, r2 = self.get_internals_water(key, monomer)
            a_s.append(a)
            r1_s.append(r1)
            r2_s.append(r2)
        #  add to dataframe
        self.monomers_df["a"] = a_s
        self.monomers_df["r1"] = r1_s
        self.monomers_df["r2"] = r2_s

    def plot_pair_monomer_E(self) -> None:
        _ = self.data[self.data["n_pairs"] == 190].copy()
        # _ = _[_["n_monomers"] == 20].copy()
        # _ = _[_["ECOL"] < -50].copy()
        # _ = _.dropna(subset=["intE"])
        _ = _[_["P_intE"] < 1]
        _ = _[_["intE"] < 1]
        print("n:", len(_))

        plot_energy_MSE(_, "intE", "P_intE", xlabel="intE [kcal/mol]",
                        ylabel="pair_monomer_E [kcal/mol]", elec="intE", CMAP="viridis")

    def plot_intE(self) -> None:
        if self.chm_df is not None:
            self.data["NBONDS"] = self.data["ELEC"] + self.data["VDW"]
            self.data["nb_intE"] = self.data["ELEC"] + self.data["VDW"]
            _ = self.data[self.data["ECOL"] < -40].copy()
            plot_energy_MSE(_, "intE", "nb_intE", xlabel="intE [kcal/mol]",
                            ylabel="NBONDS [kcal/mol]", elec="ECOL", CMAP="viridis")
        else:
            print("Data not available")

    def plot_ecol(self) -> None:
        plot_ecol(self.data)


def pairs_data(
    data,
    name="PC",
    dcm_path_="/home/boittier/homeb/water_cluster/pbe0dz_pc/{}/charmm/dcm.xyz",
):
    """

    :param data:
    :param dcm_path:
    :return:
    """

    structures, pdbs = get_structures("water_cluster")
    structure_key_pairs = {p.split(".")[0]: s for p, s in zip(pdbs, structures)}

    #  lists for the dataframe
    E_col_dcms = []
    dist_dcms = []
    min_hbond = []
    angles_dcms = []
    dih_dcms = []
    dcms_ = []
    angle_1 = []
    angle_2 = []

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
        k = fnkey.split("_")[0]
        kab = fnkey.split(".")[0]
        s = structure_key_pairs[k]

        dcm_path = dcm_path_.format(k)
        s.load_dcm(dcm_path)
        dcms = np.array(s.dcm_charges)

        # calculate electrostatic energy
        E = 0
        dists = []
        pairs = [(int(kab.split("_")[1]), int(kab.split("_")[2]))]
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
            dcm1 = dcm1[:, :-1]
            dcm2 = dcm2[:, :-1]

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

        CM1 = np.average(dcm1, axis=0)  # , weights=[1,15.99,1])
        CM2 = np.average(dcm2, axis=0)  # , weights=[1,15.99,1])
        distance = np.linalg.norm(CM1 - CM2)

        bisector1 = bisector(dcm1)
        bisector2 = bisector(dcm2)

        dih = dihedral3(
            np.array(
                [dcm1[1, :] + bisector1, dcm1[1, :], dcm2[1, :], dcm2[1, :] + bisector2]
            )
        )

        theta = angle(bisector1, bisector2)

        a1 = angle(dcm1[1, :] + bisector1, dcm1[1, :] + dcm2[1, :])
        a2 = angle(dcm2[1, :] + bisector2, dcm1[1, :] + dcm2[1, :])
        if (60 - a1 - a2) < 0:
            theta = theta * -1

        # append data
        min_hbond.append(min(dists))
        E_col_dcms.append(E)
        dist_dcms.append(distance)
        angles_dcms.append(theta)
        dih_dcms.append(dih)
        angle_1.append(a1)
        angle_2.append(a2)
        dcms_.append([dcm1, dcm2])

    #  add data to the dataframe
    data[f"ECOL_{name}"] = E_col_dcms
    data["angle_1"] = angle_1
    data["angle_2"] = angle_2
    data["dih"] = dih_dcms
    data["theta"] = angles_dcms
    data["distance"] = dist_dcms
    data["min_hbond"] = min_hbond
    data["dcms"] = dcms_

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

    return data, jax_data
