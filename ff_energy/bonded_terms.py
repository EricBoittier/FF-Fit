import numpy as np
from scipy.optimize import minimize


def harmonic(x, k, r0) -> float:
    """
    Harmonic potential
    Kb: kcal/mole/A**2
    :param x:
    :param k:
    :param r0:
    :return:
    """
    return 0.5 * k * (x - r0) ** 2


def morse(x, De, a, r0) -> float:
    """
    Morse potential
    :param x:
    :param De:
    :param a:
    :param r0:
    :return:
    """
    return De * (1 - np.exp(-a * (x - r0))) ** 2


def harmonic_angle(x, k, theta0) -> float:
    """
    Harmonic angle potential
    !Ktheta: kcal/mole/rad**2
    :param x:
    :param k:
    :param theta0:
    :return:
    """
    return 0.5 * k * (x - theta0) ** 2


def waterlike_energy_interal(kb, ka, r0, a0, a, r1, r2):
    """
    Calculate the energy of a molecule with 2 bonds and 1 angle
    :param kb:
    :param ka:
    :param r0:
    :param a0:
    :param a:
    :param r1:
    :param r2:
    :return:
    """
    E = 0
    E += harmonic(r1, kb, r0)
    E += harmonic(r2, kb, r0)
    E += harmonic_angle(a, ka, a0)
    return E


H2KCALMOL = 627.503


class FitBonded:
    """
    Takes a DF with monomer energies and internal DoFs,
    and fits the parameters of the internal DoFs
    """
    def __init__(self, df, min_m_E):
        self.df = df.copy()
        self.min_m_E = min_m_E
        self.x0 = np.array([1, 1, 1, 2])
        print("Fitting parameters: kb, ka, r0, a0")
        self.res = minimize(self.get_loss, self.x0, tol=1e-6)
        self.x = self.res.x
        self.kb, self.ka, self.r0, self.a0 = self.x
        self.df = self.get_loss_df(self.x)
        self.df["m_ENERGY"] = self.df["m_ENERGY"].astype(np.float64).interpolate()

        sum_monomer_df = self.df.groupby(
            "key", group_keys=True
        ).sum()  # .apply(lambda x: x)
        sum_monomer_df["m_E_tot"] = (
            sum_monomer_df["m_ENERGY"] + self.min_m_E * H2KCALMOL * 20
        )
        sum_monomer_df["p_m_E_tot"] = (
            sum_monomer_df["E_pred"] + self.min_m_E * H2KCALMOL * 20
        )
        self.sum_monomer_df = sum_monomer_df

    def get_loss_df(self, x):
        """Return the loss function applied to a dataframe"""
        df = self.df.copy()
        E_pred = []
        for a, r1, r2 in zip(df["a"], df["r1"], df["r2"]):
            a = np.deg2rad(a)
            E_pred.append(waterlike_energy_interal(*x, a, r1, r2))
        df["E_pred"] = E_pred
        df["m_ENERGY"] = df["m_ENERGY"] * H2KCALMOL
        df["m_ENERGY"] = df["m_ENERGY"] - df["m_ENERGY"].min()
        df["SE"] = (df["E_pred"] - df["m_ENERGY"]) ** 2
        return df

    def get_loss(self, x):
        """Return the loss function"""
        df = self.get_loss_df(x)
        MSE = df["SE"].mean()
        return MSE
