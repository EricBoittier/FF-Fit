import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pint import UnitRegistry
from scipy import stats
import os

ureg = UnitRegistry()
plt.style.use(['science', "no-latex", "nature"])

# methanol_density
#  experimental data
methanol_density = pd.read_csv("csv_data/methanol_density.csv")
methanol_density["Temp. (K)"] = methanol_density["Temp (°C)"] + 273.15

#  Equation of state and thermodynamic properties of
#  liquid methanol from 298 to 489 K and pressures to 1040 bars
# units: 10^3 bar^-1
# kappa
# isothermal compressibility
# isothermal_compress = np.array([0.120, 0.135, 0.158])*0.986923
K = [298.15, 313.15, 333.15]

# alpha
# thermal expansion
# https://pubs.acs.org/doi/pdf/10.1021/je00054a002
# https://pubs.acs.org/doi/full/10.1021/je060415l
# units: 10^3 K^-1
# therm_expan = np.array([1.204, 1.254, 1.332])/10**3
therm_expan = np.array([11.43, 11.66, 11.90, 12.15, 12.40, 12.67, 12.94]) / 10 ** 4
K_alpha = [273.15, 283.15, 293.15, 303.15, 313.15, 323.15, 333.15]

# Diffusion coefficients of methanol and water and
# the mutual diffusion coefficient in methanol-water solutions at
# 278 and 298 K units: 10^-9 m^2 s^-1
D1 = [1.462, 2.190]
K_D1 = [278.15, 298.15]
# https://pubs.acs.org/doi/pdf/10.1021/j100824a520
D2 = [1.26, 1.55, 1.91, 2.32, 2.74, 2.89, 3.88]
K_D2 = [268.2, 278.2, 288.2, 298.2, 308.2, 313.2, 328.2]
D_uncert2 = [0.05, 0.04, 0.01, 0.01, 0.20, 0.03, 0.37, 0.13]

# heat of vaporization Dionisio M.S.; Moura Ramos J.J.;
# Goncalves R.M.: The enthalpy and entropy of cavity formation
# in liquids and Corresponding States Principle.
# Can.J.Chem. 68 (1990) 1937-1949 Matyushov D.V.; Schmid R.:
# Properties of Liquids at the Boiling Point: Equation of State,
# Internal Pressure and Vaporization Entropy.
# Int.J.Phys.Chem.Ber.Bunsen-Ges. 98 (1994) 1590-1595 J/mol --> kcal/mol
H_vap = np.array([37420.00, 35200.00]) * 0.000239006
H_vap_K = [298.15, 337.80]

#  heat capacity
#  https://webbook.nist.gov/cgi/cbook.cgi?ID=C67561&Mask=2
heat_capacity = pd.read_csv("csv_data/heatcapacity_methanol.csv")

heat_capacity["cal mol K"] = \
    (heat_capacity["J mol K"].to_numpy() * ureg("J/(mol*K)")).to("cal/(mol*K)")

Cp_alpha = [273.15, 283.15, 293.15, 303.15, 313.15, 323.15, 333.15]
Cp = np.array([2391.1, 2439.5, 2494.1, 2555.1, 2622.7, 2697.3, 2779.1])

dHv = pd.read_csv("csv_data/heat_of_vap_methanol.csv")
dHv["dH"] = (dHv["Molar Enthalpy [J/mol]"].to_numpy() * ureg("J/mol")).to("kcal/mol")

isothermal_compress = np.array([0.120, 0.135, 0.158]) * 10 ** -3 * ureg("1/bar")
isothermal_compress.to("1/atm")


def plot_traj_data(df):
    n_sims = len(df)
    fig = plt.figure(figsize=(6, 3))
    dir_split = list(df["directory"])[0].split("/")
    title = f"{dir_split[-3].upper()}_{dir_split[-2]}"
    fig.suptitle(title, fontsize=8)
    gs = fig.add_gridspec(n_sims, 4, hspace=0, wspace=0.5)
    axs = gs.subplots(sharex="col")

    for i in range(n_sims):
        _df = df.iloc[i]
        axs[i, 0].plot(_df["times"], _df["temperatures"], c="red", linewidth=0.1)
        # axs[i,0].axhline(_df["avgTemp"])
        axs[i, 0].text(
            0.5, 0.6, "{:.1f}".format(_df["avgTemp"]), horizontalalignment='center',
            verticalalignment='center', transform=axs[i, 0].transAxes,
            fontsize=10)

        axs[i, 1].plot(_df["times"], _df["energies"], c="blue", linewidth=0.1)
        axs[i, 2].plot(_df["times"], _df["volumes"], c="green", linewidth=0.1)

        pressure_str = "{:.2f}\n $\pm$ {:.1f}".format(_df["average_pressure"].magnitude,
                                                      _df["sd_pressure"].magnitude)
        axs[i, 3].text(
            0.5, 0.5, pressure_str, horizontalalignment='center',
            verticalalignment='center', transform=axs[i, 3].transAxes,
            fontsize=10)
        axs[i, 3].set_axis_off()

        if i == 0:
            axs[i, 0].set_title("$T$ [K]")
            axs[i, 1].set_title("$E$ [kcal/mol]")
            axs[i, 2].set_title("       $V$ [m$^3$]")
            axs[i, 3].set_title("$P$ [atm]")

        if i == n_sims - 1:
            axs[i, 0].set_xlabel("Time (ps)")
            axs[i, 1].set_xlabel("Time (ps)")
            axs[i, 2].set_xlabel("Time (ps)")

    plt.legend()
    plt.savefig(f"figs/sim/{title}.pdf", bbox_inches="tight")


def plot_properties(data_dfs, data_labels, FONTSIZE=20, show=False):
    mosaic = """
        AB
        CD
        EF
        """
    fig = plt.figure(constrained_layout=True, figsize=(8, 8))
    ax_dict = fig.subplot_mosaic(mosaic, sharex=True, gridspec_kw={
        "bottom": 0.25,
        "top": 0.95,
        "left": 0.1,
        "right": 0.5,
        "wspace": 0,
        "hspace": 0,
    }, )

    #  density
    ax_dict["A"].plot(methanol_density["Temp. (K)"], methanol_density["Density"],
                      "-o", label="Exp.", markevery=10)
    ax_dict["A"].set_ylabel(r"$\rho$ [kg/cm$^{3}$]", fontsize=FONTSIZE)
    #  isothermal compressibility
    ax_dict["C"].set_ylabel(r"$\kappa$ [atm$^{-1}$]", fontsize=FONTSIZE)
    ax_dict["C"].plot(K, isothermal_compress, "-o", label="Exp.")
    #  coefficient of thermal expansion
    ax_dict["E"].set_ylabel(r"$\alpha$ [K$^{-1}$]", fontsize=FONTSIZE)
    ax_dict["E"].set_xlabel(r"T [K]", fontsize=FONTSIZE)
    ax_dict["E"].plot(K_alpha, therm_expan, "-o", label="Exp.")
    #  Self-diffusion coefficient
    ax_dict["B"].set_ylabel(r"$D$ [$10^{-5}$ cm$^{2}$s$^{-1}$]", fontsize=FONTSIZE)
    ax_dict["B"].yaxis.set_label_position("right")
    ax_dict["B"].tick_params(axis='y', which='both', labelleft=False,
                             labelright=True)
    ax_dict["B"].plot(K_D2, D2, "-o", label="Exp.")
    #  heat of vaporization
    ax_dict["D"].set_ylabel(r"$\Delta H_{vap}$ [kcal mol$^{-1}$]", fontsize=FONTSIZE)
    ax_dict["D"].yaxis.set_label_position("right")
    ax_dict["D"].tick_params(axis='y', which='both', labelleft=False,
                             labelright=True)
    ax_dict["D"].plot(dHv["T"], dHv["dH"], "-o", label="Exp.")
    #  heat capacity
    ax_dict["F"].set_ylabel(r"$C_{p}$ [cal mol$^{-1}$K$^{-1}$]", fontsize=FONTSIZE)
    ax_dict["F"].set_xlabel(r"T [K]", fontsize=FONTSIZE)
    ax_dict["F"].yaxis.set_label_position("right")
    ax_dict["F"].tick_params(axis='y', which='both', labelleft=False,
                             labelright=True)
    ax_dict["F"].plot(heat_capacity["K"], heat_capacity["cal mol K"], "-o",
                      label="Exp.", markevery=2)

    for i in range(len(data_dfs)):
        # just plot magnitudes
        __avg_T = [_.magnitude for _ in data_dfs[i]["avgTemp"]]
        __dens = [_.magnitude for _ in data_dfs[i]["density"]]
        __kappa = [_.magnitude for _ in data_dfs[i]["kappa"]]
        __alpha = [_.magnitude for _ in data_dfs[i]["alpha"]]
        __D = [_ for _ in data_dfs[i]["D"]]
        __dHvap = [_.magnitude for _ in data_dfs[i]["dHvap"]]
        __C_p = [_.magnitude for _ in data_dfs[i]["C_p"]]

        ax_dict["A"].plot(__avg_T, __dens, "-o",
                          label=data_labels[i])
        ax_dict["C"].plot(__avg_T, __kappa, "-o",
                          label=data_labels[i])
        ax_dict["E"].plot(__avg_T, __alpha, "-o",
                          label=data_labels[i])
        ax_dict["B"].plot(__avg_T, __D, "-o",
                          label=data_labels[i])
        ax_dict["D"].plot(__avg_T, __dHvap, "-o",
                          label=data_labels[i])
        ax_dict["F"].plot(__avg_T, __C_p, "-o",
                          label=data_labels[i])

    ax_dict["D"].set_ylim(7.5, 10)

    for i in ["A", "B", "C", "D", "E", "F"]:
        ax_dict[i].set_xlim(225, 350)

    # plt.legend()

    plt.subplots_adjust(wspace=None, hspace=None)

    plt.savefig(f"figs/prop/{data_labels[0]}.pdf", bbox_inches="tight")

    if show:
        plt.show()


def keep_line(line):
    phrases = ["OG311", "CG331", "HGP1", "HGA3"]
    for phrase in phrases:
        if line.startswith(phrase) and line.__contains__("!") and len(line.split()) > 5:
            return True
    return False


def get_params(df):
    directory = df["directory"][0]
    path = os.path.join(directory, "job.inp")
    lines = open(path).readlines()
    lines_to_keep = [line.split()[:4] for line in lines if keep_line(line)]
    lines_to_keep = [[line[0], float(line[1]), float(line[2]), float(line[3])]
                     for line in lines_to_keep]

    out_dict = {}
    for line in lines_to_keep:
        out_dict[line[0] + "_e"] = line[2]
        out_dict[line[0] + "_s"] = line[3]

    return out_dict


def test_properties(df, label, plot=False):
    """
    Test the fit to experimental properties
    :param df:
    :param label:
    :return:
    """

    output = {}

    #  simulation values
    sim_dens = [_.magnitude for _ in df["density"]]
    sim_D = [_ for _ in df["D"]]
    sim_temp = [_.magnitude for _ in df["avgTemp"]]
    dHvap_sim = [_.magnitude for _ in df["dHvap"]]
    kappa_sim = [_.magnitude for _ in df["kappa"]]

    # exp. data
    _ = methanol_density[methanol_density["Temp. (K)"] >= min(sim_temp)]
    _ = _[_["Temp. (K)"] <= max(sim_temp)]
    #  fits to exp.
    den_z = np.polyfit(_["Temp. (K)"], _["Density"], 3)
    den_fit = np.poly1d(den_z)
    D_z = np.polyfit(K_D2, D2, 3)
    D_fit = np.poly1d(D_z)

    fit_dens = den_fit(sim_temp)
    fit_D = D_fit(sim_temp)
    _iso_comp = [_.magnitude for _ in isothermal_compress]

    kappa_z = np.polyfit(K, _iso_comp, 3)
    kappa_fit = np.poly1d(kappa_z)
    fit_kappa = kappa_fit(sim_temp[2:])

    # K, isothermal_compress

    # D
    # output["D_error"] = np.mean(abs(fit_D - sim_D))
    output["D_error"] = np.sum((fit_D - sim_D) ** 2)
    # output["D_r2"] = r_value ** 2

    # Density
    # output["Dens_error"] = np.mean(abs(fit_dens - sim_dens))
    output["Dens_error"] = np.sum((fit_dens - sim_dens) ** 2)
    # output["Dens_r2"] = r_value ** 2

    # isothermal compressibility kappa
    output["kappa_error"] = np.sum(
        (fit_kappa * 10 ** 4 - np.array(kappa_sim[2:]) * 10 ** 4) ** 2)

    # Hvap
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        sim_temp, dHvap_sim)

    hvap_t1_error = abs(8.94 - (slope * 289.15 + intercept))
    hvap_t2_error = abs(8.42 - (slope * 337.85 + intercept))

    output["Hvap_error"] = np.sum([hvap_t1_error ** 2, hvap_t2_error ** 2])

    output.update(get_params(df))

    return output


def get_min_max(data):
    all_data = []
    for d in data:
        all_data.extend(d)
    min_ = min(all_data)
    max_ = max(all_data)
    return min_, max_
