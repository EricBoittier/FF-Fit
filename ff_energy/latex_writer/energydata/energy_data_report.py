from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from ff_energy.latex_writer.report import Report
from ff_energy.latex_writer.figure import Figure
from ff_energy.latex_writer.format import safe_latex_string
from ff_energy.plotting.data_plots import DataPlots
from ff_energy.utils.ffe_utils import pickle_output
from ff_energy.latex_writer.extra_data import (
    dcm_elec_, dcm_pol_no_intern
)

plt.set_loglevel("notset")

abstract = """ This report summaries the findings from 
ab initio WFT and energy decomposition analysis. These results were used to fit
classical force fields intended for the simulation. The agreement between 
the ab initio data and the data modelled by molecular mechanics force fields is analyzed.
"""

energy_cols_1 = ["M_ENERGY", "C_ENERGY", "P_intE", "intE"]


class EnergyReport:
    """
    Class for writing energy reports
    For each level of theory, we want to write a report that contains:
    - the energy histogram and summary statistics
            for monomers, pairs, and clusters.
        types include:
            - total energy
            - electrostatic energy
            - interaction energy
            - internal energy

    - summary of the fitting for:
        - internal degrees of freedom
        - interaction potentials
        - electrostatics (CP term)
        which includes correlation plots and residuals.
        CGENFF vs DFT,
        refitted CHARMM vs. DFT,
        refitted CHARMM with perfect bonded vs. DFT
        refitted CHARMM with perfect bonded and xDCM
        For the exact Coulomb and polarization figures

    1) load the data
    2) make the figures and tables
    """

    def __init__(self, report_name=None):
        self.name = None
        self.pickle_path = None
        if report_name is None:
            self.report_name = "energy_report"
        else:
            self.report_name = report_name
        #  load the data
        self.data = []
        self.data_plots = []
        self.data_keys = []
        self.data_names = []
        self.data_descriptions = []
        #  start the report
        self.report = Report(self.report_name)
        self.report.set_title("SI: Energy Decomposition Analysis")
        self.report.set_short_title("Energy Decomposition Analysis")
        self.report.add_section("\\clearpage")
        self.report.set_abstract(abstract)

    def add_data(self, index, data, key) -> None:
        """adds a column to the data
        :param index: the index of the data to add to
        :param data: the data to add
        :param key: the key to add the data under
        :return: None
        """
        self.data[index].data = \
            self.data[index].data.join(
                data, how="outer", rsuffix=key)


    def add_ff(self):
        """ Create the force field object for fitting
        :return:
        """
        # FF()


        pass

    def add_pickles(self, pickle_paths, names=None, descriptions=None):
        """
        Add the pickle paths to the data
        :param pickle_paths:
        :param names:
        :return:
        """
        if names is None:
            names = [safe_latex_string(Path(p).stem) for p in pickle_paths]
        if descriptions is None:
            descriptions = ["" for _ in pickle_paths]

        def check_cond(pp):
            return isinstance(pp, str) or isinstance(pp, Path)

        if check_cond(pickle_paths):
            pickle_paths = [pickle_paths]
        for i, path in enumerate(pickle_paths):
            #  loop and check if the path is in the data keys
            if path not in self.data_keys:
                self.data_keys.append(path)
                self.load_data(path)
                self.data_names.append(names[i])
                self.data_descriptions.append(descriptions[i])

    def generate_data_report(self):
        #  make the tables
        self.summary_tables()
        #  make the figures
        self.make_figures()

    def load_data(self, pkl_path):
        self.data.append(pd.read_pickle(pkl_path))
        self.data_plots.append(DataPlots(self.data[-1]))

    def compile_report(self):
        self.report.write_document()
        self.report.compile()

    def make_figures(self):
        for i in range(len(self.data)):
            #  subsection
            self.report.add_section("\subsection{" +
                                    safe_latex_string(
                                        self.data_names[i]
                                    )
                                    + "}")
            #  details
            self.report.add_section(
                "\subsubsection{Distribution of Energies} \n The"
                " following figures show the distribution of energies"
                " for the monomers, pairs, and clusters."
            )

            figs = self.make_mol_pics(i=i)
            for f in figs:
                self.report.add_section(f, width="0.85")

            #  Figures for single distributions
            single_keys = [
                "intE",
                "P_intE",
                "C_ENERGY",
                "M_ENERGY",
            ]

            fig = self.make_kde_figs(single_keys, i=i)
            self.report.add_section(fig, width="0.5")

            #  Figures for paired distributions
            paired_keys = [
                ("intE", "P_intE"),
                ("C_ENERGY", "M_ENERGY"),
            ]
            for key1, key2 in paired_keys:
                fig = self.make_energy_fig(key1, key2, i=i)
                self.report.add_section(fig, width="1.0")

            self.report.add_section("\\newpage \n \subsection{Fitting Results}")

        self.report.add_section("\\newpage \n \subsection{Appendix}")
        self.appendix_tables()


    def make_energy_fig(self, key1, key2, i=0):
        _ = self.data_plots[i].energy_hist(
            path=self.report.fig_path,
            key1=key1,
            key2=key2,
            label=self.data_names[i]
        )
        return Figure(
            _["path"],
            _["caption"],
            _["label"],
        )

    def make_mol_pics(self, i=0):
        _ = self.data_plots[i].mol_pics(
            ["intE", "P_intE", "C_ENERGY", "M_ENERGY"],
            self.report.fig_path,
            self.data_names[i]
        )
        figures = [
            Figure(
                path,
                caption,
                label,
            )
            for path, caption, label in zip(_[0], _[1], _[2])]

        return figures

    def make_kde_figs(self, keys, i=0):
        _ = self.data_plots[i].hist_kde(
            keys, path=self.report.fig_path,
            label=self.data_names[i]
        )
        return Figure(
            _["path"],
            _["caption"],
            _["label"],
        )

    def summary_tables(self):
        for i in range(len(self.data)):
            energy_table = (
                self.data_plots[i]
                .data[energy_cols_1]
                .describe()
                .to_latex(
                    float_format="%.2f",
                    index_names=False,
                    caption="Summary statistics for the energy data.",
                    label=f"tab:energy_stats",
                    position="b!",
                )
            )
            self.report.add_section(energy_table)

    def appendix_tables(self):
        for i in range(len(self.data)):
            energy_table = (
                self.data_plots[i]
                .data[energy_cols_1]
                .to_latex(
                    float_format="%.2f",
                    index_names=False,
                    caption="Energy data.",
                    label=f"tab:energy_raw",
                    position="b!",
                )
            )
            self.report.add_section(energy_table)


if __name__ == "__main__":
    pkl_paths = [
        "/home/boittier/Documents/phd/ff_energy/pickles/water_cluster_pbe0dz_pc.pkl",
        "/home/boittier/Documents/phd/ff_energy/pickles/dcm_pbe0dz_pc.pkl",
        "/home/boittier/Documents/phd/ff_energy/pickles/ions_ext_pbe0dz_pc.pkl",
    ]
    pkl_descriptions = [
        "200 snapshots of 20 water molecules sampled at random positions "
        "from MD in periods of 500 ps.",
        "20 snapshots of 20 DCM molecules sampled at random positions",
        "Mixed ion clusters",
    ]
    plk_names = [
        "waterpbe0dzclusters",
        "dcmpbe0dzclusters",
        "ionspbe0dzclusters",
    ]

    er = EnergyReport(report_name="Report_PBE0dz")
    er.add_pickles(pkl_paths,
                   names=plk_names,
                   descriptions=pkl_descriptions
                   )
    er.generate_data_report()
    er.compile_report()

    """
    Add extra data where available
    """
    er.add_data(1, dcm_elec_["ELEC"], "_CI")
    er.add_data(1, dcm_pol_no_intern["ELEC"], "_POL")

    pickle_output(er, "energy_report")
