from ff_energy.latex_writer.report import Report
from ff_energy.latex_writer.figure import Figure
import pandas as pd
from ff_energy.plotting.data_plots import DataPlots
import matplotlib.pyplot as plt

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

    def __init__(self, pickle_path, report_name=None):
        self.name = None
        self.pickle_path = pickle_path
        if report_name is None:
            self.report_name = "energy_report"
        else:
            self.report_name = report_name
        #  load the data
        self.data = pd.read_pickle(self.pickle_path)
        self.data_plots = DataPlots(self.data)
        #  start the report
        self.report = Report(self.report_name)
        self.report.set_title("SI: Energy Decomposition Analysis")
        self.report.set_short_title("Energy Decomposition Analysis")
        self.report.add_section("\\clearpage")
        self.report.set_abstract(abstract)

    def generate_data_report(self):
        #  make the tables
        self.make_tables()
        #  make the figures
        self.make_figures()

    def load_data(self, pkl_path):
        self.data = pd.read_pickle(pkl_path)
        self.data_plots = DataPlots(self.data)

    def compile_report(self):
        self.report.write_document()
        self.report.compile()

    def make_figures(self):
        self.report.add_section("\subsection{Distribution of Energies} \n The"
                                " following figures show the distribution of energies"
                                " for the monomers, pairs, and clusters.")

        #  Figures for single distributions
        single_keys = [
            "intE",
            "P_intE",
            "C_ENERGY",
            "M_ENERGY",
        ]
        for key in single_keys:
            fig = self.make_kde_fig(key)
            self.report.add_section(fig, width='0.25')

        #  Figures for paired distributions
        paired_keys = [
            ("intE", "P_intE"),
            ("C_ENERGY", "M_ENERGY"),
        ]
        for key1, key2 in paired_keys:
            fig = self.make_energy_fig(key1, key2)
            self.report.add_section(fig, width="1.0")

        self.report.add_section("\\newpage \n \subsection{Fitting Results}")

    def make_energy_fig(self, key1, key2):
        _ = self.data_plots.energy_hist(path=self.report.fig_path,
                                        key1=key1,
                                        key2=key2, )
        return Figure(
            _["path"],
            _["caption"],
            _["label"],
        )

    def make_kde_fig(self, key):
        _ = self.data_plots.hist_kde(path=self.report.fig_path,
                                     key=key, )
        return Figure(
            _["path"],
            _["caption"],
            _["label"],
        )

    def make_tables(self):
        energy_table = self.data_plots.data[energy_cols_1].describe(
        ).to_latex(float_format="%.2f",
                   index_names=False,
                   caption="Summary statistics for the energy data.",
                   label="tab:energy_stats",
                   position="b!",)
        self.report.add_section(energy_table)

    def report_energies(self):
        pass

    def report_fitting(self):
        pass


if __name__ == "__main__":
    pkl_path = "/home/boittier/Documents/phd/ff_energy/pickles/dcm_pbe0dz_pc.pkl"
    er = EnergyReport(pkl_path,
                      report_name="DiChloroMethane_PBE0dz",
                      )
    er.generate_data_report()
    er.compile_report()
