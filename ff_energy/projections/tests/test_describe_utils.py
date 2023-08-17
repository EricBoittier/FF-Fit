import unittest
import dscribe
import matplotlib.pyplot as plt
from ase.visualize import view
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import shutil
from pathlib import Path

from ff_energy.utils.ffe_utils import get_structures, pickle_output
from ff_energy.plotting import set_style

from ff_energy.projections.dscribe_utils import soap, soap_dist, get_dcm_soap
from ff_energy.projections.dimensionality import get_pca, get_rp_pca
from ff_energy.projections.plotting import plot_pca, plot_values_of_l


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)
        # add assertion here

    def load_data(self):
        structures2, pdbs2 = get_structures("dcm")
        atoms2 = [_.get_ase() for _ in structures2]
        return atoms2

    def get_soap(self):
        atoms2 = self.load_data()
        dcm_soap = get_dcm_soap(average="inner").create(atoms2)
        return dcm_soap

    def test_plot_pca(self):
        set_style()
        dcm_soap = self.get_soap()
        pca, new_x = get_pca(dcm_soap)
        plot_pca(pca, new_x)

    def test_plot_pca2(self):
        set_style()
        dcm_soap = self.get_soap()
        rp, pca, new_x1, new_x2 = get_rp_pca(dcm_soap)
        ax = plot_pca(pca, new_x2)
        plt.show()

    def test_values_of_l(self):
        set_style()
        atoms2 = self.load_data()
        soaps, pcas = plot_values_of_l(get_dcm_soap, atoms2)
        plt.show()


if __name__ == "__main__":
    unittest.main()
