import unittest
import pandas as pd

from ff_energy.ffe.data import pairs_data


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_pairs_data(self):
        pkl_path = "/home/boittier/Documents/phd/ff_energy/pickles/energy_report.pkl"

        er = pd.read_pickle(pkl_path)
        eg_data = er.data_plots[0].data
        print(eg_data)
        eg_dcm_path_ = "/home/boittier/homeb/water_cluster/pbe0dz_pc/{}/charmm/dcm.xyz"

        pairs_data(eg_data,
                   system="water_cluster",
                   name="test",
                   dcm_path_=eg_dcm_path_,
                   )


if __name__ == '__main__':
    unittest.main()

