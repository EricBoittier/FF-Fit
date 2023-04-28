import unittest
from ff_energy.utils import read_from_pickle
from ff_energy.potential import FF
# load the data

test_ff_fn = "pbe0_dz_kmdcm_LJ_water_cluster_ELEC_harmonic_ELEC.pkl"
test_ff = next(read_from_pickle(test_ff_fn))

class test_potential(unittest.TestCase):
    def test_ljrun(self):
        test_ff = self.get_test_ff()
        print(test_ff.get)
        self.assertEqual(True, False)  # add assertion here

    def get_test_ff(self) -> FF:
        return test_ff



if __name__ == '__main__':
    unittest.main()
