import unittest
from ff_energy.utils import read_from_pickle
from ff_energy.potential import FF
import jax.numpy as jnp
# load the data

test_ff_fn = "pbe0_dz_kmdcm_LJ_water_cluster_ELEC_harmonic_ELEC.pkl"
test_ff = next(read_from_pickle(test_ff_fn))

class test_potential(unittest.TestCase):
    def test_ljrun(self):
        test_ff = self.get_test_ff()
        test_ff.sort_data()
        test_ff.data = test_ff.data[:10].copy()
        parm = jnp.array(test_ff.get_random_parm())
        test_ff.jax_init(parm)
        # check if the number of groups is correct
        groups_set = set(test_ff.out_groups.tolist())
        self.assertEqual(len(groups_set), 10)  # add assertion here
        # check if number of targets in correct
        targets = test_ff.targets
        self.assertEqual(len(targets), 10)
        print("out_es", test_ff.out_es)

        print(test_ff.eval_func(parm))
        print(test_ff.eval_jax(parm))
        print(test_ff.get_loss(parm))
        print(test_ff.get_loss_jax(parm))

    def get_test_ff(self) -> FF:
        return test_ff



if __name__ == '__main__':
    unittest.main()
