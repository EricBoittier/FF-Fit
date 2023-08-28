import unittest

import jax.numpy

from ff_energy.utils.ffe_utils import read_from_pickle
from ff_energy.ffe.ff import FF
import jax.numpy as jnp

# load the data
# test_ff_fn = "pbe0_dz_kmdcm_LJ_water_cluster_ELEC_harmonic_ELEC.pkl"
test_ff_fn = "ff/template.ff.pkl"
test_ff = next(read_from_pickle(test_ff_fn))


class TestPotential(unittest.TestCase):
    def test_ljrun(self, N=500):
        test_ff = self.get_test_ff(N)
        parm = jnp.array(test_ff.get_random_parm())
        print("random parm:", parm)

        # check if the number of groups is correct
        groups_set = set(test_ff.out_groups.tolist())
        self.assertEqual(len(groups_set), N)  # add assertion here
        # check if number of targets in correct
        targets = test_ff.targets
        self.assertEqual(len(targets), N)
        jax_flat_test = test_ff.eval_jax_flat(parm)

        arrays_to_test = [
            test_ff.out_akps,
            test_ff.out_es,
            test_ff.out_dists,
            test_ff.out_groups,
            jax_flat_test,
        ]
        for _ in arrays_to_test:
            # check for Nans, nones, ect.
            self.assertEqual(True in jax.numpy.isnan(_), False)

        print("lenths:", [len(_) for _ in arrays_to_test])
        print("out_es", test_ff.out_es)
        print("jax_flat_test", jax_flat_test)
        print("out_akps", test_ff.out_akps)
        print("out_dists", test_ff.out_dists)

        #  check standard results
        res_standard = test_ff.eval_func(parm)
        self.assertEqual(None in res_standard, False)
        print(res_standard)

        #  check jax results
        res_jax = test_ff.eval_jax(parm)
        print("res_jax", res_jax)
        print("targets", test_ff.targets)
        self.assertEqual(True in jax.numpy.isnan(res_jax), False)
        print("loss standard", test_ff.get_loss(parm))
        print("loss jax", test_ff.get_loss_jax(parm))
        print("loss jax (grad)", test_ff.get_loss_grad(parm))

    def test_ecol(self, N=500):
        test_ff = self.get_test_ff(N)
        parm = jnp.array(test_ff.get_random_parm())
        print("random parm:", parm)
        ecol_res = test_ff.eval_coulomb(1)
        assert jnp.isclose(test_ff.dcm_ecols, ecol_res).all()

    @staticmethod
    def get_test_ff(self, N=500) -> FF:
        test_ff.sort_data()
        test_ff.data = test_ff.data[:N].copy()
        test_ff.sort_data()

        return test_ff


if __name__ == "__main__":
    unittest.main()
