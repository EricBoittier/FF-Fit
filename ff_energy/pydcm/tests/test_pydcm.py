import unittest
from pathlib import Path
from ff_energy.pydcm import mdcm_utils as dcm


DATA_PATH = Path(__file__).parents[1] / "ff_energy" / "pydcm" / "data"

test_file1 = DATA_PATH / "dcm.mdcm"

mdcm = "/home/unibas/boittier/methanol_kern/10charges.dcm"
mdcm_xyz = "/home/unibas/boittier/methanol_kern/refined.xyz"

eps = DATA_PATH /  "gaussian_0_dcm_pbe0_adz_esp.cube"
dens = DATA_PATH / "gaussian_0_dcm_pbe0_adz_dens.cube"
print(eps.absolute())
print(dens.absolute())
m = dcm.mdcm_set_up([str(eps.absolute())],
                [str(dens.absolute())],
                mdcm_cxyz = mdcm_xyz,
                mdcm_clcl = mdcm,
                local_pos=None)

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test(self):
        print(m.mdcm_clcl)
        print(m.mdcm_cxyz)
        print(m.get_rmse())


if __name__ == '__main__':
    unittest.main()
