import unittest
from ff_energy.pydcm.dcm import mdcm, mdcm_set_up, scan_fesp, scan_fdns, \
    mdcm_cxyz, mdcm_clcl, local_pos


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_dcm_fortran(self):
        m = mdcm_set_up(scan_fesp, scan_fdns,
                        local_pos=local_pos,
                        mdcm_cxyz=mdcm_cxyz,
                        mdcm_clcl=mdcm_clcl)
        print(m.get_rmse())


if __name__ == '__main__':
    unittest.main()
