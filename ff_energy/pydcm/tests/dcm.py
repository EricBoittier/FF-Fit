import os
import unittest
from ff_energy.pydcm.dcm import mdcm, mdcm_set_up, scan_fesp, scan_fdns, \
    mdcm_cxyz, mdcm_clcl, local_pos, get_clcl, optimize_mdcm
from pathlib import Path
from ff_energy.pydcm import dcm_utils as du
from ff_energy.pydcm.kernel import KernelFit


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_dcm_fortran(self):
        m = mdcm_set_up(scan_fesp, scan_fdns,
                        local_pos=local_pos,
                        mdcm_cxyz=mdcm_cxyz,
                        mdcm_clcl=mdcm_clcl)
        print(m.get_rmse())
        optimize_mdcm(m, m.mdcm_clcl, "", "test")

    def test_load_data(self):
        PICKLES = list(Path("/home/boittier/Documents/phd/ff_energy/cubes/clcl")
                       .glob("*.obj"))
        scanpath = Path("/home/boittier/Documents/phd/ff_energy/cubes/dcm/")

        def name_(x):
            if "gaussian" in str(x):
                return scanpath / "scan" / (x.name.split(".c")[0] + ".cube")
            elif "_nms_" in str(x):
                return scanpath / "nms" / (x.name.split(".c")[0] + ".cube")
            else:
                print(f"ValueError(fbad pickle name {x})")
                return None

        PICKLES = [_ for _ in PICKLES if name_(_) is not None]
        CUBES = [name_(_) for _ in PICKLES]
        x, i, y = du.get_data(CUBES, PICKLES, 5)

        return x, i, y

    def test_fit(self):
        x, i, y = self.test_load_data()
        k = KernelFit()
        k.set_data(x, i, y)
        k.fit(alpha=0.0)
        print("N X:", len(k.X))
        print("N:", len(k.ids))
        print("N test:", len(k.test_ids))
        print("N_train:", len(k.train_ids))
        print(k.r2s)
        print("sum r2s test:", sum([_[0] for _ in k.r2s]))
        print("sum r2s train:", sum([_[1] for _ in k.r2s]))
        print("n models:", len(k.r2s))

    def test_files(self):
        i = 4
        l2 = 0.0
        cube_paths = Path("/home/boittier/Documents/phd/ff_energy/cubes/dcm/scan")
        ecube_files = list(cube_paths.glob("*esp.cube"))
        dcube_files = list(cube_paths.glob("*dens.cube"))
        print(len(ecube_files), len(dcube_files))
        ecube_files.sort()
        dcube_files.sort()
        print(ecube_files[0], dcube_files[0])
        #  name of the esp and dens cube files
        e = str(ecube_files[i])
        d = str(dcube_files[i])
        #  set up the mdcm object
        m = mdcm_set_up([e], [d],
                        local_pos=local_pos,
                        mdcm_cxyz=mdcm_cxyz,
                        mdcm_clcl=mdcm_clcl)
        print("RMSE:", m.get_rmse())
        outname = ecube_files[i].name + f"_{l2}"
        optimize_mdcm(m, m.mdcm_clcl, "", outname, l2=l2)


if __name__ == '__main__':
    unittest.main()
