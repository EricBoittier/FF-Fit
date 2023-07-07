import os
import unittest
import logging
logging.log(0, "Testing ff_energy.pydcm.dcm")
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import ff_energy.utils.utils
from ff_energy.pydcm.dcm import mdcm, mdcm_set_up, scan_fesp, scan_fdns, \
    mdcm_cxyz, mdcm_clcl, local_pos, get_clcl, optimize_mdcm, eval_kernel
from ff_energy.utils.utils import make_df_same_size

from ff_energy.pydcm.dcm import FFE_PATH, espform, densform


from ff_energy.utils import dcm_utils as du

from ff_energy.pydcm.dcm import DCM_PY_PATH

from ff_energy.pydcm.kernel import KernelFit

model_test_key = "ddc0ceff-f221-45c5-bd4e-3e5e2e1eb705"

def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            test_func(self, *args, **kwargs)

    return do_test


class kMDCM_Experiments(unittest.TestCase):
    """ 
    Class of tests and experiments for kMDCM
    """
    
    def get_mdcm(self, mdcm_dict=None):
        if mdcm_dict is not None and type(mdcm_dict) is dict:
            if "scan_fesp" in mdcm_dict.keys():
                scan_fesp = mdcm_dict["scan_fesp"]
            else:
                scan_fesp = None
            if "scan_fdns" in mdcm_dict.keys():
                scan_fdns = mdcm_dict["scan_fdns"]
            else:
                scan_fdns = None
            if "mdcm_cxyz" in mdcm_dict.keys():
                mdcm_cxyz = mdcm_dict["mdcm_cxyz"]
            else:
                mdcm_cxyz = None
            if "mdcm_clcl" in mdcm_dict.keys():
                mdcm_clcl = mdcm_dict["mdcm_clcl"]
            else:
                mdcm_clcl = None
            if "local_pos" in mdcm_dict.keys():
                local_pos = mdcm_dict["local_pos"]
            else:
                local_pos = None
        else:
            scan_fesp = None
            scan_fdns = None
            local_pos = None
            mdcm_cxyz = None
            mdcm_clcl = None

        if scan_fesp is None:
            scan_fesp = [espform]
        if scan_fdns is None:
            scan_fdns = [densform]

        if mdcm_cxyz is None:
            mdcm_cxyz = FFE_PATH / "ff_energy/pydcm/sources/" \
                        "dcm8.xyz"
        if mdcm_clcl is None:
            mdcm_clcl = FFE_PATH / "ff_energy/pydcm/sources/" \
                        "dcm.mdcm"

        return mdcm_set_up(scan_fesp, scan_fdns,
                           local_pos=local_pos,
                           mdcm_cxyz=mdcm_cxyz,
                           mdcm_clcl=mdcm_clcl)

    def test_dcm_fortran(self):
        m = mdcm_set_up(scan_fesp, scan_fdns,
                        local_pos=local_pos,
                        mdcm_cxyz=mdcm_cxyz,
                        mdcm_clcl=mdcm_clcl)
        print(m.get_rmse())
        optimize_mdcm(m, m.mdcm_clcl, "", "test")

    def test_load_data(self, l2,
                       cube_path=f"{FFE_PATH}/cubes/dcm/"):
        PICKLES = list(Path(f"{FFE_PATH}/cubes/clcl/{l2}")
                       .glob("*.obj"))
        scanpath = Path(cube_path)

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
        assert len(CUBES) == len(PICKLES)
        return du.get_data(CUBES, PICKLES, 5)

    def test_standard_rmse(self,
                           k,
                           files,
                           cubes,
                           pickles,
                           cubes_pwd = None):
        """
        Test the standard RMSE
        :param k: KernelFit object
        :param files: list of filenames
        :param cubes: list of cube objects
        :param pickles: list of pickle objects
        :param cubes_pwd: path to the cubes
        :return:
        """

        if cubes_pwd is None:
            cube_paths = Path(f"{FFE_PATH}/cubes/dcm/")
        print("cubes", len(cubes))
        print("cubes path", cube_paths)
        ecube_files = list(cube_paths.glob("*/*esp.cube"))
        dcube_files = list(cube_paths.glob("*/*dens.cube"))
        print("ecube", len(ecube_files))
        print("dcube", len(dcube_files))
        print(len(cubes), len(pickles))
        rmses = eval_kernel(files, ecube_files, dcube_files)
        print("RMSEs:", rmses)
        rmse = sum(rmses) / len(rmses)
        print("RMSE:", rmse)
        k.plot_pca(rmses, title=f"Standard ({rmse:.2f})")
        pd.DataFrame({"rmses": rmses,
                      "filename": files}
                     ).to_csv("standard_.csv")

    def experiments(self):
        alphas = [0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1]
        l2s = [0.0, 0.1, 0.5, 1.0, 2.0, 4.0]
        n_factors = [2, 4, 6, 8, 10, 12]
        for alpha in alphas:
            for l2 in l2s:
                for n in n_factors:
                    print("alpha", alpha, "l2", l2, "N_factors", n_factors)
                    self.test_fit(alpha=alpha, l2=l2, n_factor=n)

    def test_N_repeats(self, n=1):
        for i in range(n):
            print("i", i)
            self.experiments()

    @ignore_warnings
    def test_fit(self,
                 alpha=0.0,
                 l2=0.0,
                 do_null=False,
                 n_factor=2,
                 do_optimize=False,
                 cubes_pwd= FFE_PATH / "cubes/dcm/",
                 mdcm_dict=None,
                 load_data=False,
                 ):
        """
        Test the kernel fit
        """

        # path to cubes
        cube_paths = Path(cubes_pwd)
        ecube_files = list(cube_paths.glob("*/*esp.cube"))
        dcube_files = list(cube_paths.glob("*/*dens.cube"))
        print("n_cubes", len(ecube_files))
        print("l2", l2)
        
        #  load mdcm object
        m = self.get_mdcm(mdcm_dict=mdcm_dict)
        print("mdcm_clcl")
        print(m.mdcm_clcl)
        
        if load_data is True:
            do_optimize = True # dumb
        # dp optimization
        if do_optimize is False:
            print("*"*80)
            print("Optimizing with l2=", l2)
            opt_rmses = eval_kernel(
                    range(140), 
                    ecube_files,
                    dcube_files,
                    opt=True, 
                    l2=l2,
                    verbose=True,
                                    )
            print("Opt RMSEs:", opt_rmses)
            opt_rmse = sum(opt_rmses) / len(opt_rmses)
            print("Opt RMSE:", opt_rmse)
        
        # unload the data
        x, i, y, cubes, pickles = self.test_load_data(l2=str(l2))
        
        #  kernel fit
        k = KernelFit()
        k.set_data(x, i, y, cubes, pickles)
        k.fit(alpha=alpha, N_factor=n_factor,l2=l2)
        
        # printing
        print("*" * 20, "Kernel Fit", "*" * 20)
        print("N X:", len(k.X))
        print("N:", len(k.ids))
        print("N test:", len(k.test_ids))
        print("N_train:", len(k.train_ids))
        print(k.r2s)
        print("sum r2s test:", sum([_[0] for _ in k.r2s]))
        print("sum r2s train:", sum([_[1] for _ in k.r2s]))
        print("n models:", len(k.r2s))
        print("r2s:", k.r2s)
        
        #   Move the local charges
        print("Moving clcls")
        files = k.move_clcls(m.mdcm_clcl, l2=l2)
        print("N files:", len(files), '\n')
        
        print("*" * 20, "Eval Results", "*" * 20)
        #  test the original model
        if do_null:
            print(" " * 20, "Eval Null", "*" * 20)
            self.test_standard_rmse(k, files, cubes, pickles)
        
        #  test the optimized model
        rmses = eval_kernel(files, ecube_files, dcube_files,
                            load_pkl=True, l2=l2)

        print("len(rmses):", len(rmses))
        
        #  Printing the rmses
        kern_rmse = self.print_rmse(rmses)
        print("RMSEs:", rmses)
        self.prepare_df(k, rmses, files, alpha=alpha, l2=l2)
        
        if do_optimize is False:
            self.prepare_df(k, opt_rmses, files, alpha=alpha, l2=l2, opt=True)

        
        print("*" * 20, "Eval Kernel", "*" * 20)
        # plot fits
        k.plot_fits(rmses)
        k.plot_pca(rmses, title=f"Kernel ({kern_rmse:.2f})", name=f"kernel_{k.uuid}")
        
        #  plot optimized
        if do_optimize is False:
            print(opt_rmses)
            print(len(opt_rmses))
            #k.plot_pca(opt_rmse, title=f"Optimized ({opt_rmse:.2f})",
            #           name=f"opt_{k.uuid}")
        #  pickle kernel
        print("Pickling kernel", k)
        self.pickle_kernel(k)
        #  write manifest
        print("Writing manifest")
        k.write_manifest(f"manifest/{k.uuid}.json")
        return k

    def print_rmse(self, rmses):
        """

        """
        print("RMSEs:", rmses)
        rmse = sum(rmses) / len(rmses)
        print("RMSE:", rmse)
        return rmse

    def prepare_df(self, k, rmses, files, alpha=0.0, l2=0.0, opt=False):
        """

        """
        class_name = ["test" if _ in k.test_ids
                      else "train" for _ in k.ids]
        if opt:
            fn = f"csvs/opt_{k.uuid}_{l2}.csv"
        else:
            fn = f"csvs/kernel_{k.uuid}_{alpha}_{l2}.csv"

        df_dict = {
            "rmse": rmses,
            "pkl": files,
            "class": class_name,
            "alpha": alpha,
            "uuid": k.uuid,
            'l2': l2,
            "type": ["nms" if "nms" in str(_) else "scan"
                     for _ in files]
        }

        pd.DataFrame(
        ff_energy.utils.utils.make_df_same_size(df_dict)
        ).to_csv(fn)

    def pickle_kernel(self, k):
        with open(f"models/kernel_{k.uuid}.pkl", "wb") as f:
            pickle.dump(k, f)

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
    from argparse import ArgumentParser
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--n_factor', type=int, default=1.0)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('unittest_args', nargs='*')
    args = parser.parse_args()
    
    k = kMDCM_Experiments()
    k.test_fit(alpha=args.alpha, n_factor=args.n_factor, l2=args.l2)

    # Now set the sys.argv to the unittest_args (leaving sys.argv[0] alone)
    #sys.argv[1:] = args.unittest_args

    #unittest.main()


