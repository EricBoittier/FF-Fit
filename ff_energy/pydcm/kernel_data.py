from ff_energy.pydcm.dcm import mdcm, mdcm_set_up, scan_fesp, scan_fdns, \
    mdcm_cxyz, mdcm_clcl, local_pos, get_clcl, optimize_mdcm
from pathlib import Path
from ff_energy.pydcm import dcm_utils as du
from ff_energy.pydcm.kernel import KernelFit

def generate_data(i, p, l2=0.0):
    cube_paths = Path(p)
    ecube_files = list(cube_paths.glob("*esp.cube"))
    dcube_files = list(cube_paths.glob("*dens.cube"))
    print(len(ecube_files), len(dcube_files))
    ecube_files.sort()
    dcube_files.sort()
    print(ecube_files[0], dcube_files[0])
    e = str(ecube_files[i])
    d = str(dcube_files[i])

    m = mdcm_set_up([e], [d],
                    local_pos=local_pos,
                    mdcm_cxyz=mdcm_cxyz,
                    mdcm_clcl=mdcm_clcl)
    print(m.get_rmse())
    outname = ecube_files[i].name + f"_{l2}"
    optimize_mdcm(m, m.mdcm_clcl, "", outname, l2=l2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=int, default=0)
    parser.add_argument("-l2", type=float, default=0.0)
    parser.add_argument("-p", type=str,
                        default="/home/boittier/Documents/phd/ff_energy/cubes/dcm/scan")
    args = parser.parse_args()
    generate_data(args.i, args.p, l2=args.l2)

