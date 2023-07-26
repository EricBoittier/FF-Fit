import MDAnalysis as mda
from pathlib import Path
import os
from MDAnalysis import Writer


def pdb_to_coords(pdb_path, n_atoms_per_res):
    lines = [_ for _ in open(pdb_path).readlines() if _[0] == "A"]
    coords = []
    total = 0

    for n in n_atoms_per_res:
        tmp_lines = lines[total:total + n]
        tmp_lines = [f"{_.split()[2][:1]} {_[31:38]} {_[39:46]} {_[47:54]}" for _ in
                     tmp_lines]
        coords.append(tmp_lines)
        total += n

    return coords


def coords_to_xyz(coords):
    """Take a list of lists coordinates and return a string in xyz format"""
    out = []
    n_atoms = sum([len(_) for _ in coords])
    out.append(f"{n_atoms}\n\n")

    for i, _ in enumerate(coords):
        for line in _:
            atom_type, x, y, z = line.split()
            out.append(f"{atom_type} {x} {y} {z}\n")
    return "".join(out)


def count_same(l):
    o = [1]
    for i, _ in enumerate(l):
        if not i == 0:
            if l[i] == l[i - 1]:
                o[-1] = o[-1] + 1
            else:
                o.append(1)
        else:
            o.append(1)

    return o

def get_dcds(dcd_dir):
    """
    Get the path to the dcd files, assumes they are named step5_1.dcd, step5_2.dcd, etc.
    TODO: make this more general... lol
    :param dcd_dir:
    :return:
    """
    dcdspath = "/home/boittier/pcbach/charmmions/"
    Ndcds = len(list(Path(dcdspath).glob("step5*dcd")))
    dcd = ["/home/boittier/pcbach/charmmions/step5_{}.dcd".format(i + 1) for i in
           range(Ndcds)]
    dcd = [_ for _ in dcd if os.path.exists(_) and (os.stat(_).st_size > 0)]
    print(dcd)
    return dcd

def get_psf(psf_dir):
    """
    Get the path to the psf file
    TODO: make this more general... lol
    :param psf_dir:
    :return:
    """
    return "/home/boittier/pcbach/charmmions/step3_pbcsetup.psf"



def save_N_atoms(u, Natoms, resname,
                 maxTries=100,
                 dr=0.05,
                 defaultR=4.750,
                 verbose=False,
                 ):
    selectStringTypes = f"(byres resname {resname})"
    selectStringDist = "(byres point {rx} {ry} {rz} {r}) or resid {resid}"

    # outfile for a xyz trajectory
    xyz_out_name = f"{Natoms}_{resname}.xyz"
    W = Writer(xyz_out_name, Natoms)

    xyz_counter = 0
    n_atoms = []
    pdb_paths = []
    for i, t in enumerate(u.trajectory):
        res_sel1 = u.select_atoms(selectStringTypes, updating=False, periodic=False)
        # loop thru the residues of the required type
        for j in [_.resid for _ in res_sel1]:
            counter = 0
            n_current = 0
            #  try to avoid an index error
            try:
                # j = j - len(res_sel1)
                #  initialize the selection
                res = u.select_atoms(selectStringTypes, updating=True,
                                     periodic=False)
                #  gather the appropriate data
                resname = res_sel1[j].resname
                rx = res_sel1[j].position[0]
                ry = res_sel1[j].position[1]
                rz = res_sel1[j].position[2]

                #  while loop to adjust r until the conditions are met
                while n_current != Natoms and counter < maxTries:
                    # format the distance string
                    _selectstringdist = selectStringDist.format(
                        rx=rx, ry=ry, rz=rz, r=defaultR, resid=j)
                    #  make the selection again
                    res = u.select_atoms(_selectstringdist,
                                         updating=False, periodic=True)
                    #  change the radius
                    if len(res) < Natoms:
                        defaultR += dr
                    else:
                        defaultR -= dr
                    counter += 1
                    n_current = len(res)

                #  exiting the while loop
                #  ....printing for debugging
                if verbose and counter >= maxTries:
                    print("conditions not met:", n_current)

                n_current = len(res)
                # check for success
                if n_current == Natoms:
                    if verbose:
                        print("success: ", n_current, defaultR, counter)
                    xyz_counter += 1
                    resids = [_.resid for _ in res]
                    # paths
                    name = f"test_coords/{Natoms}_{xyz_counter}_{resname}_{i}_{j}"
                    pdb_path = name + ".pdb"
                    xyz_path = name + ".xyz"
                    #  write
                    res.write(pdb_path)
                    res.write(xyz_path)
                    #  count the number of atoms per residue
                    natom_res = count_same(resids)
                    #  get the coords list
                    coords = pdb_to_coords(pdb_path, natom_res)
                    xyz_str = coords_to_xyz(coords)
                    n_atoms.append(float(xyz_str[:3]))
                    pdb_paths.append(pdb_path)
                    #  write to the xyz writer
                    W.write(res)

            #  handle errors
            except IndexError as e:
                if verbose:
                    print(e)

    # close the file
    W.close()

    # if the file has no atoms, delete it
    with open(xyz_out_name) as f:
        if len(f.readlines()) < 2:
            if verbose:
                print("deleting ", xyz_out_name)
            os.system(f"rm {xyz_out_name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_atoms", type=int, default=10)
    parser.add_argument("-r", "--resname", type=str, default="POT")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    args = parser.parse_args()
    print(args)
    dcds = get_dcds("")
    psf = get_psf("")
    u = mda.Universe(psf, dcds)
    save_N_atoms(u, args.n_atoms, args.resname, verbose=args.verbose)
