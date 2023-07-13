from itertools import permutations, product

def cube_to_permutable_copies(cubefile: str,
                              perm_idxs: list) -> None:
    """
    Given a list of lists of permutable atoms for each
    atom type in the cube file (with permutable atoms),
    generate the missing copies of the cube files for a given
    permutation.
    :param cubefile: string of the cube filename
    :return:
    """
    # Read the cube file
    with open(cubefile, "r") as f:
        lines = f.readlines()

    # Get the number of atoms
    natoms = int(lines[2].split()[0])
    atom_lines = lines[6:6+natoms]
    atom_dict = {i: v for i, v in enumerate(atom_lines)}
    # number of permutations
    nperms = len(perm_idxs)
    original = list(range(natoms))

    # get the combinations
    combinations = []
    for atom_type in range(nperms):
        original_order = perm_idxs[atom_type]
        combs = list(permutations(original_order))
        combinations.append(combs)
    # if more than one permutable atom type, get the cartesian product
    if len(combinations) > 1:
        combinations = list(product(*combinations))

    print(combinations)

if __name__ == "__main__":
    cube_to_permutable_copies(
        "/home/boittier/Documents/phd/ff_energy/cubes/methanol/scan/"
        "gaussian_0_meoh_pbe0_adz_esp.cube",
        [[3,4,5]]
    )