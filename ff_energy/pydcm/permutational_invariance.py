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

    # get the combinations
    combinations = []
    for atom_type in range(nperms):
        original_order = perm_idxs[atom_type]
        combs = list(permutations(original_order))
        combinations.append(combs)
    # if more than one permutable atom type, get the cartesian product
    if len(combinations) > 1:
        combinations = list(*product(*combinations))

    print(combinations)

    comb_dict = {i: {combinations[0][0][j]: _[j]
                     for j in range(len(_))}
                 for i, _ in enumerate(combinations[0])}

    # for each combination, generate the new cube file
    for i, comb in enumerate(combinations[0]):
        # get the new atom lines
        new_atom_lines = []
        for j in range(natoms):
            # if the atom is not permutable, keep the original line
            if j not in comb:
                new_atom_lines.append(atom_dict[j])
            else:
                # if the atom is permutable, get the new line
                new_atom_lines.append(atom_dict[comb_dict[i][j]])

        # get the new cube lines
        new_cube_lines = lines[:6] + new_atom_lines + lines[6+natoms:]
        if "esp" in cubefile:
            filename = cubefile.replace("_esp.cube", f"_perm_{i}_esp.cube")
        elif "dens" in cubefile:
            filename = cubefile.replace("_dens.cube", f"_perm_{i}_dens.cube")
        else:
            raise ValueError("Cube file does not contain ESP or DENS")
        print(filename, comb)
        # write the new cube file
        with open(filename, "w") as f:
                f.writelines(new_cube_lines)


if __name__ == "__main__":
    from pathlib import Path
    cubes_path = "/home/boittier/Documents/phd/ff_energy/cubes/methanol/scan/"
    # loop thru esp files
    files = Path(cubes_path).glob("*_esp.cube")
    for f in files:
        cube_to_permutable_copies(
            str(f),
            [[3,4,5]]
        )
    # loop thru density files
    files = Path(cubes_path).glob("*_dens.cube")
    for f in files:
        cube_to_permutable_copies(
            str(f),
            [[3,4,5]]
        )
