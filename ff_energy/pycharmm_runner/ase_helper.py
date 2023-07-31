from ase.io import read

def read_pdb(pdb_name):
    return read(pdb_name)

def get_masses(pdb_name):
    return read_pdb(pdb_name).get_masses()

def get_mass(pdb_name):
    return get_masses(pdb_name).sum()

def get_pmass(pdb_name):
    return int(get_mass(pdb_name)//50)

