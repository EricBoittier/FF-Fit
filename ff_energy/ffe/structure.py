import itertools
import os
import numpy as np

from ff_energy.ffe.templates import PSF
from ff_energy.ffe.geometry import sqrt_einsum_T


def valid_atom_key_pairs(atom_keys):
    atom_key_pairs = list(itertools.combinations(atom_keys, 2))
    atom_key_pairs = [(a, b) if a < b else (b, a) for a, b in atom_key_pairs]
    atom_key_pairs.extend([(a, a) for a in atom_keys])
    atom_key_pairs.sort(key=lambda x: (x[0], x[1]))
    return atom_key_pairs


atom_keys = ["OG311", "CG331", "HGP1", "HGA3", "OT", "HT"]
atom_key_pairs = valid_atom_key_pairs(atom_keys)

atom_types = {
    ("LIG", "O"): "OG311",
    ("LIG", "C"): "CG331",
    ("LIG", "H1"): "HGP1",
    ("LIG", "H2"): "HGA3",
    ("LIG", "H3"): "HGA3",
    ("LIG", "H4"): "HGA3",
    ("TIP3", "OH2"): "OT",
    ("TIP3", "H1"): "HT",
    ("TIP3", "H2"): "HT",
    # ("LIG", "O"): "OT",
    # ("LIG", "H1"): "HT",
    # ("LIG", "H"): "HT",
    # ("LIG", "H2"): "HT",
}


class Structure:
    """Class for a pdb structure"""

    def __init__(self, path, atom_types=atom_types, system_name=None):
        self.system_name = system_name
        self.path = path
        self.name = os.path.basename(path)
        self.lines = None
        self.atoms = None
        self.atomnames = None
        self.keys = None
        self.resids = None
        self.restypes = None
        self.xyzs = None
        self.chm_typ = None
        self.chm_typ_mask = None
        self.res_mask = None
        self.pairs = None
        self.distances = None
        self.distances_pairs = None
        self.distances_mask = None
        self.dcm = None
        self.dcm_charges = None
        self.dcm_charges_mask = None
        self.atom_types = atom_types

        self.read_pdb(path)

    def read_pdb(self, path):
        self.lines = open(path).readlines()
        self.atoms = [_ for _ in self.lines if _.startswith("ATOM")]
        self.atomnames = np.array([_.split()[2] for _ in self.atoms])
        self.keys = [(_[17:21].strip(), _[12:17].strip()) for _ in self.atoms]
        self.resids = [int(_[22:27].strip()) for _ in self.atoms]
        self.n_res = len(set(self.resids))
        resids_old = list(set(self.resids))
        resids_old.sort()
        resids_new = list(range(1, len(resids_old) + 1))
        self.resids = [resids_new[resids_old.index(_)] for _ in self.resids]
        # print(self.resids)
        self.restypes = [_[16:21].strip() for _ in self.atoms]
        self.xyzs = np.array(
            [[float(_[30:38]), float(_[39:46]), float(_[47:55])] for _ in self.atoms]
        )
        self.chm_typ = np.array(
            [self.atom_types[(a, b)] for a, b in zip(self.restypes, self.atomnames)]
        )
        self.chm_typ_mask = {
            ak: np.array([ak == _ for _ in self.chm_typ]) for ak in atom_keys
        }
        self.res_mask = {
            r: np.array([r == _ for _ in self.resids]) for r in list(set(self.resids))
        }

    def load_dcm(self, path):
        """Load dcm file"""
        self.n_res = len(set(self.resids))
        with open(path) as f:
            lines = f.readlines()
        self.dcm = [
            [float(_) for _ in line.split()[1:]] for line in lines[2:]
        ]  # skip first two lines
        self.dcm_charges = self.dcm[len(self.atoms) :]
        dcm_charges_per_res = len(self.dcm_charges) // self.n_res // 3
        self.dcm_charges_mask = {
            r: np.array(
                [
                    [True] * dcm_charges_per_res
                    if r == _
                    else [False] * dcm_charges_per_res
                    for _ in self.resids
                ]
            ).flatten()
            for r in list(set(self.resids))
        }

    def get_psf(self):
        """Get psf file for structure"""
        OM = ["O"]
        CM = ["C"]
        H1M = ["H1"]
        H2M = ["H2"]
        H3M = ["H3"]
        H4M = ["H4"]
        OATOM = ["O", "OH2"]
        OATOM = [_ for _ in OATOM if _ in [x[1] for x in self.atom_types.keys()]]
        H = ["H", "H1"]
        H = [_ for _ in H if _ in [x[1] for x in self.atom_types.keys()]]
        if H[0] == "H":
            H1 = ["H1"]
        if H[0] == "H1":
            H1 = ["H2"]

        METHANOL = "MEO"
        WATER = "LIG"

        if "TIP3" in [x[0] for x in self.atom_types.keys()]:
            WATER = "TIP3"
            METHANOL = "LIG"

        return PSF.render(
            OM=OM[0],
            CM=CM[0],
            H1M=H1M[0],
            H2M=H2M[0],
            H3M=H3M[0],
            H4M=H4M[0],
            O=OATOM[0],
            H=H[0],
            H1=H1[0],
            WATER=WATER,
            METHANOL=METHANOL,
        )

    def set_2body(self):
        """Set 2-body distances"""
        #  all interacting pairs
        self.pairs = list(itertools.combinations(range(1, max(self.resids) + 1), 2))
        self.distances = [[] for _ in range(len(atom_key_pairs))]
        self.distances_pairs = [{} for _ in range(len(atom_key_pairs))]

        for res_a, res_b in self.pairs:
            for i, akp in enumerate(atom_key_pairs):
                a, b = akp
                mask_a = self.chm_typ_mask[a]
                res_mask_a = self.res_mask[res_a]
                mask_b = self.chm_typ_mask[b]
                res_mask_b = self.res_mask[res_b]
                xyza_ = self.xyzs[mask_a * res_mask_a]
                xyzb_ = self.xyzs[mask_b * res_mask_b]
                xyza = np.repeat(xyza_, xyzb_.shape[0], axis=0)
                xyzb = np.repeat(xyzb_, xyza_.shape[0], axis=0)
                #  case for same atom types
                if xyza.shape[0] > 0 and xyzb.shape[0] > 0:
                    self.distances[i].append(sqrt_einsum_T(xyza.T, xyzb.T))
                    self.distances_pairs[i][(res_a, res_b)] = []
                    self.distances_pairs[i][(res_a, res_b)].append(
                        sqrt_einsum_T(xyza.T, xyzb.T)
                    )
                #  case for different atom types
                if a != b:
                    b, a = akp
                    mask_a = self.chm_typ_mask[a]
                    res_mask_a = self.res_mask[res_a]
                    mask_b = self.chm_typ_mask[b]
                    res_mask_b = self.res_mask[res_b]
                    xyza_ = self.xyzs[mask_a * res_mask_a]
                    xyzb_ = self.xyzs[mask_b * res_mask_b]
                    xyza = np.repeat(xyza_, xyzb_.shape[0], axis=0)
                    xyzb = np.repeat(xyzb_, xyza_.shape[0], axis=0)
                    if xyza.shape[0] > 0 and xyzb.shape[0] > 0:
                        self.distances[i].append(sqrt_einsum_T(xyza.T, xyzb.T))
                        self.distances_pairs[i][(res_a, res_b)].append(
                            sqrt_einsum_T(xyza.T, xyzb.T)
                        )

    def get_monomers(self):
        out = list(set(self.resids))
        out.sort()
        return out

    def get_pairs(self):
        return self.pairs

    def get_monomer_xyz(self, res):
        """returns xyz coordinates of all atoms in residue res"""
        atom_names = self.atomnames[self.res_mask[res]]
        xyz = self.xyzs[self.res_mask[res]]
        return self.get_xyz_string(xyz, atom_names)

    def get_pair_xyz(self, res_a, res_b):
        """returns xyz coordinates of all atoms in residue res"""
        atom_names = self.atomnames[self.res_mask[res_a] + self.res_mask[res_b]]
        xyz = self.xyzs[self.res_mask[res_a] + self.res_mask[res_b]]
        return self.get_xyz_string(xyz, atom_names)

    def get_cluster_xyz(self):
        """returns xyz coordinates of all atoms in cluster"""
        atom_names = self.atomnames
        xyz = self.xyzs
        return self.get_xyz_string(xyz, atom_names)

    def get_xyz_string(self, xyz, atomnames):
        """returns a string in the format atomname x y z for all atoms in xyz"""
        xyz_string = ""
        for i, atom in enumerate(atomnames):
            xyz_string += "{} {:8.3f} {:8.3f} {:8.3f}\n".format(
                atom[:1], xyz[i, 0], xyz[i, 1], xyz[i, 2]
            )
        return xyz_string

    def get_pdb(self):
        header = """HEADER
TITLE
REMARK
"""
        pdb_format = (
            "{:6s}{:5d} {:^4s}{:1s}{:4s}{:1s}{:4d}{:1s}   "
            "{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}"
            "          {:>2s}{:2s}\n"
        )
        _str = header
        for i, line in enumerate(self.atoms):
            _1 = "ATOM"
            _2 = i + 1
            _3 = self.atomnames[i]
            _4 = ""
            _5 = self.restypes[i]
            _6 = ""
            _7 = self.resids[i]
            _8 = ""
            _9 = self.xyzs[i, 0]
            _10 = self.xyzs[i, 1]
            _11 = self.xyzs[i, 2]
            _12 = 0.0
            _13 = 0.0
            _14 = self.atomnames[i]
            _15 = " "
            _ = pdb_format.format(
                _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15
            )
            _str += _
        _str += "END"
        return _str

    def save_pdb(self, path):
        with open(path, "w") as f:
            f.write(self.get_pdb())
