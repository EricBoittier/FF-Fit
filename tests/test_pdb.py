import unittest
from ff_energy.ffe.structure import Structure

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_pdb_read(self):
        s = Structure("../pdbs/water_tests/3zoj.pdb")
        print("natoms", len(s.atoms))
        print("nres", s.n_res)
        print("atomnames", s.atomnames)
        print("keys", s.keys)
        print(s.get_pdb())
        print(s.get_psf())

    def test_li_pdb_read(self):
        s = Structure("../pdbs/lithium/1_li.xyz.pdb")
        print("natoms", len(s.atoms))
        print("nres", s.n_res)
        print("atomnames", s.atomnames)
        print("keys", s.keys)
        print(s.get_pdb())
        print(s.get_psf())


if __name__ == '__main__':
    unittest.main()
