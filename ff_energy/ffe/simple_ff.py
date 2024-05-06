class FFData:
    def __init__(self,
                 name=None,
                 distances=None,
                 atom_types=None,
                 atom_groups=None,
                 num_segments=None,
                 int_energy=None,
                 elec=None,
                 ):
        self.name = name
        self.distances = distances
        self.atom_types = atom_types
        self.atom_groups = atom_groups
        self.num_segments = num_segments
        self.int_energy = int_energy
        self.elec = elec

