import uuid

class ForceField:
    """
    Simple class for FFs
    - distances
    - atom types
    - energies
     - pairwise and cluster
    """
    def __init__(self,
                 cluster,
                 pairwise,
                 atom_types,
                 groups,
                 distances,
                 name=None,
                 ):
        if name is None:
            name = f"FF{uuid.uuid4()}"
        self.cE = cluster
        self.pE = pairwise
        self.atom_types = atom_types
        self.groups = groups
        self.distances = distances
        self.numSegmentsC = len(self.groups)
        self.numSegmentsP = len(pairwise)

    def __repr__(self):
        return f"ForceField with {self.numSegmentsC} clusters " \
               f"and {self.numSegmentsP} pairwise terms"

