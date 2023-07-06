from dscribe.descriptors import SOAP
from scipy.spatial.distance import pdist, squareform
import numpy as np
from sklearn.decomposition import PCA




order_of_soap_features = """
for Z in atomic numbers in increasing order:
   for Z' in atomic numbers in increasing order:
      for l in range(l_max+1):
         for n in range(n_max):
            for n' in range(n_max):
               if (n', Z') >= (n, Z):
                  append p(\\chi)^{Z Z'}_{n n' l}` to output
"""
def soap(rcut=3.0, nmax=8, lmax=6, species=("H", "O"), average='off'):
    # Setting up the SOAP descriptor
    soap = SOAP(
        species=species,
        periodic=False,
        r_cut=rcut,
        n_max=nmax,
        l_max=lmax,
        average=average,
    )
    return soap


def soap_dist(molecule_soaps):
    molecules = np.vstack(molecule_soaps)
    distance = squareform(pdist(molecules))
    return distance


WATER_SOAP = soap()

