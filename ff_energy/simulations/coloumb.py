import numpy as np


# calculate distance between two points
def calculate_distance(rA, rB):
    """Calculate the distance between two points

    Parameters
    ----------
    rA, rB : np.array
        The coordinates of each point.

    Returns
    -------
    distance : float
        The distance between the two points.

    Examples
    --------
    >>> r1 = np.array([0, 0, 0])
    >>> r2 = np.array([0, 1, 0])
    >>> calculate_distance(r1, r2)
    1.0

    """
    if isinstance(rA, np.ndarray) and isinstance(rB, np.ndarray):
        dist_vec = (rA - rB)
        distance = np.linalg.norm(dist_vec)
    else:
        raise TypeError("rA and rB must be np.arrays")

    return distance


# calculate pairwaise distances
def calculate_distances(xyz1, xyz2):
    """Calculate the distances between two sets of coordinates

    xyz1: np.array of shape (n_atoms, 3) in angstrom
    xyz2: np.array of shape (n_atoms, 3) in angstrom

    """
    distances = np.zeros((xyz1.shape[0], xyz2.shape[0]))
    for i, atom1 in enumerate(xyz1):
        for j, atom2 in enumerate(xyz2):
            distances[i, j] = calculate_distance(atom1, atom2)
    return distances


# coluombic energy
def calculate_coulombic_energy(q1, q2, r):
    """Calculate the coulombic energy between two particles

    Parameters
    ----------
    q1, q2 : float
        The charge of each particle.
    r : float
        The distance between the particles.

    Returns
    -------
    energy : float
        The coulombic energy between the two particles.

    Examples
    --------
    >>> calculate_coulombic_energy(1.0, 1.0, 1.0)
    138.935456

    """
    if not isinstance(q1, (int, float)) or not isinstance(q2, (int, float)):
        raise TypeError("q1 and q2 must be floats or ints")
    if not isinstance(r, (int, float)):
        raise TypeError("r must be a float or int")

    k = 138.935456
    energy = k * q1 * q2 / r

    return energy
