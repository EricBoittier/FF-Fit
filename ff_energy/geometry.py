import numpy as np


def kabsch_rmsd(P, Q):
    """Calculate the RMSD between two sets of points using the Kabsch algorithm.

    Args:
        P (np.ndarray): Nx3 array of points (to fit)
        Q (np.ndarray): Nx3 array of points (ref.)

    Returns:
        float: RMSD between P and Q
    """
    P_save = P
    Q_save = Q
    P = P[:3]
    Q = Q[:3]

    # Center the data
    P -= np.mean(P, axis=0)
    Q -= np.mean(Q, axis=0)

    # Calculate the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Singular Value Decomposition
    V, S, W = np.linalg.svd(C)

    # Calculate the rotation matrix
    d = np.linalg.det(np.dot(V, W))
    R = np.dot(V, np.dot(np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]]), W))

    # Calculate the transformed coordinates
    P_transformed = np.dot(P_save, R)

    # Calculate the RMSD
    rmsd = np.sqrt(np.sum((Q_save - P_transformed) ** 2) / P_transformed.shape[0])

    return P_transformed, rmsd


# This one starts with two cross products to get a vector perpendicular to
# b2 and b1 and another perpendicular to b2 and b3. The angle between those vectors
# is the dihedral angle.
def dihedral3(p) -> float:
    """formula from Wikipedia article on "Dihedral angle"; formula was removed
    from the most recent version of article (no idea why, the article is a
    mess at the moment) but the formula can be found in at this permalink to
    an old version of the article:
    https://en.wikipedia.org/w/index.php?title=Dihedral_angle&oldid=689165217#Angle_between_three_vectors
    uses 1 sqrt, 3 cross products

    :returns: dihedral angle in degrees
    """
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b2, b1)

    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)

    y = np.dot(b0xb1_x_b1xb2, b1) * (1.0 / np.linalg.norm(b1))
    x = np.dot(b0xb1, b1xb2)

    return np.degrees(np.arctan2(y, x))


def bisector(dcm) -> np.array:
    """calculate the bisector of two vectors"""
    vector1 = dcm[1] - dcm[0]
    vector2 = dcm[1] - dcm[2]
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    bisector1 = (unit_vector1 + unit_vector2) / np.linalg.norm(
        unit_vector1 + unit_vector2
    )
    return bisector1


def angle(u, v) -> float:
    """calculate the angle between two vectors,
    return the angle in degrees"""
    # Calculate polar angle (theta) between u and v
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    theta = np.degrees(np.arccos(cos_theta))
    return theta


def dist(a: np.array, b=None) -> float:
    """calculate the length of a vector
    or the distance between two vectors"""
    if b is None:
        return np.linalg.norm(a)
    return np.linalg.norm(a - b)
