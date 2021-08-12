'''
Support linear algebra function:
    Coordinates translation
    Solutions for system of equations
    Projection calculation

'''

import numpy as np
import numpy.linalg as la


# https://github.com/falcondai/py-ransac/blob/master/ransac.py
def project_3d(coords, plane_point, unit_normal):

    points_from_point_in_plane = coords - plane_point
    proj_onto_normal_vector = np.dot(points_from_point_in_plane,
                                     unit_normal)
    proj_onto_plane = (points_from_point_in_plane -
                       proj_onto_normal_vector[:, None] * unit_normal)

    coords = plane_point + proj_onto_plane

    return coords


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def coord2d_pix(points_2d, h=64, w=64):

    d_y = points_2d[:, 1].max() - points_2d[:, 1].min()
    d_x = points_2d[:, 0].max() - points_2d[:, 0].min()

    y_max, y_min = points_2d[:, 1].max() + 0.2 * d_y, points_2d[:, 1].min() - 0.2 * d_y
    x_max, x_min = points_2d[:, 0].max() + 0.2 * d_x, points_2d[:, 0].min() - 0.2 * d_x

    d_y = y_max - y_min
    d_x = x_max - x_min

    y2h = (points_2d[:, 1] - y_min) * h / d_y
    x2w = (points_2d[:, 0] - x_min) * w / d_x

    grid_space = np.zeros((h, w, 3), dtype='uint8')
    grid_space[y2h.astype(int), x2w.astype(int), :] = (255, 255, 255)

    return grid_space

def points_svd(points):  # V is a base

    if len(points.shape) > 3:
        points = points[:, :3]

    u, s, vh = la.svd(points)
    S = np.zeros(points.shape)
    S[:s.shape[0], :s.shape[0]] = np.diag(s)
    return u, S, vh

def fit_plane_LSE(points): # Find null space of orthogonal plane, i.e. ax+by+cz+d

    assert points.shape[0] >= 3
    U, S, Vt = points_svd(points)
    null_space = Vt[-1, :]
    return null_space

def get_point_dist(points, plane):

    dists = np.abs(points @ plane) / np.sqrt(plane[0]**2 + plane[1]**2 + plane[2]**2)
    return dists

