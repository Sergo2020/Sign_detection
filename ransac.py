import numpy as np
import numpy.linalg as la


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
    # return: 1d array of size N (number of points)
    dists = np.abs(points @ plane) / np.sqrt(plane[0]**2 + plane[1]**2 + plane[2]**2)
    return dists

def fit_plane_LSE_RANSAC(points, criteria_n, iters=1000, thresh=0.01, return_outlier_list=False):
    # points: (x,y,z, 1)
    # return:
    #   plane: 1d array of four elements [a, b, c, d] of ax+by+cz+d = 0
    #   inlier_list: 1d array of size N of inlier points
    max_inlier_num = -1
    max_inlier_list = None

    N = points.shape[0]
    assert N >= 3

    for i in range(iters):
        chose_id = np.random.choice(N, 3, replace=False)
        chose_points = points[chose_id, :]
        tmp_plane = fit_plane_LSE(chose_points)

        dists = get_point_dist(points, tmp_plane)
        tmp_inlier_list = np.where(dists < thresh)[0]
        tmp_inliers = points[tmp_inlier_list, :]
        num_inliers = tmp_inliers.shape[0]
        if num_inliers > max_inlier_num:
            max_inlier_num = num_inliers
            max_inlier_list = tmp_inlier_list

            if criteria_n <= max_inlier_num:
                break

        # print('iter %d, %d inliers' % (i, max_inlier_num))

    final_points = points[max_inlier_list, :]
    plane = fit_plane_LSE(final_points)
    dists = get_point_dist(points, plane)

    sorted_idx = np.argsort(dists)
    dists = dists[sorted_idx]
    points = points[sorted_idx]

    inlier_list = np.where(dists < thresh)[0]

    if not return_outlier_list:
        return plane, points[inlier_list]
    else:
        outlier_list = np.where(dists >= thresh)[0]
        return plane, points[inlier_list], points[outlier_list]
