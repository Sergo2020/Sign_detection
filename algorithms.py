from scipy.signal import find_peaks, argrelextrema
from scipy.spatial import ConvexHull, convex_hull_plot_2d

from sklearn.neighbors import KernelDensity

import io_utils as ply
from ransac import *


class Sign_Detector:
    def __init__(self, bw=2.5):
        self._kde = KernelDensity(kernel='gaussian', bandwidth=bw)

        self.pole_val = None
        self.plane_val = None  # To be found
        self.r_threshold = None

    def fit_kde(self, data):
        self._kde.fit(data.reshape(-1, 1))

    def produce_density_arr(self, data, n_points=0, show=False):
        if n_points <= 0:
            n_points = len(data)
        dens_x = np.linspace(data.min(), data.max(), n_points).reshape(-1, 1)

        density = np.exp(self._kde.score_samples(dens_x))

        if show:
            ply.plot_fun(density, dens_x, title='Estimated density')

        return density, dens_x

    def detect_modes(self, data, density_arr, ):
        distance = 0.25 * (data.max() - data.min())  # Minimum distance is 25% from overall range
        peaks_idx = find_peaks(density_arr, distance=distance)[0]

        print(f'{len(peaks_idx)} modes are found.')

        if len(peaks_idx) != 2:
            print('Amount of modes does not match 2 - this is not a sign.')
            return 0

        # find_peaks scans from the lowest values to higher,
        # thus first value is for pole with lower R
        self.pole_val, self.plane_val = data[peaks_idx[0]], data[peaks_idx[1]]

        min_idx = argrelextrema(density_arr[peaks_idx[0]: peaks_idx[-1]], np.less)[0]
        self.r_threshold = data[peaks_idx[0] + min_idx].mean()

        return 1

    def separate_by_thresh(self, data: np.array):
        plate_idx = (data[:, -1] > self.r_threshold)
        plate_idx = plate_idx

        return data[plate_idx], data[np.bitwise_not(plate_idx)]


class Plane:
    def __init__(self, cluster_points, min_pers=0.75):

        self.coefs = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
        self.normal = np.zeros((3,))
        self.inliers = None
        self.outliers = None

        self.detect_plane_coefs(cluster_points, min_inliers_pers=min_pers)

    def detect_plane_coefs(self, points, min_inliers_pers,
                           steps=3,
                           thresh_min=0.02,
                           thresh_max=0.08):

        # xyc = np.ones((points.shape[0], 3))
        # z = points[:,-1]

        fit_points = points.copy()
        fit_points[:, -1] = 1.0

        min_n = int(min_inliers_pers * points.shape[0])

        for th in np.linspace(thresh_min, thresh_max, steps):
            plane, inliers, outliers = fit_plane_LSE_RANSAC(fit_points, min_n,
                                                            thresh=th,
                                                            return_outlier_list=True)

            if len(inliers) >= min_n:
                print(f'Excpected amount {len(inliers)} is achived with threshold {th:.2f}')
                break

        self.coefs = {'A': plane[0], 'B': plane[1], 'C': plane[2], 'D': plane[3]}

        self.normal = np.array([self.coefs['A'], self.coefs['B'], self.coefs['C']])

        self.inliers = inliers
        self.outliers = outliers

        self.inliers[:, -1] = 100
        self.outliers[:, -1] = 0

        print(f' Inliers {len(self.inliers)}/{len(points)}')

    def project_points(self, points):
        """
        points(x, y, z)

        Projects the points with coordinates x, y, z onto the plane
        defined by a*x + b*y + c*z = 1
        """

        if points.shape[1] > 3:
            coords = points[:, :3]
        else:
            coords = points

        normal_en = np.dot(self.normal, self.normal)
        unit_normal = self.normal / np.sqrt(normal_en)
        plane_point = self.inliers[0, :3].reshape(1, 3)

        points_from_point_in_plane = coords - plane_point
        proj_onto_normal_vector = np.dot(points_from_point_in_plane,
                                         unit_normal)
        proj_onto_plane = (points_from_point_in_plane -
                           proj_onto_normal_vector[:, None] * unit_normal)

        points[:, :3] = plane_point + proj_onto_plane

        return points

    def projected_2d(self, projected_points):
        if projected_points.shape[1] > 3:
            coords3d = projected_points[:, :3]
        else:
            coords3d = projected_points

        return coords3d[:,:2] /( coords3d[:,-1].reshape(-1,1) + 1e-7)

    def get_hull(self, points_2d):
        hull = ConvexHull(points_2d)

        point_simplices = np.zeros((len(points_2d), 2))

        cnt = 0
        for s in hull.simplices:
            point_simplices[cnt] = points_2d[s[0]]
            point_simplices[cnt + 1] = points_2d[s[1]]
            cnt += 2

        verticies = points_2d[hull.vertices]
        return hull, point_simplices, verticies
