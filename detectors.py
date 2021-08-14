'''
Algorithmic methods and classes:
    Bi model class - detection of bi modal density, based on KDE
    Plate class - detection of plate orientation in 3D space and 2D shape
    RANSAC method

'''

import cv2 as cv
import numpy as np
from scipy.signal import find_peaks, argrelextrema
from sklearn.neighbors import KernelDensity

import lin_alg


class BiModal:
    def __init__(self, bw: float = 2.5) -> None:
        # Bi Model density estimation detection by KDE
        # bw: float, bandwidth of a KDE

        self._kde = KernelDensity(kernel='gaussian', bandwidth=bw)  # Initialization of KDE alg.

        self.pole_val = None
        self.plane_val = None
        self.r_threshold = None

    def fit_kde(self, data: np.array) -> None:
        # KDE fit by reflectivity. Input is reflectivity vector.

        self._kde.fit(data.reshape(-1, 1))

    def produce_density_arr(self, data: np.array, n_points: int = 0) -> (np.array, np.array):
        # Generation of dummy reflectance values in the range of found reflectivity

        if n_points <= 0:
            n_points = len(data)
        dens_x = np.linspace(data.min(), data.max(), n_points).reshape(-1, 1)

        density = np.exp(self._kde.score_samples(dens_x))

        return density, dens_x

    def detect_modes(self, data: np.array, density_arr: np.array, min_dist: int = 0.25) -> int:
        # Number of modes detection and threshold by simulated density function

        distance = min_dist * (data.max() - data.min())  # Minimum distance is min_dist*(max - min) from overall range
        peaks_idx = find_peaks(density_arr, distance=distance)[0]

        print(f'{len(peaks_idx)} modes are found.')

        if len(peaks_idx) != 2:
            print('Amount of modes does not match 2 - this is not a sign.')
            return 0

        self.pole_val, self.plane_val = data[peaks_idx[0]], data[peaks_idx[1]]

        min_idx = argrelextrema(density_arr[peaks_idx[0]: peaks_idx[-1]], np.less)[0]
        self.r_threshold = data[peaks_idx[0] + min_idx].mean()

        return 1

    def separate_by_thresh(self, data: np.array) -> (np.array, np.array):
        # Separate reflectivity by detected threshold.

        plate_idx = (data[:, -1] > self.r_threshold)
        plate_idx = plate_idx

        return data[plate_idx], data[np.bitwise_not(plate_idx)]


class Plate:
    def __init__(self, min_pers: float = 0.75, pix_h: int = 64, pix_w: int = 64):
        # Sign plate class. Contains plate orientation parameters, methods for orientation detection and
        # shape related methods.

        self.coefs = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
        self.areas = {'Triangle': 0, 'Circle': 0, 'Rectangle': 0}
        self.shape = None

        self.normal = np.zeros((3,))
        self.inliers = None
        self.outliers = None

        self.min_pers = min_pers

        self.h = pix_h
        self.w = pix_w

    def detect_plane_coefs(self, points: np.array,
                           steps: int = 3,
                           thresh_min: float = 0.02,
                           thresh_max: float = 0.08) -> int:
        # Plate plane coefficient estimation by RANSAC
        # There is a loop over loss (error) values, as it was not clear what is expected loss range.

        if points.shape[1] == 4:  # Translation of coordinates to homogeneous form
            fit_points = points.copy()
            fit_points[:, -1] = 1.0
        else:
            fit_points = np.ones((len(points), 4))
            fit_points[:, :3] = points

        min_n = int(self.min_pers * points.shape[0])

        inliers = []
        outliers = []

        for th in np.linspace(thresh_min, thresh_max, steps):
            plane, inliers, outliers = fit_plane_ransac(fit_points, min_n,
                                                        thresh=th)

            if len(inliers) >= min_n:
                print(f'Excpected amount {len(inliers)} is achived with threshold {th:.2f}')
                break

        if len(inliers) < min_n:
            print(f'Only {len(inliers)}/{min_n} points form a plane. Plate is not detected')
            return 0

        self.coefs = {'A': plane[0], 'B': plane[1], 'C': plane[2], 'D': plane[3]}

        self.normal = np.array([self.coefs['A'], self.coefs['B'], self.coefs['C']])

        self.inliers = inliers
        self.outliers = outliers

        self.inliers[:, -1] = 100
        self.outliers[:, -1] = 0

        print(f'Inliers {len(self.inliers)}/{len(points)}')

        return 1

    def project_to_plate(self, points: np.array) -> np.array:
        # Projection of points to plate plane

        if points.shape[1] > 3:
            coords = points[:, :3]
        else:
            coords = points

        unit_normal = self.normal / np.linalg.norm(self.normal)
        plane_point = self.inliers[0, :3].reshape(1, 3)

        points[:, :3] = lin_alg.project_3d(coords, plane_point, unit_normal)

        return points

    def rotate_plane(self, projected_points: np.array,
                     expected_normal: np.array = np.array([0, 0, 1])) -> (np.array, np.array):
        # Rotation matrix estimation from two normals: plane_normal and expected_norm.
        # Then plane rotation is preformed by found matrix.

        matrix = lin_alg.rotation_matrix_from_vectors(self.normal, expected_normal)

        projected_points[:, :3] = projected_points[:, :3] @ matrix

        pixel_points = lin_alg.coord2d_pix(projected_points, self.h, self.w)

        return projected_points, pixel_points

    def detect_shapes(self, img: np.array, draw: bool = False) -> str:
        # Shape detection from points in pixel space.
        # Method relies on OpenCV algorithms for minimum area shape detection.

        indicies = np.where(img[:, :, 0] > 0)  # Points save as image, array with shape (H,W,3)

        points = np.array([(y, x) for y, x in zip(*indicies)])

        min_rect = cv.minAreaRect(points)
        rect_points = cv.boxPoints(min_rect)
        self.areas['Rectangle'] = cv.contourArea(rect_points)

        min_circle = cv.minEnclosingCircle(points)
        self.areas['Circle'] = np.pi * min_circle[1] ** 2

        min_triangle = cv.minEnclosingTriangle(points.reshape((-1, 1, 2)))
        self.areas['Triangle'] = min_triangle[0]

        self.shape = min(self.areas, key=self.areas.get)

        print(f'The {self.shape} shaped sign is detected.')

        if draw:
            # OpenCV has some issues with coordinates and data types.
            # Therefore, besides drawing, coordinates have to be fixed per function.
            if self.shape == 'Rectangle':
                rect_points = np.array([(x, y) for y, x in np.round(rect_points).astype(int)])
                cv.drawContours(img, [rect_points], 0, (0, 0, 255), 1)

            if self.shape == 'Circle':
                center = (tuple([int(c) for c in min_circle[0][::-1]]))
                cv.circle(img, center, int(min_circle[1]), (0, 255, 0), 1)

            if self.shape == 'Triangle':
                triag_points = np.array([(x, y) for y, x in np.round(min_triangle[1]).astype(int).squeeze(1)])
                cv.drawContours(img, [triag_points], 0, (255, 0, 0), 1)

        return img


def fit_plane_ransac(points: np.array, criteria_n: int,
                     iters: int = 1000, thresh: float = 0.01) -> (np.array, np.array, np.array):
    # RANSAC for plane fitting
    # Implementation based on https://github.com/htcr/plane-fitting

    max_inlier_num = -1
    max_inlier_list = None

    n = points.shape[0]
    assert n >= 3, f'Number of points was {n}, expected => 3'

    for i in range(iters):
        chose_id = np.random.choice(n, 3, replace=False)
        chose_points = points[chose_id, :]
        tmp_plane = lin_alg.fit_plane_lse(chose_points)

        dists = lin_alg.get_point_dist(points, tmp_plane)
        tmp_inlier_list = np.where(dists < thresh)[0]
        tmp_inliers = points[tmp_inlier_list, :]
        num_inliers = tmp_inliers.shape[0]
        if num_inliers > max_inlier_num:
            max_inlier_num = num_inliers
            max_inlier_list = tmp_inlier_list

            if criteria_n <= max_inlier_num:
                break

    final_points = points[max_inlier_list, :]
    plane = lin_alg.fit_plane_lse(points=final_points)
    dists = lin_alg.get_point_dist(points, plane)

    sorted_idx = np.argsort(dists)
    dists = dists[sorted_idx]
    points = points[sorted_idx]

    inlier_list = np.where(dists < thresh)[0]
    outlier_list = np.where(dists >= thresh)[0]

    return plane, points[inlier_list], points[outlier_list]
