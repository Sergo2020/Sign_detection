'''
Input/Output methods:
    Read data from files (*.ply, *.csv)
    Data pre processing (for sim clusters)
    Data visualization
    Detection status check
'''

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('legend', fontsize=20)
plt.rc('axes', labelsize=20)

plot_size = 3


class SceneViewer:
    def __init__(self, scene_points: np.array) -> None:
        # Scene class. Contains information of point cloud location and coordinate range.

        self.coord_system = ('x', 'y', 'z')
        self.scene_center = None
        self.scene_boundries = None

        self._set_scene(scene_points)

    def _calc_limits(self, points: np.array) -> dict:
        # Scene limit calculation by point coordinates

        limits = {}

        for ax_idx, ax in enumerate(self.coord_system):
            limits[ax] = (points[:, ax_idx].min(), points[:, ax_idx].max())

        return limits

    def _set_scene(self, scene_points: np.array) -> None:
        # Scene initialization

        self.scene_center = {k: c for (k, c) in zip(self.coord_system, scene_points[:, :-1].mean(0))}
        self.scene_boundries = self._calc_limits(scene_points)

        scale_params = np.array([(self.scene_boundries[k][1] - self.scene_boundries[k][0]) for k in self.coord_system])
        max_diff = scale_params.max() / 2
        for ax in self.coord_system:
            self.scene_boundries[ax] = (self.scene_center[ax] - max_diff,
                                        self.scene_center[ax] + max_diff)

    def show_cluster(self, points: np.array, scale2scene: bool = False, title: str = 'Title') -> None:
        # Show cluster in 3D and 2D projections.
        # If scale2scene is True scene is scale according to initial point cluster

        if scale2scene:  # Scaling all the axis according to maximal difference
            limits = self.scene_boundries
        else:
            limits = self._calc_limits(points)

        fig = plt.figure(figsize=(4 * plot_size, 4 * plot_size))
        fig.suptitle(title, fontsize=16)
        ax_3d = fig.add_subplot(2, 2, 2, projection='3d')  # 3D, xy, yz, xz
        ax_3d.title.set_text('3D Scene')
        ax_3d.set_xlim3d(*limits['x'])
        ax_3d.set_ylim3d(*limits['y'])
        ax_3d.set_zlim3d(*limits['z'])
        ax_3d.set_xlabel('x')
        ax_3d.set_ylabel('y')
        ax_3d.set_zlabel('z')
        ax_3d.scatter3D(points[:, 0], points[:, 1], points[:, 2], c=points[:, 3])
        ax_3d.xaxis.set_ticks([])
        ax_3d.yaxis.set_ticks([])
        ax_3d.zaxis.set_ticks([])

        ax_xy = fig.add_subplot(2, 2, 1)
        ax_xy.title.set_text('X-Y Projection')
        ax_xy.scatter(points[:, 0], points[:, 1], c=points[:, 3])
        ax_xy.set_xlabel('x')
        ax_xy.set_ylabel('y')
        ax_xy.set_xlim(*limits['x'])
        ax_xy.set_ylim(*limits['y'])
        ax_xy.yaxis.set_label_position("right")
        ax_xy.yaxis.tick_right()
        ax_xy.grid(True)

        ax_yz = fig.add_subplot(2, 2, 3)
        ax_yz.title.set_text('Y-Z Projection')
        ax_yz.scatter(points[:, 1], points[:, 2], c=points[:, 3])
        ax_yz.set_xlabel('y')
        ax_yz.set_ylabel('z')
        ax_yz.set_xlim(*limits['y'])
        ax_yz.set_ylim(*limits['z'])
        ax_yz.grid(True)

        ax_xz = fig.add_subplot(2, 2, 4)
        ax_xz.title.set_text('X-Z Projection')
        ax_xz.scatter(points[:, 0], points[:, 2], c=points[:, 3])
        ax_xz.set_xlabel('x')
        ax_xz.set_ylabel('z')
        ax_xz.set_xlim(*limits['x'])
        ax_xz.set_ylim(*limits['z'])
        ax_xz.yaxis.tick_right()
        ax_xz.yaxis.set_label_position("right")
        ax_xz.grid(True)

        fig.tight_layout()
        plt.show()


def read_ply(path: Path) -> np.array:
    # Extraction of points from *.ply file as np.array

    data = pd.read_csv(path, header=None).values
    n_points = int(str(data[3]).split(' ')[-1][:-2])
    points = data[10:10 + n_points]
    points = np.array([np.array(str(p)[2:-2].split(' '), dtype=float) for p in points])
    return points


def read_csv(path: Path) -> np.array:
    # Extraction of points from *csv file as np.array

    points = pd.read_csv(path, header=None).values  # Read *.csv file
    return points


def organize_points(xyz: np.array, mean: float, sig: float) -> np.array:
    # Simulate data by merging xyz coordinates with generated reflectivity by normal distribution

    n_coords = len(xyz)
    r = np.random.normal(mean, sig, size=(n_coords,))

    points = np.zeros((n_coords, 4))

    points[:, :3] = xyz
    points[:, -1] = r

    return points


def add_noise(arr: np.array, noise_level: float = 0.05):
    # Simulate noise in point cloud
    return arr + noise_level * np.random.randn(*arr.shape)


def status_report(status: int) -> None:
    # Break points for algorithm. Exits execution if status is 0
    if status != 1:
        print('The cluster is not a sign.')
        os.sys.exit()  # If not a sign - exit the execution


def prep_scene(path_plate: Path, path_cone: Path,
               noise_level: float = 0.05,
               plate_r_mean: float = 115.0,
               plate_r_std: float = 5.0,
               pole_r_mean: float = 25.0,
               pole_r_std: float = 5.0) -> np.array:
    # Preparation points for algorithm by merging plate cloud and pole cloud
    # Reflectivity for each is sampled from separate normal distributions.
    # Noise is added to coordinates.

    xyz_plate = read_ply(path_plate)
    xyz_cone = read_ply(path_cone)

    xyz_plate = add_noise(xyz_plate, noise_level)
    xyz_cone = add_noise(xyz_cone, noise_level)

    points_plate = organize_points(xyz_plate, plate_r_mean, plate_r_std)
    points_cone = organize_points(xyz_cone, pole_r_mean, pole_r_std)

    points_scene = np.concatenate((points_plate, points_cone), 0)
    points_scene = np.random.permutation(points_scene)

    return points_scene


def plot_hitogram(points: np.array, n_bins: int):
    # Histogram plot method
    y_labels = ['x values', 'y values', 'z values', 'R values']

    def plot_subhist(ax, data, bins, y_label):
        ax.hist(data, bins=bins)
        ax.set_ylabel('Amount')
        ax.set_xlabel(y_label)
        ax.grid(True)

    fig = plt.figure(figsize=(4 * plot_size, 4 * plot_size), tight_layout=True)

    for val_idx, label in enumerate(y_labels):
        c_ax = fig.add_subplot(2, 2, val_idx + 1)
        plot_subhist(c_ax, points[:, val_idx], n_bins, label)

    plt.show()


def plot_fun(arr_y: np.array, arr_x: (None, np.array) = None,
             title: str = 'Title', y_label: str = 'y', x_label: str = 'x') -> None:
    if arr_x is None:
        arr_x = range(len(arr_y))

    plt.figure(figsize=(plot_size * 2, plot_size * 2))
    plt.plot(arr_x, arr_y)
    plt.grid(True)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.tight_layout()
    plt.show()


def scatter_2d(arr_2d: np.array, hull: np.array,
               title: str = 'Title', y_label: str = 'y', x_label: str = 'x') -> None:
    plt.figure(figsize=(plot_size * 2, plot_size * 2))
    plt.scatter(arr_2d[:, 0], arr_2d[:, 1], c='b')
    plt.plot(hull[:, 0], hull[:, 1], c='r')
    plt.grid(True)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.tight_layout()
    plt.show()


def show_image(img_np: np.array,
               title: str = 'Image', y_label: str = 'y', x_label: str = 'x') -> None:
    plt.figure(figsize=(plot_size * 2, plot_size * 2))
    plt.imshow(img_np)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.axis(False)
    plt.tight_layout()
    plt.show()
