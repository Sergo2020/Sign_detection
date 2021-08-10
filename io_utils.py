import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)
plt.rc('axes', labelsize=14)

plot_size = 3


def read_ply(path):
    data = pd.read_csv(path, header=None).values  # Read *.ply file as text
    n_points = int(str(data[3]).split(' ')[-1][:-2])  # Extract number of vertices (points) from scene (cloud)

    points = data[10:10 + n_points]
    # points is a numpy array of objects - strings by the format
    # "[x y z]"
    points = np.array([np.array(str(p)[2:-2].split(' '), dtype=float) for p in points])
    return points


def read_csv(path):
    points = pd.read_csv(path, header=None).values  # Read *.csv file as text
    return points


def show_cluster(points: np.array, scale: bool = False) -> None:
    limits = {}

    for ax_idx, ax in enumerate(['x', 'y', 'z']):
        limits[ax] = (points[:, ax_idx].min(), points[:, ax_idx].max(), points[:, ax_idx].mean())

    if scale:  # Scaling all the axis according to maximal diffrence
        scale_params = np.array([(limits[k][1] - limits[k][0]) for k in limits.keys()])
        max_diff = scale_params.max() / 2
        for ax in limits.keys():
            limits[ax] = (limits[ax][-1] - max_diff, limits[ax][-1] + max_diff)

    fig = plt.figure(figsize=(4 * plot_size, 4 * plot_size))
    ax_3d = fig.add_subplot(2, 2, 1, projection='3d')  # 3D, xy, yz, xz
    ax_3d.title.set_text('3D Scene')
    ax_3d.set_xlim3d(*limits['x'])
    ax_3d.set_ylim3d(*limits['y'])
    ax_3d.set_zlim3d(*limits['z'])
    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('z')
    ax_3d.scatter3D(points[:, 0], points[:, 1], points[:, 2], c=points[:, 3])

    ax_xy = fig.add_subplot(2, 2, 2)
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


def plot_hitogram(points, n_bins):
    y_labels = ['x values', 'y values', 'z values', 'R values']

    def plot_subhist(ax, data, n_bins, y_label):
        ax.hist(data, bins=n_bins)
        ax.set_ylabel('Amount')
        ax.set_xlabel(y_label)
        ax.grid(True)

    fig = plt.figure(figsize=(4 * plot_size, 4 * plot_size), tight_layout=True)

    for val_idx, label in enumerate(y_labels):
        ax = fig.add_subplot(2, 2, val_idx + 1)
        plot_subhist(ax, points[:, val_idx], n_bins, label)

    plt.show()


def plot_fun(arr_y, arr_x=None, title='Title', y_label='y', x_label='x'):
    if arr_x is None:
        arr_x = range(len(arr_y))

    plt.figure(figsize=(plot_size * 2, plot_size * 2))
    plt.plot(arr_x, arr_y)
    plt.grid(True)
    plt.show()
