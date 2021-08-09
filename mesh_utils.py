import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

def show_cluster(points: np.array) -> None:
    limits = {}
    for ax_idx, ax in enumerate(['x', 'y', 'z']):
        limits[ax] = (points[:, ax_idx].min(), points[:, ax_idx].max())

    fig = plt.figure( figsize=(12, 12))
    ax_3d = fig.add_subplot(2, 2, 1, projection='3d')  # 3D, xy, yz, xz
    ax_3d.title.set_text('3D Scene')
    ax_3d.set_xlim3d(*limits['x'])
    ax_3d.set_ylim3d(*limits['y'])
    ax_3d.set_zlim3d(*limits['z'])
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.scatter3D(points[:, 0], points[:, 1], points[:, 2], c = points[:, 3])

    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xy.title.set_text('X-Y Projection')
    ax_xy.scatter(points[:, 0], points[:, 1])
    ax_xy.set_xlabel('y')
    ax_xy.set_ylabel('x')
    ax_xy.grid(True)

    ax_yz = fig.add_subplot(2, 2, 3)
    ax_yz.title.set_text('Y-Z Projection')
    ax_yz.scatter(points[:, 1], points[:, 2])
    ax_yz.set_xlabel('y')
    ax_yz.set_ylabel('z')
    ax_yz.grid(True)

    ax_xz = fig.add_subplot(2, 2, 4)
    ax_xz.title.set_text('X-Z Projection')
    ax_xz.scatter(points[:, 0], points[:, 2])
    ax_xz.set_xlabel('x')
    ax_xz.set_ylabel('z')
    ax_xz.grid(True)
    plt.show()

    # print(len(points))
    #
    # plt.figure(figsize=(6, 6))
    # ax = plt.axes(projection='3d')
    # ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], linewidth=0, antialiased=False)
    # plt.show()
