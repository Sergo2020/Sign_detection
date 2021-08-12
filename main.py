import os
from pathlib import Path

import numpy as np

import algorithms as alg
import io_utils as ply


# https://github.com/falcondai/py-ransac/blob/master/ransac.py


def detect_sign(points):

    scene = ply.Scene_viewer(points)
    sign_detector = alg.Sign_Detector()

    scene.show_cluster(points, True)

    sign_detector.fit_kde(points[:, -1])
    dens, dens_x = sign_detector.produce_density_arr(points[:, -1], 100, show=False)
    mode_status = sign_detector.detect_modes(dens_x, dens)

    if mode_status != 1:
        print('The cluster is not a sign.')
        os.exit()

    points_plate, points_pole = sign_detector.separate_by_thresh(points)

    scene.show_cluster(points_plate, True)
    scene.show_cluster(points_pole, True)

    plate_plane = alg.Plane(points_plate, 0.75)

    scene.show_cluster(np.concatenate((plate_plane.inliers, plate_plane.outliers), 0), True)

    projected_points = plate_plane.project_points(points_plate)
    scene.show_cluster(projected_points, True)

    points_2d = plate_plane.projected_2d(projected_points)
    hull, verticies = plate_plane.get_hull(points_2d)

    img = plate_plane.coord2d_pix(points_2d, verticies)

    ply.show_image(img)

    lines = plate_plane.detect_lines(img)
    ply.show_image(lines)
    # ply.scatter_2d(points_2d, verticies)

    # lines = plate_plane.detect_shape(verticies, ply.scatter_2d)
    # ply.scatter_2d(points_2d, lines)


# TODO: Separate points by R - DONE
# TODO: Check that all plane points are on the same surface - DONE
# TODO: Project points to plane - DONE
# TODO: Check what type of plane it is - DONE?
# TODO: Fix code and check for usability
# TODO: Blender simulation - optional

if __name__ == '__main__':
    pole_path = Path(r"Objects\rot_z\pole.ply")
    plate_path = Path(r"Objects\rot_z\triangle.ply")

    sim_points = ply.prep_scene(plate_path, pole_path)
    detect_sign(sim_points)


    # cluster_path = Path(r"Objects\cluster_2.csv")
    # points = ply.read_csv(cluster_file_path)
    # detect_sign(points)
