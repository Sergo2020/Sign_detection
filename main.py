import os
from pathlib import Path

import algorithms as alg
import io_utils as ply


# https://github.com/falcondai/py-ransac/blob/master/ransac.py


def detect_sign(cluster_file_path: Path):
    points = ply.read_csv(cluster_path)

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



# TODO: Separate points by R - DONE
# TODO: Check that all plane points are on the same surface (find normal?)
# TODO: Check what type of plane it is
# TODO: Fix code and check for usability
# TODO: Blender simulation - optional

if __name__ == '__main__':
    obj_path = Path(r"Objects\cube.ply")
    cluster_path = Path(r"Objects\cluster_2.csv")

    detect_sign(cluster_path)
