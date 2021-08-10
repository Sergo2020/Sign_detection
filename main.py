import os
from pathlib import Path

import algorithms as alg
import io_utils as ply


# https://github.com/falcondai/py-ransac/blob/master/ransac.py


def detect_sign(cluster_file_path: Path):
    points = ply.read_csv(clster_path)
    sign_detector = alg.Sign_Detector()
    sign_detector.fit_kde(points[:, -1])
    dens, dens_x = sign_detector.produce_density_arr(points[:, -1], 100, show=True)
    mode_status = sign_detector.detect_modes(dens_x, dens)

    if mode_status != 1:
        print('The cluster is not a sign.')
        os.exit()


if __name__ == '__main__':
    obj_path = Path(r"Objects\cube.ply")
    clster_path = Path(r"Objects\cluster_2.csv")

    detect_sign(clster_path)
