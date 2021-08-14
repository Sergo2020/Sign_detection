'''
Innoviz assigment:
    Sign detection and classification by point cloud.

    Simulation data is commented out below.

    Sergey Sinitsa

'''

import argparse
from pathlib import Path

import numpy as np

import detectors as alg
import io_utils as io


def detect_sign(points: np.array, visualize: bool = False, min_ratio: float = 0.75) -> None:
    """
    Sign detection method. The result will be printed out.

    points: point cluster (N,4) where each points is (x,y,z,R)
    visualize: If True every step will be plotted
    min_ratio: ratio of minimum inliers for plane coefficients estimation. Default is 0.75.

    return: None. If needed, it may be changed to return plane_plate object, as
             it contains plane orientation and shape.
    """
    scene = io.SceneViewer(points)  # Initialization of scene view
    sign_detector = alg.BiModal()  # Initialization of bi modal detector

    if visualize:
        scene.show_cluster(points, False, title='Scene preview')
        scene.show_cluster(points, True, title='Scaled scene preview')  # Cluster preview

    sign_detector.fit_kde(points[:, -1])
    dens, dens_x = sign_detector.produce_density_arr(points[:, -1], 100)  # Generation of smooth density function
    mode_status = sign_detector.detect_modes(dens_x, dens)  # Mode detection and threshold detection

    io.status_report(mode_status)  # Stops the execution if not a sign

    points_plate, points_pole = sign_detector.separate_by_thresh(points)  # Two sets of points - pole and plate

    if visualize:
        scene.show_cluster(points_plate, True, title='Plate preview')  # Plate preview
        scene.show_cluster(points_pole, True, title='Pole preview')  # Pole preview

    plate_plane = alg.Plate(min_ratio)  # At least 75% of points have to form a plane
    plate_status = plate_plane.detect_plane_coefs(points_plate)  # Detect the 3D plane coefficients and normal

    io.status_report(plate_status)  # Stops the execution if not a sign

    if visualize:
        scene.show_cluster(np.concatenate((plate_plane.inliers, plate_plane.outliers), 0),
                           True, title='Plate inliers vs outliers')  # Inliers and outliers preview

    projected_points = plate_plane.project_to_plate(points_plate)  # Project plate points to plate, including outliers

    if visualize:
        scene.show_cluster(projected_points, True, title='Projection preview')  # Preview the projected points

    projected_points, img = plate_plane.rotate_plane(projected_points)  # Align plate plane with xy plane

    if visualize:
        io.show_image(img, title='Points in pixel space')  # Preview the image

    img = plate_plane.detect_shapes(img, visualize)  # Check what shape suits the most to plate

    if visualize:
        io.show_image(img, title='Points bounded by sign estimated plate shape')


if __name__ == '__main__':
    # Provided data

    parser = argparse.ArgumentParser(description='Sign detector.')

    parser.add_argument("-c", "--cluster_path", default=Path(r"Objects\cluster_2.csv"),
                        help='Path to the *.csv cluster file.', required=False)
    parser.add_argument("-r", "--min_ratio", default=0.75,
                        help='Minimum ration of inliers for plane estimation.', required=False)
    parser.add_argument("-v", "--visual", default=False,
                        help='If True results of each step will be displayed.', required=False)

    args = parser.parse_args()

    cluster_points = io.read_csv(args.cluster_path)
    detect_sign(cluster_points, args.visual, args.min_ratio)

    # Simulated data with Blender software

    # pole_path = Path(r"Objects\straight\pole.ply")
    # plate_path = Path(r"Objects\straight\triangle.ply")
    #
    # sim_points = io.prep_scene(plate_path, pole_path, noise_level=0.01)
    # detect_sign(sim_points, True, 0.6)
