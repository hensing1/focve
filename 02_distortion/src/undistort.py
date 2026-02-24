from __future__ import annotations

import numpy as np


def create_image_points(
    width: int,
    height: int,
) -> np.ndarray[np.float]:  # H x W x 2
    points = np.zeros((height, width))

    # TODO: Implement ...

    return points


def apply_camera_matrix(
    camera_matrix: np.ndarray,  # 3 x 3
    points: np.ndarray,  # H x W x 2
) -> np.ndarray:  # H x W x 2
    result = np.zeros(points.shape)

    # TODO: Implement ...

    return result


def apply_inverse_camera_matrix(
    camera_matrix: np.ndarray,  # 3 x 3
    points: np.ndarray,  # H x W x 2
) -> np.ndarray:  # H x W x 2
    result = np.zeros(points.shape)

    # TODO: Implement ...

    return result


def distort_points(
    points: np.ndarray,  # H x W x 2
    dist_coeffs: np.ndarray,  # 8
) -> np.ndarray:  # H x W x 2

    k1 = dist_coeffs[0]
    k2 = dist_coeffs[1]

    p1 = dist_coeffs[2]
    p2 = dist_coeffs[3]

    k3 = dist_coeffs[4]
    k4 = dist_coeffs[5]
    k5 = dist_coeffs[6]
    k6 = dist_coeffs[7]

    xp = points[..., 0]
    yp = points[..., 1]

    distorted_points = np.zeros(points.shape)

    # TODO: Implement ...

    return distorted_points


def remap(
    image: np.ndarray,  # H_1 x W_1 x C
    points: np.ndarray[np.float32],  # H_2 x W_2 x 2
) -> np.ndarray:  # H_2 x W_2 x C
    # Remap without interpolation. Round to next pixel index!

    remapped = np.zeros(image.shape)

    # TODO: Implement ...

    return remapped


def undistort_image(image, dist_coeffs, camera_matrix):
    height, width = image.shape[0:2]

    points = create_image_points(width, height)
    # points : np.array[ shape=( height, width, 2 ) ]

    # Inverse camera matrix to standard coordinate system
    points = apply_inverse_camera_matrix(camera_matrix, points)

    # Distort the points according to the distortion coefficients
    distorted_points = distort_points(points, dist_coeffs)

    # Apply camera matrix to image coordinate system
    distorted_points = apply_camera_matrix(camera_matrix, distorted_points)

    # Remap the image using distorted points
    undistorted_image = remap(image, distorted_points)
    return undistorted_image
