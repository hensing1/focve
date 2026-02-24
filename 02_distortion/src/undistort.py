from __future__ import annotations

import numpy as np


def create_image_points(
    width: int,
    height: int,
) -> np.ndarray[np.float]:  # H x W x 2

    points = np.mgrid[:width, :height]
    points = np.transpose(points, (2, 1, 0)).astype(np.float32)

    # print(points.shape)
    # print(points[0, 0])

    return points


def apply_camera_matrix(
    camera_matrix: np.ndarray,  # 3 x 3
    points: np.ndarray,  # H x W x 2
) -> np.ndarray:  # H x W x 2

    # make homogenous coords
    ones = np.ones((points.shape[0], points.shape[1]), dtype=np.float32)
    # homog_points = np.concatenate((points, ones[..., np.newaxis]), axis=2)
    homog_points = np.dstack((points, ones))

    # apply the camera matrix to each of the H x W vectors
    projected_points = np.tensordot(
        homog_points, camera_matrix, axes=([2], [1])
    )

    # un-homogenize
    projected_points /= projected_points[:, :, 2:]

    return projected_points[:, :, :2]


def apply_inverse_camera_matrix(
    camera_matrix: np.ndarray,  # 3 x 3
    points: np.ndarray,  # H x W x 2
) -> np.ndarray:  # H x W x 2

    return apply_camera_matrix(np.linalg.inv(camera_matrix), points)


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

    r_2 = xp * xp + yp * yp
    r_4 = r_2 * r_2
    r_6 = r_4 * r_2

    c = (1 + k1*r_2 + k2*r_4 + k3*r_6) / (1 + k4*r_2 + k5*r_4 + k6*r_6)

    xpyp = xp * yp
    x_dist = xp * c + 2*p1*xpyp + p2*(r_2 + 2*xp*xp)
    y_dist = yp * c + p1*(r_2 + 2*yp*yp) + 2*p2*xpyp

    distorted_points = np.stack((x_dist, y_dist), axis=2)

    return distorted_points


def remap(
    image: np.ndarray,  # H_1 x W_1 x C
    points: np.ndarray[np.float32],  # H_2 x W_2 x 2
) -> np.ndarray:  # H_2 x W_2 x C
    # Remap without interpolation. Round to next pixel index!

    coords = np.round(points).astype(np.int32)

    remapped = np.zeros_like(image)

    xs = coords[..., 0]
    ys = coords[..., 1]
    x_mask = (xs >= 0) & (xs < image.shape[1])
    y_mask = (ys >= 0) & (ys < image.shape[0])

    mask = x_mask & y_mask

    remapped = np.zeros((points.shape[0], points.shape[1], image.shape[2]))

    remapped[mask] = image[ys[mask], xs[mask]]

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
