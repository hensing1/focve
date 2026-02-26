from __future__ import annotations

import pathlib
import sys

import numpy as np
import scipy.interpolate
import scipy.spatial.transform
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
from tsdf_fusion import (
    TSDFGrid,
    compute_tsdf_and_weight_from_depth,
    integrate_values,
    project_points,
)


def unproject_points(
    pixels: torch.Tensor,  # N x 2
    depths: torch.Tensor,  # N x 1
    K: torch.Tensor,  # 3 x 3
    T: torch.Tensor,  # 4 x 4
) -> torch.Tensor:  # N x 3
    N = pixels.shape[0]

    pixels_homogenous = torch.cat([pixels, torch.ones((N, 1))], dim=1)

    points_camera = depths * (torch.linalg.inv(K) @ pixels_homogenous.T).T

    points_camera_homogenous = torch.cat([points_camera, torch.ones((N, 1))], dim=1)

    points_homogenous = (torch.linalg.inv(T) @ points_camera_homogenous.T).T

    return points_homogenous[:, 0:3]


def test_project_points():
    w, h = 640, 480
    samples = 10000

    pixels = torch.cat(
        [
            torch.distributions.Uniform(0, w).sample([samples, 1]),
            torch.distributions.Uniform(0, h).sample([samples, 1]),
        ],
        dim=1,
    )
    depths = torch.distributions.Uniform(0.5, 5.0).sample([samples, 1])

    f_x = 525
    f_y = 525
    c_x = 319.5
    c_y = 239.5
    K = torch.tensor(
        [
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )

    axis = np.array([1, -2, 3], dtype=np.float32)
    axis = axis / np.linalg.norm(axis)
    R = scipy.spatial.transform.Rotation.from_rotvec(np.pi / 2 * axis).as_matrix()
    t = np.array([-2.4, 3.7, -0.1], dtype=np.float32)
    T = torch.eye(4, dtype=torch.float32)
    T[0:3, 0:3] = torch.from_numpy(R)
    T[0:3, 3] = torch.from_numpy(t)

    points = unproject_points(pixels, depths, K, T)

    pixels_reprojected, depths_reprojected = project_points(points, K, T)

    assert torch.allclose(pixels_reprojected, pixels, atol=1e-2)
    assert torch.allclose(depths_reprojected, depths, atol=1e-2)


def test_compute_tsdf_and_weight_from_depth():
    samples = 10000

    depths = torch.distributions.Uniform(0.5, 5.0).sample([samples, 1])
    points_z = torch.distributions.Uniform(0.001, 10.0).sample([samples, 1])
    truncation_region = 1.5

    points_tsdf, points_w = compute_tsdf_and_weight_from_depth(
        depths, points_z, truncation_region
    )

    assert (points_tsdf >= -truncation_region).all()
    assert (points_tsdf <= truncation_region).all()
    assert (points_tsdf[points_z < depths] > torch.zeros(1)).all()  # Positive outside
    assert (points_tsdf[points_z > depths] < torch.zeros(1)).all()  # Negative inside

    assert (points_w >= torch.zeros(1)).all()
    assert points_w.nonzero().numel() > 0
    assert (points_w[points_tsdf == -truncation_region] == torch.zeros(1)).all()

    # Test case where depth is 0 due to background pixels, but e.g. the weight is somehow set incorrectly and not 0

    depths = torch.zeros([1, 1])
    points_z = torch.tensor([[2.0]])
    truncation_region = 1.5

    points_tsdf, points_w = compute_tsdf_and_weight_from_depth(
        depths, points_z, truncation_region
    )

    assert (points_tsdf >= -truncation_region).all()
    assert (points_tsdf <= truncation_region).all()
    assert (points_tsdf[points_z < depths] > torch.zeros(1)).all()  # Positive outside
    assert (points_tsdf[points_z > depths] < torch.zeros(1)).all()  # Negative inside

    assert (points_w >= torch.zeros(1)).all()
    assert points_w.nonzero().numel() == 0 
    assert (points_w[points_tsdf == -truncation_region] == torch.zeros(1)).all()


def test_integrate_values():
    samples = 10000

    TSDF_old = torch.linspace(-2, 4, samples)[:, None]
    COLOR_old = torch.distributions.Uniform(0, 1).sample([samples, 3])
    W_old_one = torch.ones(samples)[:, None]
    W_old_zero = torch.zeros(samples)[:, None]

    tsdf = torch.linspace(5, -1, samples)[:, None]
    color = torch.distributions.Uniform(0, 1).sample([samples, 3])
    w_one = torch.ones(samples)[:, None]
    w_zero = torch.zeros(samples)[:, None]

    TSDF_new, COLOR_new, W_new = integrate_values(
        TSDF_old, COLOR_old, W_old_one, tsdf, color, w_one
    )
    assert torch.allclose(TSDF_new, torch.full((1,), 1.5))
    assert (COLOR_new >= torch.zeros(1)).all()
    assert (COLOR_new <= torch.ones(1)).all()
    assert torch.allclose(W_new, torch.full((1,), 2.0))

    TSDF_new, COLOR_new, W_new = integrate_values(
        TSDF_old, COLOR_old, W_old_zero, tsdf, color, w_one
    )
    assert torch.allclose(TSDF_new, tsdf)
    assert torch.allclose(COLOR_new, color)
    assert torch.allclose(W_new, w_one)

    TSDF_new, COLOR_new, W_new = integrate_values(
        TSDF_old, COLOR_old, W_old_one, tsdf, color, w_zero
    )
    assert torch.allclose(TSDF_new, TSDF_old)
    assert torch.allclose(COLOR_new, COLOR_old)
    assert torch.allclose(W_new, W_old_one)

    TSDF_new, COLOR_new, W_new = integrate_values(
        TSDF_old, COLOR_old, W_old_zero, tsdf, color, w_zero
    )
    assert torch.isnan(TSDF_new).nonzero().numel() == 0
    assert torch.isnan(COLOR_new).nonzero().numel() == 0
    assert torch.allclose(W_new, W_old_zero)


def test_TSDFGrid_integrate():
    resolution = 200
    truncation_region = 1.5

    bound_low = torch.tensor([-2, -2, 0], dtype=torch.float32)
    bound_size = 6
    grid = TSDFGrid(
        resolution,
        bound_low,
        bound_size,
        truncation_region,
    )

    w, h = 640, 480
    pixels = torch.stack(
        torch.meshgrid(
            [
                torch.arange(w, dtype=torch.float32),
                torch.arange(h, dtype=torch.float32),
            ],
            indexing="xy",
        ),
        dim=2,
    )  # h x w x 2

    depth = (
        0.25
        * (
            torch.sin(pixels.reshape([-1, 2])[:, 0] / 100)
            + torch.cos(pixels.reshape([-1, 2])[:, 1] / 100)
        )
        + 3
    ).reshape([h, w, 1])
    rgb = torch.distributions.Uniform(0, 1).sample([h, w, 3])

    f_x = 525
    f_y = 525
    c_x = 319.5
    c_y = 239.5
    K = torch.tensor(
        [
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )

    T = torch.eye(4, dtype=torch.float32)

    grid.integrate(K, T, rgb, depth)

    points = unproject_points(pixels.reshape([-1, 2]), depth.reshape([-1, 1]), K, T)

    grid_points = grid.points.reshape([resolution, resolution, resolution, 3]).numpy()
    grid_tsdf = grid.tsdf.reshape([resolution, resolution, resolution]).numpy()

    pixels_tsdf = scipy.interpolate.interpn(
        (grid_points[:, 0, 0, 0], grid_points[0, :, 0, 1], grid_points[0, 0, :, 2]),
        grid_tsdf,
        points,
        method="linear",
    )

    # TSDF at surface should be roughly 0
    # -> Pixels close to the border are allowed to have larger values
    assert np.abs(pixels_tsdf.min()) < bound_size / resolution
    assert np.abs(np.percentile(pixels_tsdf, 95)) < bound_size / resolution
