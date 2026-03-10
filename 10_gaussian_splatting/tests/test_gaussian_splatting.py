import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))

import numpy as np
import torch
from gaussian_splatting import (
    compute_alpha_blending_weights,
    compute_image_positional_gradients,
    compute_jacobians,
    get_bounding_boxes,
    get_covariance_2d,
)
from utils import get_covariance_3d


def test_compute_jacobians():

    j_expected = torch.tensor(
        [
            [[100.0000, 0.0000, -300.0000], [0.0000, 250.0000, 750.0000]],
            [[-100.0000, 0.0000, -100.0000], [0.0000, -250.0000, -0.0000]],
            [[-50.0000, 0.0000, 50.0000], [0.0000, -125.0000, -62.5000]],
        ]
    )

    means = torch.tensor([[3.0, -3.0, 1.0], [1.0, 0.0, -1], [-2.0, 1.0, -2.0]])
    fx = 100
    fy = 250
    j = compute_jacobians(means, fx, fy)
    assert torch.allclose(j, j_expected), "Jacobian not computed correctly"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    means_dev = means.to(device)
    j_dev = compute_jacobians(means_dev, fx, fy)
    j_expected_dev = j_expected.to(device)
    assert j_expected_dev.device == j_dev.device, "Jacobian not on the correct device"


def test_get_covariance_2d():

    covs_2d_expected = torch.tensor(
        [
            [[65.7456, -119.0082], [-119.0082, 229.5786]],
            [[13.1574, -9.5937], [-9.5937, 46.0110]],
            [[8.7919, -8.9097], [-8.9097, 48.1457]],
        ]
    )

    scales = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
    orientations = torch.tensor([[1.0, 0.5, -0.5], [0.5, 1.0, 0.5], [-0.5, 0.5, 1.0]])

    covs_3d = get_covariance_3d(scales=scales, orientations=orientations)
    jacobians = torch.tensor(
        [
            [[1.0000, 0.0000, -3.0000], [0.0000, 2.5000, 7.5000]],
            [[-1.0000, 0.0000, -1.0000], [0.0000, -2.5000, -0.0000]],
            [[-0.5000, 0.0000, 0.5000], [0.0000, -1.2500, -0.6250]],
        ]
    )
    covs_2d = get_covariance_2d(jacobians, covs_3d)

    assert torch.allclose(
        covs_2d, covs_2d_expected
    ), "Covariances not computed correctly"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    jacobians_dev = jacobians.to(device)
    covs_3d_dev = covs_3d.to(device)
    covs_2d_dev = get_covariance_2d(jacobians_dev, covs_3d_dev)
    covs_2d_expected_dev = covs_2d_expected.to(device)
    assert (
        covs_2d_dev.device == covs_2d_expected_dev.device
    ), "Covariances not on the correct device"


def test_get_bounding_boxes():

    covs_2d = torch.tensor(
        [
            [[65.7456, -119.0082], [-119.0082, 229.5786]],
            [[13.1574, -9.5937], [-9.5937, 46.0110]],
            [[8.7919, -8.9097], [-8.9097, 48.1457]],
        ]
    )
    bb_min, bb_max = get_bounding_boxes(covs_2d, num_std_devs=3.0)

    assert bb_min.shape == torch.Size(
        [3, 2]
    ), "Minimum bounds do not have the correct shape"
    assert bb_max.shape == torch.Size(
        [3, 2]
    ), "Maximum bounds do not have the correct shape"
    assert torch.all((bb_max - bb_min) > 0), "Minimum larger than maximum found"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    covs_2d_dev = covs_2d.to(device)
    bb_min_dev, bb_max_dev = get_bounding_boxes(covs_2d_dev, num_std_devs=3.0)
    assert bb_min_dev.device == device, "Minimum bounds not on the correct device"
    assert bb_max_dev.device == device, "Maximum bounds not on the correct device"

    # Compute bounds for axis-aligned ellipse with 3*num_std_devs, should not be smaller than exact bounds
    covs_2d_simple = torch.tensor([[[9.0, 0.0], [0.0, 4.0]]])
    bb_min_simple_expected = torch.tensor([[-9.0, -6.0]])
    bb_max_simple_expected = torch.tensor([[9.0, 6.0]])
    bb_min_simple, bb_max_simple = get_bounding_boxes(covs_2d_simple, num_std_devs=3.0)
    assert torch.all(
        bb_min_simple <= bb_min_simple_expected
    ), "Minimum bounds do not include num_std_devs standard deviations in simple example, bounds too small"
    assert torch.all(
        bb_max_simple >= bb_max_simple_expected
    ), "Maximum bounds do not include num_std_devs standard deviations in simple example, bounds too small"

    # Compute bounds for rotated ellipse with 3*num_std_devs, should not be smaller than exact bounds
    covs_2d_rot = torch.tensor([[[2.5, 1.5], [1.5, 2.5]]])
    bb_min_rot_expected = torch.tensor([[-4.743416, -4.743416]])
    bb_max_rot_expected = torch.tensor([[4.743416, 4.743416]])
    bb_min_rot, bb_max_rot = get_bounding_boxes(covs_2d_rot, num_std_devs=3.0)
    assert torch.all(
        bb_min_rot <= bb_min_rot_expected
    ), "Minimum bounds do not include num_std_devs standard deviations in rotated example, bounds too small"
    assert torch.all(
        bb_max_rot >= bb_max_rot_expected
    ), "Maximum bounds do not include num_std_devs standard deviations in rotated example, bounds too small"


def test_compute_alpha_blending_weights():

    alpha_expected = torch.tensor(
        [
            [
                [0.2253, 0.2896, 0.2909, 0.2284],
                [0.2652, 0.3000, 0.2652, 0.1832],
                [0.2909, 0.2896, 0.2253, 0.1369],
                [0.2973, 0.2605, 0.1783, 0.0954],
            ],
            [
                [0.2817, 0.2178, 0.1539, 0.0994],
                [0.2840, 0.2154, 0.1494, 0.0947],
                [0.2790, 0.2078, 0.1414, 0.0880],
                [0.2672, 0.1953, 0.1305, 0.0797],
            ],
            [
                [0.0051, 0.0136, 0.0314, 0.0631],
                [0.0072, 0.0188, 0.0422, 0.0826],
                [0.0100, 0.0252, 0.0553, 0.1054],
                [0.0134, 0.0330, 0.0706, 0.1312],
            ],
        ]
    )

    h = w = 4
    ys = torch.arange(0, h)
    xs = torch.arange(0, w)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    pixel_coords = torch.stack((grid_x, grid_y), dim=-1).float()
    means_2d = torch.tensor([[1.0, 1.0], [-3.0, 3.0], [6.0, 8.0]])
    covs_2d = torch.tensor(
        [
            [[65.7456, -119.0082], [-119.0082, 229.5786]],
            [[13.1574, -9.5937], [-9.5937, 46.0110]],
            [[8.7919, -8.9097], [-8.9097, 48.1457]],
        ]
    )

    inv_covs_2d = torch.inverse(covs_2d)
    opacities = torch.tensor([0.3, 0.4, 0.5])[:, None]

    alpha = compute_alpha_blending_weights(
        means_2d=means_2d,
        pixel_coords=pixel_coords,
        inv_covs_2d=inv_covs_2d,
        opacities=opacities,
    )
    assert torch.allclose(
        alpha, alpha_expected, atol=1e-4
    ), "Alpha blending weights not computed correctly"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    means_2d_dev = means_2d.to(device)
    pixel_coords_dev = pixel_coords.to(device)
    inv_covs_2d_dev = inv_covs_2d.to(device)
    opacities_dev = opacities.to(device)
    alpha_dev = compute_alpha_blending_weights(
        means_2d=means_2d_dev,
        pixel_coords=pixel_coords_dev,
        inv_covs_2d=inv_covs_2d_dev,
        opacities=opacities_dev,
    )
    alpha_expected_dev = alpha_expected.to(device)
    assert (
        alpha_dev.device == alpha_expected_dev.device
    ), "Alpha blending weights not on the correct device"


def test_compute_image_positional_gradients():

    grad_expected = torch.tensor([[10.0, 20.0], [10.0, 20.0]])

    uv = torch.tensor(
        [
            [0.0, 3.0],
            [2.0, 1.0],
        ],
        requires_grad=True,
    )
    loss = (uv @ uv.T).mean() ** 2

    grad = compute_image_positional_gradients(uv, loss)
    assert torch.allclose(
        grad, grad_expected
    ), "Image positional gradients not computed correctly"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    uv_dev = uv.to(device)
    loss_dev = (uv_dev @ uv_dev.T).mean() ** 2
    grad_dev = compute_image_positional_gradients(uv_dev, loss_dev)
    grad_expected_dev = grad_expected.to(device)
    assert (
        grad_dev.device == grad_expected_dev.device
    ), "Image positional gradients not on the correct device"
