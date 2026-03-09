import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

import numpy as np
import torch
import torch.nn as nn
from src.minimal_nerf import (
    PositionalEncoding,
    create_rays,
    forward,
    stratified_sampling,
    volumetric_rendering,
)


def test_positional_encoding():
    # Test 1: n_output_dims
    encoding = PositionalEncoding(2)
    assert encoding.n_output_dims == 12

    # Test 2: Phases are evaluated correctly
    encoding = PositionalEncoding(4)
    x = torch.rand([5, 3]).float()
    phases = encoding._build_phases(x)
    assert phases.shape == (5, 3, 4)
    for i in range(5):
        for j in range(3):
            for k in range(4):
                torch.testing.assert_close(phases[i, j, k], 2**k * torch.pi * x[i, j]),

    # Test 3: sin/cos are applied correctly
    encoding = PositionalEncoding(4)
    x = torch.rand([5, 3]).float()
    phases = encoding._build_phases(x)
    sincos = encoding._apply_sincos(phases)
    assert sincos.shape == (5, 3, 4, 2)
    for i in range(5):
        for j in range(3):
            for k in range(4):
                torch.testing.assert_close(
                    sincos[i, j, k],
                    torch.stack(
                        [
                            torch.sin(2**k * torch.pi * x[i, j]),
                            torch.cos(2**k * torch.pi * x[i, j]),
                        ]
                    ),
                )

    # Test 4: Outputs correct encoding
    encoding = PositionalEncoding(4)
    x = torch.rand([5, 3]).float()
    outputs = encoding(x)
    assert outputs.shape == (5, 24)
    for i in range(5):
        for j in range(3):
            for k in range(4):
                torch.testing.assert_close(
                    outputs[i, 2 * k + 8 * j : 2 * (k + 1) + 8 * j],
                    torch.stack(
                        [
                            torch.sin(2**k * torch.pi * x[i, j]),
                            torch.cos(2**k * torch.pi * x[i, j]),
                        ]
                    ),
                )

    # Test 5: L = 0 returns the input
    encoding = PositionalEncoding(0)
    x = torch.rand(1, 3)
    torch.testing.assert_close(encoding(x), torch.empty(1, 0))

    # Test 6: Frequencies are multiplies of PI
    encoding = PositionalEncoding(7)
    x = torch.stack([torch.zeros(3), torch.ones(3)], dim=0)
    expected_zeros = torch.tensor([[0, 1] * 7] * 3).flatten().float()
    expected_ones = torch.tensor([[0, -1] + [0, 1] * 6] * 3).flatten().float()
    expected = torch.stack([expected_zeros, expected_ones], dim=0)
    torch.testing.assert_close(encoding(x), expected)

    if False:
        # Test 7: Support arbitrary batch sizes
        encoding = PositionalEncoding(3)
        assert encoding(torch.rand(3)).shape == (
            18,
        ), f"Expected ouput have shape (18,), but got {encoding(torch.rand(3)).shape}"
        assert encoding(torch.rand(2, 3)).shape == (
            2,
            18,
        ), f"Expected ouput have shape (2, 18), but got {encoding(torch.rand(2, 3)).shape}"
        assert encoding(torch.rand(2, 5, 3)).shape == (
            2,
            5,
            18,
        ), f"Expected ouput have shape (2, 5, 18), but got {encoding(torch.rand(2, 5, 3)).shape}"
        assert encoding(torch.rand(2, 5, 7, 3)).shape == (
            2,
            5,
            7,
            18,
        ), f"Expected ouput have shape (2, 5, 7, 18), but got {encoding(torch.rand(2, 5, 7, 3)).shape}"

    # Test 8: Propagates gradients
    x = torch.rand(1, 3, requires_grad=True)
    y = encoding(x)
    y.sum().backward()
    assert x.grad is not None, f"Expected x to have a gradient, but got None."

    # Test 9: Output is on the correct device
    encoding = PositionalEncoding(3).cuda()
    output = encoding(torch.rand(2 * 6, 3).cuda())
    assert output.device == torch.device(
        "cuda:0"
    ), f"Expected ouput to be on cuda:0, but got {output.device}."


def test_create_rays():
    # Define image dimensions and focal length
    H, W = 100, 100
    focal = 50.0

    # ---------------------------------------------------------------------
    # Using a corner pixel at (i, j) = (0, 0)
    # Compute the unrotated ray direction for the corner pixel:
    # x = (0 - 50) / 50 = -1
    # y = -(0 - 50) / 50 = 1
    # z = -1
    # Thus, the expected unnormalized direction is [-1, 1, -1],
    # and its normalized version is:
    base_dir = torch.tensor([-1.0, 1.0, -1.0])
    base_dir_normalized = base_dir / base_dir.norm()

    # Define the corner pixel coordinates.
    i = torch.tensor([0.0])
    j = torch.tensor([0.0])

    # Test 1: Identity Pose (simple corner pixel test)
    # Identity pose: rotation is identity and translation is zero.
    pose = torch.eye(4)
    origins, dirs = create_rays(i, j, pose, H, W, focal)

    # For the corner pixel, expected direction is the normalized base_dir.
    expected_origin = pose[:3, 3]  # [0, 0, 0]
    assert torch.allclose(
        dirs[0], base_dir_normalized, atol=1e-5
    ), f"Direction mismatch (identity pose): {dirs[0]} vs {base_dir_normalized}"
    assert torch.allclose(
        origins[0], expected_origin, atol=1e-5
    ), f"Origin mismatch (identity pose): {origins[0]} vs {expected_origin}"
    assert (
        origins.shape == dirs.shape
    ), f"Origins shape {origins.shape} != directions shape {dirs.shape}"
    norms = torch.norm(dirs, dim=-1)
    assert torch.allclose(
        norms, torch.ones_like(norms), atol=1e-5
    ), f"Directions not normalized (identity pose): norms = {norms}"

    # Test 2: Non-Identity Pose with Rotation around Y-axis by 90 degrees
    # Define a 90-degree rotation around the y-axis:
    # R_y = [[0,  0, -1],
    #        [0,  1,  0],
    #        [1,  0,  0]]
    R_y = torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    # Set a translation, e.g., t = [1, 2, 3]
    t = torch.tensor([1.0, 2.0, 3.0])
    pose_y = torch.cat([R_y, t.reshape(3, 1)], dim=1)

    # For the corner pixel, the unrotated ray is base_dir.
    # After applying R_y, the rotated vector is:
    # [ (0*-1 + 0*1 + -1*-1),
    #   (0*-1 + 1*1 +  0*-1),
    #   (1*-1 + 0*1 +  0*-1) ]
    # = [ 1, 1, -1 ]
    rotated_y = torch.tensor([1.0, 1.0, -1.0])
    expected_dir_y = rotated_y / rotated_y.norm()
    expected_origin_y = t  # Broadcast to every pixel.

    origins_y, dirs_y = create_rays(i, j, pose_y, H, W, focal)
    assert torch.allclose(
        dirs_y[0], expected_dir_y, atol=1e-5
    ), f"Direction mismatch (Y-axis rotation): {dirs_y[0]} vs {expected_dir_y}"
    assert torch.allclose(
        origins_y[0], expected_origin_y, atol=1e-5
    ), f"Origin mismatch (Y-axis rotation): {origins_y[0]} vs {expected_origin_y}"

    # Test 3: Non-Identity Pose with Rotation around X-axis by 90 degrees
    # Define a 90-degree rotation around the x-axis:
    # R_x = [[1,  0,  0],
    #        [0,  0, -1],
    #        [0,  1,  0]]
    R_x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    # Use a different translation, e.g., t2 = [4, 5, 6]
    t2 = torch.tensor([4.0, 5.0, 6.0])
    pose_x = torch.cat([R_x, t2.reshape(3, 1)], dim=1)

    # For the corner pixel, the unrotated ray is base_dir.
    # After applying R_x, the rotated vector is:
    # [ 1*-1 + 0*1 + 0*-1,
    #   0*-1 + 0*1 + -1*-1,
    #   0*-1 + 1*1 + 0*-1 ]
    # = [ -1, 1, 1 ]
    rotated_x = torch.tensor([-1.0, 1.0, 1.0])
    expected_dir_x = rotated_x / rotated_x.norm()
    expected_origin_x = t2  # Broadcast to every pixel.

    origins_x, dirs_x = create_rays(i, j, pose_x, H, W, focal)
    assert torch.allclose(
        dirs_x[0], expected_dir_x, atol=1e-5
    ), f"Direction mismatch (X-axis rotation): {dirs_x[0]} vs {expected_dir_x}"
    assert torch.allclose(
        origins_x[0], expected_origin_x, atol=1e-5
    ), f"Origin mismatch (X-axis rotation): {origins_x[0]} vs {expected_origin_x}"

    # Test 4: Outputs are on the correct device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    i_dev = i.to(device)
    j_dev = j.to(device)
    pose_dev = pose.to(device)
    origins_dev, dirs_dev = create_rays(i_dev, j_dev, pose_dev, H, W, focal)
    assert (
        origins_dev.device == device
    ), f"Origins on device {origins_dev.device}, expected {device}"
    assert (
        dirs_dev.device == device
    ), f"Directions on device {dirs_dev.device}, expected {device}"

    # Test 5: Grid of Pixel Coordinates using identity pose
    ii, jj = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    ii = ii.to(torch.float32).reshape(-1)
    jj = jj.to(torch.float32).reshape(-1)
    origins_grid, dirs_grid = create_rays(ii, jj, pose, H, W, focal)
    # Expect grid shapes: (W, H, 3)
    assert (
        origins_grid.shape == dirs_grid.shape == (W * H, 3)
    ), f"Unexpected grid shape: {origins_grid.shape}"
    grid_norms = torch.norm(dirs_grid, dim=-1)
    assert torch.allclose(
        grid_norms, torch.ones_like(grid_norms), atol=1e-5
    ), "Grid directions not normalized"

    # Test 6: Non-Identity Pose with Rotation around Y-axis by 90 degrees using pixel with different x and y coordinates
    # Using a pixel at (i, j) = (75, 50)
    # Compute the unrotated ray direction for the pixel:
    # x = (75 - 50) / 50 = 0.5
    # y = -(50 - 50) / 50 = 0.0
    # z = -1
    # Thus, the expected unnormalized direction is [0.5, 0.0, -1]

    # Define the pixel coordinates.
    i_coord = torch.tensor([75.0])
    j_coord = torch.tensor([50.0])

    # For the pixel, the unrotated ray is base_dir.
    # After applying R_y, the rotated vector is:
    # [ (0*0.5 + 0*0 + -1*-1),
    #   (0*0.5 + 1*0 +  0*-1),
    #   (1*0.5 + 0*0 +  0*-1) ]
    # = [ 1, 0, 0.5 ]
    rotated_coord = torch.tensor([1.0, 0.0, 0.5])
    expected_dir_coord = (rotated_coord / rotated_coord.norm())
    expected_origin_coord = t  # Broadcast to every pixel.

    origins_coord, dirs_coord = create_rays(i_coord, j_coord, pose_y, H, W, focal)
    assert torch.allclose(
        dirs_coord[0], expected_dir_coord, atol=1e-5
    ), f"Direction mismatch (different xy-coordinates): {dirs_coord[0]} vs {expected_dir_coord}"
    assert torch.allclose(
        origins_coord[0], expected_origin_coord, atol=1e-5
    ), f"Origin mismatch (different xy-coordinates): {origins_coord[0]} vs {expected_origin_coord}"


def test_stratified_sampling():
    # Test 1: Correct number of intervals
    origins = torch.zeros((10, 3))
    dirs = torch.zeros((10, 3))
    n_intervals = 5
    t_vals, _, _ = stratified_sampling(origins, dirs, n_intervals, 0.0, 1.0)
    assert (
        t_vals.shape[-1] == n_intervals
    ), "Test 1 failed: Incorrect number of interval boundaries"

    # Test 2: Correct tval spacing
    near, far = 2.0, 10.0
    n_intervals = 5
    t_vals = stratified_sampling(origins, dirs, n_intervals, near, far)[0]
    expected_spacing = 2.0
    diffs = t_vals[0, 1:] - t_vals[0, :-1]
    assert torch.allclose(
        diffs, torch.full_like(diffs, expected_spacing)
    ), "Test 2 failed: Incorrect spacing"

    # Test 3: Sample positions within intervals
    origins = torch.zeros((1, 3))
    dirs = torch.tensor([[1.0, 0.0, 0.0]])
    n_intervals = 5
    near, far = 0.0, 1.0
    t_vals, points, _ = stratified_sampling(origins, dirs, n_intervals, near, far)
    t_pos = points[..., 0].squeeze()
    for i in range(n_intervals - 1):
        lower = t_vals[0, i].item()
        upper = t_vals[0, i + 1].item()
        assert (
            lower <= t_pos[i] < upper
        ), f"Test 3 failed: Sample {i} not in interval [{lower}, {upper})"

    # Test 4: Samples are stratified (jittered within intervals)
    torch.manual_seed(42)
    n_intervals = 3
    t_vals, points, _ = stratified_sampling(origins, dirs, n_intervals, near, far)
    t_pos = points[..., 0].squeeze()
    expected_rands = torch.tensor([0.8823, 0.9150])
    expected_t_pos = torch.tensor([0.0 + 0.5 * 0.8823, 0.5 + 0.5 * 0.9150])
    assert torch.allclose(
        t_pos, expected_t_pos, atol=1e-4
    ), "Test 4 failed: Samples not stratified correctly"

    # Test 5: Broadcast directions correctly
    origins = torch.rand((5, 3))
    dirs = torch.rand((5, 3))
    n_intervals = 4
    _, _, dirs_out = stratified_sampling(origins, dirs, n_intervals, 0.0, 1.0)
    assert dirs_out.shape == (
        5,
        n_intervals - 1,
        3,
    ), "Test 5 failed: Incorrect directions shape"

    # Test 6: Outputs on correct device
    # CPU
    origins_cpu = torch.zeros((1, 3), device="cpu")
    dirs_cpu = torch.zeros((1, 3), device="cpu")
    t_vals_cpu, points_cpu, dirs_cpu = stratified_sampling(
        origins_cpu, dirs_cpu, 5, 0.0, 1.0
    )
    assert (
        t_vals_cpu.device == origins_cpu.device
    ), "Test 6 failed: CPU t_vals device mismatch"
    assert (
        points_cpu.device == origins_cpu.device
    ), "Test 6 failed: CPU points device mismatch"
    assert (
        dirs_cpu.device == origins_cpu.device
    ), "Test 6 failed: CPU dirs device mismatch"

    # CUDA (if available)
    if torch.cuda.is_available():
        origins_cuda = torch.zeros((1, 3), device="cuda")
        dirs_cuda = torch.zeros((1, 3), device="cuda")
        t_vals_cuda, points_cuda, dirs_cuda = stratified_sampling(
            origins_cuda, dirs_cuda, 5, 0.0, 1.0
        )
        assert (
            t_vals_cuda.device == origins_cuda.device
        ), "Test 6 failed: CUDA t_vals device mismatch"
        assert (
            points_cuda.device == origins_cuda.device
        ), "Test 6 failed: CUDA points device mismatch"
        assert (
            dirs_cuda.device == origins_cuda.device
        ), "Test 6 failed: CUDA dirs device mismatch"


def test_volumetric_rendering():
    # Test 1: Correct outputs for simple case
    # Use a batch size of 1, 3 sample points, and constant sigma and rgb values.
    batch_size, n_samples, channels = 1, 2, 3
    t_vals = torch.tensor([[0.0, 1.0, 2.0]])
    # Use sigma=1.0 for all samples (shape: [1, 3, 1])
    sigma = torch.ones((batch_size, n_samples))
    # Use rgb = ones (so that pixel color is the weighted sum of ones)
    rgb = torch.ones((batch_size, n_samples, channels))
    # Calculate expected behavior manually:
    T, expected = 1, 0
    for _ in range(n_samples):
        expected += T * (1 - math.exp(-1))
        T *= math.exp(-1)
    # Sum of weights = ~0.95020164, so pixel_colors should be ones * that scalar.
    expected = torch.full((batch_size, channels), expected)
    output = volumetric_rendering(rgb, sigma, t_vals)
    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    # Test 2: Returns zero if either rgb, sigma, or t_vals is zero
    # Case a: Zero rgb (non-zero sigma and t_vals)
    rgb_zero = torch.zeros((batch_size, n_samples, channels))
    out_rgb_zero = volumetric_rendering(rgb_zero, sigma, t_vals)
    torch.testing.assert_close(out_rgb_zero, torch.zeros_like(out_rgb_zero))

    # Case b: Zero sigma (non-zero rgb and t_vals)
    sigma_zero = torch.zeros((batch_size, n_samples, 1))
    out_sigma_zero = volumetric_rendering(rgb, sigma_zero, t_vals)
    torch.testing.assert_close(out_sigma_zero, torch.zeros_like(out_sigma_zero))

    # Case c: Zero t_vals (non-zero rgb and sigma)
    # With t_vals all zero, delta becomes zero so alpha=0 and output should be zero.
    t_vals_zero = torch.zeros((batch_size, n_samples))
    out_tvals_zero = volumetric_rendering(rgb, sigma, t_vals_zero)
    torch.testing.assert_close(out_tvals_zero, torch.zeros_like(out_tvals_zero))

    # Test 3: Outputs are on the correct device
    # Create tensors on the same device (CPU in this case, or GPU if available).
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t_vals_dev = t_vals.to(device)
    sigma_dev = sigma.to(device)
    rgb_dev = rgb.to(device)
    out_dev = volumetric_rendering(rgb_dev, sigma_dev, t_vals_dev)
    assert out_dev.device == device, "Output is not on the expected device."

    # Test 4: Propagates gradients
    # Create inputs with requires_grad=True and compute a scalar loss.
    t_vals_grad = t_vals.clone().detach().requires_grad_(True)
    sigma_grad = sigma.clone().detach().requires_grad_(True)
    rgb_grad = rgb.clone().detach().requires_grad_(True)
    out_grad = volumetric_rendering(rgb_grad, sigma_grad, t_vals_grad)
    loss = out_grad.sum()
    loss.backward()
    # Check that gradients are not None.
    assert t_vals_grad.grad is not None, "No gradients for t_vals."
    assert sigma_grad.grad is not None, "No gradients for sigma."
    assert rgb_grad.grad is not None, "No gradients for rgb."


# Define a constant model that returns constant sigma and rgb values.
class ConstModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pts, dirs):
        # Return constant values regardless of input:
        # sigma shape: (num_points), rgb shape: (num_points, 3)
        sigma = torch.ones(pts.shape[0])
        rgb = torch.ones(pts.shape[0], 3) * 0.5
        return sigma, rgb


class TinyModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Include a dummy parameter to check gradients.
        self.param = nn.Linear(3, 1)

    def forward(self, pts, dirs):
        rgb = torch.ones(pts.shape[0], 3) * 0.5

        return self.param(pts) + self.param(dirs), rgb


def test_forward():
    # Define image dimensions and focal length.
    H, W = 100, 100
    focal = 50.0

    # Set sampling and depth parameters.
    N_samples = 10  # number of samples along each ray
    near, far = 1.0, 2.0

    # Create dummy pixel coordinate tensors and a simple camera pose.
    batch_size = 1
    i = torch.zeros(batch_size)  # dummy pixel coordinate i
    j = torch.zeros(batch_size)  # dummy pixel coordinate j
    pose = torch.eye(4)  # a 4x4 camera pose matrix (extrinsics)

    # Test 1: Check that forward pass returns expected colors
    model = ConstModel()

    # Call forward and get the predicted colors.
    colors_pred = forward(model, i, j, pose, H, W, focal, N_samples, near, far)

    T, expected = 1, 0
    for _ in range(10):
        expected += T * (1 - math.exp(-1 / 10)) * 0.5
        T *= math.exp(-1 / 10)

    expected = torch.full((batch_size, 3), expected)
    assert torch.allclose(
        colors_pred, expected
    ), f"Expected {expected}, got {colors_pred}"

    # Test 2:  Check that gradients are being computed.
    model = TinyModel()
    colors_pred = forward(model, i, j, pose, H, W, focal, N_samples, near, far)
    loss = colors_pred.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"

    print("test_forward passed")


if __name__ == "__main__":
    test_positional_encoding()
