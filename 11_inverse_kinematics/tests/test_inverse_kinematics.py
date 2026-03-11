import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
from inverse_kinematics import (
    apply_changes,
    compute_joint_positions,
    ik_jacobian,
    ik_shift,
    ik_solve,
)


def drag_test(points, n_iterations, step_size, parents, angles, lengths):
    for i in range(n_iterations):
        target_positions = points.clone()
        target_positions[-1] += torch.tensor([step_size, 0])

        jacobian = ik_jacobian(parents, angles, lengths)
        shift = ik_shift(points, target_positions)
        changes = ik_solve(jacobian, shift)
        points, angles = apply_changes(points, angles, changes, parents, lengths)

    return points


def assert_close(expected: torch.Tensor, actual: torch.Tensor, rtol: float = 1e-05, atol: float = 1e-08) -> None:
    __tracebackhide__ = True
    def msg(str) -> str:
        return f"{str}\nExpected: {expected}\nActual: {actual}"
    torch.testing.assert_close(expected, actual, rtol=rtol, atol=atol, msg=msg)


def test_positions():
    parents = torch.tensor([-1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32)
    angles = torch.tensor([torch.inf, 0, 90, 180, 270, 180, 270, 0, 90]) * (
        torch.pi / 180
    )  # first angle not used, just for nicer indexing
    lengths = torch.tensor([0, 1, 1, 1, 1, 2, 2, 2, 2])
    points = compute_joint_positions(torch.tensor([0.0, 0.0]), angles, lengths, parents)
    reference = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [-2.0, 0.0],
            [-2.0, -2.0],
            [0.0, -2.0],
            [0.0, 0.0],
        ]
    )
    assert_close(points, reference, rtol = 1e-5, atol = 1e-5)

    parents = torch.tensor([-1, 0, 0, 1, 3, 2, 5, 4, 6], dtype=torch.int32)
    angles = torch.tensor([torch.inf, 0, 180, 270, 180, 90, 0, 90, 270]) * (
        torch.pi / 180
    )  # first angle not used, just for nicer indexing
    lengths = torch.tensor([0, 1, 1, 1, 1, 1, 1, 0.5, 0.5])
    points = compute_joint_positions(torch.tensor([0.0, 0.0]), angles, lengths, parents)
    reference = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [1.0, -1.0],
            [0.0, -1.0],
            [-1.0, 1.0],
            [0.0, 1.0],
            [0.0, -0.5],
            [0.0, 0.5],
        ]
    )
    assert_close(points, reference, rtol = 1e-5, atol = 1e-5)


def test_jacobian():
    reference = torch.tensor(
        [
            [1.0, 0.0, 0.0000e00, 0.0000e00, 0.0000e00, 0.0e00, 0.0000e00],
            [0.0, 1.0, 0.0000e00, 0.0000e00, 0.0000e00, 0.0e00, 0.0000e00],
            [1.0, 0.0, 1.0000e01, 0.0000e00, 0.0000e00, 0.0e00, 0.0000e00],
            [0.0, 1.0, -4.3711e-07, 0.0000e00, 0.0000e00, 0.0e00, 0.0000e00],
            [1.0, 0.0, 1.0000e01, 4.0000e00, 0.0000e00, 0.0e00, 0.0000e00],
            [0.0, 1.0, -4.3711e-07, 6.9282e00, 0.0000e00, 0.0e00, 0.0000e00],
            [1.0, 0.0, 1.0000e01, 4.0000e00, -2.5000e00, 0.0e00, 0.0000e00],
            [0.0, 1.0, -4.3711e-07, 6.9282e00, 4.3301e00, 0.0e00, 0.0000e00],
            [1.0, 0.0, 1.0000e01, 4.0000e00, -2.5000e00, -0.0e00, 0.0000e00],
            [0.0, 1.0, -4.3711e-07, 6.9282e00, 4.3301e00, 1.0e01, 0.0000e00],
            [1.0, 0.0, 1.0000e01, 4.0000e00, 0.0000e00, 0.0e00, 1.0000e01],
            [0.0, 1.0, -4.3711e-07, 6.9282e00, 0.0000e00, 0.0e00, -4.3711e-07],
        ]
    )

    parents = torch.tensor([-1, 0, 1, 2, 3, 2], dtype=torch.int32)
    angles = torch.tensor([torch.inf, -90, -30, 30, 0, -90]) * (
        torch.pi / 180
    )  # first angle not used, just for nicer indexing
    lengths = torch.tensor([0, 10, 8, 5, 10, 10])

    jacobian = ik_jacobian(parents, angles, lengths)
    assert_close(jacobian, reference, rtol = 1e-3, atol = 1e-5)

    parents = torch.tensor([-1, 0, 1, 2, 3], dtype=torch.int32)
    angles = torch.tensor([torch.inf, -90, -30, 30, 0]) * (
        torch.pi / 180
    )  # first angle not used, just for nicer indexing
    lengths = torch.tensor([0, 10, 8, 5, 10])

    jacobian = ik_jacobian(parents, angles, lengths)
    assert_close(jacobian, reference[:-2, :-1], rtol = 1e-3, atol = 1e-5)

    parents = torch.tensor([-1, 0, 1, 2], dtype=torch.int32)
    angles = torch.tensor([torch.inf, -90, -30, 30]) * (
        torch.pi / 180
    )  # first angle not used, just for nicer indexing
    lengths = torch.tensor([0, 10, 8, 5])

    jacobian = ik_jacobian(parents, angles, lengths)
    assert_close(jacobian, reference[:-4, :-2], rtol = 1e-3, atol = 1e-5)


def test_shift():
    n_iterations = 1000
    counter = 0
    for i in range(n_iterations):
        n = torch.randint(5, 10, (1,)).item()
        current_positions = torch.rand((n, 2))
        reference = torch.rand(2 * n)
        target_positions = current_positions + reference.view(n, 2)

        shift = ik_shift(current_positions, target_positions)
        if not torch.allclose(shift, reference, rtol=1e-4, atol=1e-5):
            counter += 1
    assert counter < 0.03 * n_iterations


def test_solve():
    n_iterations = 1000
    counter = 0
    for i in range(n_iterations):
        n = torch.randint(5, 10, (1,)).item()
        m = torch.randint(n - 4, n - 1, (1,)).item()
        matrix = 2 * torch.rand((n, m)) + 1 + 5 * torch.eye(n)[:, :m]
        reference = torch.rand((m))
        vector = matrix @ reference

        solution = ik_solve(matrix, vector)
        if not torch.allclose(solution, reference, rtol=1e-2, atol=1e-5):
            counter += 1
    assert counter < 0.03 * n_iterations


def test_changes():
    parents = torch.tensor([-1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32)
    angles_init = torch.tensor([torch.inf, 0, 90, 180, 270, 180, 270, 0, 90]) * (
        torch.pi / 180
    )  # first angle not used, just for nicer indexing
    lengths = torch.tensor([0, 1, 1, 1, 1, 2, 2, 2, 2])
    positions_init = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [-2.0, 0.0],
            [-2.0, -2.0],
            [0.0, -2.0],
            [0.0, 0.0],
        ]
    )
    changes = torch.cat(
        (
            torch.tensor([1.0, 2.0]),
            torch.tensor([270, 180, -180, 0, 0, -90, 90, -90]) * torch.pi / 180,
        )
    )
    positions, angles = apply_changes(
        positions_init, angles_init, changes, parents, lengths
    )
    positions_reference = torch.tensor(
        [
            [1.0, 2.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, -1.0],
            [0.0, -1.0],
            [-2.0, -1.0],
            [-2.0, 1.0],
            [0.0, 1.0],
        ]
    )
    angles_reference = torch.tensor([torch.inf, 270, 270, 0, 270, 180, 180, 90, 0]) * (
        torch.pi / 180
    )
    assert_close(positions, positions_reference, rtol = 1e-5, atol = 1e-5)
    assert_close(angles, angles_reference, rtol = 1e-5, atol = 1e-5)


def test_drag_arm():
    # arm with 4 links
    parents = torch.tensor([-1, 0, 1, 2, 3], dtype=torch.int32)
    angles = torch.tensor([torch.inf, -90, -30, 30, 0]) * (
        torch.pi / 180
    )  # first angle not used, just for nicer indexing
    lengths = torch.tensor([0, 10, 8, 5, 10])

    # create all joint positions
    points = compute_joint_positions(torch.tensor([0.0, 0.0]), angles, lengths, parents)

    points = drag_test(points, 1000, 3.0, parents, angles, lengths)
    relative_coordinates = points - points[0]
    expected_corrdinates = torch.stack(
        (torch.cumsum(lengths, dim=0), torch.zeros_like(lengths)), dim=1
    )

    assert torch.linalg.norm(relative_coordinates[:, 1]).item() < 1e-5
    assert torch.linalg.norm(relative_coordinates - expected_corrdinates).item() < 1e-5


def test_drag_skeleton():
    # skeleton
    parents = torch.tensor(
        [-1, 0, 1, 1, 3, 4, 1, 6, 7, 0, 9, 10, 0, 12, 13], dtype=torch.int32
    )
    angles = torch.tensor(
        [torch.inf, -90, -90, 0, 60, 90, 180, 120, 90, 45, 90, 90, 135, 90, 90]
    ) * (
        torch.pi / 180
    )  # first angle not used, just for nicer indexing
    lengths = torch.tensor([0, 15, 8, 5, 10, 10, 5, 10, 10, 5, 10, 10, 5, 10, 10])

    # create all joint positions
    points = compute_joint_positions(torch.tensor([0.0, 0.0]), angles, lengths, parents)

    points = drag_test(points, 1000, 3.0, parents, angles, lengths)
    relative_coordinates = points - points[0]
    assert torch.linalg.norm(relative_coordinates[:, 1]).item() < 1e-1
