from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch
from unittest.mock import patch
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))

from tensors import (
    get_info,
    swap_matrix_quadrant,
    create_matrix_from_vectors,
    interleave_vectors,
    make_broadcastable_1,
    make_broadcastable_2,
    make_broadcastable_3,
    batch_scalar_product,
    transpose_matrices,
    normalize_and_abs,
    replace_near_zero,
    select_matrices_from_batch,
    generate_and_convert_tensors,
    max_column_sum,
)


tasks_solved = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


@pytest.mark.skipif(1 not in tasks_solved, reason="Task 1 not solved")
def test_info() -> None:
    matrix = torch.rand(2, 3)
    num_dim, data_type, shape, dev = get_info(matrix)
    assert data_type == torch.get_default_dtype()
    assert num_dim == 2
    assert shape == torch.Size([2, 3])
    assert dev == torch.get_default_device()

    matrix = torch.zeros(4, 5, 2, dtype=torch.int)
    num_dim, data_type, shape, dev = get_info(matrix)
    assert data_type == torch.int
    assert num_dim == 3
    assert shape == torch.Size([4, 5, 2])
    assert dev == torch.get_default_device()


@pytest.mark.skipif(2 not in tasks_solved, reason="Task 2 not solved")
def test_slicing() -> None:
    matrix = torch.arange(1, 17).reshape(4, 4)
    matrix_copy = torch.arange(1, 17).reshape(4, 4)
    expected_output = torch.tensor(
        [[11, 12, 3, 4], [15, 16, 7, 8], [9, 10, 1, 2], [13, 14, 5, 6]]
    )
    output = swap_matrix_quadrant(matrix)
    assert output.shape == expected_output.shape
    assert torch.allclose(expected_output, output, atol=1e-5)
    assert torch.allclose(matrix, matrix_copy, atol=1e-5)


@pytest.mark.skipif(3 not in tasks_solved, reason="Task 3 not solved")
def test_operations() -> None:
    tensor = torch.tensor([1.0, 2.0, -1.0, 5.0, -0.5])
    minval, maxval = -2.0, 2.0
    output = normalize_and_abs(tensor, minval, maxval)
    expected_output = torch.tensor([0.5, 1.0, 0.5, 2.5, 0.25])
    assert output.shape == expected_output.shape
    assert torch.allclose(expected_output, output, atol=1e-5)

    minval, maxval = torch.tensor([0.0] * 5), torch.tensor([4.0] * 5)
    output = normalize_and_abs(tensor, minval, maxval)
    expected_output = torch.tensor([0.5, 0.0, 1.5, 1.5, 1.25])
    assert output.shape == expected_output.shape
    assert torch.allclose(expected_output, output, atol=1e-5)


@pytest.mark.skipif(4 not in tasks_solved, reason="Task 4 not solved")
def test_masking():
    tensor = torch.tensor([[-2.0, 0.5, 1.0], [1.5, -1.0, -0.3]])
    expected_output = torch.tensor([[-2.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    output = replace_near_zero(tensor)
    assert output.shape == expected_output.shape
    assert torch.allclose(expected_output, output, atol=1e-5)


@pytest.mark.skipif(5 not in tasks_solved, reason="Task 5 not solved")
def test_matrix_selection():
    matrices = torch.arange(-45, 45).reshape([10, 3, 3])
    indices = [1, 2, 3, 5, 8]
    output = select_matrices_from_batch(matrices, indices)
    expected_output = torch.tensor(
        [
            [[-36, -35, -34], [-33, -32, -31], [-30, -29, -28]],
            [[-27, -26, -25], [-24, -23, -22], [-21, -20, -19]],
            [[-18, -17, -16], [-15, -14, -13], [-12, -11, -10]],
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            [[27, 28, 29], [30, 31, 32], [33, 34, 35]],
        ]
    )
    assert output.shape == expected_output.shape
    assert torch.allclose(expected_output, output, atol=1e-5)


@pytest.mark.skipif(6 not in tasks_solved, reason="Task 6 not solved")
def test_generation_and_conversion():
    np_array = np.array([-1.0, 2.5, 3.0], dtype=np.float32)
    use_device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    result_from_numpy, result_zeros, result_filled, result_uint8, result_device = (
        generate_and_convert_tensors(np_array, use_device)
    )
    expected_zeros = torch.tensor([0.0, 0.0, 0.0])
    expected_filled = torch.tensor(
        [[42, 42, 42, 42], [42, 42, 42, 42], [42, 42, 42, 42], [42, 42, 42, 42]]
    )
    expected_from_numpy = torch.tensor([-1.0, 2.5, 3.0])
    expected_uint8 = torch.tensor([255, 2, 3], dtype=torch.uint8)
    expected_device = torch.tensor([255, 2, 3], device=use_device, dtype=torch.uint8)

    assert result_zeros.shape == expected_zeros.shape
    assert torch.allclose(expected_zeros, result_zeros, atol=1e-5)
    assert result_filled.shape == expected_filled.shape
    assert torch.allclose(
        expected_filled, result_filled.to(expected_filled.dtype), atol=1e-5
    )
    assert result_filled.dtype in [
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.int8,
        torch.short,
        torch.int,
        torch.long,
    ]
    assert result_from_numpy.shape == expected_from_numpy.shape
    assert torch.allclose(expected_from_numpy, result_from_numpy, atol=1e-5)
    assert result_uint8.shape == expected_uint8.shape
    assert torch.allclose(expected_uint8, result_uint8, atol=1e-5)
    assert result_uint8.dtype == torch.uint8
    assert result_device.shape == expected_device.shape
    assert torch.allclose(expected_device, result_device, atol=1e-5)
    assert result_device.dtype == torch.uint8
    assert result_device.device == use_device


@pytest.mark.skipif(7 not in tasks_solved, reason="Task 7 not solved")
def test_dimensions():
    matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    output = max_column_sum(matrix)
    expected_output = torch.tensor([18])
    print(output, expected_output)
    assert output.shape == expected_output.shape
    assert torch.allclose(expected_output, output, atol=1e-5)


@pytest.mark.skipif(8 not in tasks_solved, reason="Task 8 not solved")
def test_concat() -> None:
    x, y, z = torch.tensor([1, 4]), torch.tensor([2, 5]), torch.tensor([3, 6])
    output = create_matrix_from_vectors(x, y, z)
    expected_output = torch.tensor([[1, 2, 3], [4, 5, 6]])
    assert torch.allclose(expected_output, output, atol=1e-5)


@pytest.mark.skipif(9 not in tasks_solved, reason="Task 9 not solved")
def test_reshape() -> None:
    x = torch.tensor([1, 3, 5])
    y = torch.tensor([2, 4, 6])
    output = interleave_vectors(x, y)
    expected_output = torch.tensor([1, 2, 3, 4, 5, 6])
    assert output.shape == expected_output.shape
    assert torch.allclose(expected_output, output, atol=1e-5)


@pytest.mark.skipif(10 not in tasks_solved, reason="Task 10 not solved")
def test_transposing() -> None:
    matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
    output = transpose_matrices(matrix)
    expected_output = torch.tensor([[1, 4], [2, 5], [3, 6]])
    assert output.shape == expected_output.shape
    assert torch.allclose(expected_output, output, atol=1e-5)

    matrices = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    output = transpose_matrices(matrices)
    expected_output = torch.tensor(
        [[[1, 4], [2, 5], [3, 6]], [[7, 10], [8, 11], [9, 12]]]
    )
    assert output.shape == expected_output.shape
    assert torch.allclose(expected_output, output, atol=1e-5)


@pytest.mark.skipif(11 not in tasks_solved, reason="Task 11 not solved")
def test_broadcasting() -> None:
    matrix1 = torch.rand(2, 3, 2)
    matrix2 = torch.rand(2, 3)
    result1, result2 = make_broadcastable_1(matrix1, matrix2)
    assert (result1 + result2).shape == torch.Size([2, 3, 2])

    matrix1 = torch.rand(2, 3, 4, 5, 1)
    matrix2 = torch.rand(2, 4, 1)
    result1, result2 = make_broadcastable_2(matrix1, matrix2)
    assert (result1 + result2).shape in [
        torch.Size([2, 3, 4, 5, 1]),
        torch.Size([2, 3, 4, 5]),
    ]

    matrix1 = torch.rand(3, 1, 2)
    matrix2 = torch.rand(3, 2, 7)
    result1, result2 = make_broadcastable_3(matrix1, matrix2)
    assert (result1 + result2).shape in [
        torch.Size([3, 2, 7]),
        torch.Size([3, 1, 2, 7]),
    ]

    matrix1 = torch.rand(1000, 3)
    matrix2 = torch.rand(1000, 3)
    expected_output = (matrix1 * matrix2).sum(dim=1)
    output = batch_scalar_product(matrix1, matrix2)
    assert torch.allclose(expected_output, output, atol=1e-5)
    with patch("torch.bmm") as mock_bmm:
        output = batch_scalar_product(matrix1, matrix2)
        mock_bmm.assert_called()
