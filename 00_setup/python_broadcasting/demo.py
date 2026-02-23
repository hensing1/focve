# /// script
# dependencies = [
#   "numpy",
#   "torch",
# ]
# ///

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
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


def demo1():
    print("##### Exercise 1 #####\n")
    T = torch.arange(1, 17).reshape(4, 4)
    print("Tensor: ")
    print(T)
    num_dim, data_type, shape, dev = get_info(T)
    print("Number of dimensions:", num_dim)
    print("Data type:", data_type)
    print("Shape:", shape)
    print("Device:", dev)
    print("\n")


def demo2():
    print("##### Exercise 2 #####\n")
    T = torch.arange(17, 33).reshape(4, 4)
    print("Tensor: ")
    print(T)
    output = swap_matrix_quadrant(T)
    print("Swapped Quadrants:")
    print(output)
    print("\n")


def demo3():
    print("##### Exercise 3 #####\n")
    T = torch.tensor([7.5, 2, -4, -10, 5])
    minval, maxval = -5.0, 5.0
    print("Tensor: ")
    print(T)
    print(f"minval = {minval}, maxval = {maxval}")
    output = normalize_and_abs(T, minval, maxval)
    print("Absolute values of tensor normalized to [-1, 1]:")
    print(output)
    print("\n")


def demo4():
    print("##### Exercise 4 #####\n")
    T = torch.tensor([[1.2, 0.7, -1.5, -0.5], [0.34, 1.4, -2, 1]])
    print("Tensor: ")
    print(T)
    output = replace_near_zero(T)
    print("Replaced values from [-1, 1] with 0:")
    print(output)
    print("\n")


def demo5():
    print("##### Exercise 5 #####\n")
    T = torch.arange(20).reshape([5, 2, 2])
    indices = [1, 3]
    print("Tensor: ")
    print(T)
    print("Indices:", indices)
    print()
    output = select_matrices_from_batch(T, indices)
    print("Selected matrices:")
    print(output)
    print("\n")


def demo6():
    print("##### Exercise 6 #####\n")
    np_array = np.array([1.5, 42, 275], dtype=np.float32)
    use_device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print("Numpy array:", np_array)
    print("Device:", use_device)
    result_from_numpy, result_zeros, result_filled, result_uint8, result_device = (
        generate_and_convert_tensors(np_array, use_device)
    )
    print("Tensor from numpy array:", result_from_numpy)
    print("Zero-tensor with the same size:", result_zeros)
    print("4x4 tensor containing integer 42:")
    print(result_filled)
    print("Data type of 4x4 tensor:", result_filled.dtype)
    print("8-bit unsigned int tensor:", result_uint8)
    print("Data type 8-bit unsigned int tensor:", result_uint8.dtype)
    print("Tensor moved to device:", result_device)
    print("Device of tensor moved to device:", result_device.device)
    print("Data type of tensor moved to device:", result_device.dtype)
    print("\n")


def demo7():
    print("##### Exercise 7 #####\n")
    x, y, z = (
        torch.tensor([-2.0, 5.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([3.14, 42.0]),
    )
    print("Vectors:")
    print("x:", x)
    print("y:", y)
    print("z:", z)
    output = create_matrix_from_vectors(x, y, z)
    print("Matrix with vectors as columns:")
    print(output)
    print("\n")


def demo8():
    print("##### Exercise 8 #####\n")
    T = torch.tensor([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
    print("Tensor: ")
    print(T)
    output = max_column_sum(T)
    print("Maximum of the sum of each column:", output)
    print("Shape:", output.shape)
    print("\n")


def demo9():
    print("##### Exercise 9 #####\n")
    x, y = torch.tensor([9, 7, 5]), torch.tensor([8, 6, 4])
    print("Vectors:")
    print("x:", x)
    print("y:", y)
    output = interleave_vectors(x, y)
    print("Interleaved vector:", output)
    print("\n")


def demo10():
    print("##### Exercise 10 #####\n")
    T = torch.tensor([[10, 12, 14], [16, 18, 20]])
    print("Tensor:")
    print(T)
    print("Shape:", T.shape)
    output = transpose_matrices(T)
    print("Transposed tensor:")
    print(output)
    print("Shape:", output.shape)
    print()
    T = torch.tensor([[[21, 22, 23], [24, 25, 26]], [[27, 28, 29], [30, 31, 32]]])
    print("Tensors:")
    print(T)
    print("Shape:", T.shape)
    output = transpose_matrices(T)
    print("Transposed tensors:")
    print(output)
    print("Shape:", output.shape)
    print("\n")


def demo11():
    print("##### Exercise 11 #####\n")
    print("### Part (a) ###\n")
    T1 = torch.arange(12).reshape((2, 3, 2))
    T2 = torch.arange(6).reshape((2, 3))
    print("Tensors:")
    print(T1)
    print(T2)
    result1, result2 = make_broadcastable_1(T1, T2)
    result = result1 + result2
    print("Elementwise sum of broadcasted tensors:")
    print(result)
    print()
    T1 = torch.arange(120).reshape((2, 3, 4, 5, 1))
    T2 = torch.arange(8).reshape((2, 4, 1))
    print("Tensor shapes:")
    print(T1.shape)
    print(T2.shape)
    result1, result2 = make_broadcastable_2(T1, T2)
    result = result1 + result2
    print("Elementwise sum of broadcasted tensors (selected entries):")
    print(result[0, 0, 0, 0, 0], result[0, 1, 2, 3, 0], result[-1, -1, -1, -1, -1])
    print()
    T1 = torch.arange(6).reshape((3, 1, 2))
    T2 = torch.arange(42).reshape((3, 2, 7))
    print("Tensor shapes:")
    print(T1.shape)
    print(T2.shape)
    result1, result2 = make_broadcastable_3(T1, T2)
    result = result1 + result2
    print("Elementwise sum of broadcasted tensors (selected entries):")
    if result.ndim == 3:
        print(result[0, 0, 0], result[1, 1, 4], result[-1, -1, -1])
    else:
        print(result[0, 0, 0, 0], result[1, 0, 1, 4], result[-1, 0, -1, -1])
    print()
    print("### Part (b) ###\n")
    T1 = torch.arange(3000).reshape((3, 1000)).t() / 1000
    T2 = torch.zeros_like(T1)
    T2[:500] = T1[500:]
    T2[500:] = T1[:500]
    output = batch_scalar_product(T1, T2)
    print("Example scalar product:")
    print("Vector 1:      ", T1[42])
    print("Vector 2:      ", T2[42])
    print("Scalar product:", output[42])
    print()
    print("Example scalar product:")
    print("Vector 1:      ", T1[256])
    print("Vector 2:      ", T2[256])
    print("Scalar product:", output[256])
    print()
    print("Example scalar product:")
    print("Vector 1:      ", T1[750])
    print("Vector 2:      ", T2[750])
    print("Scalar product:", output[750])
    print("\n")


def main() -> None:
    if 1 in tasks_solved:
        demo1()
    if 2 in tasks_solved:
        demo2()
    if 3 in tasks_solved:
        demo3()
    if 4 in tasks_solved:
        demo4()
    if 5 in tasks_solved:
        demo5()
    if 6 in tasks_solved:
        demo6()
    if 7 in tasks_solved:
        demo7()
    if 8 in tasks_solved:
        demo8()
    if 9 in tasks_solved:
        demo9()
    if 10 in tasks_solved:
        demo10()
    if 11 in tasks_solved:
        demo11()


if __name__ == "__main__":
    main()
