from __future__ import annotations

from typing import Any

import numpy as np
import torch


# 1. Tensor information
def get_info(
    T: torch.Tensor,  # unknown
) -> tuple[int, Any, Any, Any]:
    num_dim, data_type, shape, dev = None, None, None, None

    # TODO: get the number of dimensions, the data type, and the shape of T, and the device that T is stored on

    return num_dim, data_type, shape, dev


# 2. Slicing
def swap_matrix_quadrant(
    T: torch.Tensor,  # 4 x 4
) -> torch.Tensor:  # 4 x 4
    T_result = torch.zeros(4, 4)

    # TODO: Swap the top left and bottom right 2x2 matrix quadrants of T

    return T_result


# 3. Elementary and Elementwise Operations
def normalize_and_abs(
    T: torch.Tensor,  # N
    minval: float | torch.Tensor,  # scalar | N
    maxval: float | torch.Tensor,  # scalar | N
) -> torch.Tensor:  # N
    T_result = torch.zeros(T.shape[0])

    # TODO: Normalize the entries of the sum of T1 and T2 between -1 and 1 based on the given minimum and maximum value, and then compute the absolute value of each element

    return T_result


# 4. Boolean Array Indexing
def replace_near_zero(
    T: torch.Tensor,  # unknown
) -> torch.Tensor:  # unknown

    # TODO: Replace all entries of T that are in [-1,1] with 0

    return T


# 5. Integer Array Indexing
def select_matrices_from_batch(
    T: torch.Tensor,  # B x N x M
    indices: list,  # K
) -> torch.tensor:  # K x N x M
    selected_T = torch.zeros(len(indices), T.shape[1], T.shape[2])

    # TODO: Create a tensor of matrices from the batch of matrices given in T using the given indices

    return selected_T


# 6. Tensor Generation and Data Type and Device Conversion
def generate_and_convert_tensors(
    data: np.array,  # unknown
    device: torch.device,
) -> tuple[
    torch.Tensor,  # unknown
    torch.Tensor,  # unknown
    torch.Tensor,  # 4 x 4
    torch.Tensor,  # unknown
    torch.Tensor,  # unknown
]:
    tensor_from_numpy = None
    tensor_zeros = None
    tensor_filled = None
    tensor_uint8 = None
    tensor_device = None

    # TODO: create a tensor from the given numpy array

    # TODO: create a tensor containing only zeros of the same shape as the tensor created from the numpy array

    # TODO: create a tensor of size 4 x 4 filled with the value 42

    # TODO: convert the tensor created from the numpy array to 8-bit unsigned integers

    # TODO: move the 8-bit unsigned integer tensor to the given device

    return tensor_from_numpy, tensor_zeros, tensor_filled, tensor_uint8, tensor_device


# 7. Dimensions as Arguments
def max_column_sum(
    T: torch.Tensor,  # N x M
) -> torch.Tensor:  # 1
    result = 0

    # TODO: Sum up the values of each column of T and then compute the maximum while making sure the resulting tensor is 1D with size 1

    return result


# 8. Concatenation and Stacking
def create_matrix_from_vectors(
    x: torch.Tensor,  # 2
    y: torch.Tensor,  # 2
    z: torch.Tensor,  # 2
) -> torch.Tensor:  # 2 x 3
    T = torch.zeros(2, 3)

    # TODO: create a matrix from the given vectors with the vectors as its columns

    return T


# 9. Reshape and View
def interleave_vectors(
    x: torch.Tensor,  # length
    y: torch.Tensor,  # length
) -> torch.Tensor:  # 2*length
    interleaved = torch.zeros(2 * x.shape[0])

    # TODO: Interleave the entries of the two vectors

    return interleaved


# 10. Transposing Tensors
def transpose_matrices(
    T: torch.Tensor,  # N x M or B x N x M
) -> torch.Tensor:  # M x N or B x M x N
    T_transposed = None

    # TODO: Transpose the given matrix or the given batch of matrices

    return T_transposed


# 11. Broadcasting and Singleton Dimensions
def make_broadcastable_1(
    T1: torch.Tensor,  # 2 x 3 x 2
    T2: torch.Tensor,  # 2 x 3
) -> tuple[
    torch.Tensor,  # unknown
    torch.Tensor,  # unknown
]:
    T1_broadcastable = T1
    T2_broadcastable = T2

    # TODO: make the tensors broadcastable by using singleton dimensions

    return T1_broadcastable, T2_broadcastable


# 11. Broadcasting and Singleton Dimensions
def make_broadcastable_2(
    T1: torch.Tensor,  # 2 x 3 x 4 x 5 x 1
    T2: torch.Tensor,  # 2 x 4 x 1
) -> tuple[
    torch.Tensor,  # unknown
    torch.Tensor,  # unknown
]:
    T1_broadcastable = T1
    T2_broadcastable = T2

    # TODO: make the tensors broadcastable by using singleton dimensions

    return T1_broadcastable, T2_broadcastable


# 11. Broadcasting and Singleton Dimensions
def make_broadcastable_3(
    T1: torch.Tensor,  # 3 x 1 x 2
    T2: torch.Tensor,  # 3 x 2 x 7
) -> tuple[
    torch.Tensor,  # unknown
    torch.Tensor,  # unknown
]:
    T1_broadcastable = T1
    T2_broadcastable = T2

    # TODO: make the tensors broadcastable by using singleton dimensions

    return T1_broadcastable, T2_broadcastable


# 11. Broadcasting and Singleton Dimensions
def batch_scalar_product(
    T1: torch.Tensor,  # batchsize x 3
    T2: torch.Tensor,  # batchsize x 3
) -> torch.Tensor:  # batchsize
    result = torch.zeros(T1.shape[0])

    # TODO: compute the batched dot product of the two batches of vectors by using singleton dimensions and torch.bmm

    return result
