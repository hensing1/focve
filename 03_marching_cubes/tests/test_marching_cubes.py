from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
from marching_cubes import (
    compute_marching_cubes_index,
    interpolate_edge_vertex,
    marching_cubes,
    neighboring_voxels,
)
from marching_cubes_table import edge_table


def sdf_sphere(
    center: torch.Tensor, radius: float, samples: torch.Tensor
) -> torch.Tensor:
    return torch.linalg.norm(samples - torch.unsqueeze(center, 0), dim=1) - radius


def sdf_gradient_sphere(
    center: torch.Tensor, radius: float, samples: torch.Tensor
) -> torch.Tensor:
    return torch.nn.functional.normalize(samples - torch.unsqueeze(center, 0), dim=1)


def triangle_normals(
    vertices: torch.Tensor, faces: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    p0 = vertices[faces[:, 0], :]
    p1 = vertices[faces[:, 1], :]
    p2 = vertices[faces[:, 2], :]

    normal = torch.cross(p1 - p0, p2 - p0, dim=1)
    return torch.nn.functional.normalize(normal, dim=1), (p0 + p1 + p2) / 3


def test_neighboring_voxels():
    positions = neighboring_voxels(np.array((0, 0, 0), dtype=np.int64))

    assert positions.shape[0] == 8
    assert positions.shape[1] == 3
    assert positions.dtype == np.int64

    indices = np.ravel_multi_index(
        (positions[:, 2], positions[:, 1], positions[:, 0]), (2, 2, 2)
    )

    # Check order
    assert np.array_equal(indices, np.array((0, 1, 3, 2, 4, 5, 7, 6), dtype=np.int64))

    # Check absolute position
    voxel_start = np.array((-4, 8, 1), dtype=np.int64)
    assert np.array_equal(neighboring_voxels(voxel_start) - voxel_start, positions)


def bits(n):
    return bin(n).count("1")


def test_compute_marching_cubes_index():
    # Case 1: Empty
    assert (
        bits(
            edge_table[
                compute_marching_cubes_index(
                    sdf_values=np.array((0, 0, 0, 0, 0, 0, 0, 0), dtype=np.float32),
                    isovalue=0.5,
                )
            ]
        )
        == 0
    )
    assert (
        bits(
            edge_table[
                compute_marching_cubes_index(
                    sdf_values=np.array((1, 1, 1, 1, 1, 1, 1, 1), dtype=np.float32),
                    isovalue=0.5,
                )
            ]
        )
        == 0
    )

    # Case 2: One corner
    assert (
        bits(
            edge_table[
                compute_marching_cubes_index(
                    sdf_values=np.array((1, 0, 0, 0, 0, 0, 0, 0), dtype=np.float32),
                    isovalue=0.5,
                )
            ]
        )
        == 3
    )

    # Case 5: Lower half and upper half opposing signs
    assert (
        bits(
            edge_table[
                compute_marching_cubes_index(
                    sdf_values=np.array((1, 1, 0, 0, 1, 1, 0, 0), dtype=np.float32),
                    isovalue=0.5,
                )
            ]
        )
        == 4
    )

    # Case 10: Two corners
    assert (
        bits(
            edge_table[
                compute_marching_cubes_index(
                    sdf_values=np.array((1, 0, 0, 0, 0, 0, 1, 0), dtype=np.float32),
                    isovalue=0.5,
                )
            ]
        )
        == 6
    )


def test_interpolate_edge_vertex():
    p1 = np.array((0, 1, -2), dtype=np.float32)
    p2 = np.array((2, -1, 6), dtype=np.float32)

    # Midpoint
    assert np.allclose(
        interpolate_edge_vertex(p1, p2, val1=-1, val2=1, isovalue=0),
        np.array((1, 0, 2), dtype=np.float32),
    )

    # p1
    assert np.allclose(
        interpolate_edge_vertex(p1, p2, val1=-1, val2=1, isovalue=-1),
        p1,
    )

    # p2
    assert np.allclose(
        interpolate_edge_vertex(p1, p2, val1=-1, val2=1, isovalue=1),
        p2,
    )


def test_marching_cubes(size: int = 50) -> None:
    grid_1d = torch.arange(size)

    grid_verts = torch.stack(
        torch.meshgrid(3 * [grid_1d], indexing="ij"),
        dim=0,
    )  # 3 x size x size x size
    grid_verts = grid_verts.reshape([3, -1]).T.to(dtype=torch.float32)  # size^3 x 3

    center = torch.tensor(
        [size / 2 + 1.5, size / 2, size / 2 - 1.5],
        dtype=torch.float32,
    )
    radius = 0.8731945 * (
        size / 2
    )  # Use very uneven fraction to avoid zero values at the vertices

    sdf = sdf_sphere(center, radius, grid_verts)
    sdf = sdf.reshape([*(3 * [size])])

    vertices, faces = marching_cubes(sdf, 0)

    # Distance of vertices to center should be the radius
    assert torch.allclose(
        torch.linalg.norm(vertices - center, dim=1),
        torch.full((vertices.shape[0],), radius),
        atol=1e-2,
    )

    # Vertices should be close to isosurface
    sdf_verts = sdf_sphere(center, radius, vertices)
    assert torch.allclose(sdf_verts, torch.zeros_like(sdf_verts), atol=1e-2)


def test_marching_cubes_triangle_orientation(size: int = 50) -> None:
    grid_1d = torch.arange(size)

    grid_verts = torch.stack(
        torch.meshgrid(3 * [grid_1d], indexing="ij"),
        dim=0,
    )  # 3 x size x size x size
    grid_verts = grid_verts.reshape([3, -1]).T.to(dtype=torch.float32)  # size^3 x 3

    center = torch.full((3,), size / 2, dtype=torch.float32)
    radius = 0.8731945 * (
        size / 2
    )  # Use very uneven fraction to avoid zero values at the vertices

    sdf = sdf_sphere(center, radius, grid_verts)
    sdf = sdf.reshape([*(3 * [size])])

    vertices, faces = marching_cubes(sdf, 0)

    normals, triangle_centers = triangle_normals(vertices, faces)

    sdf_gradient = sdf_gradient_sphere(center, radius, triangle_centers)

    # Normals should point outwards, so < n | grad sdf > approx 1
    assert torch.allclose(
        (normals * sdf_gradient).sum(dim=1), torch.ones(normals.shape[0]), atol=1e-2
    )
