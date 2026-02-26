from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from marching_cubes_table import X, edge_table, triangle_table


def neighboring_voxels(
    voxel_start: np.ndarray  # 3
) -> np.ndarray:  # 8 x 3
    voxels = np.empty((8, 3), dtype=np.int64)

    # TODO: Implement ...
    offsets = np.array(
        [
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [1, 1, 0],  # 2
            [0, 1, 0],  # 3
            [0, 0, 1],  # 4
            [1, 0, 1],  # 5
            [1, 1, 1],  # 6
            [0, 1, 1],  # 7
        ],
        dtype=np.int64,
    )
    voxels[:] = voxel_start[np.newaxis, :] + offsets

    return voxels


def collect_sdf_values(
    grid: np.ndarray,  # N x N x N
    positions: np.ndarray  # 8 x 3
) -> np.ndarray:  # 8
    sdf_values = np.empty(8, dtype=np.float32)

    #TODO: Implement ...

    for i, (x, y, z) in enumerate(positions):
        sdf_values[i] = grid[x, y, z]

    return sdf_values


def compute_marching_cubes_index(
    sdf_values: np.ndarray,  # 8
    isovalue: float
) -> int:
    cube_index = 0

    # TODO: Implement ...

    for i, value in enumerate(sdf_values):
        if value > isovalue:
            cube_index |= 1 << i

    return cube_index


def interpolate_edge_vertex(
    p1: np.ndarray,  # 3
    p2: np.ndarray,  # 3
    val1: float,
    val2: float,
    isovalue: float,
) -> np.ndarray:  # 3
    p = np.empty_like(p1)

    #TODO: Implement ...
    e = 1e-8
    if abs(isovalue - val1) < e:
        return p1
    if abs(isovalue - val2) < e:
        return p2
    if abs(val2 - val1) < e:
        return p1

    t = (isovalue - val1) / (val2 - val1)
    p[:] = p1 + t * (p2 - p1)

    return p


def marching_cubes(
    volume: torch.Tensor,
    isovalue: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid = volume.cpu().numpy()

    size = grid.shape
    vertices = []

    for x in tqdm(range(size[0] - 1)):
        for y in range(size[1] - 1):
            for z in range(size[2] - 1):
                # 1. Construct local cube
                positions = neighboring_voxels(np.array((x, y, z), dtype=np.int64))

                # 2. Collect SDF values
                sdf_values = collect_sdf_values(grid, positions)

                # 3. Compute MC index
                cube_index = compute_marching_cubes_index(sdf_values, isovalue)

                # 3.1 No edges -> early out
                if edge_table[cube_index] == 0:
                    continue

                # 4. Compute edge vertices
                edge_to_vertex_table = [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 0],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                    [7, 4],
                    [0, 4],
                    [1, 5],
                    [2, 6],
                    [3, 7],
                ]

                edge_vertex = np.empty((12, 3), dtype=np.float32)
                for i in range(12):
                    corner_i = edge_to_vertex_table[i][0]
                    corner_j = edge_to_vertex_table[i][1]

                    if edge_table[cube_index] & (1 << i):
                        edge_vertex[i] = interpolate_edge_vertex(
                            positions[corner_i].astype(np.float32),
                            positions[corner_j].astype(np.float32),
                            sdf_values[corner_i],
                            sdf_values[corner_j],
                            isovalue,
                        )

                # 5. Triangulate edge vertices
                for i in range(5):
                    if (
                        triangle_table[cube_index][3 * i + 0] == X
                        or triangle_table[cube_index][3 * i + 1] == X
                        or triangle_table[cube_index][3 * i + 2] == X
                    ):
                        break

                    v1 = edge_vertex[triangle_table[cube_index][3 * i + 0]]
                    v2 = edge_vertex[triangle_table[cube_index][3 * i + 1]]
                    v3 = edge_vertex[triangle_table[cube_index][3 * i + 2]]

                    vertices.append(v1)
                    vertices.append(v2)
                    vertices.append(v3)

    vertices = np.stack(vertices, dtype=np.float32).reshape(-1, 3)

    torch_verts = torch.from_numpy(vertices).to(volume.device)

    torch_faces = (
        torch.arange(torch_verts.shape[0], dtype=torch.int64)
        .to(volume.device)
        .reshape(-1, 3)
    )

    return torch_verts, torch_faces
