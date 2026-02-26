# /// script
# dependencies = [
#   "numpy",
#   "Pillow",
#   "polyscope",
#   "scikit-image",
#   "scipy",
#   "torch",
#   "tqdm",
# ]
# ///

from __future__ import annotations

import pathlib
import sys

import numpy as np
import polyscope as ps
import scipy.interpolate
import skimage.measure
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from tsdf_fusion import TSDFGrid


class BunnyDataset:
    def __init__(self, data_dir: pathlib.Path) -> None:

        self.T = self.load_extrinsics(data_dir / "views.bin")
        self.rgb, self.depth = self.load_images(data_dir, len(self.T))

        f_x = 416.6
        f_y = 416.6
        c_x = 320.0
        c_y = 240.0
        self.K = torch.tensor(
            [
                [f_x, 0, c_x],
                [0, f_y, c_y],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )

    def __iter__(self):
        for i in range(len(self.T)):
            yield self.K, self.T[i], self.rgb[i], self.depth[i]

    def __len__(self) -> int:
        return len(self.T)

    def load_extrinsics(self, path: pathlib.Path) -> list[torch.Tensor]:
        T = torch.from_numpy(np.fromfile(path, dtype=np.float32)).reshape(-1, 4, 4)

        T = T.transpose(2, 1)
        T = (
            torch.tensor([1, -1, -1, 1], dtype=torch.float32).diag()
            @ T
            @ torch.tensor([1, -1, -1, 1], dtype=torch.float32).diag()
        )

        return [T[i, :, :] for i in range(T.shape[0])]

    def load_images(
        self,
        data_dir: pathlib.Path,
        num_images: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        # RGB
        rgbs = []

        for i in range(num_images):
            rgbs.append(
                (
                    torch.from_numpy(
                        np.array(Image.open(data_dir / "rgb" / f"rgb_{i}.png"))
                    ).to(torch.float32)
                    / 255
                )[:, :, 0:3]
            )

        # Depth
        depths = []
        for i in range(num_images):
            depths.append(
                (
                    torch.from_numpy(
                        np.array(Image.open(data_dir / "depth" / f"depth_{i}.png"))
                    ).to(torch.float32)
                    / 1000
                )[:, :, None]
            )

        return rgbs, depths


def determine_grid_bounds(
    camera_positions: torch.Tensor, relative_size: float
) -> tuple[torch.Tensor, torch.Tensor]:
    bb_min, _ = torch.min(camera_positions, dim=0)
    bb_max, _ = torch.max(camera_positions, dim=0)

    bb_size, _ = torch.max(bb_max - bb_min, dim=0)

    bound_low = (0.5 * bb_min + 0.5 * bb_max) - relative_size * 0.5 * bb_size
    bound_high = (0.5 * bb_min + 0.5 * bb_max) + relative_size * 0.5 * bb_size

    return bound_low, bound_high


def main() -> None:
    data_dir = pathlib.Path(__file__).parent / "assets" / "data"

    dataset = BunnyDataset(data_dir)

    camera_positions = []
    for _, T, _, _ in dataset:
        camera_positions.append(torch.linalg.inv(T)[0:3, 3])
    camera_positions = torch.stack(camera_positions)

    bound_low, bound_high = determine_grid_bounds(camera_positions, 0.4)
    bound_size, _ = torch.max(bound_high - bound_low, dim=0)

    resolution = 150
    truncation_region = 0.1
    grid = TSDFGrid(
        resolution,
        bound_low,
        bound_size.item(),
        truncation_region,
    )

    # Fuse data
    for K, T, rgb, depth in tqdm(dataset):
        grid.integrate(K, T, rgb, depth)

    grid_tsdf = grid.tsdf.reshape([resolution, resolution, resolution]).flip([1, 2])
    grid_color = grid.color.reshape([resolution, resolution, resolution, 3]).flip(
        [1, 2]
    )
    grid_weight = (
        (grid.weight > 4.5).reshape([resolution, resolution, resolution]).flip([1, 2])
    )
    grid_weight_vis = (
        (grid.weight).reshape([resolution, resolution, resolution]).flip([1, 2])
    )

    # Extract surface
    vertices, faces, _, _ = skimage.measure.marching_cubes(
        grid_tsdf.numpy(), 0, method="lorensen", mask=grid_weight.numpy()
    )

    vertex_colors = scipy.interpolate.interpn(
        (np.arange(resolution), np.arange(resolution), np.arange(resolution)),
        grid_color.numpy(),
        vertices,
        method="linear",
    )

    # Map vertices to global coordinates
    bound_low, bound_high = determine_grid_bounds(
        camera_positions * torch.tensor([1, -1, -1], dtype=torch.float32), 0.4
    )

    vertices = bound_low.numpy() + vertices / (resolution - 1) * np.max(
        bound_high.numpy() - bound_low.numpy()
    )

    ps.init()
    ps.set_ground_plane_mode("tile")

    ps_plane = ps.add_scene_slice_plane()
    ps_plane.set_draw_widget(True)

    # Cameras
    cameras = []
    for j, (_, T, rgb, depth) in enumerate(dataset):
        w, h = rgb.shape[1], rgb.shape[0]
        ps_K = ps.CameraIntrinsics(fov_vertical_deg=60, aspect=w / h)

        ps_T = ps.CameraExtrinsics(
            mat=(
                torch.tensor([1, -1, -1, 1], dtype=torch.float32).diag()
                @ T
                @ torch.tensor([1, -1, -1, 1], dtype=torch.float32).diag()
            ).numpy()
        )

        ps_camera = ps.register_camera_view(
            f"Camera {j}", ps.CameraParameters(ps_K, ps_T)
        )

        ps_camera.add_color_image_quantity(
            "RGB",
            rgb,
            # enabled=True,
            show_in_camera_billboard=True,
        )
        ps_camera.add_scalar_image_quantity(
            "Depth",
            depth.squeeze(),
            cmap="turbo",
            vminmax=(0, 5),
            enabled=True,
            show_in_camera_billboard=True,
        )
        ps_camera.set_ignore_slice_plane(ps_plane, True)

        cameras.append(ps_camera)

    # Voxel Grid
    ps_grid = ps.register_volume_grid(
        "Voxel Grid",
        (resolution, resolution, resolution),
        bound_low.numpy(),
        bound_high.numpy(),
        edge_width=1,
        cube_size_factor=0.01,
    )
    ps_grid.add_scalar_quantity(
        "TSDF",
        grid_tsdf.numpy(),
        defined_on="nodes",
        vminmax=(-truncation_region, truncation_region),
        cmap="coolwarm",
        enabled=True,
    )
    ps_grid.add_scalar_quantity(
        "W",
        grid_weight_vis.numpy(),
        defined_on="nodes",
        cmap="viridis",
    )

    # Reconstruction
    ps_reconstruction = ps.register_surface_mesh(
        "Reconstruction", vertices, faces, material="flat"
    )
    ps_reconstruction.add_color_quantity(
        "Color", vertex_colors, defined_on="vertices", enabled=True
    )
    ps_reconstruction.set_ignore_slice_plane(ps_plane, True)

    ps.show()


if __name__ == "__main__":
    main()
