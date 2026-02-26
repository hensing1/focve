from __future__ import annotations

import torch


def project_points(
    points: torch.Tensor,  # N x 3
    K: torch.Tensor,  # 3 x 3
    T: torch.Tensor,  # 4 x 4
) -> tuple[
    torch.Tensor,  # N x 2
    torch.Tensor,  # N x 1
]:
    N = points.shape[0]
    pixels = torch.empty((N, 2), dtype=torch.float32)
    points_z = torch.empty((N, 1), dtype=torch.float32)

    # TODO Implement ...

    return pixels, points_z


def compute_tsdf_and_weight_from_depth(
    depths: torch.Tensor,  # N x 1
    points_z: torch.Tensor,  # N x 1
    truncation_region: float,
) -> tuple[
    torch.Tensor,  # N x 1
    torch.Tensor,  # N x 1
]:
    points_tsdf = torch.empty_like(depths)
    points_w = torch.empty_like(depths)

    # TODO Implement ...

    return points_tsdf, points_w


def integrate_values(
    TSDF_old: torch.Tensor,  # N x 1
    COLOR_old: torch.Tensor,  # N x 3
    W_old: torch.Tensor,  # N x 1
    tsdf: torch.Tensor,  # N x 1
    color: torch.Tensor,  # N x 3
    w: torch.Tensor,  # N x 1
) -> tuple[
    torch.Tensor,  # N x 1
    torch.Tensor,  # N x 3
    torch.Tensor,  # N x 1
]:
    TSDF_new = torch.empty_like(TSDF_old)
    COLOR_new = torch.empty_like(COLOR_old)
    W_new = torch.empty_like(W_old)

    # TODO Implement ...

    return TSDF_new, COLOR_new, W_new


class TSDFGrid:
    def __init__(
        self,
        resolution: int,
        bound_low: torch.Tensor,
        bound_size: float,
        truncation_region: float,
    ) -> None:
        grid_x = bound_low[0] + torch.linspace(
            0, bound_size, resolution, dtype=torch.float32
        )
        grid_y = bound_low[1] + torch.linspace(
            0, bound_size, resolution, dtype=torch.float32
        )
        grid_z = bound_low[2] + torch.linspace(
            0, bound_size, resolution, dtype=torch.float32
        )

        grid_verts = torch.stack(
            torch.meshgrid([grid_x, grid_y, grid_z], indexing="ij"),
            dim=0,
        )  # 3 x size x size x size
        self.points = grid_verts.permute([1, 2, 3, 0]).reshape([-1, 3])  # size^3 x 3

        self.tsdf = torch.ones((resolution**3, 1), dtype=torch.float32)
        self.color = torch.zeros((resolution**3, 3), dtype=torch.float32)
        self.weight = torch.zeros((resolution**3, 1), dtype=torch.float32)

        self.truncation_region = truncation_region

    def integrate(
        self,
        K: torch.Tensor,
        T: torch.Tensor,
        rgb: torch.Tensor,
        depth: torch.Tensor,
    ) -> None:
        w, h = rgb.shape[1], rgb.shape[0]

        pixels, points_z = project_points(self.points, K, T)

        valid = (
            (
                torch.asarray(points_z[:, 0] > 0.1)
                & torch.asarray(pixels[:, 0] >= 0)
                & torch.asarray(pixels[:, 0] <= w - 1)
                & torch.asarray(pixels[:, 1] >= 0)
                & torch.asarray(pixels[:, 1] <= h - 1)
            )
            .nonzero()
            .squeeze()
        )

        # Nearest Neighbor lookup
        pixels_nn = torch.round(pixels[valid, :]).to(torch.int64)
        depths = depth[pixels_nn[:, 1], pixels_nn[:, 0], :]
        rgbs = rgb[pixels_nn[:, 1], pixels_nn[:, 0], :]

        point_tsdf, point_w = compute_tsdf_and_weight_from_depth(
            depths, points_z[valid, :], self.truncation_region
        )
        point_color = rgbs

        self.tsdf[valid, :], self.color[valid, :], self.weight[valid, :] = (
            integrate_values(
                self.tsdf[valid, :],
                self.color[valid, :],
                self.weight[valid, :],
                point_tsdf,
                point_color,
                point_w,
            )
        )
