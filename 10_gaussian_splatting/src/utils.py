import pathlib
from typing import NamedTuple
import numpy as np
from PIL import Image
import torch


def sh_basis_l2(dirs):
    """
    Compute real SH basis (up to L=2, 9 coefficients).
    dirs: [N, 3] normalized direction vectors
    returns: [N, 9] SH basis values
    """
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]

    Y = torch.stack([
        0.282095 * torch.ones_like(x),                      # l=0, m=0
        0.488603 * y,                                       # l=1, m=-1
        0.488603 * z,                                       # l=1, m=0
        0.488603 * x,                                       # l=1, m=1
        1.092548 * x * y,                                   # l=2, m=-2
        1.092548 * y * z,                                   # l=2, m=-1
        0.315392 * (3 * z ** 2 - 1),                        # l=2, m=0
        1.092548 * x * z,                                   # l=2, m=1
        0.546274 * (x ** 2 - y ** 2),                       # l=2, m=2
    ], dim=1)  # [N, 9]

    return Y


def sh_basis_l3(dirs):
    """
    Compute real SH basis functions up to degree L=3 (16 coefficients).
    Input:
        dirs: [N, 3] normalized direction vectors
    Output:
        [N, 16] real SH basis values
    """
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    yz = y * z
    xz = x * z

    Y = torch.stack([
        # L=0
        0.282095 * torch.ones_like(x),                 # Y00

        # L=1
        0.488603 * y,                                  # Y1-1
        0.488603 * z,                                  # Y10
        0.488603 * x,                                  # Y11

        # L=2
        1.092548 * xy,                                 # Y2-2
        1.092548 * yz,                                 # Y2-1
        0.315392 * (3 * zz - 1),                       # Y20
        1.092548 * xz,                                 # Y2+1
        0.546274 * (xx - yy),                          # Y2+2

        # L=3
        0.590044 * y * (3 * xx - yy),                  # Y3-3
        2.890611 * xy * z,                             # Y3-2
        0.457046 * y * (5 * zz - 1),                   # Y3-1
        0.373176 * z * (5 * zz - 3),                   # Y30
        0.457046 * x * (5 * zz - 1),                   # Y3+1
        1.445306 * z * (xx - yy),                      # Y3+2
        0.590044 * x * (xx - 3 * yy),                  # Y3+3
    ], dim=1)  # Shape: [N, 16]

    return Y


def sh_coeffs_to_rgb(coeffs: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """
    Evaluate color in a direction using SH coeffs up to L=2.

    Inputs:
        coeffs: [N, 3, 9] - SH coefficients for RGB
        dirs: [N, 3] - direction vectors (should be normalized)
    Returns:
        colors: [N, 3] - RGB color in each direction
    """

    if len(dirs) == 0:
        return torch.zeros((0, 3), device=coeffs.device)
    
    Y = sh_basis_l2(dirs)
    assert 0.99 < dirs.norm(dim=1).max() <= 1.001
    colors = torch.einsum('nci,ni->nc', coeffs, Y)
    return colors


def load_texture(path: str) -> torch.Tensor:
    texture = (
        (torch.from_numpy(np.array(Image.open(path))).cuda().to(torch.float32) / 255.0)
        .unsqueeze(0)
        .flip(1)
    )

    return texture


def load_extrinsics(path: pathlib.Path) -> list[torch.Tensor]:
    T = torch.from_numpy(np.fromfile(path, dtype=np.float32)).reshape(-1, 4, 4)

    T = T.transpose(2, 1)

    return [T[i, :, :] for i in range(T.shape[0])]

def convert_intrinsics_cv_to_gl(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: float,
    height: float,
    zNear: float,
    zFar: float,
) -> torch.Tensor:
    P_clip = torch.zeros((4, 4), dtype=torch.float32).cuda()

    P_clip[0, 0] = 2.0 * fx / width
    P_clip[1, 1] = 2.0 * fy / height
    P_clip[0, 2] = 1.0 - 2.0 * cx / width
    P_clip[1, 2] = 2.0 * cy / height - 1
    P_clip[3, 2] = -1.0
    P_clip[2, 2] = -(zFar + zNear) / (zFar - zNear)
    P_clip[2, 3] = -(2.0 * zFar * zNear) / (zFar - zNear)

    return P_clip


class Camera:
    def __init__(
        self,
        V: torch.Tensor,
        P: torch.Tensor,
        resolution: tuple[int, int],
        image: torch.Tensor,
        focal_length: float
    ) -> None:
        self.V = V
        self.P = P
        self.resolution = resolution
        self.V_inv = torch.linalg.inv(V)
        self.position = self.V_inv[:3, -1]
        self.direction = self.V_inv[:3, 2]
        self.image = image
        self.focal_length = focal_length
     
    def __eq__(self, value):
        if not isinstance(value, Camera):
            return False
        else:
            return (
                torch.allclose(self.V, value.V)
                and torch.allclose(self.P, value.P)
                and self.resolution == value.resolution
            )

def so3_exp_map(omega):
    theta = torch.norm(omega, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = omega / theta
    half_theta = 0.5 * theta
    w = torch.cos(half_theta)
    xyz = axis * torch.sin(half_theta)
    return torch.cat([w, xyz], dim=-1)  # quaternion [w, x, y, z]


def quaternion_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    assert q.shape[-1] == 4, "Input must be (..., 4) shaped quaternion"

    x, y, z, w = q.unbind(-1)

    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    rot = torch.stack([
        torch.stack([1 - 2 * (yy + zz),     2 * (xy - zw),         2 * (xz + yw)], dim=-1),
        torch.stack([2 * (xy + zw),         1 - 2 * (xx + zz),     2 * (yz - xw)], dim=-1),
        torch.stack([2 * (xz - yw),         2 * (yz + xw),         1 - 2 * (xx + yy)], dim=-1)
    ], dim=-2)

    return rot

def get_initial_orientations(num_splats: int) -> torch.Tensor:
    return torch.rand(num_splats, 3).cuda() * torch.pi * 2

def get_initial_scales(num_splats: int) -> torch.Tensor:
    return 0.001 + torch.rand(num_splats, 3).cuda() * 0.05

def get_initial_colors(num_splats: int) -> torch.Tensor:
    initial_colors = torch.zeros(num_splats, 3, 9).cuda()
    initial_colors[:, :, 0] = torch.rand(num_splats, 3).cuda() * (1 / 0.282095)
    return initial_colors

def get_initial_opacities(num_splats: int) -> torch.Tensor:
    return torch.randn(num_splats, 1).abs().cuda() + 0.1

class GaussianSplats(NamedTuple):
    means_: torch.Tensor         # (N, 3)
    orientations_: torch.Tensor  # (N, 3, 3)
    scales_: torch.Tensor        # (N, 3)
    colors_: torch.Tensor        # (N, 3, 16)
    opacities_: torch.Tensor     # (N, 1)
    
    @staticmethod
    def initalize_from_means(
        initial_means: torch.Tensor
    ) -> "GaussianSplats":      
        num_splats = initial_means.shape[0]  
        return GaussianSplats(
            means_=initial_means,
            orientations_=get_initial_orientations(num_splats),
            scales_=get_initial_scales(num_splats),
            colors_=get_initial_colors(num_splats),
            opacities_=get_initial_opacities(num_splats)
        )

    def __getitem__(self, idx):
        return GaussianSplats(
            means_=self.means_[idx],
            orientations_=self.orientations_[idx],
            scales_=self.scales_[idx],
            colors_=self.colors_[idx],
            opacities_=self.opacities_[idx]
        )

    def clone(self):
        return GaussianSplats(
            means_=self.means_.clone(),
            orientations_=self.orientations_.clone(),
            scales_=self.scales_.clone(),
            colors_=self.colors_.clone(),
            opacities_=self.opacities_.clone()
        )

    @staticmethod
    def cat(splats: list["GaussianSplats"]):
        return GaussianSplats(
            means_=torch.cat([s.means_ for s in splats]).requires_grad_(True),
            orientations_=torch.cat([s.orientations_ for s in splats]).requires_grad_(True),
            scales_=torch.cat([s.scales_ for s in splats]).requires_grad_(True),
            colors_=torch.cat([s.colors_ for s in splats]).requires_grad_(True),
            opacities_=torch.cat([s.opacities_ for s in splats]).requires_grad_(True)
        )

    @property
    def means(self):
        return self.means_
    
    @means.setter
    def means(self, value):
        self.means_[:] = value  
        

    @property
    def orientations(self):
        return self.orientations_
    
    @orientations.setter
    def orientations(self, value):
        self.orientations_[:] = value
        

    @property
    def scales(self):
        return self.scales_ * (1/4)
    
    @scales.setter
    def scales(self, value):
        self.scales_[:] = value * 4


    @property
    def colors(self):
        return torch.abs(self.colors_)
    
    @colors.setter
    def colors(self, value):
        self.colors_[:] = value
        

    @property
    def opacities(self):
        return torch.abs(self.opacities_)
        
    @opacities.setter
    def opacities(self, value):
        self.opacities_[:] = value



def get_covariance_3d(scales: torch.Tensor, orientations: torch.Tensor) -> torch.Tensor:
    quats = so3_exp_map(orientations)
    R = quaternion_to_rotmat(quats)
    scale_matrices = torch.diag_embed(scales.clamp(min=1e-3) ** 2)
    return R @ scale_matrices @ R.transpose(1, 2)


def project_to_screen(
        vertices_world: torch.Tensor, # (N, 3)
        camera: Camera
) -> torch.Tensor: # (N, 2)
    """
    Projects 3D vertices to 2D screen coordinates.

    Returns a tensor (N, 2), where tensor[k] is the 2D screen coordinates of
    the k-th vertex from vertices_world.
    """
    P = camera.P
    V = camera.V

    vertices = torch.cat(
        [vertices_world, torch.ones((vertices_world.shape[0], 1)).cuda()], dim=1
    ).contiguous()

    vertices = (P @ V @ vertices.T).T
    vertices_xy = vertices[..., :2] / vertices[..., 2:3]
    vertices_xy = vertices_xy.contiguous()

    assert vertices_xy.shape == (vertices_world.shape[0], 2)
    return vertices_xy
