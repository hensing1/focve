from __future__ import annotations

import pathlib

import numpy as np
import nvdiffrast.torch as nv
import torch
from PIL import Image


def convert_intrinsics_cv_to_gl(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: float,
    height: float,
    zNear: float,
    zFar: float,
) -> torch.Tensor:  # 4 x 4
    P_clip = torch.zeros((4, 4), dtype=torch.float32).cuda()

    P_clip[0, 0] = 2 * fx / width
    P_clip[1, 1] = 2 * fy / height
    P_clip[0, 2] = 1 - 2 * cx / width
    P_clip[1, 2] = 2 * cy / height - 1
    P_clip[2, 2] = ((-zFar - zNear) / (zFar - zNear))
    P_clip[2, 3] = ((-2 * zFar * zNear) / (zFar - zNear))
    P_clip[3, 2] = -1

    return P_clip


def gradient_descent(
    parameters: torch.Tensor,  # 1 x h x w x 3
    gradients: torch.Tensor,  # 1 x h x w x 3
    lr: float,
) -> torch.Tensor:  # 1 x h x w x 3

    return parameters - lr * gradients


def loss_L1(
    estimate: torch.Tensor,  # 1 x h x w x 3
    target: torch.Tensor,  # 1 x h x w x 3
) -> torch.Tensor:  # 1

    return torch.sum(torch.abs(estimate - target))


def optimize_step(
    renderer: DifferentiableRenderer,
    optimizer: torch.optim.Optimizer,
    scene: Scene,
    cameras: list[Camera],
) -> torch.Tensor:  # 1

    optimizer.zero_grad()
    L = torch.tensor([0.0], requires_grad=True).cuda()
    for camera in cameras:
        estimate = renderer.render(scene, camera)
        L = L + loss_L1(estimate, camera.image)
    L.backward()
    optimizer.step()
    return L


def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(
        f <= 0.04045,
        f / 12.92,
        torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4),
    )


def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(
        f <= 0.0031308,
        f * 12.92,
        torch.pow(torch.clamp(f, 0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
    )


def distribution_ggx(
    NdotH: torch.Tensor,  # N x 1
    roughness: torch.Tensor,  # N x 1
) -> torch.Tensor:  # N x 1
    a = roughness**2
    a2 = a**2
    NdotH2 = NdotH**2

    nom = a2
    denom = NdotH2 * (a2 - 1.0) + 1.0
    denom = torch.pi * denom * denom

    return nom / denom


def geometry_ggx(
    NdotV: torch.Tensor,  # N x 1
    roughness: torch.Tensor,  # N x 1
) -> torch.Tensor:  # N x 1
    a = roughness**2
    a2 = a**2
    NdotV2 = NdotV**2

    L = (-1 + torch.sqrt((NdotV2 * (1.0 - a2) + a2) / torch.clamp(NdotV2, 1e-5))) / 2.0

    return L


def geometry_smith(
    NdotL: torch.Tensor,  # N x 1
    NdotV: torch.Tensor,  # N x 1
    roughness: torch.Tensor,  # N x 1
) -> torch.Tensor:  # N x 1
    L1 = geometry_ggx(NdotL, roughness)
    L2 = geometry_ggx(NdotV, roughness)

    return 1.0 / (1.0 + L1 + L2)


def fresnel_schlick(
    F0: torch.Tensor,  # N x 3
    VdotH: torch.Tensor,  # N x 1
) -> torch.Tensor:  # N x 3
    p = torch.clamp(1.0 - VdotH, 0.0, 1.0)
    return F0 + (1 - F0) * p**5


def bsdf(
    light_dir: torch.Tensor,  # N x 3
    view_dirs: torch.Tensor,  # N x 3
    vertex_normals: torch.Tensor,  # N x 3
    diffuse: torch.Tensor,  # N x 3
    metallic: torch.Tensor,  # N x 1
    roughness: torch.Tensor,  # N x 1
) -> torch.Tensor:  # N x 3
    H = torch.nn.functional.normalize(light_dir + view_dirs, dim=1)
    NdotH = torch.clamp(torch.sum(H * vertex_normals, dim=1).unsqueeze(-1), 0)
    NdotL = torch.clamp(torch.sum(light_dir * vertex_normals, dim=1).unsqueeze(-1), 0)
    NdotV = torch.clamp(torch.sum(view_dirs * vertex_normals, dim=1).unsqueeze(-1), 0)
    VdotH = torch.clamp(torch.sum(view_dirs * H, dim=1).unsqueeze(-1), 0)
    F0 = (1.0 - metallic) * 0.04 + metallic * diffuse

    NDF = distribution_ggx(NdotH, roughness)
    G = geometry_smith(NdotL, NdotV, roughness)
    F = fresnel_schlick(F0, VdotH)

    numerator = NDF * G * F
    denom = torch.clamp(4.0 * NdotV * NdotL, 1e-3)
    specular = numerator / denom

    color_diffuse = (1.0 - metallic) * diffuse

    return specular + color_diffuse / torch.pi


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


class GradientDescent(torch.optim.Optimizer):
    def __init__(self, params, lr):
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data

                param.data = gradient_descent(param.data, grad, lr)


class Camera:
    def __init__(
        self,
        V: torch.Tensor,
        P: torch.Tensor,
        resolution: tuple[int, int],
        image: torch.Tensor,
    ) -> None:
        self.V = V
        self.P = P
        self.resolution = resolution
        self.V_inv = torch.linalg.inv(V)
        self.position = self.V_inv[:3, -1]
        self.direction = self.V_inv[:3, 2]
        self.image = image


class Material:
    def __init__(
        self,
        diffuse_texture: torch.Tensor,
        metallic_texture: torch.Tensor,
        roughtness_texture: torch.Tensor,
    ) -> None:
        self.diffuse_texture = diffuse_texture
        self.metallic_texture = metallic_texture
        self.roughness_texture = roughtness_texture


class Model:
    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        normals: torch.Tensor,
        uvs: torch.Tensor,
        material: Material,
    ) -> None:
        self.vertices = vertices
        self.faces = faces
        self.normals = normals
        self.uvs = uvs
        self.material = material


class PointLight:
    def __init__(self, position, color, intensity):
        self.position = position
        self.color = color
        self.intensity = intensity


class Scene:
    def __init__(self):
        self.model: Model = None
        self.lights: list[PointLight] = []

    def addModel(self, model: Model):
        self.model = model

    def addLight(self, light: PointLight):
        self.lights.append(light)


class DifferentiableRenderer:
    def __init__(self):
        self.ctx = nv.RasterizeCudaContext()
        self.optimizer = None

    def render(self, scene: Scene, camera: Camera) -> torch.Tensor:
        model = scene.model
        vertices_world = model.vertices
        normals = model.normals
        uvs = model.uvs
        faces = model.faces
        P = camera.P
        V = camera.V

        vertices = torch.cat(
            [vertices_world, torch.ones((vertices_world.shape[0], 1)).cuda()], dim=1
        ).contiguous()

        vertices = (P @ V @ vertices.T).T
        vertices = vertices.unsqueeze(0).contiguous()
        normals = normals.unsqueeze(0)
        uvs = uvs.unsqueeze(0)

        attr = torch.cat([vertices_world.unsqueeze(0), uvs, normals], -1).contiguous()

        rast, _ = nv.rasterize(self.ctx, vertices, faces, camera.resolution)
        valid = rast[..., -1] != 0
        attr, _ = nv.interpolate(attr, rast, faces)
        vertices_world = attr[..., :3].contiguous()
        uvs = attr[..., 3:5].contiguous()
        normals = attr[..., 5:].contiguous()

        diffuse = nv.texture(model.material.diffuse_texture, uvs)
        metallic = nv.texture(model.material.metallic_texture, uvs)
        roughness = nv.texture(model.material.roughness_texture, uvs)

        diffuse = nv.antialias(diffuse, rast, vertices, faces)
        metallic = nv.antialias(metallic, rast, vertices, faces)
        roughness = nv.antialias(roughness, rast, vertices, faces)

        output = torch.zeros_like(diffuse)

        # Shading
        view_dirs = camera.position - vertices_world[valid]
        view_dirs = torch.nn.functional.normalize(view_dirs, dim=1)
        vertex_normals = normals[valid]
        diffuse = diffuse[valid]
        metallic = metallic[valid]
        roughness = roughness[valid]
        for light in scene.lights:
            light_dirs = light.position - vertices_world[valid]
            light_r2 = torch.sum(light_dirs * light_dirs, dim=1).unsqueeze(-1)
            light_dirs = torch.nn.functional.normalize(light_dirs, dim=1)
            cos = torch.clamp(torch.sum(light_dirs * vertex_normals, dim=1)[:, None], 0)
            bsdf_values = bsdf(
                light_dirs, view_dirs, vertex_normals, diffuse, metallic, roughness
            )
            light_weight = light.intensity * light.color / light_r2

            output[valid] = output[valid] + light_weight * bsdf_values * cos

        # Tonemap
        output = rgb_to_srgb(output).clamp(0, 1)
        return output

    def optimize_step(self, scene: Scene, cameras: list[Camera]) -> float:
        diffuse = scene.model.material.diffuse_texture
        if not self.optimizer:
            self.optimizer = GradientDescent([diffuse], lr=0.001)

        L = optimize_step(self, self.optimizer, scene, cameras)

        diffuse = diffuse.clamp(0, 1)

        return L.item()
