from __future__ import annotations

import nvdiffrast.torch as nv
import torch


def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    result = torch.where(
        f <= 0.04045,
        f / 12.92,
        torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4),
    )
    return result


def rgb_to_srgb(
    f: torch.Tensor  # N x 3
) -> torch.Tensor:  # N x 3
    return torch.where(
        f <= 0.0031308,
        f * 12.92,
        1.055 * torch.pow(f, 1/2.4) - 0.055
    )


def distribution_ggx(
    NdotH: torch.Tensor,  # N x 1
    roughness: torch.Tensor,  # N x 1
) -> torch.Tensor:  # N x 1

    alpha = roughness ** 2
    alpha_sq = alpha ** 2
    temp = (NdotH ** 2) * (alpha_sq - 1) + 1

    return alpha_sq / (torch.pi * (temp ** 2))


def geometry_ggx(
    NdotV: torch.Tensor,  # N x 1
    roughness: torch.Tensor,  # N x 1
) -> torch.Tensor:  # N x 1

    NdotV_sq = NdotV ** 2
    alpha = roughness ** 2
    alpha_sq = alpha ** 2

    result = (-1 + torch.sqrt((NdotV_sq*(1-alpha_sq) + alpha_sq)/torch.clamp(NdotV_sq, 1e-5))) / 2

    return result


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

    return F0 + (1-F0) * ((1-VdotH) ** 5)


def brdf(
    light_dir: torch.Tensor,  # N x 3
    view_dirs: torch.Tensor,  # N x 3
    vertex_normals: torch.Tensor,  # N x 3
    base_color: torch.Tensor,  # N x 3
    metallic: torch.Tensor,  # N x 1
    roughness: torch.Tensor,  # N x 1
) -> torch.Tensor:  # N x 3
    H = torch.nn.functional.normalize(light_dir + view_dirs, dim=1)
    NdotH = torch.clamp(torch.sum(H * vertex_normals, dim=1).unsqueeze(-1), 0)
    NdotL = torch.clamp(torch.sum(light_dir * vertex_normals, dim=1).unsqueeze(-1), 0)
    NdotV = torch.clamp(torch.sum(view_dirs * vertex_normals, dim=1).unsqueeze(-1), 0)
    VdotH = torch.clamp(torch.sum(view_dirs * H, dim=1).unsqueeze(-1), 0)
    F0 = (1.0 - metallic) * 0.04 + metallic * base_color

    diffuse = (1 - metallic) * base_color / torch.pi
    specular = (
        distribution_ggx(NdotH, roughness) *
        geometry_smith(NdotL, NdotV, roughness) *
        fresnel_schlick(F0, VdotH)
    ) / (torch.clamp(4 * NdotL * NdotV, 1e-5))

    return diffuse + specular


class Camera:
    def __init__(
        self, V: torch.Tensor, P: torch.Tensor, resolution: tuple[int, int]
    ) -> None:
        self.V = V
        self.P = P
        self.resolution = resolution
        self.V_inv = torch.linalg.inv(V)
        self.position = self.V_inv[:3, -1]
        self.direction = self.V_inv[:3, 2]


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


class ForwardRenderer:
    def __init__(self):
        self.ctx = nv.RasterizeGLContext()

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
            bsdf_values = brdf(
                light_dirs, view_dirs, vertex_normals, diffuse, metallic, roughness
            )
            light_weight = light.intensity * light.color / light_r2

            output[valid] += light_weight * bsdf_values * cos

        # Tonemap
        output = torch.clamp(rgb_to_srgb(output), 0, 1)
        return output
