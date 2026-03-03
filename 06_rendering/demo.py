# /// script
# dependencies = [
#   "numpy",
#   "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git",
#   "Pillow",
#   "polyscope",
#   "torch",
#   "trimesh",
# ]
# ///

from __future__ import annotations

import pathlib
import sys

import numpy as np
import polyscope as ps
import torch
import trimesh
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from forward_renderer import (
    Camera,
    ForwardRenderer,
    Material,
    Model,
    PointLight,
    Scene,
    srgb_to_rgb,
)


def load_texture(path: str) -> torch.Tensor:
    texture = (
        (torch.from_numpy(np.array(Image.open(path))).cuda().to(torch.float32) / 255.0)
        .unsqueeze(0)
        .flip(1)
    )

    return texture


if __name__ == "__main__":
    mesh = trimesh.load("assets/bunny.obj")

    vertices = torch.from_numpy(mesh.vertices).cuda().to(torch.float32)
    vertices[:, 1] -= 0.6
    normals = torch.from_numpy(mesh.vertex_normals.copy()).cuda().to(torch.float32)
    uvs = torch.from_numpy(mesh.visual.uv).cuda().to(torch.float32)
    faces = torch.from_numpy(mesh.faces).cuda().to(torch.int32)

    diffuse = srgb_to_rgb(
        load_texture("assets/bunny_diffuse.png")[..., :3].contiguous()
    )
    metallic = load_texture("assets/bunny_metallic.png")[..., 0:1].contiguous()
    roughness = load_texture("assets/bunny_roughness.png")[..., 0:1].contiguous()

    P = torch.tensor(
        [
            [1.29904, 0, 0, 0],
            [0, 1.73205, 0, 0],
            [0, 0, -1.00002, -1],
            [0, 0, -0.0200002, 0],
        ],
        dtype=torch.float32,
    ).T.cuda()

    V = torch.tensor(
        [
            [9.5106e-01, -0.0000e00, 3.0902e-01, 1.1921e-07],
            [1.5451e-01, 8.6603e-01, -4.7553e-01, -0.0000e00],
            [-2.6762e-01, 5.0000e-01, 8.2364e-01, -4.0000e00],
            [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
        ]
    ).cuda()

    w = 640
    h = 480

    # Define a scene
    material = Material(diffuse, metallic, roughness)
    model = Model(vertices, faces, normals, uvs, material)
    camera = Camera(V, P, (h, w))

    scene = Scene()
    scene.addModel(model)
    light = PointLight(
        torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float32).cuda(),
        torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32).cuda(),
        10.0,
    )
    scene.addLight(light)
    light = PointLight(
        torch.tensor([[2.0, 0.5, 0.0]], dtype=torch.float32).cuda(),
        torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32).cuda(),
        10.0,
    )
    scene.addLight(light)
    light = PointLight(
        torch.tensor([[-2.0, 0.5, 0.0]], dtype=torch.float32).cuda(),
        torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32).cuda(),
        10.0,
    )
    scene.addLight(light)
    light = PointLight(
        torch.tensor([[0.0, 0.5, 2.0]], dtype=torch.float32).cuda(),
        torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32).cuda(),
        10.0,
    )
    scene.addLight(light)
    light = PointLight(
        torch.tensor([[0.0, 0.5, -2.0]], dtype=torch.float32).cuda(),
        torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32).cuda(),
        10.0,
    )
    scene.addLight(light)

    renderer = ForwardRenderer()
    img = renderer.render(scene, camera)

    img = img[0].flip(0).cpu().numpy()

    ps.init()
    ps.set_ground_plane_mode("tile")
    ps.set_background_color((0, 0, 0))

    ps_mesh = ps.register_surface_mesh(
        "Bunny", vertices.cpu().numpy(), faces.cpu().numpy(), material="flat"
    )
    ps_mesh.add_parameterization_quantity(
        "UVs", uvs.cpu().numpy(), defined_on="vertices"
    )
    ps_mesh.add_color_quantity(
        "Texture",
        diffuse[0].flip(0).cpu().numpy() ** (1.0 / 2.2),
        defined_on="texture",
        param_name="UVs",
        enabled=True,
    )
    ps_K = ps.CameraIntrinsics(fov_vertical_deg=60, aspect=w / h)
    ps_T = ps.CameraExtrinsics(mat=V.cpu().numpy())

    ps_camera = ps.register_camera_view(
        f"Camera", ps.CameraParameters(ps_K, ps_T), widget_color=(1.0, 1.0, 1.0)
    )
    ps_camera.add_color_image_quantity(
        "Rendering", img, show_in_camera_billboard=True, enabled=True
    )

    ps.add_color_image_quantity(
        "color_img",
        img,
        enabled=True,
    )

    for i, light in enumerate(scene.lights):
        light_position = light.position.cpu().numpy()
        light_color = light.color.cpu().numpy().flatten()
        ps_light = ps.register_point_cloud(f"Lights {i}", light_position)
        ps_light.set_material("flat")
        ps_light.set_radius(0.05)
        ps_light.set_color(light_color)

    ps.show()
