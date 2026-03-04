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

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from differentiable_renderer import (
    Camera,
    DifferentiableRenderer,
    Material,
    Model,
    PointLight,
    Scene,
    convert_intrinsics_cv_to_gl,
    load_extrinsics,
    load_texture,
)


def perspective(fov_y: float, aspect: float, zNear: float, zFar: float) -> torch.Tensor:
    tanHalfFovy = np.tan(fov_y / 2.0)

    result = torch.zeros((4, 4), dtype=torch.float32).cuda()
    result[0][0] = 1.0 / (aspect * tanHalfFovy)
    result[1][1] = 1.0 / (tanHalfFovy)
    result[2][2] = -(zFar + zNear) / (zFar - zNear)
    result[2][3] = -1.0
    result[3][2] = -(2.0 * zFar * zNear) / (zFar - zNear)
    return result.T


if __name__ == "__main__":
    mesh = trimesh.load("assets/bunny.obj")

    vertices = torch.from_numpy(mesh.vertices).cuda().to(torch.float32)
    vertices[:, 1] -= 0.6
    normals = torch.from_numpy(mesh.vertex_normals.copy()).cuda().to(torch.float32)
    uvs = torch.from_numpy(mesh.visual.uv).cuda().to(torch.float32)
    faces = torch.from_numpy(mesh.faces).cuda().to(torch.int32)

    diffuse = torch.distributions.Uniform(0.3, 0.7).sample([1, 300, 300, 3]).cuda()
    diffuse.requires_grad = True
    metallic = load_texture("assets/bunny_metallic.png")[..., 0:1].contiguous()
    roughness = load_texture("assets/bunny_roughness.png")[..., 0:1].contiguous()

    targets = []
    for i in range(100):
        targets.append(
            load_texture(f"assets/targets/target_{i}.png")[..., 0:3].contiguous()
        )

    w = 640
    h = 480
    fx = 416.6
    fy = 416.6
    cx = 320.0
    cy = 240.0
    P = convert_intrinsics_cv_to_gl(fx, fy, cx, cy, w, h, 0.01, 1000.0)

    views = load_extrinsics("./assets/views.bin")

    # Define a scene
    material = Material(diffuse, metallic, roughness)
    model = Model(vertices, faces, normals, uvs, material)

    cameras = []
    for i in range(len(views)):
        cam = Camera(views[i].cuda(), P, (h, w), targets[i])
        cameras.append(cam)

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

    renderer = DifferentiableRenderer()

    camera_indices = [5, 36, 66, 75, 52, 87, 96, 45, 7, 12]
    optimized_cameras = [cameras[index] for index in camera_indices]

    ps.init()
    ps.set_always_redraw(True)
    ps.set_ground_plane_mode("tile")
    ps.set_background_color((0, 0, 0))

    for i, light in enumerate(scene.lights):
        light_position = light.position.cpu().numpy()
        light_color = light.color.cpu().numpy().flatten()
        ps_light = ps.register_point_cloud(f"Lights {i}", light_position)
        ps_light.set_material("flat")
        ps_light.set_radius(0.05)
        ps_light.set_color(light_color)

    for i, camera in enumerate(optimized_cameras):
        ps_K = ps.CameraIntrinsics(fov_vertical_deg=60, aspect=w / h)
        ps_T = ps.CameraExtrinsics(mat=camera.V.cpu().numpy())

        ps_camera = ps.register_camera_view(
            f"Camera_{camera_indices[i]}",
            ps.CameraParameters(ps_K, ps_T),
            widget_color=(1.0, 1.0, 1.0),
        )
        ps_camera.add_color_image_quantity(
            "Rendering",
            camera.image.squeeze(0).cpu().flip(0).numpy(),
            show_in_camera_billboard=True,
            enabled=True,
        )

    while not ps.window_requests_close():

        view = ps.get_camera_view_matrix().astype(np.float32)
        if np.any(np.isnan(view)):
            view = np.eye(4, dtype=np.float32)
        view = torch.from_numpy(view).cuda()

        size = ps.get_window_size()
        fov = ps.get_view_camera_parameters().get_fov_vertical_deg()
        size_down = (size[0] // 2, size[1] // 2)

        P = perspective(np.deg2rad(fov), size[0] / size[1], 0.01, 100.0)

        camera = Camera(view, P, size_down, None)

        renderer.optimize_step(
            scene,
            optimized_cameras,
        )

        with torch.no_grad():
            img = renderer.render(scene, camera)

        img = img[0].detach().flip(0).cpu().numpy()

        ps.add_raw_color_render_image_quantity(
            "color_img", np.zeros(size_down, dtype=np.float32), img
        )

        ps_mesh = ps.register_surface_mesh(
            "Bunny", vertices.cpu().numpy(), faces.cpu().numpy(), material="flat"
        )
        ps_mesh.add_parameterization_quantity(
            "UVs", uvs.cpu().numpy(), defined_on="vertices"
        )

        diffuse_ = torch.clamp(diffuse[0].detach().flip(0), 0, 1).cpu().numpy() ** (
            1.0 / 2.4
        )

        ps_mesh.add_color_quantity(
            "Texture",
            diffuse_,
            defined_on="texture",
            param_name="UVs",
            enabled=True,
        )

        ps.frame_tick()
