from __future__ import annotations

import pathlib
import sys

import torch
import numpy as np

import trimesh

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
from differentiable_renderer import (
    gradient_descent,
    convert_intrinsics_cv_to_gl,
    loss_L1,
    Camera,
    Model,
    Scene,
    PointLight,
    DifferentiableRenderer,
    Material,
    load_texture,
    load_extrinsics,
)


def test_convert_intrinsics_cv_to_gl():
    w = 640
    h = 480
    fx = 416.6
    fy = 416.6
    cx = 320.0
    cy = 240.0
    n = 0.01
    f = 1000.0
    P = convert_intrinsics_cv_to_gl(fx, fy, cx, cy, w, h, n, f)
    target = torch.tensor(
        [
            [1.3020, 0.0000, 0.0000, 0.0000],
            [0.0000, 1.7360, 0.0000, 0.0000],
            [0.0000, 0.0000, -1.0000, -0.0200],
            [0.0000, 0.0000, -1.0000, 0.0000],
        ],
        device="cuda:0",
    )

    assert torch.allclose(torch.round(P, decimals=3), target, 1e-4)

    w = 640
    h = 480
    fx = 418.9
    fy = 416.6
    cx = 318.0
    cy = 230.0
    n = 0.01
    f = 100.0
    P = convert_intrinsics_cv_to_gl(fx, fy, cx, cy, w, h, n, f)
    target = torch.tensor(
        [
            [1.3090, 0.0000, 0.0060, 0.0000],
            [0.0000, 1.7360, -0.0420, 0.0000],
            [0.0000, 0.0000, -1.0000, -0.0200],
            [0.0000, 0.0000, -1.0000, 0.0000],
        ],
        device="cuda:0",
    )
    assert torch.allclose(torch.round(P, decimals=3), target, 1e-4)

    w = 1280
    h = 720
    fx = 837.8
    fy = 833.2
    cx = 636
    cy = 460
    n = 0.01
    f = 100.0
    P = convert_intrinsics_cv_to_gl(fx, fy, cx, cy, w, h, n, f)
    target = torch.tensor(
        [
            [1.3090, 0.0000, 0.0060, 0.0000],
            [0.0000, 2.3140, 0.2780, 0.0000],
            [0.0000, 0.0000, -1.0000, -0.0200],
            [0.0000, 0.0000, -1.0000, 0.0000],
        ],
        device="cuda:0",
    )
    assert torch.allclose(torch.round(P, decimals=3), target, 1e-4)

    w = 1920
    h = 1080
    fx = 2317.65
    fy = 2317.65
    cx = 960.0
    cy = 960.0
    n = 0.1
    f = 1000.0
    P = convert_intrinsics_cv_to_gl(fx, fy, cx, cy, w, h, n, f)
    target = torch.tensor(
        [
            [2.4140, 0.0000, 0.0000, 0.0000],
            [0.0000, 4.2920, 0.7780, 0.0000],
            [0.0000, 0.0000, -1.0000, -0.2000],
            [0.0000, 0.0000, -1.0000, 0.0000],
        ],
        device="cuda:0",
    )
    assert torch.allclose(torch.round(P, decimals=3), target, 1e-4)


def test_gradient_descent():
    parameters = torch.tensor([5.0])

    lr = 0.1
    expected = torch.tensor([4.0, 3.2, 2.56, 2.048, 1.6384])
    results = []
    for i in range(5):
        gradients = 2.0 * parameters

        updated = gradient_descent(parameters, gradients, lr)
        assert updated.shape == parameters.shape

        parameters = updated
        results.append(updated.item())

    assert torch.allclose(torch.tensor(results), expected)

    parameters = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape((1, 2, 2, 1))

    lr = 0.1
    expected = [
        torch.tensor([[[[0.8000], [1.6000]], [[2.4000], [3.2000]]]]),
        torch.tensor([[[[0.6400], [1.2800]], [[1.9200], [2.5600]]]]),
        torch.tensor([[[[0.5120], [1.0240]], [[1.5360], [2.0480]]]]),
    ]
    for i in range(3):
        gradients = 2.0 * parameters

        updated = gradient_descent(parameters, gradients, lr)
        assert updated.shape == parameters.shape
        assert torch.allclose(expected[i], updated)

        parameters = updated


def test_loss_L1():
    estimate = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape((1, 2, 2, 1))
    target = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape((1, 2, 2, 1))

    loss = loss_L1(estimate, target)

    assert torch.allclose(loss, torch.zeros_like(loss))

    estimate = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape((1, 2, 2, 1))
    target = torch.tensor([0.0, 2.0, 3.0, 4.0]).reshape((1, 2, 2, 1))

    loss = loss_L1(estimate, target)

    assert torch.allclose(loss, torch.ones_like(loss))

    estimate = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape((1, 2, 2, 1))
    target = torch.tensor([0.0, 2.0, 0.0, 4.0]).reshape((1, 2, 2, 1))

    loss = loss_L1(estimate, target)

    assert torch.allclose(loss, torch.full_like(loss, 4))

    estimate = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape((1, 2, 2, 1))
    target = torch.tensor([0.0, -2.0, 0.0, 4.0]).reshape((1, 2, 2, 1))

    loss = loss_L1(estimate, target)

    assert torch.allclose(loss, torch.full_like(loss, 8))


def test_optimize_step():
    mesh = trimesh.load("assets/bunny.obj")

    vertices = torch.from_numpy(mesh.vertices).cuda().to(torch.float32)
    vertices[:, 1] -= 0.6
    normals = torch.from_numpy(np.array(mesh.vertex_normals)).cuda().to(torch.float32)
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

    n_epochs = 30

    losses = []
    for _ in range(n_epochs):
        L = renderer.optimize_step(
            scene,
            optimized_cameras,
        )

        assert torch.all(torch.isfinite(diffuse.grad))

        losses.append(L)

    losses = torch.tensor(losses)
    differences = losses[1:] - losses[:-1]
    assert torch.count_nonzero(differences < 0) >= 0.9 * (n_epochs - 1)

    # Coverage

    with torch.no_grad():
        diffuse.fill_(0.5)
        diffuse.grad.zero_()

    camera_indices = [5]
    optimized_cameras = [cameras[index] for index in camera_indices]

    renderer.optimize_step(
        scene,
        optimized_cameras,
    )
    grads_one = (torch.abs(diffuse.grad) > 0).to(torch.float32)

    with torch.no_grad():
        diffuse.fill_(0.5)
        diffuse.grad.zero_()

    camera_indices = [5, 36]
    optimized_cameras = [cameras[index] for index in camera_indices]

    renderer.optimize_step(
        scene,
        optimized_cameras,
    )
    grads_two = (torch.abs(diffuse.grad) > 0).to(torch.float32)

    with torch.no_grad():
        diffuse.fill_(0.5)
        diffuse.grad.zero_()

    camera_indices = [5, 36, 66]
    optimized_cameras = [cameras[index] for index in camera_indices]

    renderer.optimize_step(
        scene,
        optimized_cameras,
    )
    grads_three = (torch.abs(diffuse.grad) > 0).to(torch.float32)

    assert torch.count_nonzero(grads_three) > torch.count_nonzero(grads_two)
    assert torch.count_nonzero(grads_two) > torch.count_nonzero(grads_one)

    with torch.no_grad():
        diffuse.fill_(0.5)
        diffuse.grad.zero_()

    camera_indices = [5, 36, 66, 75, 52, 87, 96, 45, 7, 12]
    optimized_cameras = [cameras[index] for index in camera_indices]

    renderer.optimize_step(
        scene,
        optimized_cameras,
    )
    coverage = torch.count_nonzero(diffuse.grad) / diffuse.numel()

    assert coverage > 0.45
