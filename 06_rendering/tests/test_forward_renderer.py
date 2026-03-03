from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
from forward_renderer import (
    rgb_to_srgb,
    srgb_to_rgb,
    distribution_ggx,
    geometry_ggx,
    fresnel_schlick,
    brdf,
)


def test_rgb_to_srgb():
    rgb = torch.linspace(0, 1, 1500).reshape(-1,3)
    srgb = rgb_to_srgb(rgb)
    rgb_rec = srgb_to_rgb(srgb)

    assert torch.allclose(rgb, rgb_rec)


def test_distribution_ggx():
    torch.manual_seed(0)

    NdotH = torch.distributions.Uniform(0, 1).sample([100, 1])
    roughness = torch.zeros((100, 1))

    D = distribution_ggx(NdotH, roughness)

    assert torch.allclose(D, torch.zeros_like(D))

    NdotH = torch.distributions.Uniform(0, 1).sample([100, 1])
    roughness = torch.ones((100, 1))

    D = distribution_ggx(NdotH, roughness)

    assert torch.allclose(D, torch.full_like(D, 1.0 / torch.pi))

    NdotH = torch.distributions.Uniform(0, 1).sample([100, 1])
    roughness = torch.distributions.Uniform(0, 1).sample([100, 1])

    D = distribution_ggx(NdotH, roughness)

    assert torch.all(D > 0.0)

    NdotH = torch.linspace(0, 1, 10).reshape(10, 1)
    roughness = torch.full((10, 1), 0.8)

    D = distribution_ggx(NdotH, roughness)
    expected = torch.tensor([0.130380, 0.132301, 0.138328, 0.149329, 0.167077, 0.194957, 0.239645, 0.315499, 0.458061, 0.777124]).reshape(10, 1)

    assert torch.allclose(D, expected)


def test_geometry_schlick_ggx():
    torch.manual_seed(0)

    roughness = torch.distributions.Uniform(0, 1).sample([100, 1])
    NdotV = torch.zeros((100, 1))

    G = geometry_ggx(NdotV, roughness)

    assert torch.all(torch.isfinite(G))

    NdotV = torch.ones((100, 1))

    G = geometry_ggx(NdotV, roughness)

    assert torch.allclose(G, torch.zeros_like(G), 1e-5)

    roughness = torch.distributions.Uniform(0, 1).sample([100, 1])
    NdotV = torch.distributions.Uniform(0, 1).sample([100, 1])

    G = geometry_ggx(NdotV, roughness)

    assert G.min() >= 0.0

    roughness = torch.linspace(0.01, 1, 10).reshape(10, 1)
    NdotV = torch.linspace(0, 1, 10).flip(0).reshape(10, 1)

    G = geometry_ggx(NdotV, roughness)
    expected = torch.tensor([0.0000000, 0.0000138, 0.0004566, 0.0041587, 0.0224591, 0.0915079, 0.3080980, 0.9252541, 3.0774915, 157.6138916]).reshape(10, 1)
    
    assert torch.allclose(G, expected, atol=1e-05)


def test_fresnel_schlick():
    torch.manual_seed(0)

    F0 = torch.distributions.Uniform(0, 1).sample([2, 3])
    VdotH = torch.tensor([0.0, 1.0]).reshape(-1, 1)

    F = fresnel_schlick(F0, VdotH)

    assert torch.allclose(F[0], torch.ones_like(F[0]))
    assert torch.allclose(F[1], F0[1])

    F0 = torch.distributions.Uniform(0, 1).sample([100, 3])
    VdotH = torch.distributions.Uniform(0, 1).sample([100, 1])

    F = fresnel_schlick(F0, VdotH)

    assert F.min() >= 0.0
    assert F.max() <= 1.0

    F0 = torch.full((10, 3), 0.3)
    VdotH = torch.linspace(0, 1, 30).reshape(10, 3)

    F = fresnel_schlick(F0, VdotH)
    expected = torch.tensor([[1.000000, 0.887352, 0.789696],
                             [0.705485, 0.633279, 0.571747],
                             [0.519658, 0.475882, 0.439381],
                             [0.409209, 0.384504, 0.364487],
                             [0.348457, 0.335786, 0.325916],
                             [0.318355, 0.312671, 0.308492],
                             [0.305496, 0.303413, 0.302015],
                             [0.301118, 0.300574, 0.300265],
                             [0.300107, 0.300035, 0.300008],
                             [0.300001, 0.300000, 0.300000]])

    assert torch.allclose(F, expected)


def test_brdf():
    torch.manual_seed(0)

    light_dirs = torch.distributions.Uniform(-1, 1).sample([100, 3])
    light_dirs = torch.nn.functional.normalize(light_dirs, dim=1)
    view_dirs = torch.distributions.Uniform(-1, 1).sample([100, 3])
    view_dirs = torch.nn.functional.normalize(view_dirs, dim=1)
    normals = torch.distributions.Uniform(-1, 1).sample([100, 3])
    normals = torch.nn.functional.normalize(normals, dim=1)

    NdotL = torch.sum(light_dirs * normals, dim=1)
    light_dirs[NdotL < 0] = -light_dirs[NdotL < 0]
    NdotV = torch.sum(view_dirs * normals, dim=1)
    view_dirs[NdotV < 0] = -view_dirs[NdotV < 0]

    diffuse = torch.distributions.Uniform(0, 1).sample([100, 3])
    metallic = torch.distributions.Uniform(0, 1).sample([100, 1])
    roughness = torch.distributions.Uniform(0, 1).sample([100, 1])

    values = brdf(light_dirs, view_dirs, normals, diffuse, metallic, roughness)

    assert torch.all(torch.isfinite(values))
    assert torch.all(values >= 0.0)

    values_rec = brdf(view_dirs, light_dirs, normals, diffuse, metallic, roughness)

    assert torch.allclose(values, values_rec)

    light_dir = torch.tensor([[1.0, 0.0, 0.0]])
    view_dir = torch.tensor([[1.0, 0.0, 0.0]])
    normal = torch.tensor([[0.0, 0.0, 1.0]])

    diffuse = torch.distributions.Uniform(0, 1).sample([1, 3])
    metallic = torch.distributions.Uniform(0, 1).sample([1, 1])
    roughness = torch.distributions.Uniform(0, 1).sample([1, 1])

    values = brdf(light_dir, view_dir, normal, diffuse, metallic, roughness)

    assert torch.all(torch.isfinite(values))

    light_dir = torch.nn.functional.normalize(torch.linspace(0, 1, 30).reshape(3, 10).T, p=2.0, dim=1)
    view_dir = torch.nn.functional.normalize(torch.linspace(0, 1, 30).flip(0).reshape(10, 3), p=2.0, dim=1)
    normal = torch.nn.functional.normalize(torch.concat([torch.linspace(0, 1, 30)[10:],torch.linspace(0, 1, 30)[:10]]).reshape(10,3), p=2.0, dim=1)

    diffuse = torch.linspace(0, 1, 30).reshape(10, 3)
    metallic = torch.linspace(0, 1, 10).reshape(10, 1)
    roughness = torch.concat([torch.linspace(0, 1, 10)[5:], torch.linspace(0, 1, 10)[:5]]).reshape(10, 1)

    values = brdf(light_dir, view_dir, normal, diffuse, metallic, roughness)
    expected = torch.tensor([[0.015865, 0.026841, 0.037818],
                             [0.043801, 0.054741, 0.065681],
                             [0.067232, 0.077360, 0.087489],
                             [0.083137, 0.091981, 0.100825],
                             [0.090649, 0.098046, 0.105443],
                             [0.073175, 0.078053, 0.082931],
                             [0.065905, 0.069566, 0.073228],
                             [0.100611, 0.105366, 0.110120],
                             [3.717299, 3.871263, 4.025228],
                             [0.482172, 0.500029, 0.517886]])

    assert torch.allclose(values, expected, atol=1e-5)
