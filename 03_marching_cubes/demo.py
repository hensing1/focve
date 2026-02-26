# /// script
# dependencies = [
#   "numpy",
#   "polyscope",
#   "torch",
#   "tqdm",
# ]
# ///

from __future__ import annotations

import pathlib
import sys

import numpy as np
import polyscope as ps
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from marching_cubes import marching_cubes


def load_sdf(file: pathlib.Path) -> torch.Tensor:
    assert file.suffix == ".npz"
    field = np.load(file)

    assert "distance" in field.keys()
    sdf = field["distance"]

    # Can hardcode size and order of grid, since we know the data
    sdf = sdf.reshape([100, 100, 100]).transpose((2, 1, 0))

    return torch.from_numpy(sdf)


def main() -> None:
    data_dir = pathlib.Path(__file__).parent / "assets" / "data"

    vertices_dinosaur, faces_dinosaur = marching_cubes(
        load_sdf(data_dir / "Dinosaur.npz"),
        0,
    )

    vertices_mobius, faces_mobius = marching_cubes(
        load_sdf(data_dir / "Mobius.npz"),
        0,
    )

    vertices_tiefighter, faces_tiefighter = marching_cubes(
        load_sdf(data_dir / "TieFighter.npz"),
        0,
    )

    ps.set_allow_headless_backends(True)
    ps.init()

    offset = 75
    render_color = (0.890, 0.612, 0.110)

    ps_dinosaur = ps.register_surface_mesh(
        "Dinosaur",
        vertices_dinosaur.cpu().numpy() + np.array([-offset, 0, 0]),
        faces_dinosaur.cpu().numpy(),
        back_face_policy="cull",
    )
    ps_dinosaur.set_color(render_color)

    ps_mobius = ps.register_surface_mesh(
        "Mobius",
        vertices_mobius.cpu().numpy(),
        faces_mobius.cpu().numpy(),
        back_face_policy="cull",
    )
    ps_mobius.set_color(render_color)

    ps_tiefighter = ps.register_surface_mesh(
        "TieFighter",
        vertices_tiefighter.cpu().numpy() + np.array([offset, 0, 0]),
        faces_tiefighter.cpu().numpy(),
        back_face_policy="cull",
    )
    ps_tiefighter.set_color(render_color)

    if ps.is_headless():
        ps.screenshot()
    else:
        ps.show()


if __name__ == "__main__":
    main()
