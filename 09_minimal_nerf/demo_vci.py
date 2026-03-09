# /// script
# dependencies = [
#   "numpy",
#   "torch",
#   "torchvision",
#   "imageio",
#   "tqdm",
#   "tinycudann @ git+https://github.com/NVlabs/tiny-cuda-nn/@075158a70b87dba8729188a9cadc9411cfa4b71d#subdirectory=bindings/torch",
# ]
# ///

import json
import os
import pathlib
import random

import imageio.v2 as imageio
import numpy as np
import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.minimal_nerf import (
    PositionalEncoding,
    stratified_sampling,
    volumetric_rendering,
)
from torch.utils.data import Dataset
from torchvision.utils import save_image
from tqdm import tqdm

""" 
INFO

This demo runs the NeRF-code you wrote with your own, captured VCI data set, 
using a single static frame.
To run the demo, perform the following steps:
- Make a directory called 'vci' within the assets folder: mkdir ./assets/vci
- Copy your data set into the directory
- Complete the two functions 'create_rays' and 'forward' with your solution from the 
    assignment. Note that now, instead of the focal length, you need to use the given
    values from the instrinsics matrix, so adjust your code accordingly. (Both marked 
    with 'TODO')
- To select a different frame from your sequence of multi-view captures for 
    reconstruction, or to select a different camera for rendering training or testing 
    views, change the corresponding variables. (Also marked with "TODO")
- In the case of parts of the reconstruction being "cut off", you might need to adjust
    the bounding box limits of the scene. (Also marked with "TODO")
You will probably notice, that the reconstructed results are still of relatively low 
quality. Feel free to play around with the implementation and add the NeRF components 
missing from our minimal NeRF implementation!
"""


def create_rays(
    i: torch.Tensor,  # N
    j: torch.Tensor,  # N
    pose: torch.Tensor,  # N x 4 x 4 or 4 x 4
    H: int,
    W: int,
    K: torch.Tensor,  # N x 3 x 3 or 3 x 3
) -> tuple[
    torch.Tensor,  # N x 3
    torch.Tensor,  # N x 3
]:
    T_cv_to_blender = torch.diag(
        torch.tensor([1, -1, -1], dtype=pose.dtype, device=pose.device)
    )
    T_cv_to_blender_homogeneous = torch.diag(
        torch.tensor([1, -1, -1, 1], dtype=pose.dtype, device=pose.device)
    )

    origins = torch.zeros((i.shape[0], 3), device=i.device)
    dirs = torch.zeros((i.shape[0], 3), device=i.device)

    # TODO: Transform pixel coordinates into normalized ray directions.
    # Copy your solution from minimal_nerf.py and adjust it to use 'K' instead of 'focal'

    return origins, dirs


def forward(
    model: nn.Module,
    i: torch.Tensor,  # N
    j: torch.Tensor,  # N
    pose: torch.Tensor,  # N x 4 x 4 or 4 x 4
    H: int,
    W: int,
    K: torch.Tensor,  # N x 3 x 3 or 3 x 3
    N_samples: int,
    near: float,
    far: float,
) -> torch.Tensor:  # N x 3
    colors_pred = torch.zeros((i.shape[0], 3), device=i.device)

    # TODO: Implement the forward pass of the NeRF model.
    # Copy your solution from minimal_nerf.py and adjust it to use 'K' instead of 'focal'

    return colors_pred


class InstantNGP(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 8,
                "n_features_per_level": 4,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
            },
            dtype=torch.float32,
        )
        self.dir_encoding = PositionalEncoding(4)
        self.network = nn.Sequential(
            nn.Linear(self.pos_encoding.n_output_dims, 64), nn.ReLU(), nn.Linear(64, 16)
        )
        self.rgb_network = nn.Sequential(
            nn.Linear(16 + self.dir_encoding.n_output_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        if False:
            self.network = tcnn.Network(
                n_input_dims=self.pos_encoding.n_output_dims,
                n_output_dims=16,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            ).to(torch.float32)
            self.rgb_network = tcnn.NetworkWithInputEncoding(
                n_input_dims=3 + 16,
                n_output_dims=3,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3,
                            "otype": "SphericalHarmonics",
                            "degree": 4,
                        },
                        {"otype": "Identity"},
                    ],
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            ).to(torch.float32)

        # TODO: Adjust the bounding box limits if necessary
        self.register_buffer("aabb_min", torch.tensor([[-2.0, -2.0, -0.25]]))
        self.register_buffer("aabb_max", torch.tensor([[2.0, 2.0, 2.5]]))

    def _contract(self, x):
        """Contracts points towards the origin (Eq 10 of arxiv.org/abs/2111.12077)."""
        # Clamping to 1 produces correct scale inside |x| < 1
        x_mag_sq = torch.clamp(torch.sum(x**2, axis=-1, keepdims=True), min=1.0)
        scale = (2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq
        z = scale * x
        return (z + 2) / 4

    def forward(self, x, d):
        # x = self._contract(x)
        x = torch.clamp(
            (x - self.aabb_min) / (self.aabb_max - self.aabb_min), min=0, max=1
        )
        features = self.pos_encoding(x)
        features = self.network(features)
        rgb = self.rgb_network(torch.cat((self.dir_encoding(d), features), dim=-1))
        return torch.exp(features[..., 0]), torch.sigmoid(rgb)


class VCIDataset(Dataset):
    def __init__(self, datadir, split="train", img_size=(666, 576)):
        with open(os.path.join(datadir, f"calibration_dome.json"), "r") as fp:
            meta = json.load(fp)
        self.img_size = img_size
        self.imgs = []
        self.poses = []
        self.Ks = []
        frame = "00000"  # TODO: Select the frame of your data you want to reconstruct
        subfolders = [
            f for f in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, f))
        ]
        subfolders.remove("background")
        subfolders.remove("preview")
        single_frames = subfolders[0][:5] == "frame"
        for cam in meta["cameras"]:
            cid = cam["camera_id"]
            if single_frames:
                img_path = os.path.join(datadir, "frame_" + frame, "rgb", cid + ".jpg")
                mask_path = os.path.join(
                    datadir, "frame_" + frame, "mask", "mask_" + cid + ".png"
                )
                if not os.path.isfile(img_path):
                    continue
            else:
                if not os.path.isdir(os.path.join(datadir, cid)):
                    continue
                img_path = os.path.join(
                    datadir, cid, "rgb", cid + "_F" + frame + ".jpg"
                )
                mask_path = os.path.join(
                    datadir, cid, "mask", cid + "_F" + frame + ".png"
                )
            img = imageio.imread(img_path)
            img = (img / 255.0).astype(np.float32)
            mask = imageio.imread(mask_path)
            mask = (mask / 255.0).astype(np.float32)[..., None]
            img = img[..., :3] * mask + (1 - mask) * 0.0
            self.imgs.append(img)
            pose = np.array(cam["extrinsics"]["view_matrix"]).reshape(4, 4)
            pose = self._fix_coordinate_system(pose)
            self.poses.append(pose)
            K = np.array(cam["intrinsics"]["camera_matrix"]).reshape(3, 3)
            w_org, h_org = cam["intrinsics"]["resolution"]
            factor = img.shape[1] / w_org
            K[:2, :] *= factor
            self.Ks.append(K)
        num_imgs = len(self.imgs)
        test = np.arange(0, num_imgs, 10)
        splits = {"train": np.delete(np.arange(num_imgs), test), "test": test}
        self.imgs = np.stack(self.imgs).astype(np.float32)[splits[split]]
        self.poses = np.stack(self.poses).astype(np.float32)[splits[split]]
        self.Ks = np.stack(self.Ks).astype(np.float32)[splits[split]]
        self.H, self.W = self.imgs[0].shape[:2]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.poses[idx]

    def _fix_coordinate_system(self, extrinsics):
        inv_extrinsics = np.linalg.inv(extrinsics)
        rotation = inv_extrinsics[:3, :3]
        position = inv_extrinsics[:3, -1]
        mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        extrinsics[:3, :3] = mat @ rotation.T @ mat
        extrinsics[:3, -1] = -mat @ rotation.T @ position
        extrinsics = np.linalg.inv(extrinsics)
        t_gl = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        extrinsics[:3, :] = t_gl @ extrinsics[:3, :]
        up_is_down = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        extrinsics[:3, :] = up_is_down @ extrinsics[:3, :]
        return extrinsics


class TrainDataset(VCIDataset, torch.utils.data.IterableDataset):
    def __init__(self, datadir, split="train", img_size=(666, 576)):
        super().__init__(datadir, split, img_size)

    def __iter__(self):
        return self

    def __next__(self):
        img_idx = random.randint(0, len(self) - 1)
        i = random.randint(0, self.W - 1)
        j = random.randint(0, self.H - 1)
        pose = self.poses[img_idx]
        K = self.Ks[img_idx]
        colors_gt = self.imgs[img_idx, j, i]
        return i, j, pose, K, colors_gt


class Trainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = InstantNGP()
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.pos_encoding.parameters()},
                {"params": self.model.network.parameters(), "weight_decay": 1e-6},
                {
                    "params": self.model.rgb_network.parameters(),
                    "weight_decay": 1e-6,
                },
            ],
            lr=0.001,
            eps=1e-15,
        )
        self.loss_fn = nn.MSELoss()
        self.n_samples = 128
        self.near = 1.5
        self.far = 7.5

    def train_step(
        self,
        pixels,
        cameras,
        colors_gt,
    ):
        colors_pred = forward(
            self.model,
            *pixels,
            *cameras,
            self.n_samples,
            self.near,
            self.far,
        )

        # Loss and optimize
        loss = self.loss_fn(colors_pred, colors_gt)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def render_image(self, *camera, chunk=1024):
        """Render an image using the trained model"""
        self.model.eval()
        # Generate rays for the entire image
        _, H, W, _ = camera
        i, j = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="ij")
        device = self.model.network[0].weight.device
        i_chunks = torch.split(i.flatten(), chunk)
        j_chunks = torch.split(j.flatten(), chunk)

        # Process in chunks to avoid memory issues
        all_colors = []
        for pixels in zip(i_chunks, j_chunks):
            pixels = (t.to(device) for t in pixels)
            colors = forward(
                self.model, *pixels, *camera, self.n_samples, self.near, self.far
            )
            all_colors.append(colors.cpu())

        image = torch.cat(all_colors).reshape(W, H, 3)
        self.model.train()
        return image.permute(2, 1, 0)  # CxHxW


def main():
    # torch.autograd.set_detect_anomaly(True)
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda")
    trainer = Trainer().to(device)
    batch_size = 2**14

    # Load datasets
    data_dir = pathlib.Path(__file__).parent / "assets" / "vci"
    train_dataset = TrainDataset(data_dir, split="train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size, num_workers=8, pin_memory=True
    )
    test_dataset = VCIDataset(data_dir, split="test")

    # Create output directories
    os.makedirs("train_renders_vci", exist_ok=True)
    os.makedirs("test_renders_vci", exist_ok=True)

    # Select fixed images for visualization
    train_idx = 0  # TODO: Select the training view you want to render
    test_idx = 0  # TODO: Select the test view you want to render
    train_pose = torch.as_tensor(train_dataset.poses[train_idx]).to(device)
    train_K = torch.as_tensor(train_dataset.Ks[train_idx]).to(device)
    test_pose = torch.as_tensor(test_dataset.poses[test_idx]).to(device)
    test_K = torch.as_tensor(test_dataset.Ks[test_idx]).to(device)
    H, W = train_dataset.img_size

    pbar = tqdm(range(2500))
    for epoch, batch in zip(pbar, train_dataloader):
        i, j, poses, K, colors_gt = (t.to(device, non_blocking=True) for t in batch)
        trainer.train_step((i, j), (poses, H, W, K), colors_gt)

        # Rendering phase
        if epoch % 100 == 0:  # Render every epoch
            # Render training view
            train_render = trainer.render_image(train_pose, H, W, train_K)
            save_image(train_render, f"train_renders_vci/iter_{epoch//100:04d}.png")

            # Render test view
            test_render = trainer.render_image(test_pose, H, W, test_K)
            save_image(test_render, f"test_renders_vci/iter_{epoch//100:04d}.png")


if __name__ == "__main__":
    main()
