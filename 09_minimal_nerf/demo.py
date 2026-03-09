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
from torch.utils.data import Dataset
from torchvision.utils import save_image
from tqdm import tqdm

from src.minimal_nerf import PositionalEncoding, forward


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

        self.register_buffer("aabb_min", torch.tensor([[-3, -3, -3]]))
        self.register_buffer("aabb_max", torch.tensor([[3, 3, 3]]))

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


class BlenderDataset(Dataset):
    def __init__(self, datadir, split="train", img_size=(800, 800)):
        with open(os.path.join(datadir, f"transforms_{split}.json"), "r") as fp:
            meta = json.load(fp)
        self.img_size = img_size
        self.focal = 0.5 * img_size[0] / np.tan(0.5 * meta["camera_angle_x"])
        self.imgs = []
        self.poses = []
        for frame in meta["frames"]:
            img_path = os.path.join(datadir, frame["file_path"][2:] + ".png")
            img = imageio.imread(img_path)
            img = (img / 255.0).astype(np.float32)
            if img.shape[-1] == 4:
                img = img[..., :3] * img[..., 3:] + (1 - img[..., 3:]) * 0.0
            self.imgs.append(img)
            pose = np.stack(frame["transform_matrix"])
            self.poses.append(pose)
        self.imgs = np.stack(self.imgs).astype(np.float32)
        self.poses = np.stack(self.poses).astype(np.float32)
        self.H, self.W = self.imgs[0].shape[:2]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.poses[idx]


class TrainDataset(BlenderDataset, torch.utils.data.IterableDataset):
    def __init__(self, datadir, split="train", img_size=(800, 800)):
        super().__init__(datadir, split, img_size)

    def __iter__(self):
        return self

    def __next__(self):
        img_idx = random.randint(0, len(self) - 1)
        i = random.randint(0, self.W - 1)
        j = random.randint(0, self.H - 1)
        pose = self.poses[img_idx]
        colors_gt = self.imgs[img_idx, j, i]
        return i, j, pose, colors_gt


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
        self.near = 2.0
        self.far = 6.0

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

        image = torch.cat(all_colors).reshape(H, W, 3)
        self.model.train()
        return image.permute(2, 1, 0)  # CxHxW


def main():
    # torch.autograd.set_detect_anomaly(True)
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda")
    trainer = Trainer().to(device)
    batch_size = 2**14

    # Load datasets
    data_dir = pathlib.Path(__file__).parent / "assets" / "data"
    train_dataset = TrainDataset(data_dir, split="train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size, num_workers=8, pin_memory=True
    )
    test_dataset = BlenderDataset(data_dir, split="test")

    # Create output directories
    os.makedirs("train_renders", exist_ok=True)
    os.makedirs("test_renders", exist_ok=True)

    # Select fixed images for visualization
    train_pose = torch.as_tensor(train_dataset.poses[0]).to(device)
    test_pose = torch.as_tensor(test_dataset.poses[0]).to(device)
    H, W = train_dataset.img_size
    focal = train_dataset.focal

    pbar = tqdm(range(2500))
    for epoch, batch in zip(pbar, train_dataloader):
        i, j, poses, colors_gt = (t.to(device, non_blocking=True) for t in batch)
        trainer.train_step((i, j), (poses, H, W, focal), colors_gt)

        # Rendering phase
        if epoch % 100 == 0:  # Render every epoch
            # Render training view
            train_render = trainer.render_image(train_pose, H, W, focal)
            save_image(train_render, f"train_renders/iter_{epoch//100:04d}.png")

            # Render test view
            test_render = trainer.render_image(test_pose, H, W, focal)
            save_image(test_render, f"test_renders/iter_{epoch//100:04d}.png")


if __name__ == "__main__":
    main()
