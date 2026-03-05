# /// script
# dependencies = [
#   "numpy",
#   "Pillow",
#   "polyscope",
#   "scikit-image",
#   "scipy",
#   "torch",
#   "tqdm",
# ]
# ///

import multiprocessing as mp
import pathlib

import numpy as np
import polyscope as ps
import skimage.measure as measure
import torch
import torch.nn as nn
from PIL import Image
from src.neural_sdf import Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm


# ------------------------------
# Dataset & helpers
# ------------------------------
class BunnyDataset:
    def __init__(self, data_dir: pathlib.Path) -> None:

        self.T = self.load_extrinsics(data_dir / "views.bin")
        self.rgb, self.depth = self.load_images(data_dir, len(self.T))

        f_x = 416.6
        f_y = 416.6
        c_x = 320.0
        c_y = 240.0
        self.K = torch.tensor(
            [
                [f_x, 0, c_x],
                [0, f_y, c_y],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )

    def __iter__(self):
        for i in range(len(self.T)):
            yield self.K, self.T[i], self.rgb[i], self.depth[i]

    def __len__(self) -> int:
        return len(self.T)

    def load_extrinsics(self, path: pathlib.Path) -> list[torch.Tensor]:
        T = torch.from_numpy(np.fromfile(path, dtype=np.float32)).reshape(-1, 4, 4)

        T = T.transpose(2, 1)
        T = (
            torch.tensor([1, -1, -1, 1], dtype=torch.float32).diag()
            @ T
            @ torch.tensor([1, -1, -1, 1], dtype=torch.float32).diag()
        )

        return [T[i, :, :] for i in range(T.shape[0])]

    def load_images(
        self,
        data_dir: pathlib.Path,
        num_images: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        # RGB
        rgbs = []

        for i in range(num_images):
            rgbs.append(
                (
                    torch.from_numpy(
                        np.array(Image.open(data_dir / "rgb" / f"rgb_{i}.png"))
                    ).to(torch.float32)
                    / 255
                )[:, :, 0:3]
            )

        # Depth
        depths = []
        for i in range(num_images):
            depths.append(
                (
                    torch.from_numpy(
                        np.array(Image.open(data_dir / "depth" / f"depth_{i}.png"))
                    ).to(torch.float32)
                    / 1000
                )[:, :, None]
            )

        return rgbs, depths


def unproject(cam2pix, world2cam, depth):
    height, width = depth.shape[:2]
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    pixels = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()

    # Unproject to camera space
    rays = torch.linalg.solve(cam2pix, pixels.reshape(-1, 3).T).T
    points = rays * depth.reshape(-1, 1)

    # Transform to world space
    homogeneous = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
    world_points = (torch.linalg.inv(world2cam) @ homogeneous.T).T[:, :3]
    world_points[..., 1:] *= -1

    return world_points


def random_points_in_box(box_min, box_max, n):
    dims = box_min.shape[0]
    rand_vals = np.random.uniform(0, 1, (n, dims))
    points = box_min + rand_vals * (box_max - box_min)
    return points


def random_points_on_box_surface(box_min, box_max, n):
    dims = box_min.shape[0]
    fixed_dims = np.random.randint(0, dims, (n,))
    rand_vals = np.random.uniform(0, 1, (n, dims))
    points = box_min + rand_vals * (box_max - box_min)
    choose_max = np.random.uniform(0, 1, n) < 0.5
    fixed_coords = np.where(
        choose_max,
        box_max[fixed_dims],
        box_min[fixed_dims],
    )
    points[np.arange(n), fixed_dims] = fixed_coords
    return points


class SurfaceSampleDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        # Load Bunny dummy dataset
        data_dir = pathlib.Path(__file__).parent / "assets" / "data"
        dataset = BunnyDataset(data_dir)
        points = []
        for cam2pix, world2cam, _, depth in dataset:
            all_points = unproject(cam2pix, world2cam, depth)
            valid_depth = (depth > 1e-3).view(-1, 1).expand(-1, 3)
            points.append(all_points[valid_depth].view(-1, 3).numpy(force=True))
        points = np.concatenate(points)
        # Get bounding box from the point cloud
        self.aabb_min = points.min(axis=0)
        self.aabb_max = points.max(axis=0)
        padding = 0.25
        self.aabb_min -= padding
        self.aabb_max += padding

        rng_order = np.random.shuffle(np.arange(len(points)))
        self.points = points[rng_order][0].astype(np.float32)
        self.volume_points = random_points_in_box(
            self.aabb_min, self.aabb_max, len(self.points)
        ).astype(np.float32)
        self.boundary_points = random_points_on_box_surface(
            self.aabb_min, self.aabb_max, len(self.points)
        ).astype(np.float32)
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        surface_point = self.points[self.idx]
        volume_point = self.volume_points[self.idx]
        boundary_point = self.boundary_points[self.idx]

        self.idx += 1
        if self.idx == len(self.points):
            self.idx = 0
            rng_order = np.random.shuffle(np.arange(len(self.points)))
            self.points = self.points[rng_order][0]
            self.volume_points = random_points_in_box(
                self.aabb_min, self.aabb_max, len(self.points)
            ).astype(np.float32)
            self.boundary_points = random_points_on_box_surface(
                self.aabb_min, self.aabb_max, len(self.points)
            ).astype(np.float32)
        return surface_point, volume_point, boundary_point


def get_voxel_center(voxel_size, aabb_min, aabb_max):
    grid_resolution = ((aabb_max - aabb_min) // voxel_size).astype(np.int32)
    x = torch.linspace(aabb_min[0], aabb_max[0], grid_resolution[0])
    y = torch.linspace(aabb_min[1], aabb_max[1], grid_resolution[1])
    z = torch.linspace(aabb_min[2], aabb_max[2], grid_resolution[2])
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
    return torch.stack([grid_x, grid_y, grid_z], dim=-1)


class SinActivation(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return torch.sin(x)


# ---------------------------------------------
# Multiprocessing training worker (GPU process)
# ---------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device {device}")

class TrainingWorker:
    def __init__(self):
        super().__init__()
        self.dl = None

    def train(
        self,
        vol_shape: tuple,
        shared_volume: mp.RawArray,  # mp.RawArray('f', ...)
        vol_lock: mp.Lock,
        update_event: mp.Event,
        stop_event: mp.Event,
        mesh_queue: mp.Queue,
        num_epochs: int,
        voxel_size: float,
    ):
        """
        Runs the training on a dedicated process with its *own* CUDA context.
        Avoids inheriting CUDA state by using the 'spawn' start method.
        """

        # IMPORTANT: do not touch CUDA in the parent. All CUDA usage is inside this worker.
        dataset = SurfaceSampleDataset()

        # Use spawn context for any *nested* DataLoader workers to avoid forking after CUDA init.
        # If your PyTorch version doesn't support multiprocessing_context, set num_workers=0.
        try:
            self.dl = DataLoader(
                dataset,
                batch_size=2**16,
                num_workers=mp.cpu_count(),
                multiprocessing_context=mp.get_context("spawn"),
                persistent_workers=True,
            )
        except TypeError:
            # Fallback for older PyTorch
            self.dl = DataLoader(dataset, batch_size=2**16, num_workers=0)

        trainer = Trainer(
            n_neurons=512,
            activation_fn=SinActivation(),
            learn_rate=0.005,
            surface_weight=1.0,
            eikonal_weight=0.5,
            boundary_weight=1.0,
        ).to(device)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer,
            T_max=num_epochs,  # eta_min=0.0025
        )
        # Grid for evaluation & mesh extraction
        aabb_min = dataset.aabb_min
        aabb_max = dataset.aabb_max
        grid_points = get_voxel_center(voxel_size, aabb_min, aabb_max)

        # Convenience view over the shared RawArray
        shared_np = np.frombuffer(shared_volume, dtype=np.float32).reshape(vol_shape)

        pbar = tqdm(enumerate(self.dl), total=num_epochs)
        ema_surface, ema_eikonal, ema_boundary, ema_combined = 1.0, 1.0, 1.0, 1.0

        for epoch, (surface_pts, volume_pts, boundary_pts) in pbar:
            if stop_event.is_set():
                break
            if epoch >= num_epochs:
                break

            # Push data to GPU and update model
            surface_pts = surface_pts.to(device)
            volume_pts = volume_pts.to(device)
            boundary_pts = boundary_pts.to(device)

            surface_loss, eikonal_loss, boundary_loss, combined_loss = trainer.step(
                surface_pts, volume_pts, boundary_pts
            )
            lr_scheduler.step()

            # Periodically publish visualization data to the parent process
            if epoch % 10 == 0:
                with torch.no_grad():
                    sdf_vals = trainer.eval(grid_points.view(-1, 3).to(device))
                    sdf_vals = (
                        sdf_vals.reshape(*grid_points.shape[:-1]).cpu().detach().numpy()
                    )

                    try:
                        spacing = (aabb_max - aabb_min) / np.array(
                            grid_points.shape[:-1]
                        )
                        vertices, faces, _, _ = measure.marching_cubes(
                            sdf_vals, 0.0, spacing=spacing
                        )
                        vertices += aabb_min
                    except Exception:
                        vertices, faces = None, None

                # Write into shared memory under a lock to avoid tearing
                with vol_lock:
                    shared_np[...] = sdf_vals
                update_event.set()  # signal new volume available

                # Send the latest mesh, dropping older ones if GUI is behind
                if vertices is not None and faces is not None:
                    try:
                        # Clear any stale entry so the queue always holds at most the latest
                        while not mesh_queue.empty():
                            mesh_queue.get_nowait()
                        mesh_queue.put_nowait(
                            (vertices.astype(np.float32), faces.astype(np.int32))
                        )
                    except Exception:
                        pass

            ema_surface = 0.9 * ema_surface + (1 - 0.9) * surface_loss.item()
            ema_eikonal = 0.9 * ema_eikonal + (1 - 0.9) * eikonal_loss.item()
            ema_boundary = 0.9 * ema_boundary + (1 - 0.9) * boundary_loss.item()
            ema_combined = 0.9 * ema_combined + (1 - 0.9) * combined_loss.item()
            pbar.set_description(
                f"Iteration {epoch:4d} | Surface: {ema_surface:.4f} | Eikonal: {ema_eikonal:.4f} | Boundary: {ema_boundary:.4f}| Combined: {ema_combined:.4f}"
            )


# --------------------------
# GUI process (main process)
# --------------------------


def main() -> None:
    # Use spawn to avoid inheriting CUDA state. Must be set before creating any processes.
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # start method was already set
        pass

    # Build dataset *in the GUI process* only to compute the bounds and grid
    dataset = SurfaceSampleDataset()

    voxel_size = 0.05
    grid_points = get_voxel_center(voxel_size, dataset.aabb_min, dataset.aabb_max)
    vol_shape = tuple(list(grid_points.shape[:-1]))

    # Shared memory for the dense SDF volume (float32), plus sync primitives
    ctx = mp.get_context("spawn")
    shared_volume = ctx.RawArray("f", int(np.prod(vol_shape)))  # contiguous buffer
    vol_lock = ctx.Lock()  # writer/reader lock
    update_event = ctx.Event()  # signals a new volume
    stop_event = ctx.Event()  # signals training to stop
    mesh_queue = ctx.Queue(maxsize=1)  # latest mesh only

    # Initialize shared buffer to zeros
    shared_np_parent = np.frombuffer(shared_volume, dtype=np.float32).reshape(vol_shape)
    shared_np_parent.fill(0.0)

    # Initialize Polyscope in the GUI process
    ps.set_allow_headless_backends(True)
    ps.init()
    ps.set_ground_plane_mode("tile")
    ps_plane = ps.add_scene_slice_plane()
    ps_plane.set_draw_widget(True)

    ps_volume = ps.register_volume_grid(
        "Volume",
        vol_shape,
        dataset.aabb_min,
        dataset.aabb_max,
        edge_width=1,
        edge_color=(0.4, 0.4, 0.4),
        cube_size_factor=0.01,
    )
    ps_volume.add_scalar_quantity(
        "Signed Distance",
        shared_np_parent.copy(),
        defined_on="nodes",
        enabled=True,
        cmap="coolwarm",
        vminmax=(-2.0, 2.0),
        isolines_enabled=True,
        isoline_width=0.01,
    )

    # Visualization callback that consumes updates from the training worker
    def visualization_callback():
        # Update volume if the worker signaled new data
        if update_event.is_set():
            with vol_lock:
                current_volume = (
                    np.frombuffer(shared_volume, dtype=np.float32)
                    .reshape(vol_shape)
                    .copy()
                )
            update_event.clear()
            ps_volume.add_scalar_quantity(
                "Signed Distance",
                current_volume,
                defined_on="nodes",
                enabled=True,
                cmap="coolwarm",
                vminmax=(-2.0, 2.0),
                isolines_enabled=True,
                isoline_width=0.01,
            )

        # Drain at most one latest mesh from the queue and display it
        try:
            vertices, faces = None, None
            while not mesh_queue.empty():
                vertices, faces = mesh_queue.get_nowait()
            if vertices is not None and faces is not None:
                ps_mesh = ps.register_surface_mesh(
                    "Isosurface", vertices.copy(), faces.copy()
                )
                ps_mesh.set_ignore_slice_plane(ps_plane, True)
        except Exception:
            pass

    ps.set_user_callback(visualization_callback)

    # Launch the training process (GPU work lives there)
    training_worker = TrainingWorker()
    proc = ctx.Process(
        target=training_worker.train,
        args=(
            vol_shape,
            shared_volume,
            vol_lock,
            update_event,
            stop_event,
            mesh_queue,
            1000,  # num_epochs
            voxel_size,
        ),
        daemon=False,  # explicit: let us shut it down cleanly
    )
    proc.start()

    # Show the Polyscope UI (blocks until closed)
    try:
        if ps.is_headless():
            ps.screenshot(transparent_bg=False)
        else:
            ps.show()
    finally:
        if training_worker.dl is not None:
            del training_worker.dl._iterator
        # Tell the worker to stop and wait a bit for a graceful exit
        stop_event.set()
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.terminate()


if __name__ == "__main__":
    main()
