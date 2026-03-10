# /// script
# dependencies = [
#   "numpy",
#   "polyscope",
#   "torch",
#   "trimesh",
#   "tqdm",
#   "fused_ssim @ git+https://github.com/rahul-goel/fused-ssim/",
#   "Pillow",
# ]
# ///

from __future__ import annotations
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parents[0] / "src"))

from random import randint
from time import perf_counter, sleep

import numpy as np
import polyscope as ps
import torch
import threading
import polyscope.imgui as psim
import trimesh
from fused_ssim import fused_ssim
from PIL import Image
import argparse
from tqdm import tqdm


from gaussian_splatting import (
    compute_image_positional_gradients,
    grow_and_prune,
    rasterize,
    reset_splats,
)
from utils import (
    Camera,
    GaussianSplats,
    load_texture,
    convert_intrinsics_cv_to_gl,
    load_extrinsics,
)

torch.cuda.init()
torch.set_float32_matmul_precision('high')


def perspective(fov_y: float, aspect: float, zNear: float, zFar: float) -> torch.Tensor:
    tanHalfFovy = np.tan(fov_y / 2.0)

    result = torch.zeros((4, 4), dtype=torch.float32).cuda()
    result[0][0] = 1.0 / (aspect * tanHalfFovy)
    result[1][1] = 1.0 / (tanHalfFovy)
    result[2][2] = -(zFar + zNear) / (zFar - zNear)
    result[2][3] = -1.0
    result[3][2] = -(2.0 * zFar * zNear) / (zFar - zNear)
    return result.T


def splats_to_pointcloud(splats: GaussianSplats) -> torch.Tensor:
    pc = ps.register_point_cloud(
        name="Splats",
        points=splats.means.detach().cpu().numpy(),
        radius=0.01,
    )
    pc.add_color_quantity("rgb", (splats.colors[:, :, 0] * 0.282095).detach().cpu().numpy(), enabled=True)
    pc.add_scalar_quantity("opacity", splats.opacities[:, 0].detach().cpu().numpy())
    pc.set_transparency_quantity("opacity")
    
    radii = torch.exp(splats.scales).amax(dim=1).sqrt().detach().cpu().numpy()
    pc.add_scalar_quantity("radius", radii * 100)
    pc.set_point_radius_quantity("radius")

    return pc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gui", action="store_true", help="Run without Polyscope GUI")
    parser.add_argument("--use-torch-compile", action="store_true", help="Use torch.compile to speed up optimization (linux only)")
    args = parser.parse_args()
    
    mesh = trimesh.load("assets/bunny.obj")

    vertices = torch.from_numpy(mesh.vertices).cuda().to(torch.float32)
    vertices[:, 1] -= 0.6
    print("num loaded vertices:", vertices.shape)
    
    targets = []
    for i in range(100):
        targets.append(
            load_texture(f"assets/targets/target_{i}.png")[..., 0:3].contiguous()
        )

    render_ds = 1
    w = 640 // render_ds
    h = 480 // render_ds
    focal_length = 416.6 / render_ds
    cx = 320.0 / render_ds
    cy = 240.0 / render_ds
    P = convert_intrinsics_cv_to_gl(focal_length, focal_length, cx, cy, w, h, 0.01, 100.0)

    views = load_extrinsics("./assets/views.bin")
    
    rand_idxs = torch.randperm(len(vertices)) #[:5000]
    initial_means = vertices[rand_idxs].contiguous()
    num_splats = initial_means.shape[0]
    initial_colors = torch.zeros(num_splats, 3, 9).cuda()
    initial_colors[:, :, 0] = torch.rand(num_splats, 3).cuda() * (1 / 0.282095)
    
    splats = GaussianSplats.initalize_from_means(initial_means)
    
    if not args.no_gui:
        ps.set_allow_headless_backends(True)
        ps.init()
        ps.set_always_redraw(True)
        ps.set_ground_plane_mode("none")
        ps.set_background_color((0, 0, 0))
    
    cameras = []
    for i in range(len(views)):
        cam = Camera(views[i].cuda(), P, (h, w), targets[i], focal_length=focal_length)
        cameras.append(cam)
    
    camera_indices = list(range(100))
    optimized_cameras = [cameras[index] for index in camera_indices]
    
    if not args.no_gui:
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
        
    for t in splats:
        t.requires_grad = True
        
    base_lr = 1
        
    optimizer = torch.optim.RAdam([
        {'params': splats.means_, 'lr': 1e-2 * base_lr},
        {'params': splats.orientations_, 'lr': 1e-0 * base_lr},
        {'params': splats.scales_, 'lr': 1e-3 * base_lr},
        {'params': splats.colors_, 'lr': 1e-0 * base_lr},
        {'params': splats.opacities_, 'lr': 5e-1 * base_lr},
    ])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    
    if not args.no_gui:
        ps_splats = splats_to_pointcloud(splats)

    last_start = perf_counter()
    lock = threading.Lock()
    is_running = True
    loss_cpu = 0
    num_iterations = 0
    opt_fps = 0
    last_render_start = perf_counter()
    sh_order = 0
    is_updated_since_last_shown = True
    
    grow_and_prune_at = list(range(1000, 11_000, 1000))
    reset_at = list(range(4800, 11_000, 4800))
    activate_next_sh_order_at = [3_000, 8_000]
    optimize_every = 10
    save_img_every = 499
    
    should_pause_opt = False
    
    if torch.cuda.is_available() and args.use_torch_compile:
        print("Using torch.compile to speed up optimization (may take some minutes to compile)")
        
        if sys.platform == "win32":
            print("Warning: Using aot_eager backend for torch.compile on Windows (might be slower than uncompiled)")
            rasterize = torch.compile(rasterize, backend="aot_eager", dynamic=True)
        else:
            rasterize = torch.compile(rasterize, backend="inductor")
    
    def optimize_loop():
        global is_running, loss_cpu, num_iterations, last_render_start, opt_fps, splats, sh_order, is_updated_since_last_shown
        
        with tqdm() as pbar:
            while is_running:
                
                if should_pause_opt:
                    sleep(1.0)
                    continue
                
                cur_start = perf_counter()
                opt_fps = 1 / (cur_start - last_render_start)
                last_render_start = cur_start
                
                loss = torch.tensor(0.0).cuda()
                at_least_one_rendered = False
                grad_2d = []
                area_2d = []
                
                
                for _ in range(5):
                    cam_idx = randint(0, len(optimized_cameras) - 1)
                    camera = optimized_cameras[cam_idx]
                    
                    render_result = rasterize(
                        image_size=(camera.resolution[0], camera.resolution[1]),
                        tile_size=(32, 32),
                        splats=splats,
                        camera=camera,
                        num_random_tiles=16,
                        sh_order=sh_order,
                    )
                    rendered = render_result["image"]
                    valid_mask = render_result["valid_mask"]
                    target = camera.image[0].cuda()[::render_ds, ::render_ds, :3]
                    
                    if rendered is not None:
                        
                        l1_loss = torch.nn.functional.l1_loss(
                            input=rendered[:, :, :3] * valid_mask[:, :, None],
                            target=target * valid_mask[:, :, None]
                        )
                        ssim_loss = 1 - fused_ssim(
                            (rendered[:, :, :3] * valid_mask[:, :, None])[None],
                            (target * valid_mask[:, :, None])[None]
                        )
                        loss = loss + l1_loss + 0.2 * ssim_loss
                        
                        at_least_one_rendered = True
                        
                        if num_iterations in grow_and_prune_at:
                            grads = compute_image_positional_gradients(
                                means_2d=render_result["means_2d"],
                                image_loss=l1_loss
                            )
                            grad_2d.append(grads)
                            area_2d.append(render_result["area_2d"])
                            
                if at_least_one_rendered:
                    loss.backward()
                    
                if num_iterations % optimize_every == 0:
                    
                    for p in splats:
                        if p.grad is not None:
                            torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0, out=p.grad)
                    torch.nn.utils.clip_grad_norm_(splats, max_norm=1.0, error_if_nonfinite=False)
                    
                    optimizer.step()
                    scheduler.step()        
                    optimizer.zero_grad()
                    loss_cpu = loss.item()
                    is_updated_since_last_shown = True
                    
                if num_iterations % save_img_every == 0:
                    with torch.no_grad():
                        print("Rendering output image at iteration", num_iterations)
                        out_camera = optimized_cameras[71]
                        render_result = rasterize(
                            image_size=(out_camera.resolution[0], out_camera.resolution[1]),
                            tile_size=(32, 32),
                            splats=splats,
                            camera=out_camera,
                            num_random_tiles=None,
                            sh_order=sh_order,
                        )
                        rendered = render_result["image"]
                        if rendered is not None:
                            img = ((rendered[:, :, :3].clamp(0, 1)) * 255).to(torch.uint8).cpu().numpy()[::-1]
                            im = Image.fromarray(img)
                            path = f"out_{num_iterations:05d}.png"
                            im.save(path)
                            print("Saved image as", path)
                            
                        if num_iterations == 0:
                            target_img = (out_camera.image[0, ::render_ds, ::render_ds, :3] * 255).to(torch.uint8).cpu().numpy()[::-1]
                            im = Image.fromarray(target_img)
                            path = f"out_target.png"
                            im.save(path)
                            print("Saved target image as", path)
                    
                if num_iterations in grow_and_prune_at:
                    with torch.no_grad():
                        
                        if len(grad_2d) > 0:
                            grad_2d = torch.stack(grad_2d, dim=0).mean(dim=0).norm(dim=-1)
                            area_2d = torch.stack(area_2d, dim=0).mean(dim=0)
                            
                            splats = grow_and_prune(
                                splats=splats,
                                grad_mag_2d=grad_2d,
                                area_2d=area_2d,
                                grad_2d_magnitude_threshold=0.000002, # duplicate
                                area_2d_threshold=max(10, 100 - (num_iterations / 250)), # splitting
                                opacity_threshold=0.5, # removing
                                optimizer=optimizer
                            )
                        
                if num_iterations in reset_at:
                    with torch.no_grad():
                        reset_splats(splats)
                        
                if num_iterations in activate_next_sh_order_at:
                    sh_order = min(sh_order + 1, 2)
                    print("Activating SH order", sh_order)
                    
                num_iterations += 1         
                pbar.update(1)         
                    
                    
    threading.Thread(target=optimize_loop).start()
    
    last_iteration_shown = -1
    should_render_splats = True
    should_prioritize_opt = False
    should_render_fullres = False
    last_camera = None
    
    def ps_loop():
        global last_start, ps_splats, should_render_splats, should_prioritize_opt, last_iteration_shown, should_pause_opt, should_render_fullres, is_updated_since_last_shown, last_camera
        
        _, should_render_splats = psim.Checkbox("render splats", should_render_splats)
        _, should_render_fullres = psim.Checkbox("render full resolution", should_render_fullres)
        _, should_prioritize_opt = psim.Checkbox("prioritize optimization (reduces Polyscope FPS)", should_prioritize_opt)
        _, should_pause_opt = psim.Checkbox("pause optimization (prevents crashes when interacting with UI)", should_pause_opt)
        
        max_fps = 1 if (should_prioritize_opt and not should_pause_opt) else 120
        
            
        cur_start = perf_counter()
        ps_fps = 1 / (cur_start - last_start)
        last_start = cur_start
        
        psim.TextUnformatted(
            f"Polyscope FPS: {ps_fps:.4f}\n"
            f"Optimize it/s: {opt_fps:.4f}\n"
            f"Loss: {loss_cpu:.4f}\n"
            f"Iterations: {num_iterations}\n"
            f"Num splats: {splats.means.shape[0]}\n"
            f"Lr: {scheduler.get_last_lr()[0]:.6f}\n"
        )
        
        if not should_render_splats and num_iterations > last_iteration_shown:
            
            ps_splats.remove()
            ps_splats = splats_to_pointcloud(splats)
            last_iteration_shown = num_iterations
        
        if should_render_splats:
            
            # downscale factor
            ds_factor = 1 if should_render_fullres else 2
            
            view = ps.get_camera_view_matrix().astype(np.float32)
            if np.any(np.isnan(view)):
                view = np.eye(4, dtype=np.float32)
            view = torch.from_numpy(view).cuda()

            size = ps.get_window_size()
            
            if size[0] > 0 and size[1] > 0:
            
                fov = ps.get_view_camera_parameters().get_fov_vertical_deg()
                
                size_down = (size[1] // ds_factor, size[0] // ds_factor)

                P = perspective(np.deg2rad(fov), size_down[1] / size_down[0], 0.01, 100.0)

                # convert fov to focal length
                camera = Camera(
                    V=view,
                    P=P,
                    resolution=size_down,
                    image=None,
                    focal_length=0.5 * size_down[0] / np.tan(np.deg2rad(fov) / 2)
                )
                
                if is_updated_since_last_shown or last_camera != camera:
                    is_updated_since_last_shown = False
                
                    with torch.no_grad():
                        render_result = rasterize(
                            image_size=size_down,
                            tile_size=(32, 32),
                            splats=splats,
                            camera=camera,
                        )
                        
                        if render_result["image"] is not None:
                            image = render_result["image"].clamp(0, 1)

                            if image is not None:
                                ps.add_raw_color_render_image_quantity(
                                    name="color_img",
                                    depth_values=np.zeros(size_down, dtype=np.float32),
                                    color_values=image[:, :, :3].cpu().numpy()[::-1],
                                    enabled=should_render_splats
                                )
                                
                                
                last_camera = camera
                
        else:
            ps.add_raw_color_render_image_quantity(
                name="color_img",
                depth_values=np.zeros((10, 10), dtype=np.float32),
                color_values=np.zeros((10, 10, 3), dtype=np.float32),
                enabled=False
            )
        
        sleep(1/max(1, max_fps))
    
    if args.no_gui:
        try:
            while True:
                sleep(1 / 30)
        except KeyboardInterrupt:
            print("Exiting...")
    else:
        ps.set_user_callback(ps_loop)
        if ps.is_headless():
            ps.screenshot(transparent_bg=False)
        else:
            ps.show()
    
    is_running = False
