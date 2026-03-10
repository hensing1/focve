from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))

import torch
from utils import (
    Camera,
    GaussianSplats,
    get_covariance_3d,
    get_initial_opacities,
    get_initial_scales,
    project_to_screen,
    sh_coeffs_to_rgb,
)


def compute_jacobians(
    means: torch.Tensor,  # N x 3
    fx: float,
    fy: float,
) -> torch.Tensor:  # N x 2 x 3
    """
    Computes the Jacobians of the projection function with respect to the 3D
    coordinates of the splats.

    Returns a tensor (N, 2, 3), where tensor[k] is the Jacobian of the
    projection function at the k-th splat.
    """
    J = torch.zeros((means.shape[0], 2, 3), device=means.device)

    # TODO: implement

    return J


def get_covariance_2d(
    jacobians: torch.Tensor,  # N x 2 x 3
    cov_3d: torch.Tensor,  # N x 3 x 3
) -> torch.Tensor:  # N x 2 x 2
    """
    Computes the covariance of the 2D screen coordinates of the splats.

    Returns a tensor (N, 2, 2), where tensor[k] is the covariance of the 2D
    screen coordinates of the k-th splat.
    """
    cov_2d = torch.zeros((jacobians.shape[0], 2, 2), device=jacobians.device)

    # TODO: implement

    return cov_2d


def get_bounding_boxes(
    covs_2d,  # N x 2 x 2
    num_std_devs=3.0,
) -> tuple[
    torch.Tensor,  # N x 2
    torch.Tensor,  # N x 2
]:
    """
    Computes the bounding boxes of the splats in 2D screen coordinates.

    Returns two tensors (min_xy, max_xy), where min_xy[k] and max_xy[k] are the
    minimum and maximum 2D screen coordinates of the k-th splat, such that
    num_std_devs standard deviations in each direction are included.
    """

    min_xy = torch.zeros((covs_2d.shape[0], 2), device=covs_2d.device)
    max_xy = torch.zeros((covs_2d.shape[0], 2), device=covs_2d.device)

    # TODO: implement

    return min_xy, max_xy


def compute_alpha_blending_weights(
    means_2d: torch.Tensor,  # N x 2
    pixel_coords: torch.Tensor,  # H x W x 2
    inv_covs_2d: torch.Tensor,  # N x 2 x 2
    opacities: torch.Tensor,  # N x 1
) -> torch.Tensor:  # N x H x W
    """
    Computes the alpha blending weights for each gaussian at each pixel.

    Returns a tensor (N, H, W), where tensor[k, y, x] is the blending weight
    of the k-th gaussian at pixel (y, x).
    """
    alpha = torch.zeros(
        (means_2d.shape[0], *pixel_coords.shape[:2]), device=means_2d.device
    )

    # TODO: implement

    return alpha


def compute_image_positional_gradients(
    means_2d: torch.Tensor,  # N x 2
    image_loss: torch.Tensor,  # scalar
) -> torch.Tensor:  # N x 2
    """
    Computes the gradients of the image loss w.r.t. to the 2D pixel position
    of the splats.

    Returns a tensor (N, 2), where tensor[k] is the gradient of the image loss
    w.r.t. the 2D image coordinates of the k-th splat.
    """
    grads = torch.zeros(means_2d.shape, device=means_2d.device)

    # TODO: implement

    return grads


def compute_2d_quantities(splats: GaussianSplats, camera: Camera):
    means_camera = splats.means @ camera.V[:3, :3].T + camera.V[:3, -1]
    cov_3d = get_covariance_3d(splats.scales, splats.orientations)
    J = compute_jacobians(means_camera, camera.focal_length, camera.focal_length)
    cov_2d = get_covariance_2d(J, cov_3d)
    means_2d = project_to_screen(splats.means, camera)

    return {
        "means_2d": means_2d,
        "depths": -means_camera[:, 2],
        "covs_2d": cov_2d,
    }


def rasterize_tile(
    image_size,  # (H, W)
    tile_size,  # (H, W)
    origin,  # (2,)
    means_2d,  # (N, 2)
    depths,  # (N,)
    inv_covs_2d,  # (N, 2, 2)
    colors,  # (N, 3)
    opacities,  # (N, 1)
):
    H, W = image_size
    h, w = tile_size
    device = means_2d.device

    means_2d = torch.stack(
        [
            (means_2d[:, 0] + 1) / 2 * W - origin[0],
            (means_2d[:, 1] + 1) / 2 * H - origin[1],
        ],
        dim=-1,
    )

    ys = torch.arange(0, h, device=device)
    xs = torch.arange(0, w, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1).float() + 0.5

    alpha = compute_alpha_blending_weights(
        means_2d=means_2d,
        pixel_coords=grid,
        inv_covs_2d=inv_covs_2d,
        opacities=opacities,
    )[..., None]

    sort_order = torch.argsort(depths)
    alpha = alpha[sort_order]
    colors = colors[sort_order]

    one = torch.ones((1, h, w), device=alpha.device, dtype=alpha.dtype)
    alpha_shifted = torch.cat([one, 1 - alpha[..., 0]], dim=0)
    T = torch.cumprod(alpha_shifted, dim=0)[:-1, ..., None]

    weights = T * alpha
    rgb_contrib = weights * colors[:, None, None]

    final_rgb = rgb_contrib.sum(dim=0)
    final_alpha = weights.sum(dim=0)

    image = torch.cat([final_rgb, final_alpha], dim=-1)
    return image


def rasterize(
    image_size,  # (H, W)
    tile_size,  # (H, W)
    splats: GaussianSplats,
    camera: Camera,
    std_devs=3.0,
    num_random_tiles=None,
    sh_order=2,
):
    # get params

    H, W = image_size
    h, w = tile_size
    device = splats.means.device

    # transform to 2d

    quants = compute_2d_quantities(splats, camera)
    all_means_2d = quants["means_2d"]
    depths = quants["depths"]
    covs_2d = quants["covs_2d"]
    colors = splats.colors
    opacities = splats.opacities

    # compute bounding boxes for each gaussian

    min_xy, max_xy = get_bounding_boxes(covs_2d=covs_2d, num_std_devs=std_devs)

    min_xy = torch.stack(
        [
            min_xy[:, 0] + (all_means_2d[:, 0] + 1) / 2 * W,
            min_xy[:, 1] + (all_means_2d[:, 1] + 1) / 2 * H,
        ],
        dim=-1,
    )
    max_xy = torch.stack(
        [
            max_xy[:, 0] + (all_means_2d[:, 0] + 1) / 2 * W,
            max_xy[:, 1] + (all_means_2d[:, 1] + 1) / 2 * H,
        ],
        dim=-1,
    )

    # compute area

    eigvals, _ = torch.linalg.eigh(covs_2d)  # eigvals: (N, 2), eigvecs: (N, 2, 2)
    std_devs = torch.sqrt(torch.clamp(eigvals, min=1e-6))  # (N, 2)
    area_2d = (
        (std_devs[:, 0] * std_devs[:, 1])
        * (std_devs[:, 0] * std_devs[:, 1])
        * (3.0**2)
        * 3.14
        * 4
    )

    # remove splats that are behind the camera

    mask = depths > 0
    means_2d = all_means_2d[mask]
    depths = depths[mask]
    covs_2d = covs_2d[mask]
    colors = colors[mask]
    opacities = opacities[mask]
    min_xy = min_xy[mask]
    max_xy = max_xy[mask]

    # reset colors for unused spherical harmonics coefficients

    with torch.no_grad():
        colors[:, :, (sh_order + 1) ** 2 :] = 0.0

    # compute colors from spherical harmonics coefficients

    colors = colors.clone()
    directions = camera.position[None, :] - splats.means[mask]
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
    colors = sh_coeffs_to_rgb(colors, directions)

    # if there are no splats, return None

    if means_2d.shape[0] == 0:
        return {"image": None, "valid_mask": None, "means_2d": None, "area_2d": None}

    # get inverse covariances

    inv_covs_2d = torch.linalg.inv(covs_2d)

    # compute all tile origins

    x_coords = torch.arange(0, W, w, device=device)
    y_coords = torch.arange(0, H, h, device=device)
    y_coords, x_coords = torch.meshgrid(y_coords, x_coords, indexing="ij")
    origins = torch.stack((x_coords.flatten(), y_coords.flatten()), dim=1)

    # allocate final image and mask of filled tiles

    image = torch.zeros((H, W, 4), device=device)
    tile_mask = torch.zeros((H, W), device=device, dtype=torch.bool)

    # render all tiles (or a random subset)

    if num_random_tiles is not None:
        origins = origins[torch.randperm(len(origins))[:num_random_tiles]]

    at_least_one_tile = False

    for origin in origins:

        # find all gaussians that intersect with the bbox of the tile

        mask = (
            (min_xy[:, 0] < origin[0] + w)
            & (max_xy[:, 0] > origin[0])
            & (min_xy[:, 1] < origin[1] + h)
            & (max_xy[:, 1] > origin[1])
        )

        if mask.sum() > 0:

            # render tile

            tile = rasterize_tile(
                image_size=image_size,
                tile_size=tile_size,
                origin=origin,
                means_2d=means_2d[mask],
                depths=depths[mask],
                inv_covs_2d=inv_covs_2d[mask],
                colors=colors[mask],
                opacities=opacities[mask],
            )

            # paste the tile onto the final image

            x0, y0 = origin
            x1, y1 = origin[0] + tile_size[0], origin[1] + tile_size[1]
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(W, x1)
            y1 = min(H, y1)

            image[y0:y1, x0:x1] = tile[y0 - y0 : y1 - y0, x0 - x0 : x1 - x0]
            tile_mask[y0:y1, x0:x1] = True
            at_least_one_tile = True

    # if no tiles were rendered, return None

    if not at_least_one_tile:
        image = None

    # prevent saturation

    if image is not None:
        image = -torch.exp(-image) + 1.0

    return {
        "image": image,  # (H, W, 4)
        "valid_mask": tile_mask,  # (H, W)
        "means_2d": all_means_2d,  # (N, 3)
        "area_2d": area_2d,  # (N,)
    }


def _resize_moment(moment, mask_remaining, mask_split, mask_dup):
    """
    Resize a 1st-dimension-indexed Adam-like moment buffer to match:
      new = [remaining, split_original, split_new, dup_original, dup_new]
    For split/dup, the 'new' item gets zeros_like() state.
    """
    if moment is None:
        return None

    # Select along dim=0 using masks, preserve remaining and "originals"

    parts = [
        moment[mask_remaining],  # keep
        moment[mask_split],  # the split originals stay
        torch.zeros_like(moment[mask_split]),  # the split "new" gets zero state
        moment[mask_dup],  # the dup originals stay
        torch.zeros_like(moment[mask_dup]),  # the dup "new" gets zero state
    ]
    return torch.cat(parts, dim=0)


def _carry_over_state(
    optim, old_param, new_param, mask_remaining, mask_split, mask_dup
):
    """
    Move/resize Adam/RAdam state from old_param to new_param.
    """
    prev = optim.state.get(old_param, None)
    # Make sure there is a state dict for the new param
    state = optim.state.setdefault(new_param, {})
    if prev is None or len(prev) == 0:
        # No previous state (first step) – leave empty; PyTorch will init lazily
        return

    # Step is scalar per parameter tensor – carry it over
    if "step" in prev:
        state["step"] = torch.zeros_like(prev["step"])

    # Resize first-moment and second-moment buffers along dim=0
    for k in ("exp_avg", "exp_avg_sq"):
        buf = prev.get(k, None)
        resized = _resize_moment(buf, mask_remaining, mask_split, mask_dup)
        if resized is not None:
            state[k] = resized


def grow_and_prune(
    splats: GaussianSplats,
    grad_mag_2d: torch.Tensor,  # (N,)
    area_2d: torch.Tensor,  # (N,)
    grad_2d_magnitude_threshold: float,
    area_2d_threshold: float,
    opacity_threshold: float,
    optimizer: torch.optim.Optimizer,
):
    # decide which splats to split / duplicate / remove

    mask_split = area_2d > area_2d_threshold
    mask_remaining = ~mask_split

    splats_to_split = splats[mask_split]
    splats_to_split_dup = splats_to_split.clone()

    if mask_split.any():
        split_cov_3d = get_covariance_3d(
            splats_to_split.scales_ / 5, splats_to_split.orientations_
        )
        mv_normal = torch.distributions.MultivariateNormal(
            splats_to_split.means_, split_cov_3d
        )
        # modify originals
        splats_to_split.means = mv_normal.sample()
        splats_to_split.scales = splats_to_split.scales / 2.5
        # create their twins
        splats_to_split_dup.means = mv_normal.sample()
        splats_to_split_dup.scales = splats_to_split_dup.scales / 2.5

    mask_dup = mask_remaining & (grad_mag_2d > grad_2d_magnitude_threshold)
    mask_remaining &= ~mask_dup

    splats_to_dup = splats[mask_dup]
    splats_to_dup_dup = splats_to_dup.clone()

    mask_remove = mask_remaining & (splats.opacities_[:, 0] < opacity_threshold)
    mask_remove = mask_remove | (area_2d < 0.01)
    mask_remaining &= ~mask_remove

    # save old params (per field) to access optimizer.state

    old_params = {
        "means": splats.means_,
        "orientations": splats.orientations_,
        "scales": splats.scales_,
        "colors": splats.colors_,
        "opacities": splats.opacities_,
    }

    # build new tensors per field with the same concatenation pattern

    def cat_field(getter):
        return torch.cat(
            [
                getter(splats)[mask_remaining],
                getter(splats_to_split) if mask_split.any() else getter(splats)[:0],
                getter(splats_to_split_dup) if mask_split.any() else getter(splats)[:0],
                getter(splats_to_dup) if mask_dup.any() else getter(splats)[:0],
                getter(splats_to_dup_dup) if mask_dup.any() else getter(splats)[:0],
            ],
            dim=0,
        ).requires_grad_(True)

    new_splats = GaussianSplats(
        means_=cat_field(lambda s: s.means_),
        orientations_=cat_field(lambda s: s.orientations_),
        scales_=cat_field(lambda s: s.scales_),
        colors_=cat_field(lambda s: s.colors_),
        opacities_=cat_field(lambda s: s.opacities_),
    )

    # re-point optimizer param groups to new tensors (one group per field)

    field_order = ["means", "orientations", "scales", "colors", "opacities"]
    new_params = {
        "means": new_splats.means_,
        "orientations": new_splats.orientations_,
        "scales": new_splats.scales_,
        "colors": new_splats.colors_,
        "opacities": new_splats.opacities_,
    }

    if len(optimizer.param_groups) != 5:
        raise RuntimeError("Expected 5 param groups (one per field).")

    # Reassign the single tensor for each group

    for group, key in zip(optimizer.param_groups, field_order):
        group["params"] = [new_params[key]]

    # carry over/resize optimizer state per field tensor

    for key in field_order:
        _carry_over_state(
            optimizer,
            old_param=old_params[key],
            new_param=new_params[key],
            mask_remaining=mask_remaining,
            mask_split=mask_split,
            mask_dup=mask_dup,
        )

    print(
        "splitting",
        mask_split.sum().item(),
        "duplicating",
        mask_dup.sum().item(),
        "removing",
        mask_remove.sum().item(),
        "->",
        len(new_splats.means_),
    )

    return new_splats


def reset_splats(splats: GaussianSplats):
    splats.opacities_[:] = get_initial_opacities(splats.opacities.shape[0])
    splats.scales_[:] = get_initial_scales(splats.scales.shape[0])
