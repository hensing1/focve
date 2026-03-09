import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, L: int) -> None:
        super().__init__()
        self.L = L
        # NOTE: A registered buffer becomes an instance attribute.
        # You can get the frequencies tensor with self.frequencies.
        self.register_buffer("frequencies", 2.0 ** torch.arange(L) * torch.pi)

    @property
    def n_output_dims(self) -> int:
        dim = 0

        # TODO: Return the dimension of an encoded vector.

        return dim

    def _build_phases(
        self,
        x: torch.Tensor,  # N x 3
    ) -> torch.Tensor:  # N x 3 x L
        """
        Multiplies each three-dimensional input point with each of the frequencies.

        Args:
            x (torch.Tensor): batch of three-dimensional points with shape [...,3].

        Returns:
            torch.Tensor: The broadcasted product of self.frequencies and x with shape [...,3,self.L].
        """
        phases = torch.zeros((*x.shape, self.L), device=x.device)

        # TODO: Multiply the frequencies with the input positions.

        return phases

    def _apply_sincos(
        self,
        phases: torch.Tensor,  # N x 3 x L
    ) -> torch.Tensor:  # N x 3 x L x 2
        """
        Applies both sin and cos to each phase 2^k*π*x for all frequencie bands.

        Args:
            phase: (torch.Tensor): batch of angle-like quantities with shape [...,3,self.L].

        Returns:
            torch.Tensor: Stacked tensor of sin/cos values with shape [...,3,self.L,2].
        """
        result = torch.zeros((*phases.shape, 2), device=phases.device)

        # TODO: Apply sin/cos to the phase values {2^k*π*x}_{k=1...L}.

        return result

    def forward(
        self,
        x: torch.Tensor,  # N x 3
    ) -> torch.Tensor:  # N x self.n_output_dims
        """
        Evaluates the positional encoding of batched three-dimensional points with self.L frequencies.

        Args:
            x (torch.Tensor): batch of three-dimensional points with shape [...,3].

        Returns:
            torch.Tensor: The positional encodings of x with shape [...,self.n_output_dims].

        """
        encodings = torch.zeros((x.shape[0], self.n_output_dims), device=x.device)

        # TODO: Implement the encoding function $\gamma(x)$. Use helper functions _build_phases and _apply_sincos.

        return encodings


def create_rays(
    i: torch.Tensor,  # N
    j: torch.Tensor,  # N
    pose: torch.Tensor,  # N x 4 x 4 or 4 x 4
    H: int,
    W: int,
    focal: float,
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

    return origins, dirs


def stratified_sampling(
    origins: torch.Tensor,  # N x 3
    dirs: torch.Tensor,  # N x 3
    n_interval_bounds: int,
    near: float,
    far: float,
) -> tuple[
    torch.Tensor,  # N x n_intervals
    torch.Tensor,  # N x (n_intervals-1) x 3
    torch.Tensor,  # N x (n_intervals-1) x 3
]:
    t_vals = torch.zeros((origins.shape[0], n_interval_bounds), device=origins.device)
    points = torch.zeros(
        (origins.shape[0], n_interval_bounds - 1, 3), device=origins.device
    )
    dirs_per_point = torch.zeros(
        (origins.shape[0], n_interval_bounds - 1, 3), device=origins.device
    )

    # TODO: Implements stratified sampling of the intervals [near,far]. Returns interval boundaries, sample positions, and directions.

    return t_vals, points, dirs_per_point


def volumetric_rendering(
    rgb: torch.Tensor,  # N x (n_intervals-1) x 3
    density: torch.Tensor,  # N x (n_intervals-1)
    t_vals: torch.Tensor,  # N x n_intervals
) -> torch.Tensor:  # N x 3
    pixel_colors = torch.zeros((rgb.shape[0], 3), device=rgb.device)

    # TODO: Implement the NeRF's volumetric rendering equation.

    return pixel_colors


def forward(
    model: nn.Module,
    i: torch.Tensor,  # N
    j: torch.Tensor,  # N
    pose: torch.Tensor,  # N x 4 x 4 or 4 x 4
    H: int,
    W: int,
    focal: float,
    N_samples: int,
    near: float,
    far: float,
) -> torch.Tensor:  # N x 3
    colors_pred = torch.zeros((i.shape[0], 3), device=i.device)

    # TODO: Implement the forward pass of the NeRF model.

    return colors_pred
