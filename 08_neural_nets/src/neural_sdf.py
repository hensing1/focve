from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SDFMLP(nn.Module):
    def __init__(self, n_neurons: int, activation_fn: nn.Module) -> None:
        super().__init__()

        self.net = None

        # TODO: Setup the neural network layer

        return

    def forward(
        self,
        points: torch.Tensor,  # batchsize x 3
    ) -> torch.Tensor:  # batchsize x 1
        values = torch.zeros(points.shape[0], 1)

        # TODO: Implement the forward function

        return values


def compute_network_gradient(
    network: nn.Module,
    points: torch.Tensor,  # batchsize x 3
) -> torch.Tensor:  # batchsize x 3
    gradients = torch.zeros_like(points, requires_grad=True)

    # TODO: Implement the compute_network_gradient function

    return gradients


class SurfaceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        values: torch.Tensor,  # batchsize x 1
    ) -> torch.Tensor:  # scalar
        result = torch.tensor(0.0, requires_grad=True, device=values.device)

        # TODO: Implement the forward function

        return result


class EikonalLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        gradients: torch.Tensor,  # batchsize x 3
    ) -> torch.Tensor:  # scalar
        result = torch.tensor(0.0, requires_grad=True, device=gradients.device)

        # TODO: Implement the forward function

        return result


class BoundaryLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        values: torch.Tensor,  # batchsize x 1
    ) -> torch.Tensor:  # scalar
        result = torch.tensor(0.0, requires_grad=True, device=values.device)

        # TODO: Implement the forward function

        return result

class Trainer(nn.Module):
    def __init__(
        self,
        n_neurons: int,
        activation_fn: nn.Module,
        learn_rate: float,
        surface_weight: float,
        eikonal_weight: float,
        boundary_weight: float,
    ) -> None:
        super().__init__()
        self.surface_loss_fn = SurfaceLoss()
        self.eikonal_loss_fn = EikonalLoss()
        self.boundary_loss_fn = BoundaryLoss()
        self.surface_lambda = surface_weight
        self.eikonal_lambda = eikonal_weight
        self.boundary_lambda = boundary_weight

        self.model = None
        self.optimizer = None

        # TODO: Setup the MLP and optimizer

        return

    def step(
        self,
        surface_points: torch.Tensor,  # batchsize x 3
        volume_points: torch.Tensor,  # batchsize x 3
        boundary_points: torch.Tensor,  # batchsize x 3
    ) -> tuple[
        torch.Tensor,  # scalar
        torch.Tensor,  # scalar
        torch.Tensor,  # scalar
        torch.Tensor,  # scalar
    ]:
        surface_loss = torch.tensor(
            0.0, requires_grad=True, device=surface_points.device
        )
        eikonal_loss = torch.tensor(
            0.0, requires_grad=True, device=volume_points.device
        )
        boundary_loss = torch.tensor(
            0.0, requires_grad=True, device=boundary_points.device
        )
        combined_loss = torch.tensor(
            0.0, requires_grad=True, device=surface_points.device
        )

        # TODO: Implement the step function

        return (
            surface_loss,
            eikonal_loss,
            boundary_loss,
            combined_loss,
        )

    def eval(
        self,
        points: torch.Tensor,  # batchsize x 3
    ) -> torch.Tensor:  # batchsize x 1
        values = torch.zeros(points.shape[0], 1, device=points.device)

        # TODO: Implement the eval function

        return values
