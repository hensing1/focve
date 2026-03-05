from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SDFMLP(nn.Module):
    def __init__(self, n_neurons: int, activation_fn: nn.Module) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, n_neurons),
            activation_fn,
            nn.Linear(n_neurons, n_neurons),
            activation_fn,
            nn.Linear(n_neurons, n_neurons),
            activation_fn,
            nn.Linear(n_neurons, 1)
        )

    def forward(
        self,
        points: torch.Tensor,  # batchsize x 3
    ) -> torch.Tensor:  # batchsize x 1
        return self.net(points)


def compute_network_gradient(
    network: nn.Module,
    points: torch.Tensor,  # batchsize x 3
) -> torch.Tensor:  # batchsize x 3
    points.requires_grad_(True)

    output = network.forward(points)
    gradients = torch.autograd.grad(outputs=output,
                                    inputs=points,
                                    grad_outputs=torch.ones_like(output),
                                    create_graph=True)[0]
    # output.backward(torch.ones_like(output))

    return gradients


class SurfaceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        values: torch.Tensor,  # batchsize x 1
    ) -> torch.Tensor:  # scalar

        result = torch.sum(torch.abs(values)) / torch.numel(values)
        result.requires_grad_(True)

        return result


class EikonalLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        gradients: torch.Tensor,  # batchsize x 3
    ) -> torch.Tensor:  # scalar

        result = torch.sum((torch.linalg.vector_norm(gradients, dim=-1) - 1) ** 2) / gradients.shape[0]
        result.requires_grad_(True)

        return result


class BoundaryLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        values: torch.Tensor,  # batchsize x 1
    ) -> torch.Tensor:  # scalar
        result = torch.sum(torch.max(torch.zeros_like(values), -values)) / torch.numel(values)
        result.requires_grad_(True)

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

        self.model = SDFMLP(n_neurons, activation_fn)
        self.optimizer = torch.optim.Adam(self.model.parameters(), learn_rate)


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

        self.optimizer.zero_grad()
        surface_loss = self.surface_loss_fn.forward(self.model.forward(surface_points))
        eikonal_loss = self.eikonal_loss_fn.forward(compute_network_gradient(self.model, volume_points))
        boundary_loss = self.boundary_loss_fn.forward(self.model.forward(boundary_points))
        combined_loss = self.surface_lambda * surface_loss + \
                        self.eikonal_lambda * eikonal_loss + \
                        self.boundary_lambda * boundary_loss
        combined_loss.backward()
        self.optimizer.step()

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
        # values = torch.zeros(points.shape[0], 1, device=points.device)

        # TODO: Implement the eval function
        # self.model.training = False
        self.model.eval()
        with torch.no_grad():
            values = self.model.forward(points)
        self.model.train()

        return values
