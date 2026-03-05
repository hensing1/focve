import pathlib
import sys
from unittest import skip
from unittest.mock import MagicMock

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from demo import SurfaceSampleDataset
from src.neural_sdf import (
    SDFMLP,
    BoundaryLoss,
    EikonalLoss,
    SurfaceLoss,
    Trainer,
    compute_network_gradient,
)
from torch.utils.data import DataLoader, TensorDataset


# Custom network that squares its input elementwise.
class Square(nn.Module):
    def forward(self, x):
        return x**2


# Custom network that always returns a constant.
class TinyNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            self.layer.weight.set_(torch.ones_like(self.layer.weight))

    def forward(self, x):
        # Return a constant scalar (or tensor) regardless of the input.
        # This makes the output independent of x, so the gradient should be zero.
        return F.relu(self.layer(x))


# Custom network that cubes its input elementwise.
class Cube(nn.Module):
    def forward(self, x):
        return x**3


class MockLoss(nn.Module):
    def __init__(self, return_value):
        super().__init__()
        self._val = return_value

    def forward(self, x):
        return self._val.requires_grad_(True)


def test_SDFMLP_architecture():
    """Ensure that the network has the expected architecture."""
    torch.manual_seed(0)

    n_neurons = 64
    model = SDFMLP(n_neurons=n_neurons, activation_fn=nn.ELU())

    # Test 1: Trivial check that inputs are mapped to the correct output shape
    dummy_input = torch.randn(128, 3)
    dummy_output = model(dummy_input)
    assert dummy_output.shape == (
        128,
        1,
    ), f"Expected network to output to have shape (128,1) but got {dummy_output.shape}"

    # Test 2: Check that the neural network has the correct number of Linear layers
    # Iterate over all modules in the model (including nested ones)
    expected_linear_count = 4  # Based on the example network
    linear_count = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            linear_count += 1
    assert (
        linear_count == expected_linear_count
    ), f"Expected {expected_linear_count} Linear layers, found {linear_count}"

    # Test 3: Verify the shapes of the weight/bias of each Linear layer
    # Iterate over all modules in the model (including nested ones)
    # Collect all Linear layers in execution order
    linear_layers = [
        module for module in model.modules() if isinstance(module, nn.Linear)
    ]
    # Expected parameter shapes based on network architecture
    for i, layer in enumerate(linear_layers):
        if i == 0:
            weight_shape = (n_neurons, 3)
            bias_shape = (n_neurons,)
        elif i == len(linear_layers) - 1:
            weight_shape = (1, n_neurons)
            bias_shape = (1,)
        else:
            weight_shape = (n_neurons, n_neurons)
            bias_shape = (n_neurons,)

        # Test weight dimensions
        assert layer.weight.shape == weight_shape, (
            f"Layer {i} weight shape mismatch: "
            f"expected {weight_shape}, got {layer.weight.shape}"
        )
        # Test bias existence and dimensions
        if layer.bias is None:
            raise AssertionError(f"Layer {i} is missing bias term")
        assert layer.bias.shape == bias_shape, (
            f"Layer {i} bias shape mismatch: "
            f"expected {bias_shape}, got {layer.bias.shape}"
        )

    # Test 4: Check that the non-linear function is applied inbetween each of the linear layers
    # Register forward hooks on Linear and ELU modules
    execution_order, hooks = [], []  # Tracks types of modules as they execute
    for module in model.modules():
        if module is model:  # Skip the root module
            continue
        if isinstance(module, (nn.Linear, nn.ELU)):
            # Capture the module type at hook creation time
            module_type = type(module)
            hook = module.register_forward_hook(
                lambda m, inp, out, mt=module_type: execution_order.append(mt)
            )
            hooks.append(hook)
    # Perform a forward pass to trigger hooks
    dummy_input = torch.randn(1, 3)
    model(dummy_input)
    # Cleanup hooks
    for hook in hooks:
        hook.remove()
    # Check that the network executed the layers in the correct order
    expected_execution_order = [nn.Linear, nn.ELU] * 3 + [nn.Linear]
    assert len(execution_order) == len(expected_execution_order), (
        f"Expected {len(expected_execution_order)} layers to be applied, "
        f"found {len(execution_order)}"
    )
    for i, (layer_type, expected_type) in enumerate(
        zip(execution_order, expected_execution_order)
    ):
        assert (
            layer_type == expected_type
        ), f"Expected layer at depth {i} to be of type {expected_type} but got {layer_type} instead."


def test_compute_network_gradient():
    """Tests compute the gradient d/dx f(x) for a callable function f."""
    # Test 1: Simple Linear Network
    linear = nn.Linear(3, 2, bias=False)
    # Set the weights manually. Note: nn.Linear.weight shape is (out_features, in_features)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0, 2.0, 1.0], [3.0, 4.0, 2.0]]))
    x1 = torch.tensor([[1.0, 2.0, 3.0]])
    grad1 = compute_network_gradient(linear, x1)
    # The output is y = [1*1+2*2+1*3, 3*1+4*2+2*3] = [8, 17].
    # The sum of outputs is sum(y) = (1+3)*x1 + (2+4)*x2 + (1+2)*x3 = 4*x1 + 6*x2 + 3*x3.
    expected_grad1 = torch.tensor([[4.0, 6.0, 3.0]])
    assert torch.allclose(
        grad1, expected_grad1
    ), f"Test 1 failed: expected {expected_grad1}, got {grad1}"
    print("Test 1 passed: Simple Linear Network")

    # Test 2: Elementwise Nonlinearity (Square Function)
    square_net = Square()
    x2 = torch.tensor([[2.0, -3.0, 4.0]])
    grad2 = compute_network_gradient(square_net, x2)
    # For f(x) = x^2, df/dx = 2x.
    expected_grad2 = 2 * x2  # [4.0, -6.0, 8.0]
    assert torch.allclose(
        grad2, expected_grad2
    ), f"Test 2 failed: expected {expected_grad2}, got {grad2}"
    print("Test 2 passed: Elementwise Square Function")

    # Test 3: ReLU Activation Network
    relu = nn.ReLU()
    x3 = torch.tensor([[-1.0, 0.0, 1.0]])
    grad3 = compute_network_gradient(relu, x3)
    # For ReLU, the derivative is 0 for inputs <= 0 and 1 for inputs > 0.
    # Here, we expect [0, 0, 1] (PyTorch sets the derivative at 0 to 0).
    expected_grad3 = torch.tensor([[0.0, 0.0, 1.0]])
    assert torch.allclose(
        grad3, expected_grad3
    ), f"Test 3 failed: expected {expected_grad3}, got {grad3}"
    print("Test 3 passed: ReLU Activation Network")

    # Test 4: Batched Input using Square Function
    x4 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    grad4 = compute_network_gradient(square_net, x4)
    expected_grad4 = 2 * x4  # Elementwise gradient for f(x)=x^2.
    assert (
        grad4.shape == x4.shape
    ), f"Test 4 failed: expected shape {x4.shape}, got {grad4.shape}"
    assert torch.allclose(
        grad4, expected_grad4
    ), f"Test 4 failed: expected {expected_grad4}, got {grad4}"
    print("Test 4 passed: Batched Input")

    # Test 5: Graph Retention Test
    # Verify that the gradient returned is connected to the computational graph,
    # so that we can compute second derivatives without error.
    net = TinyNet()
    x5 = torch.tensor([[3.0, 2.0, 1.0]])
    grad5 = compute_network_gradient(net, x5)
    # Since f(x)=x^2, first derivative is 2x and second derivative is 2.
    grad_sum = torch.sum(grad5**2)
    grad_sum.backward()
    expected_second_deriv = torch.tensor([[2.0, 2.0, 2.0]])
    assert torch.allclose(
        net.layer.weight.grad, expected_second_deriv
    ), f"Test 5 failed: expected {expected_second_deriv}, got {net.layer.weight.grad}"
    print("Test 5 passed: Graph Retention via retain_graph=True")

    # Test 6: Higher-Order Gradient Check using Cube Network
    cube_net = Cube()
    # For higher-order tests, ensure x requires grad.
    x6 = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    grad6 = compute_network_gradient(cube_net, x6)
    # For f(x) = x^3, the first derivative is 3*x^2.
    expected_grad6 = 3 * x6**2
    assert torch.allclose(
        grad6, expected_grad6
    ), f"Test 6 (first derivative) failed: expected {expected_grad6}, got {grad6}"
    # Now test that the gradient itself is differentiable: compute the second derivative.
    # Differentiating grad6.sum() with respect to x6 should yield 6*x.
    second_grad6 = torch.autograd.grad(grad6.sum(), x6, create_graph=True)[0]
    expected_second_grad6 = 6 * x6
    assert torch.allclose(
        second_grad6, expected_second_grad6
    ), f"Test 6 (second derivative) failed: expected {expected_second_grad6}, got {second_grad6}"
    print("Test 6 passed: Higher-Order Gradient Check")


def test_SurfaceLoss():
    loss_fn = SurfaceLoss()

    # 1. Zero Input Test:
    # Input: tensor of zeros. Expected loss: 0.
    zeros = torch.zeros(2, 3)
    loss = loss_fn(zeros)
    assert torch.isclose(
        loss, torch.tensor(0.0)
    ), f"Zero input test failed: expected 0, got {loss.item()}"

    # 2. Positive Values Test:
    # Input: tensor with positive values [1, 2, 3]. Expected loss: mean([1^2, 2^2, 3^2]).
    pos = torch.tensor([[1.0, 2.0, 3.0]])
    loss = loss_fn(pos)
    expected = torch.tensor((1 + 2 + 3) / 3.0)
    assert torch.isclose(
        loss, expected
    ), f"Positive values test failed: expected {expected.item()}, got {loss.item()}"

    # 3. Negative Values Test:
    # Input: tensor with negative values [-1, -2, -3]. Expected loss: same as positive values.
    neg = torch.tensor([[-1.0, -2.0, -3.0]])
    loss = loss_fn(neg)
    expected = torch.tensor((1 + 2 + 3) / 3.0)
    assert torch.isclose(
        loss, expected
    ), f"Negative values test failed: expected {expected.item()}, got {loss.item()}"

    # 4. Mixed Values Test:
    # Input: tensor with both negative and positive values [-1, 0, 1]. Expected loss: mean([1, 0, 1]) = 2/3.
    mixed = torch.tensor([[-1.0, 0.0, 1.0]])
    loss = loss_fn(mixed)
    expected = torch.tensor((1 + 0 + 1) / 3.0)
    assert torch.isclose(
        loss, expected
    ), f"Mixed values test failed: expected {expected.item()}, got {loss.item()}"

    # 5. Same Element Test:
    # Input: tensor with three times the same value [5]. Expected loss: 5.
    single = torch.tensor([[5.0, 5.0, 5.0]])
    loss = loss_fn(single)
    expected = torch.tensor(5.0)
    assert torch.isclose(
        loss, expected
    ), f"Same element test failed: expected {expected.item()}, got {loss.item()}"

    # 6. Larger Batched Tensor Test:
    batched = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [-1.0, -2.0, -3.0],
            [-4.0, -5.0, -6.0],
        ]
    )
    loss = loss_fn(batched)
    expected = torch.tensor((1 + 2 + 3 + 4 + 5 + 6) / 6.0)
    assert torch.isclose(
        loss, expected
    ), f"Larger Batched test failed: expected {expected.item()}, got {loss.item()}"

    # 7. Gradient Propagation Test:
    # Input: tensor with requires_grad=True to verify that gradients are computed.
    grad_input = torch.tensor([[1.0, -2.0, 3.0]], requires_grad=True)
    loss = loss_fn(grad_input)
    loss.backward()
    assert (
        grad_input.grad is not None
    ), "Gradient propagation test failed: no gradients computed."

    # 8. Data Type and Precision Test:
    # Compare outputs using float32 and float64.
    tensor_float32 = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    tensor_float64 = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    loss32 = loss_fn(tensor_float32)
    loss64 = loss_fn(tensor_float64)
    expected_float32 = torch.tensor((1 + 2 + 3) / 3.0, dtype=torch.float32)
    expected_float64 = torch.tensor((1 + 2 + 3) / 3.0, dtype=torch.float64)
    assert (
        loss32.dtype == expected_float32.dtype
    ), f"Float32 test failed: expected dtype float32, got {loss32.dtype}"
    assert torch.isclose(
        loss32, expected_float32
    ), f"Float32 test failed: expected {expected_float32.item()}, got {loss32.item()}"
    assert (
        loss64.dtype == expected_float64.dtype
    ), f"Float64 test failed: expected dtype float32, got {loss32.dtype}"
    assert torch.isclose(
        loss64, expected_float64
    ), f"Float64 test failed: expected {expected_float64.item()}, got {loss64.item()}"


def test_EikonalLoss():
    """Eikonal loss returns avg((||g||-1)^2) for all inputs shapes."""
    loss_fn = EikonalLoss()
    torch.manual_seed(0)

    # 1. Unit Vector Input: each row is a unit vector.
    unit_vectors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    loss = loss_fn(unit_vectors)
    assert torch.isclose(
        loss, torch.tensor(0.0)
    ), f"Unit vectors loss expected to be 0, got {loss}"

    # 2. Zero Tensor Input: each vector has norm 0, so (0-1)^2 = 1.
    zeros = torch.zeros(2, 3)
    loss = loss_fn(zeros)
    assert torch.isclose(
        loss, torch.tensor(1.0)
    ), f"Zeros loss expected to be 1, got {loss}"

    # 3. Single Vector Case: one single vector (reshaped to 2D for consistent norm computation).
    single_vector = torch.tensor([[0.6, 0.8, 0.0]])  # norm is 1.
    loss = loss_fn(single_vector.unsqueeze(0))
    assert torch.isclose(
        loss, torch.tensor(0.0)
    ), f"Single vector unit norm loss expected to be 0, got {loss}"

    # 4. Multiple Vectors with Known Norms:
    # First vector: [1, 0] (norm 1 → loss 0), second vector: [0, 0] (norm 0 → loss 1). Mean loss = 0.5.
    known_vectors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    loss = loss_fn(known_vectors)
    assert torch.isclose(
        loss, torch.tensor(0.5)
    ), f"Known vectors loss expected to be 0.5, got {loss}"

    # 5. Vectors with Scaled Values: if scaled by 2, norm becomes 2 so loss is (2-1)^2 = 1.
    scaled_vectors = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    loss = loss_fn(scaled_vectors)
    assert torch.isclose(
        loss, torch.tensor(1.0)
    ), f"Scaled vectors loss expected to be 1, got {loss}"

    # 6. Larger Batch Input: tensor of shape (batch, N, d). Ensure that the output is a scalar.
    high_dim_tensor = F.normalize(torch.randn(256, 3), dim=-1)
    loss = loss_fn(high_dim_tensor)
    assert torch.isclose(
        loss, torch.tensor(0.0)
    ), f"Larger batch input loss expected to be 0, got {loss}"
    assert (
        loss.ndim == 0
    ), f"Expected scalar loss for larger batch input, got shape {loss.shape}"

    # 7. Gradient Computation Check: verify that gradients are computed.
    input_tensor = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True)
    loss = loss_fn(input_tensor)
    loss.backward()
    assert input_tensor.grad is not None, "Expected gradients to be computed."

    # 8. Data Type and Shape Robustness: test different dtypes.
    tensor_float32 = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )
    tensor_float64 = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    loss1 = loss_fn(tensor_float32)
    loss2 = loss_fn(tensor_float64)
    assert (
        loss1.dtype == torch.float32
    ), f"Float32 test failed: expected dtype float32, got {loss1.dtype}"
    assert torch.isclose(
        loss1, torch.tensor(0.0, dtype=torch.float32)
    ), "Expected loss 0 for float32 unit vectors."
    assert (
        loss2.dtype == torch.float64
    ), f"Float32 test failed: expected dtype float64, got {loss2.dtype}"
    assert torch.isclose(
        loss2, torch.tensor(0.0, dtype=torch.float64)
    ), "Expected loss 0 for float64 unit vectors."


def test_BoundaryLoss():
    loss_fn = BoundaryLoss()

    # 1. All Positive Values Test
    # For positive values, -v is negative so relu returns 0.
    positive_tensor = torch.tensor([[1.0, 2.0, 3.0]])
    loss = loss_fn(positive_tensor)
    expected = torch.tensor(0.0)
    assert torch.isclose(
        loss, expected
    ), f"All positive test failed: expected {expected.item()}, got {loss.item()}"

    # 2. All Negative Values Test
    # For each negative value v, relu(-v) = -v.
    # Example: [-1, -2, -3] → squares: [1, 4, 9], mean = (1+4+9)/3 = 14/3.
    negative_tensor = torch.tensor([[-1.0, -2.0, -3.0]])
    loss = loss_fn(negative_tensor)
    expected = torch.tensor((1 + 2 + 3) / 3)
    assert torch.isclose(
        loss, expected, atol=1e-5
    ), f"All negative test failed: expected {expected.item()}, got {loss.item()}"

    # 3. Mixed Values Test
    # Example: [-1, 0, 2] → for -1: relu(1)=1, for 0: relu(0)=0, for 2: relu(-2)=0, mean = (1+0+0)/3.
    mixed_tensor = torch.tensor([[-1.0, 0.0, 2.0]])
    loss = loss_fn(mixed_tensor)
    expected = torch.tensor(1 / 3)
    assert torch.isclose(
        loss, expected, atol=1e-5
    ), f"Mixed values test failed: expected {expected.item()}, got {loss.item()}"

    # 4. Zero Tensor Test
    # A tensor of zeros should produce 0 loss.
    zeros = torch.zeros(2, 3)
    loss = loss_fn(zeros)
    expected = torch.tensor(0.0)
    assert torch.isclose(
        loss, expected
    ), f"Zero tensor test failed: expected {expected.item()}, got {loss.item()}"

    # 5. Same Element Test
    # Negative element three times: [-5] -> relu(5)=5, loss=5.
    single_negative = torch.tensor([[-5.0, -5.0, -5.0]])
    loss = loss_fn(single_negative)
    expected = torch.tensor(5.0)
    assert torch.isclose(
        loss, expected
    ), f"Same element negative test failed: expected {expected.item()}, got {loss.item()}"

    # Positive element three times: [5] -> relu(-5)=0.
    single_positive = torch.tensor([[5.0, 5.0, 5.0]])
    loss = loss_fn(single_positive)
    expected = torch.tensor(0.0)
    assert torch.isclose(
        loss, expected
    ), f"Same element positive test failed: expected {expected.item()}, got {loss.item()}"

    # 6. Multi-Dimensional Tensor Test
    # For a 2D tensor:
    #   [[-2.0,  2.0, 0.0],
    #    [-4.0,  0.0, 0.0]]
    # Computation:
    #   - For -2.0: relu(1)=2.
    #   - For  2.0: relu(-2)=0.
    #   - For -4.0: relu(3)=4.
    #   - For  0.0: relu(0)=0.
    # Mean = (2 + 0 + 2 + 0)/6 = 1.0.
    multi_tensor = torch.tensor([[-2.0, 2.0, 0.0], [-4.0, 0.0, 0.0]])
    loss = loss_fn(multi_tensor)
    expected = torch.tensor(1.0)
    assert torch.isclose(
        loss, expected, atol=1e-5
    ), f"Multi-dimensional tensor test failed: expected {expected.item()}, got {loss.item()}"

    # 7. Gradient Propagation Test
    # Verify that gradients are computed when the input requires grad.
    grad_input = torch.tensor([[-1.0, 2.0, -3.0]], requires_grad=True)
    loss = loss_fn(grad_input)
    loss.backward()
    assert (
        grad_input.grad is not None
    ), "Gradient propagation test failed: no gradients computed."

    # 8. Data Type and Precision Test
    # Verify that the loss is computed correctly with different dtypes.
    tensor_float32 = torch.tensor([[-1.0, 2.0, -3.0]], dtype=torch.float32)
    tensor_float64 = torch.tensor([[-1.0, 2.0, -3.0]], dtype=torch.float64)
    loss32 = loss_fn(tensor_float32)
    loss64 = loss_fn(tensor_float64)
    expected_float32 = torch.tensor((1 + 0 + 3) / 3, dtype=torch.float32)
    expected_float64 = torch.tensor((1 + 0 + 3) / 3, dtype=torch.float64)
    assert (
        loss32.dtype == expected_float32.dtype
    ), f"Float32 test failed: expected dtype float32, got {loss32.dtype}"
    assert torch.isclose(
        loss32, expected_float32, atol=1e-5
    ), f"Float32 test failed: expected {expected_float32.item()}, got {loss32.item()}"
    assert (
        loss64.dtype == expected_float64.dtype
    ), f"Float64 test failed: expected dtype float64, got {loss64.dtype}"
    assert torch.isclose(
        loss64, expected_float64, atol=1e-5
    ), f"Float64 test failed: expected {expected_float64.item()}, got {loss64.item()}"


def test_Trainer():
    torch.manual_seed(0)

    trainer = Trainer(128, nn.ELU(), 0.032, 0.33, 0.66, 0.99)

    # Test 1: Verify that the constructur is correct.
    # Verify model architecture
    assert isinstance(trainer.model, SDFMLP)
    # Verify optimizer configuration
    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert trainer.optimizer.param_groups[0]["lr"] == 0.032
    # Verify losses and weights
    assert isinstance(trainer.surface_loss_fn, SurfaceLoss)
    assert isinstance(trainer.eikonal_loss_fn, EikonalLoss)
    assert isinstance(trainer.boundary_loss_fn, BoundaryLoss)
    assert trainer.surface_lambda == 0.33
    assert trainer.eikonal_lambda == 0.66
    assert trainer.boundary_lambda == 0.99

    # Test 2: Trainer.step returns have expected type and shape
    losses = trainer.step(torch.randn(10, 3), torch.randn(5, 3), torch.randn(20, 3))
    assert (
        len(losses) == 4
    ), f"Expects Trainer.step to have four return values, but got {len(losses)}."
    assert all(
        [torch.is_tensor(t) for t in losses]
    ), "Expects all Trainer.step return values to have type torch.Tensor"
    assert all(
        [t.shape == () for t in losses]
    ), "Expects all Trainer.step return values to be scalar tensors."

    # Test 2: Optimizer gradients are reset at the start of step()
    mock_optimizer = MagicMock()
    original_optimizer = trainer.optimizer
    trainer.optimizer = mock_optimizer
    # Execute step and check zero_grad call
    trainer.step(torch.randn(10, 3), torch.randn(5, 3), torch.randn(20, 3))
    mock_optimizer.zero_grad.assert_called_once()
    trainer.optimizer = original_optimizer  # Restore optimizer after test

    # Test 3: Losses are computed from the correct inputs and scaled properly
    # Mock losses to return known values
    trainer.surface_loss_fn = MockLoss(return_value=torch.tensor(2.0))
    trainer.boundary_loss_fn = MockLoss(return_value=torch.tensor(4.0))
    trainer.eikonal_loss_fn = MockLoss(return_value=torch.tensor(3.0))
    # Mock gradient computation
    surface_loss, eikonal_loss, boundary_loss, combined_loss = trainer.step(
        torch.randn(10, 3), torch.randn(5, 3), torch.randn(20, 3)
    )
    # Verify loss values and scaling
    expected_total = (0.33 * 2.0) + (0.66 * 3.0) + (0.99 * 4.0)
    assert torch.allclose(
        surface_loss, torch.tensor(2.0)
    ), "Expects Trainer.step to return the unweighted surface loss as its first return value."
    assert torch.allclose(
        eikonal_loss, torch.tensor(3.0)
    ), "Expects Trainer.step to return the unweighted Eikonal loss as its second return value."
    assert torch.allclose(
        boundary_loss, torch.tensor(4.0)
    ), "Expects Trainer.step to return the unweighted boundary loss as its third return value."
    assert (
        abs(expected_total - combined_loss.item()) < 1e-6
    ), "Expects Trainer.step to return the weighted combined loss as its fourth return value."  # 2.0 + 0.3 + 4.0 = 6.3

    # Test 4: Gradients propagate, and parameters update after optimizer.step().
    trainer = Trainer(128, nn.ELU(), 0.01, 1.0, 1.0, 1.0)
    initial_params = [p.data.detach().clone() for p in trainer.model.parameters()]
    trainer.step(torch.randn(10, 3), torch.randn(5, 3), torch.randn(20, 3))
    # Verify parameters changed
    for p_initial, p_updated in zip(initial_params, trainer.model.parameters()):
        assert not torch.allclose(
            p_initial, p_updated
        ), "Calling Trainer.step is expected to update the network parameter."

    # Test 5: eval() temporarily switches the model to eval mode and back to train
    assert trainer.model.training
    with torch.no_grad():
        trainer.eval(torch.randn(5, 3))
    assert (
        trainer.model.training
    ), "Trainer.eval is expected to return the model into training mode."  # Should revert to training mode
    # Verify eval mode during call
    torch.manual_seed(0)
    # Replace model with mock model which behaves different in training and evaluation mode
    mock_model = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(3, 1))
    original_model = trainer.model
    trainer.model = mock_model
    # Compute expected outputs of the mock model
    test_inputs = torch.rand(7, 3)
    mock_model.eval()
    test_eval_ouputs = mock_model(test_inputs)
    mock_model.train()
    # Get Trainer.eval output with mock model
    outputs = trainer.eval(test_inputs)
    assert torch.allclose(
        outputs, test_eval_ouputs
    ), "Trainer.eval is expected to disable training mode during the evaluation."
    trainer.model = original_model  # Replace mock model with the original one.


def test_training_correct():
    """Trains network to model the SDF of a sphere and validates its outputs."""
    torch.manual_seed(0)

    # Define the sphere SDF
    def sphere_sdf(X):
        sdf_values = torch.sqrt(torch.sum(X**2, dim=1)) - 0.5  # Sphere SDF (radius 0.5)
        return sdf_values.unsqueeze(1)  # Add output dimension

    # Create dataset and dataloader
    num_samples = 9996 * 2
    surface_points = 0.5 * F.normalize(torch.randn(num_samples, 3), dim=-1)
    volume_points = torch.rand(num_samples, 3) * 2 - 1  # Random points in [-1, 1]^3
    boundary_points = []
    for dim in range(3):
        plane_points = 2 * torch.rand(num_samples // 6, 3) - 1
        plane_points[:, dim] = 1.0
        boundary_points.append(plane_points.clone())
        plane_points = 2 * torch.rand(num_samples // 6, 3) - 1
        plane_points[:, dim] = -1.0
        boundary_points.append(plane_points.clone())
    boundary_points = torch.cat(boundary_points, dim=0)
    dataset = TensorDataset(surface_points, volume_points, boundary_points)
    dataloader = DataLoader(dataset, batch_size=9996, shuffle=True)

    # Step
    trainer = Trainer(128, nn.ELU(), 0.01, 1.0, 1.0, 1.0)
    for _ in range(500):
        for surface, volume, boundary in dataloader:
            trainer.step(surface, volume, boundary)

    num_samples = 10000
    volume_points = torch.rand(num_samples, 3) * 2 - 1
    # Check that SDF are somewhat similar to the expected SDF values.
    expected_sdf_vals = sphere_sdf(volume_points)
    sdf_vals = trainer.eval(volume_points).detach().cpu().numpy()
    np.testing.assert_allclose(sdf_vals, expected_sdf_vals, atol=0.15)
    # Check that the norm is close to one everywhere.
    grads = compute_network_gradient(trainer.model, volume_points)
    assert (
        torch.mean((grads.norm(dim=-1) - 1) ** 2) < 1e-2
    ), f"Expectes trained model gradient to be close to 1 everywhere, but got an Eikonal loss of {torch.median((grads.norm(dim=-1) - 1) ** 2)} instead."
