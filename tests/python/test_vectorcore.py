import numpy as np
import pytest
import vectorcore as vc


def assert_close(actual, expected, rtol=1e-5, atol=1e-6):
    np.testing.assert_allclose(actual.numpy(), np.asarray(expected, dtype=np.float32),
                               rtol=rtol, atol=atol)


def test_creation_metadata_and_indexing():
    value = vc.Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    assert value.shape == [2, 3]
    assert value.strides == [3, 1]
    assert value.ndim == 2
    assert value.numel == 6
    assert value.at([1, 1]) == 5
    assert "requires_grad=true" in repr(value)


def test_numpy_elementwise_and_broadcast_reference():
    rng = np.random.default_rng(12)
    a = rng.normal(size=(2, 3, 4)).astype(np.float32)
    b = rng.normal(size=(1, 4)).astype(np.float32)
    va, vb = vc.Tensor(a), vc.Tensor(b)
    assert_close(va + vb, a + b)
    assert_close(va - vb, a - b)
    assert_close(va * vb, a * b)
    assert_close(va / (vb + 3.0), a / (b + 3.0))
    with pytest.raises(Exception):
        _ = va + vc.Tensor(np.zeros((2, 2), dtype=np.float32))


def test_matmul_reductions_activations_reference():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(7, 5)).astype(np.float32)
    b = rng.normal(size=(5, 3)).astype(np.float32)
    assert_close(vc.Tensor(a) @ vc.Tensor(b), a @ b, rtol=2e-5)
    assert_close(vc.Tensor(a).T, a.T)
    assert np.isclose(vc.Tensor(a).sum().item(), a.sum(), rtol=1e-5)
    assert np.isclose(vc.Tensor(a).mean().item(), a.mean(), rtol=1e-5)
    assert_close(vc.Tensor(a).relu(), np.maximum(a, 0))
    assert_close(vc.Tensor(a).sigmoid(), 1 / (1 + np.exp(-a)))


def test_autograd_broadcast_and_repeated_path():
    x_np = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -1.0]], dtype=np.float32)
    b_np = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    x = vc.Tensor(x_np, requires_grad=True)
    b = vc.Tensor(b_np, requires_grad=True)
    loss = ((x + b) * (x + b)).mean()
    loss.backward()
    expected_x = 2 * (x_np + b_np) / x_np.size
    assert_close(x.grad, expected_x)
    assert_close(b.grad, expected_x.sum(axis=0))


def test_gradient_finite_difference():
    values = np.array([-0.7, 0.1, 1.3], dtype=np.float32)
    x = vc.Tensor(values, requires_grad=True)
    (x.sigmoid() * x).sum().backward()
    analytical = x.grad.numpy()
    epsilon = 1e-3
    numerical = np.empty_like(values)
    for index in range(values.size):
        plus, minus = values.copy(), values.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        f_plus = (vc.Tensor(plus).sigmoid() * vc.Tensor(plus)).sum().item()
        f_minus = (vc.Tensor(minus).sigmoid() * vc.Tensor(minus)).sum().item()
        numerical[index] = (f_plus - f_minus) / (2 * epsilon)
    np.testing.assert_allclose(analytical, numerical, rtol=2e-3, atol=2e-3)


def test_pytorch_gradient_reference():
    torch = pytest.importorskip("torch")
    x_np = np.array([[0.2, -0.3, 0.7], [1.1, 0.4, -0.8]], dtype=np.float32)
    w_np = np.array([[0.5, -0.2], [0.1, 0.9], [-0.4, 0.3]], dtype=np.float32)
    b_np = np.array([0.2, -0.1], dtype=np.float32)
    x, w, b = (vc.Tensor(x_np, requires_grad=True),
               vc.Tensor(w_np, requires_grad=True),
               vc.Tensor(b_np, requires_grad=True))
    ((x @ w) + b).sigmoid().mean().backward()

    tx = torch.tensor(x_np, requires_grad=True)
    tw = torch.tensor(w_np, requires_grad=True)
    tb = torch.tensor(b_np, requires_grad=True)
    torch.sigmoid((tx @ tw) + tb).mean().backward()
    np.testing.assert_allclose(x.grad.numpy(), tx.grad.numpy(), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(w.grad.numpy(), tw.grad.numpy(), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(b.grad.numpy(), tb.grad.numpy(), rtol=2e-5, atol=2e-6)


def test_two_layer_network_trains():
    x = vc.Tensor([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
    y = vc.Tensor([[0.0], [0.0], [0.0], [1.0], [1.0]])
    first, second = vc.Linear(1, 4, 1), vc.Linear(4, 1, 2)
    params = [first.weight, first.bias, second.weight, second.bias]
    initial = vc.mse_loss(second(first(x).relu()).sigmoid(), y).item()
    for _ in range(400):
        prediction = second(first(x).relu()).sigmoid()
        loss = vc.mse_loss(prediction, y)
        loss.backward()
        vc.sgd_step(params, 0.25)
        vc.zero_grad(params)
    final = vc.mse_loss(second(first(x).relu()).sigmoid(), y).item()
    assert final < initial * 0.35


@pytest.mark.skipif(not vc.metal_available(), reason="Metal backend unavailable")
def test_metal_cpu_parity():
    rng = np.random.default_rng(8)
    a = rng.normal(size=(19, 13)).astype(np.float32)
    b = rng.normal(size=(13, 11)).astype(np.float32)
    cpu_a, cpu_b = vc.Tensor(a), vc.Tensor(b)
    gpu_a, gpu_b = cpu_a.to("metal"), cpu_b.to("metal")
    assert_close((gpu_a @ gpu_b).to("cpu"), a @ b, rtol=2e-4, atol=2e-4)
    bias = vc.Tensor(np.arange(11, dtype=np.float32)).to("metal")
    assert_close(((gpu_a @ gpu_b) + bias).relu().to("cpu"),
                 np.maximum(a @ b + np.arange(11), 0), rtol=2e-4, atol=2e-4)
    assert np.isclose(gpu_a.sum().item(), a.sum(), rtol=2e-4, atol=2e-4)
    x = vc.Tensor([[-1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]],
                  requires_grad=True, device="metal")
    bias = vc.Tensor([0.5, 1.0, -2.0], requires_grad=True, device="metal")
    (x + bias).relu().sum().backward()
    assert_close(x.grad.to("cpu"), [[0, 1, 1], [0, 1, 1]])
    assert_close(bias.grad.to("cpu"), [0, 2, 2])
