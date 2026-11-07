"""Autograd correctness: finite-difference gradcheck for every differentiable
operation on every available device, in-place modification detection, and
leaf-only gradient retention."""

import zlib

import numpy as np
import pytest
import vectorcore as vc

DEVICES = ["cpu"] + (["metal"] if vc.metal_available() else [])


def uniform(rng, shape, low=-1.0, high=1.0):
    return rng.uniform(low, high, size=shape).astype(np.float32)


def away_from_zero(rng, shape):
    """Values with |x| in [0.2, 1] so ReLU's kink is never inside the stencil."""
    magnitude = rng.uniform(0.2, 1.0, size=shape)
    sign = rng.choice([-1.0, 1.0], size=shape)
    return (magnitude * sign).astype(np.float32)


def positive(rng, shape):
    """Values in [1, 2], safe denominators for division."""
    return uniform(rng, shape, 1.0, 2.0)


def signed_permutation(rng, size):
    """A signed permutation matrix plus small noise. Multiplying an
    away_from_zero input by it keeps every hidden pre-activation at least ~0.17
    from zero (with both signs present), so a ReLU after it is differentiable
    across the finite-difference stencil."""
    matrix = np.zeros((size, size))
    matrix[np.arange(size), rng.permutation(size)] = rng.choice([-1.0, 1.0], size=size)
    return (matrix + rng.uniform(-0.01, 0.01, size=(size, size))).astype(np.float32)


def gradcheck(fn, inputs, device, eps=5e-3, rtol=1e-2, atol=1e-2):
    """Compare autograd against central differences of a random projection.

    Projecting the output onto random weights checks every column of the
    Jacobian at once instead of assuming all-ones upstream gradients.
    """
    leaves = [vc.Tensor(x, requires_grad=True, device=device) for x in inputs]
    output = fn(*leaves)
    weights_np = np.random.default_rng(99).uniform(
        0.5, 1.5, size=output.shape).astype(np.float32)
    weights = vc.Tensor(weights_np, device=output.device)
    (output * weights).sum().backward()

    def objective(values):
        tensors = [vc.Tensor(v, device=device) for v in values]
        result = fn(*tensors)
        return float((result * vc.Tensor(weights_np, device=result.device)).sum().item())

    for index, (leaf, base) in enumerate(zip(leaves, inputs)):
        assert leaf.grad is not None, f"input {index} received no gradient"
        analytic = leaf.grad.to("cpu").numpy()
        numeric = np.zeros_like(base, dtype=np.float64)
        flat = numeric.reshape(-1)
        for position in range(base.size):
            plus = [v.copy() for v in inputs]
            minus = [v.copy() for v in inputs]
            plus[index].reshape(-1)[position] += eps
            minus[index].reshape(-1)[position] -= eps
            flat[position] = (objective(plus) - objective(minus)) / (2 * eps)
        np.testing.assert_allclose(
            analytic, numeric, rtol=rtol, atol=atol,
            err_msg=f"gradient mismatch for input {index} on {device}")


# (name, function, input factories). Each factory takes an RNG.
CASES = [
    ("add_broadcast_row", lambda a, b: a + b,
     [lambda r: uniform(r, (2, 3)), lambda r: uniform(r, (3,))]),
    ("add_broadcast_outer", lambda a, b: a + b,
     [lambda r: uniform(r, (2, 1)), lambda r: uniform(r, (1, 3))]),
    ("subtract_broadcast_column", lambda a, b: a - b,
     [lambda r: uniform(r, (2, 3)), lambda r: uniform(r, (2, 1))]),
    ("multiply_rank3_broadcast", lambda a, b: a * b,
     [lambda r: uniform(r, (2, 3, 4)), lambda r: uniform(r, (4,))]),
    ("divide_broadcast", lambda a, b: a / b,
     [lambda r: uniform(r, (2, 3)), lambda r: positive(r, (3,))]),
    ("divide_scalar_tensor", lambda a, b: a / b,
     [lambda r: uniform(r, (2, 3)), lambda r: positive(r, ())]),
    ("negate", lambda a: -a, [lambda r: uniform(r, (2, 3))]),
    ("scalar_affine", lambda a: a * 2.5 + 1.0, [lambda r: uniform(r, (4,))]),
    ("scalar_reverse", lambda a: 3.0 / a - (1.0 - a), [lambda r: positive(r, (4,))]),
    ("matmul", lambda a, b: a @ b,
     [lambda r: uniform(r, (3, 4)), lambda r: uniform(r, (4, 2))]),
    ("matmul_vector_shaped", lambda a, b: a @ b,
     [lambda r: uniform(r, (1, 5)), lambda r: uniform(r, (5, 1))]),
    ("transpose", lambda a: a.T, [lambda r: uniform(r, (3, 2))]),
    ("sum", lambda a: a.sum(), [lambda r: uniform(r, (2, 3))]),
    ("mean", lambda a: a.mean(), [lambda r: uniform(r, (2, 3))]),
    ("relu", lambda a: a.relu(), [lambda r: away_from_zero(r, (3, 4))]),
    ("sigmoid", lambda a: a.sigmoid(), [lambda r: uniform(r, (3, 4), -3, 3)]),
    ("reshape", lambda a: a.reshape([3, 2]) * 2.0, [lambda r: uniform(r, (2, 3))]),
    ("flatten", lambda a: a.flatten(), [lambda r: uniform(r, (2, 3))]),
    ("reused_input", lambda a: a * a + a, [lambda r: uniform(r, (5,))]),
    ("linear_sigmoid", lambda x, w, b: (x @ w + b).sigmoid(),
     [lambda r: uniform(r, (4, 3)), lambda r: uniform(r, (3, 2)),
      lambda r: uniform(r, (2,))]),
    ("two_layer_mse", lambda x, w1, w2, y: (((x @ w1).relu() @ w2 - y) *
                                            ((x @ w1).relu() @ w2 - y)).mean(),
     [lambda r: away_from_zero(r, (4, 3)), lambda r: signed_permutation(r, 3),
      lambda r: uniform(r, (3, 1)), lambda r: uniform(r, (4, 1))]),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,fn,factories", CASES, ids=[case[0] for case in CASES])
def test_gradcheck(name, fn, factories, device):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    inputs = [factory(rng) for factory in factories]
    gradcheck(fn, inputs, device)


@pytest.mark.skipif(not vc.metal_available(), reason="Metal backend unavailable")
def test_gradcheck_device_transfer():
    rng = np.random.default_rng(5)
    gradcheck(lambda a: (a.to("metal") * 3.0).relu().to("cpu"),
              [away_from_zero(rng, (2, 3))], "cpu")


@pytest.mark.parametrize("device", DEVICES)
def test_relu_kink_values_are_finite(device):
    # Inputs that are exactly zero must not produce NaN gradients.
    x = vc.Tensor([0.0, -0.0, 1.0], requires_grad=True, device=device)
    x.relu().sum().backward()
    assert np.all(np.isfinite(x.grad.to("cpu").numpy()))


# --- In-place modification detection ------------------------------------


def test_version_counts_in_place_writes():
    w = vc.Tensor([1.0, 2.0], requires_grad=True)
    start = w._version
    view = w.reshape([2, 1])
    assert view._version == start
    _ = (w * 1.0).sum()          # out-of-place ops never bump the version
    assert w._version == start
    (w * 1.0).sum().backward()
    vc.sgd_step([w], 0.1)
    assert w._version == start + 1
    assert view._version == w._version, "views share one version counter"


def test_sgd_between_forward_and_backward_is_rejected():
    w = vc.Tensor([2.0], requires_grad=True)
    x = vc.Tensor([3.0], requires_grad=True)
    loss = (w * x).sum()
    (w * 1.0).sum().backward()
    vc.sgd_step([w], 1.0)
    with pytest.raises(RuntimeError, match="modified in place"):
        loss.backward()


def test_mutation_through_view_is_rejected():
    w = vc.Tensor([[0.5, -1.0], [2.0, 1.5]], requires_grad=True)
    x = vc.Tensor([[1.0, 2.0]], requires_grad=True)
    loss = (x @ w).sum()
    flat = w.flatten()
    # Mutate through the view with a fake gradient step.
    flat_leaf = flat.detach()
    flat_leaf.requires_grad = True
    (flat_leaf * 1.0).sum().backward()
    vc.sgd_step([flat_leaf], 1.0)  # shares storage with w
    with pytest.raises(RuntimeError, match="modified in place"):
        loss.backward()


def test_mutation_of_saved_output_is_rejected():
    x = vc.Tensor([0.5, -0.5], requires_grad=True)
    y = x.sigmoid()
    loss = y.sum()
    y_storage = y.detach()
    y_storage.requires_grad = True
    (y_storage * 1.0).sum().backward()
    vc.sgd_step([y_storage], 1.0)  # rewrites sigmoid's saved output
    with pytest.raises(RuntimeError, match="modified in place"):
        loss.backward()


def test_normal_training_loop_still_works():
    w = vc.Tensor([[0.0]], requires_grad=True)
    x = vc.Tensor([[1.0], [2.0], [3.0]])
    y = vc.Tensor([[2.0], [4.0], [6.0]])
    for _ in range(200):
        loss = vc.mse_loss(x @ w, y)
        loss.backward()
        vc.sgd_step([w], 0.05)
        vc.zero_grad([w])
    assert abs(w.item() - 2.0) < 1e-3


def test_unrelated_mutation_after_backward_is_allowed():
    w = vc.Tensor([2.0], requires_grad=True)
    loss = (w * w).sum()
    loss.backward()
    vc.sgd_step([w], 0.1)  # after backward: fine
    assert np.isclose(w.item(), 2.0 - 0.1 * 4.0)


# --- Leaf-only gradient retention ---------------------------------------


def test_only_leaves_keep_gradients():
    a = vc.Tensor([1.0, 2.0], requires_grad=True)
    hidden = a * 2.0
    (hidden * hidden).sum().backward()
    assert a.is_leaf and not hidden.is_leaf
    assert a.grad is not None
    assert hidden.grad is None
    np.testing.assert_allclose(a.grad.numpy(), 8.0 * np.array([1.0, 2.0]))


def test_retain_grad_opts_in_for_intermediates():
    a = vc.Tensor([1.0, 2.0], requires_grad=True)
    hidden = a * 2.0
    hidden.retain_grad()
    assert hidden.retains_grad
    (hidden * hidden).sum().backward()
    np.testing.assert_allclose(hidden.grad.numpy(), 2.0 * np.array([2.0, 4.0]))


def test_retain_grad_requires_grad():
    with pytest.raises(RuntimeError):
        vc.Tensor([1.0]).retain_grad()
