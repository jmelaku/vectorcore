"""Forward parity, error handling, stability and training tests for the
classification building blocks: exp/log, axis reductions, softmax,
cross-entropy, batched matmul and SGD on every device."""

import numpy as np
import pytest
import vectorcore as vc

DEVICES = ["cpu"] + (["metal"] if vc.metal_available() else [])
metal_only = pytest.mark.skipif(not vc.metal_available(), reason="Metal backend unavailable")


def cpu_numpy(tensor):
    return tensor.to("cpu").numpy()


def close(actual, expected, rtol=2e-5, atol=2e-5):
    np.testing.assert_allclose(cpu_numpy(actual), np.asarray(expected, dtype=np.float32),
                               rtol=rtol, atol=atol)


def np_log_softmax(x, axis):
    shifted = x - x.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


# --- forward parity -------------------------------------------------------


def test_exp_log(device):
    x = np.random.default_rng(1).uniform(0.1, 3.0, size=(4, 5)).astype(np.float32)
    t = vc.Tensor(x, device=device)
    close(t.exp(), np.exp(x), rtol=1e-4)
    close(t.log(), np.log(x), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -3])
@pytest.mark.parametrize("keepdim", [False, True])
def test_axis_reductions_match_numpy(device, axis, keepdim):
    x = np.random.default_rng(2).normal(size=(3, 4, 5)).astype(np.float32)
    t = vc.Tensor(x, device=device)
    close(t.sum(axis, keepdim), x.sum(axis=axis, keepdims=keepdim), rtol=1e-4, atol=1e-5)
    close(t.mean(axis, keepdim), x.mean(axis=axis, keepdims=keepdim), rtol=1e-4, atol=1e-5)
    close(t.max(axis, keepdim), x.max(axis=axis, keepdims=keepdim))
    close(t.argmax(axis, keepdim), x.argmax(axis=axis).reshape(
        x.max(axis=axis, keepdims=keepdim).shape))


def test_reductions_without_axis(device):
    x = np.random.default_rng(3).normal(size=(3, 4)).astype(np.float32)
    t = vc.Tensor(x, device=device)
    assert np.isclose(t.max().item(), x.max())
    assert t.argmax().item() == x.argmax()
    assert t.sum(keepdim=True).shape == [1, 1]
    assert t.max(keepdim=True).shape == [1, 1]


def test_argmax_ties_pick_first(device):
    t = vc.Tensor([[1.0, 3.0, 3.0], [2.0, 2.0, 0.0]], device=device)
    close(t.argmax(1), [1, 0])


def test_max_gradient_goes_to_one_element_on_ties(device):
    t = vc.Tensor([[1.0, 3.0, 3.0]], requires_grad=True, device=device)
    t.max(1).sum().backward()
    close(t.grad, [[0, 1, 0]])


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_softmax_and_log_softmax(device, axis):
    x = np.random.default_rng(4).normal(size=(3, 5)).astype(np.float32) * 3
    t = vc.Tensor(x, device=device)
    expected = np_log_softmax(x, axis)
    close(t.log_softmax(axis), expected, rtol=1e-4, atol=1e-5)
    close(t.softmax(axis), np.exp(expected), rtol=1e-4, atol=1e-6)
    close(vc.softmax(t, axis).sum(axis), np.ones(x.sum(axis=axis).shape), rtol=1e-5)


def test_softmax_is_stable_for_huge_logits(device):
    t = vc.Tensor([[1000.0, 1001.0, 1002.0], [-1000.0, 0.0, 1000.0]],
                  requires_grad=True, device=device)
    probabilities = t.softmax(-1)
    values = cpu_numpy(probabilities)
    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(values[0], np.exp(np_log_softmax(
        np.array([0.0, 1.0, 2.0]), 0)), rtol=1e-5)
    vc.cross_entropy(t, [2, 0]).backward()
    assert np.all(np.isfinite(cpu_numpy(t.grad)))


def test_batched_matmul_matches_numpy(device):
    rng = np.random.default_rng(5)
    cases = [((2, 3, 4), (2, 4, 5)), ((2, 3, 4), (4, 5)), ((3, 4), (2, 4, 5)),
             ((2, 1, 3, 4), (3, 4, 2)), ((1, 17, 19), (4, 19, 18))]
    for a_shape, b_shape in cases:
        a = rng.normal(size=a_shape).astype(np.float32)
        b = rng.normal(size=b_shape).astype(np.float32)
        result = vc.Tensor(a, device=device) @ vc.Tensor(b, device=device)
        # NumPy 2.0 + macOS Accelerate emits spurious FP warnings on small matmuls.
        with np.errstate(all="ignore"):
            expected = np.matmul(a, b)
        assert result.shape == list(expected.shape)
        close(result, expected, rtol=2e-4, atol=2e-4)


def test_transpose_swaps_last_two_axes(device):
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    close(vc.Tensor(x, device=device).transpose(), np.swapaxes(x, -1, -2))


def test_cross_entropy_matches_pytorch(device):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(6)
    logits = rng.normal(size=(6, 4)).astype(np.float32) * 2
    targets = [0, 3, 1, 1, 2, 0]
    t = vc.Tensor(logits, requires_grad=True, device=device)
    loss = vc.cross_entropy(t, targets)
    loss.backward()
    tt = torch.tensor(logits, requires_grad=True)
    reference = torch.nn.functional.cross_entropy(tt, torch.tensor(targets))
    reference.backward()
    assert np.isclose(loss.item(), reference.item(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(cpu_numpy(t.grad), tt.grad.numpy(), rtol=1e-4, atol=1e-6)


def test_cross_entropy_accepts_numpy_targets():
    logits = vc.Tensor(np.zeros((3, 2), dtype=np.float32))
    loss = vc.cross_entropy(logits, np.array([0, 1, 1], dtype=np.int64))
    assert np.isclose(loss.item(), np.log(2.0))


# --- error handling -------------------------------------------------------


@pytest.mark.parametrize("call", [
    lambda: vc.Tensor([[1.0, 2.0]]).sum(2),
    lambda: vc.Tensor([[1.0, 2.0]]).max(-3),
    lambda: vc.Tensor(3.0).sum(0),
    lambda: vc.Tensor(np.zeros((2, 0), dtype=np.float32)).max(1),
    lambda: vc.Tensor(np.zeros((2, 0), dtype=np.float32)).mean(1),
    lambda: vc.Tensor(np.ones((2, 3, 4), dtype=np.float32)) @ vc.Tensor(np.ones((3, 5, 2), dtype=np.float32)),
    lambda: vc.Tensor(np.ones((2, 3), dtype=np.float32)) @ vc.Tensor(np.ones((4, 2), dtype=np.float32)),
    lambda: vc.Tensor([1.0, 2.0]) @ vc.Tensor([1.0, 2.0]),
    lambda: vc.cross_entropy(vc.Tensor([1.0, 2.0]), [0]),
    lambda: vc.cross_entropy(vc.Tensor([[1.0, 2.0]]), [0, 1]),
    lambda: vc.cross_entropy(vc.Tensor([[1.0, 2.0]]), [2]),
])
def test_invalid_inputs_raise(call):
    with pytest.raises((ValueError, IndexError, RuntimeError)):
        call()


def test_sum_of_empty_axis_is_zero():
    close(vc.Tensor(np.zeros((2, 0), dtype=np.float32)).sum(1), [0.0, 0.0])


# --- optimizer on Metal ---------------------------------------------------


@metal_only
def test_sgd_on_metal_matches_cpu():
    rng = np.random.default_rng(7)
    w_np = rng.normal(size=(3, 2)).astype(np.float32)
    x_np = rng.normal(size=(4, 3)).astype(np.float32)
    results = {}
    for device in ["cpu", "metal"]:
        w = vc.Tensor(w_np, requires_grad=True, device=device)
        x = vc.Tensor(x_np, device=device)
        for _ in range(5):
            loss = vc.cross_entropy(x @ w, [0, 1, 1, 0])
            loss.backward()
            vc.sgd_step([w], 0.5)
            vc.zero_grad([w])
        results[device] = cpu_numpy(w)
    np.testing.assert_allclose(results["metal"], results["cpu"], rtol=1e-4, atol=1e-5)


@metal_only
def test_metal_sgd_bumps_version_and_is_detected():
    w = vc.Tensor([[1.0], [2.0]], requires_grad=True, device="metal")
    x = vc.Tensor([[1.0, 1.0]], device="metal")
    # w * w needs the old value of w for its own gradient. (For x @ w alone,
    # dw = x^T @ g never reads w, so mutating w would correctly not be an error.)
    stale = (w * w).sum()
    (x @ w).sum().backward()
    before = w._version
    vc.sgd_step([w], 0.1)
    assert w._version == before + 1
    with pytest.raises(RuntimeError, match="modified in place"):
        stale.backward()


# --- end-to-end training --------------------------------------------------


def make_spiral(points_per_class=60, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    features, labels = [], []
    for label in range(classes):
        radius = np.linspace(0.05, 1.0, points_per_class)
        angle = (np.linspace(label * 4.0, (label + 1) * 4.0, points_per_class)
                 + rng.normal(scale=0.2, size=points_per_class))
        features.append(np.column_stack([radius * np.sin(angle), radius * np.cos(angle)]))
        labels.extend([label] * points_per_class)
    return np.concatenate(features).astype(np.float32), labels


def test_mlp_learns_nonlinear_spiral(device):
    features, labels = make_spiral()
    x = vc.Tensor(features, device=device)
    first, second = vc.Linear(2, 64, 1), vc.Linear(64, 3, 2)
    params = [first.weight, first.bias, second.weight, second.bias]
    if device != "cpu":
        params = [p.to(device).detach() for p in params]
        for p in params:
            p.requires_grad = True
        first.weight, first.bias, second.weight, second.bias = params
    for _ in range(600):
        loss = vc.cross_entropy(second(first(x).relu()), labels)
        loss.backward()
        vc.sgd_step(params, 1.0)
        vc.zero_grad(params)
    predictions = cpu_numpy(second(first(x).relu()).argmax(1)).astype(int)
    accuracy = (predictions == np.array(labels)).mean()
    assert accuracy > 0.95, f"accuracy {accuracy:.3f} on {device}"
