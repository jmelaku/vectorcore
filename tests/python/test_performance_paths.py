"""Correctness of the optimized kernels: the packed/microkernel GEMM at edge
sizes, every elementwise fast path, parallel reductions, and asynchronous
Metal execution (ordering, in-place updates, buffer reuse)."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import vectorcore as vc

DEVICES = ["cpu"] + (["metal"] if vc.metal_available() else [])
metal_only = pytest.mark.skipif(not vc.metal_available(), reason="Metal backend unavailable")


def reference_matmul(a, b):
    with np.errstate(all="ignore"):  # NumPy 2.0 + Accelerate emits spurious FP flags
        return np.matmul(a.astype(np.float64), b.astype(np.float64))


# Shapes chosen to hit: the small-matrix path, exact microkernel multiples,
# ragged edges in both directions, k spanning several 256-deep panels,
# n spanning two 4092-wide B blocks, and tall/skinny/wide extremes.
GEMM_SHAPES = [
    (3, 5, 7), (4, 33, 33), (8, 256, 12), (16, 512, 24), (9, 257, 13),
    (63, 100, 97), (64, 64, 96), (65, 300, 95), (127, 513, 129),
    (200, 37, 1), (1, 300, 300), (300, 1, 300), (8, 3, 4100), (777, 64, 5),
    # Metal simdgroup kernel: fully interior 64x64 tiles with k % 8 == 0 take
    # the direct-load path; the others mix it with the staged edge path.
    (128, 64, 128), (192, 256, 64), (128, 60, 128), (130, 64, 128), (128, 64, 70),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("m,k,n", GEMM_SHAPES)
def test_matmul_edge_shapes(device, m, k, n):
    rng = np.random.default_rng(m * 10007 + k * 101 + n)
    a = rng.uniform(-1, 1, size=(m, k)).astype(np.float32)
    b = rng.uniform(-1, 1, size=(k, n)).astype(np.float32)
    result = (vc.Tensor(a, device=device) @ vc.Tensor(b, device=device)).to("cpu").numpy()
    np.testing.assert_allclose(result, reference_matmul(a, b), rtol=1e-4, atol=1e-4 * np.sqrt(k))


def test_large_matmul_and_gradients_match_numpy():
    rng = np.random.default_rng(11)
    a = rng.normal(size=(300, 400)).astype(np.float32)
    b = rng.normal(size=(400, 250)).astype(np.float32)
    ta = vc.Tensor(a, requires_grad=True)
    tb = vc.Tensor(b, requires_grad=True)
    (ta @ tb).sum().backward()
    np.testing.assert_allclose(ta.grad.numpy(), np.broadcast_to(b.sum(axis=1), a.shape),
                               rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(tb.grad.numpy(),
                               np.broadcast_to(a.sum(axis=0)[:, None], b.shape),
                               rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("m,k,n", [(128, 64, 192), (64, 128, 64), (100, 72, 90), (128, 60, 128)])
def test_matmul_gradients_use_transposed_kernels(device, m, k, n):
    # Backward runs op(A) @ op(B) with transpose flags; on Metal this covers
    # the TRANS_A / TRANS_B pipeline variants on aligned and ragged shapes.
    rng = np.random.default_rng(m + k + n)
    a = rng.uniform(-1, 1, size=(m, k)).astype(np.float32)
    b = rng.uniform(-1, 1, size=(k, n)).astype(np.float32)
    g = rng.uniform(-1, 1, size=(m, n)).astype(np.float32)
    ta = vc.Tensor(a, requires_grad=True, device=device)
    tb = vc.Tensor(b, requires_grad=True, device=device)
    (ta @ tb).backward(vc.Tensor(g, device=device))
    np.testing.assert_allclose(ta.grad.to("cpu").numpy(), reference_matmul(g, b.T),
                               rtol=1e-4, atol=1e-4 * np.sqrt(n))
    np.testing.assert_allclose(tb.grad.to("cpu").numpy(), reference_matmul(a.T, g),
                               rtol=1e-4, atol=1e-4 * np.sqrt(m))


@metal_only
def test_gemm_kernel_selection():
    kernel = vc.metal_gemm_kernel()
    if os.environ.get("VECTORCORE_METAL_GEMM") == "tiled":
        assert kernel == "tiled"
    else:
        # "tiled (simdgroup kernel unavailable: ...)" means the fast kernel was
        # wanted on this GPU but failed to build: that is a bug, not a fallback.
        assert kernel in ("simdgroup", "tiled"), kernel


@metal_only
@pytest.mark.skipif(os.environ.get("VECTORCORE_METAL_GEMM") == "tiled",
                    reason="already running with the tiled kernel")
def test_tiled_fallback_kernel_passes_matmul_tests():
    # Re-run the matmul and gradient tests in a fresh process with the
    # portable kernel forced, so the fallback path stays covered.
    here = Path(__file__).parent
    env = dict(os.environ, VECTORCORE_METAL_GEMM="tiled")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         str(here / "test_performance_paths.py"), str(here / "test_classifier_ops.py"),
         str(here / "test_autograd_correctness.py"),
         "-k", "matmul or gradcheck or gemm_kernel or mlp"],
        env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]


BROADCAST_CASES = [
    ((1000, 300), (1000, 300)),   # same shape, parallel
    ((1000, 300), ()),            # scalar right
    ((), (1000, 300)),            # scalar left
    ((1000, 300), (300,)),        # row vector
    ((1000, 300), (1000, 1)),     # column vector
    ((1000, 1), (1, 300)),        # outer product
    ((40, 1, 300), (1, 50, 300)), # middle-axis broadcast
    ((40, 50, 1), (50, 7)),       # last-axis broadcast with rank change
    ((3,), (2, 1)),               # tiny general path
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("a_shape,b_shape", BROADCAST_CASES)
def test_elementwise_paths(device, a_shape, b_shape):
    rng = np.random.default_rng(len(a_shape) * 13 + len(b_shape))
    a = rng.uniform(0.5, 2, size=a_shape).astype(np.float32)
    b = rng.uniform(0.5, 2, size=b_shape).astype(np.float32)
    ta, tb = vc.Tensor(a, device=device), vc.Tensor(b, device=device)
    for vc_op, np_op in ((lambda x, y: x + y, np.add), (lambda x, y: x - y, np.subtract),
                         (lambda x, y: x * y, np.multiply), (lambda x, y: x / y, np.divide)):
        np.testing.assert_allclose(vc_op(ta, tb).to("cpu").numpy(), np_op(a, b),
                                   rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_large_unary_ops(device):
    x = np.random.default_rng(3).uniform(-5, 5, size=(700, 300)).astype(np.float32)
    t = vc.Tensor(x, device=device)
    np.testing.assert_allclose(t.relu().to("cpu").numpy(), np.maximum(x, 0))
    np.testing.assert_allclose(t.sigmoid().to("cpu").numpy(), 1 / (1 + np.exp(-x)),
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(t.exp().to("cpu").numpy(), np.exp(x), rtol=1e-5)
    np.testing.assert_allclose((-t).to("cpu").numpy(), -x)


def test_parallel_sum_is_deterministic_and_accurate():
    x = np.random.default_rng(4).normal(size=3_000_001).astype(np.float32)
    t = vc.Tensor(x)
    values = {t.sum().item() for _ in range(5)}
    assert len(values) == 1, "parallel sum must not depend on scheduling"
    assert np.isclose(values.pop(), x.astype(np.float64).sum(), rtol=1e-6, atol=1e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_large_axis_reductions(device):
    x = np.random.default_rng(5).normal(size=(64, 500, 33)).astype(np.float32)
    t = vc.Tensor(x, device=device)
    for axis in (0, 1, 2):
        np.testing.assert_allclose(t.sum(axis).to("cpu").numpy(), x.sum(axis=axis),
                                   rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(t.max(axis).to("cpu").numpy(), x.max(axis=axis))
        np.testing.assert_array_equal(t.argmax(axis).to("cpu").numpy(), x.argmax(axis=axis))


@pytest.mark.parametrize("device", DEVICES)
def test_broadcast_gradient_reduction_shapes(device):
    # Exercises sum_to_shape for leading, middle and trailing broadcast axes.
    rng = np.random.default_rng(6)
    for big, small in [((8, 5, 7), (7,)), ((8, 5, 7), (5, 1)), ((8, 5, 7), (8, 1, 7)),
                       ((8, 5, 7), (1, 1, 1)), ((8, 5, 7), ())]:
        a = vc.Tensor(rng.normal(size=big).astype(np.float32), device=device)
        b_np = rng.normal(size=small).astype(np.float32)
        b = vc.Tensor(b_np, requires_grad=True, device=device)
        weights = rng.normal(size=big).astype(np.float32)
        ((a + b) * vc.Tensor(weights, device=device)).sum().backward()
        axes = tuple(range(len(big) - len(small))) + tuple(
            len(big) - len(small) + i for i, d in enumerate(small) if d == 1)
        expected = weights.sum(axis=axes).reshape(small) if small else weights.sum()
        np.testing.assert_allclose(b.grad.to("cpu").numpy(), expected, rtol=1e-4, atol=1e-4)


# --- asynchronous Metal execution ----------------------------------------


@metal_only
def test_metal_long_dependency_chain_without_sync():
    # Thousands of queued dependent ops must execute in order and only be
    # waited on when the host reads the result.
    x = vc.Tensor(np.zeros(1000, dtype=np.float32), device="metal")
    for _ in range(2000):
        x = x + 1.0
    np.testing.assert_allclose(x.to("cpu").numpy(), 2000.0)


@metal_only
def test_metal_in_place_update_is_ordered():
    w = vc.Tensor(np.ones(256, dtype=np.float32), requires_grad=True, device="metal")
    before = (w * 2.0)  # queued before the update; must see the old values
    (w * 1.0).sum().backward()
    vc.sgd_step([w], 0.5)  # w -> 0.5, queued in place
    after = (w * 2.0)
    np.testing.assert_allclose(before.to("cpu").numpy(), 2.0)
    np.testing.assert_allclose(after.to("cpu").numpy(), 1.0)


@metal_only
def test_metal_buffer_reuse_does_not_corrupt_results():
    # Allocate and drop many temporaries so pooled buffers are recycled while
    # earlier work may still be queued.
    rng = np.random.default_rng(7)
    expected, results = [], []
    for i in range(200):
        size = int(rng.integers(1, 5000))
        data = rng.normal(size=size).astype(np.float32)
        t = vc.Tensor(data, device="metal")
        tmp = (t * 3.0).relu()   # dropped immediately after use
        results.append((tmp - t).sum())
        expected.append(float((np.maximum(data * 3.0, 0) - data).astype(np.float64).sum()))
        del t, tmp
    for got, want in zip(results, expected):
        assert np.isclose(got.item(), want, rtol=1e-4, atol=1e-3)


@metal_only
def test_metal_synchronize_and_errors_surface():
    vc.synchronize()
    x = vc.Tensor([1.0, 2.0], device="metal")
    y = x * 2.0
    vc.synchronize()
    np.testing.assert_allclose(y.to("cpu").numpy(), [2.0, 4.0])
