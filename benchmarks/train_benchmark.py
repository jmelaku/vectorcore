#!/usr/bin/env python3
"""End-to-end training-step benchmark.

Times one full SGD step (forward, cross-entropy, backward, update) for two
MLP classifiers: the small spiral model from examples/ and an MNIST-shaped
model (784 -> 256 -> 10, batch 256) on synthetic data. Reports the median
step time in milliseconds after warmup; each step ends with a host read of
the loss so asynchronous GPU work is included.
"""

import argparse
import statistics
import time

import numpy as np
import vectorcore as vc

try:
    import torch
except ImportError:
    torch = None

CONFIGS = {
    "spiral_2-64-3_b300": (300, 2, 64, 3),
    "mnist_784-256-10_b256": (256, 784, 256, 10),
}


def vc_step_fn(batch, features, hidden, classes, device, seed=0):
    rng = np.random.default_rng(seed)
    x = vc.Tensor(rng.normal(size=(batch, features)).astype(np.float32), device=device)
    targets = rng.integers(0, classes, size=batch).tolist()

    def param(shape, scale):
        tensor = vc.Tensor((rng.normal(size=shape) * scale).astype(np.float32), device=device)
        tensor.requires_grad = True
        return tensor

    w1, b1 = param((features, hidden), features ** -0.5), param((hidden,), 0.0)
    w2, b2 = param((hidden, classes), hidden ** -0.5), param((classes,), 0.0)
    params = [w1, b1, w2, b2]

    def step():
        loss = vc.cross_entropy(((x @ w1 + b1).relu()) @ w2 + b2, targets)
        loss.backward()
        vc.sgd_step(params, 0.1)
        vc.zero_grad(params)
        return loss.item()

    return step


def torch_step_fn(batch, features, hidden, classes, device, seed=0):
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.normal(size=(batch, features)).astype(np.float32), device=device)
    targets = torch.tensor(rng.integers(0, classes, size=batch), device=device)
    model = torch.nn.Sequential(torch.nn.Linear(features, hidden), torch.nn.ReLU(),
                                torch.nn.Linear(hidden, classes)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def step():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), targets)
        loss.backward()
        optimizer.step()
        return loss.item()

    return step


def median_ms(step, warmups, repeats):
    for _ in range(warmups):
        step()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        step()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--no-torch", action="store_true")
    args = parser.parse_args()
    print("model,backend,median_step_ms")
    for name, config in CONFIGS.items():
        backends = [("vectorcore_cpu", lambda: vc_step_fn(*config, "cpu"))]
        if vc.metal_available():
            backends.append(("vectorcore_metal", lambda: vc_step_fn(*config, "metal")))
        if torch is not None and not args.no_torch:
            backends.append(("torch_cpu", lambda: torch_step_fn(*config, "cpu")))
            if torch.backends.mps.is_available():
                backends.append(("torch_mps", lambda: torch_step_fn(*config, "mps")))
        for backend, make in backends:
            print(f"{name},{backend},{median_ms(make(), args.warmups, args.repeats):.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
