"""Train a two-layer MLP classifier on the three-arm spiral dataset.

The spiral is not linearly separable, so a correct softmax/cross-entropy,
ReLU and backpropagation path is needed to reach high accuracy.

    PYTHONPATH=build .venv/bin/python examples/spiral_classifier.py
    PYTHONPATH=build .venv/bin/python examples/spiral_classifier.py --device metal
"""

import argparse
import time

import numpy as np
import vectorcore as vc


def make_spiral(points_per_class, classes, seed):
    rng = np.random.default_rng(seed)
    features, labels = [], []
    for label in range(classes):
        radius = np.linspace(0.05, 1.0, points_per_class)
        angle = (np.linspace(label * 4.0, (label + 1) * 4.0, points_per_class)
                 + rng.normal(scale=0.2, size=points_per_class))
        features.append(np.column_stack([radius * np.sin(angle), radius * np.cos(angle)]))
        labels.extend([label] * points_per_class)
    order = rng.permutation(len(labels))
    return np.concatenate(features).astype(np.float32)[order], np.array(labels)[order]


def parameter(values, device):
    tensor = vc.Tensor(values, device=device)
    tensor.requires_grad = True
    return tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu", choices=["cpu", "metal"])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1.0)
    args = parser.parse_args()
    if args.device == "metal" and not vc.metal_available():
        raise SystemExit("Metal is unavailable on this machine")

    classes = 3
    features, labels = make_spiral(points_per_class=100, classes=classes, seed=0)
    split = int(0.8 * len(labels))
    train_x = vc.Tensor(features[:split], device=args.device)
    test_x = vc.Tensor(features[split:], device=args.device)
    train_y, test_y = labels[:split].tolist(), labels[split:]

    rng = np.random.default_rng(1)
    w1 = parameter(rng.normal(size=(2, args.hidden)).astype(np.float32), args.device)
    b1 = parameter(np.zeros(args.hidden, dtype=np.float32), args.device)
    w2 = parameter((rng.normal(size=(args.hidden, classes)) / np.sqrt(args.hidden))
                   .astype(np.float32), args.device)
    b2 = parameter(np.zeros(classes, dtype=np.float32), args.device)
    params = [w1, b1, w2, b2]

    def logits(x):
        return ((x @ w1 + b1).relu()) @ w2 + b2

    def accuracy(x, y):
        predicted = logits(x).argmax(1).to("cpu").numpy().astype(int)
        return float((predicted == np.asarray(y)).mean())

    start = time.perf_counter()
    for step in range(args.steps + 1):
        loss = vc.cross_entropy(logits(train_x), train_y)
        if step % (args.steps // 5) == 0:
            print(f"step={step:5d} loss={loss.item():.4f} "
                  f"train_acc={accuracy(train_x, train_y):.3f}")
        if step == args.steps:
            break
        loss.backward()
        vc.sgd_step(params, args.lr)
        vc.zero_grad(params)
    elapsed = time.perf_counter() - start
    print(f"test_acc={accuracy(test_x, test_y):.3f} device={args.device} "
          f"time={elapsed:.2f}s")


if __name__ == "__main__":
    main()
