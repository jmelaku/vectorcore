#!/usr/bin/env python3
"""Fair, reproducible operation benchmarks.

Inputs are allocated before timing. Each framework performs a warmup, every GPU
result is synchronized, and the reported value is the median wall-clock time.
VectorCore Metal currently synchronizes each operation internally. Transfers are
reported separately and are never included in kernel measurements.
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


def measure(function, synchronize=lambda: None, warmups=2, repeats=7):
    for _ in range(warmups):
        function()
        synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(samples)


def print_row(size, operation, backend, milliseconds):
    print(f"{size},{operation},{backend},{milliseconds:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="include 1024 and 2048 matrices")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    sizes = [128, 256, 512] + ([1024, 2048] if args.full else [])
    rng = np.random.default_rng(2026)
    print("size,operation,backend,median_ms")

    # Some Accelerate builds leave benign floating exception flags set after
    # highly optimized GEMM. Results are not consumed here, and warnings would
    # pollute the CSV stream, so keep NumPy's warning policy local to this run.
    np.seterr(all="ignore")
    for size in sizes:
        a = rng.normal(size=(size, size)).astype(np.float32)
        b = rng.normal(size=(size, size)).astype(np.float32)
        va, vb = vc.Tensor(a), vc.Tensor(b)
        for name, numpy_op, vc_op in (
            ("add", lambda: a + b, lambda: va + vb),
            ("multiply", lambda: a * b, lambda: va * vb),
            ("matmul", lambda: a @ b, lambda: va @ vb),
        ):
            print_row(size, name, "numpy", measure(numpy_op, repeats=args.repeats))
            print_row(size, name, "vectorcore_cpu", measure(vc_op, repeats=args.repeats))

        if vc.metal_available():
            ma, mb = va.to("metal"), vb.to("metal")
            for name, operation in (
                ("add", lambda: ma + mb),
                ("multiply", lambda: ma * mb),
                ("matmul", lambda: ma @ mb),
            ):
                print_row(size, name, "vectorcore_metal",
                          measure(operation, repeats=args.repeats))
            print_row(size, "cpu_to_metal", "vectorcore_transfer",
                      measure(lambda: va.to("metal"), repeats=args.repeats))
            print_row(size, "metal_to_cpu", "vectorcore_transfer",
                      measure(lambda: ma.to("cpu"), repeats=args.repeats))

        if torch is not None:
            ta, tb = torch.from_numpy(a), torch.from_numpy(b)
            for name, operation in (
                ("add", lambda: ta + tb),
                ("multiply", lambda: ta * tb),
                ("matmul", lambda: ta @ tb),
            ):
                print_row(size, name, "torch_cpu", measure(operation, repeats=args.repeats))
            if torch.backends.mps.is_available():
                ga, gb = ta.to("mps"), tb.to("mps")
                sync = torch.mps.synchronize
                for name, operation in (
                    ("add", lambda: ga + gb),
                    ("multiply", lambda: ga * gb),
                    ("matmul", lambda: ga @ gb),
                ):
                    print_row(size, name, "torch_mps",
                              measure(operation, sync, repeats=args.repeats))


if __name__ == "__main__":
    main()
