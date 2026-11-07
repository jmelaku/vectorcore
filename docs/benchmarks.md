# Reproducible benchmarks

## Performance work: A/B comparison (2026-11-07)

The CPU kernels (thread pool, elementwise fast paths, packed NEON GEMM, transposed-operand backward, fast gradient reduction) and the Metal runtime (asynchronous command buffers, buffer pool) were compared against the previous version by building both and running them **alternately, four rounds each, in the same session**; values are medians of per-round medians in milliseconds. The machine (Apple M4, 16 GB) had heavy unrelated load during the run (a virtual machine and a game; load average 14-29), which inflates every absolute time, but both versions saw the same conditions, so the ratios are the meaningful result. Re-run on an idle machine for clean absolute numbers (commands below).

| Measurement | Previous | Current | Speedup |
|---|---:|---:|---:|
| add 512x512, CPU | 2.019 | 0.106 | 19.1x |
| add 2048x2048, CPU | 28.010 | 0.917 | 30.5x |
| matmul 512, CPU | 8.497 | 1.718 | 4.9x |
| matmul 2048, CPU | 1134.0 | 164.4 | 6.9x |
| add 512x512, Metal | 0.706 | 0.424 | 1.7x |
| add 2048x2048, Metal | 3.537 | 1.748 | 2.0x |
| matmul 512 / 1024, Metal (5 rounds, after pipeline specialisation) | 1.203 / 6.728 | 1.143 / 7.215 | ~1x |
| training step, spiral MLP (2-64-3, batch 300), CPU | 1.596 | 0.749 | 2.1x |
| training step, spiral MLP, Metal | 29.593 | 0.964 | 30.7x |
| training step, MNIST-shaped MLP (784-256-10, batch 256), CPU | 19.849 | 6.346 | 3.1x |
| training step, MNIST-shaped MLP, Metal | 37.255 | 1.756 | 21.2x |

In a quieter moment earlier the same day, the current CPU GEMM measured 0.37 ms for 256x784x256 (279 GFLOP/s), 0.67 ms at 512 (399 GFLOP/s) and 32 ms at 2048 (537 GFLOP/s); the MNIST-shaped step took 1.5 ms on CPU and 3.1 ms on Metal, against 0.41 ms (PyTorch CPU) and 1.25 ms (PyTorch MPS).

Single-operation Metal timings still include one command-buffer round trip because the benchmark synchronizes after every operation; the training-step gains come from queuing a whole step and synchronizing once. Standalone Metal GEMM is unchanged: it is a simple tiled kernel and remains several times slower than PyTorch MPS at large sizes.

Commands:

```bash
PYTHONPATH=build .venv/bin/python benchmarks/benchmark.py --full --repeats 5
PYTHONPATH=build .venv/bin/python benchmarks/train_benchmark.py
```

`train_benchmark.py` times a complete SGD step (forward, cross-entropy, backward, update, and a host read of the loss so asynchronous GPU work is included) for VectorCore and, when installed, PyTorch CPU/MPS.

## Metal GEMM: simdgroup kernel vs tiled kernel (2026-11-07)

Same build, kernel selected per process with `VECTORCORE_METAL_GEMM=tiled`, three alternating rounds, median milliseconds including one synchronize. The GPU was shared with a game and a Docker VM during the run (load average 26-42), so absolute times are pessimistic; the ratio is the result.

| Shape (m x k x n) | Tiled | Simdgroup | Speedup |
|---|---:|---:|---:|
| 256 x 784 x 256 | 1.398 | 1.355 | 1.0x |
| 512 x 512 x 512 | 2.022 | 2.015 | 1.0x |
| 1024 x 1024 x 1024 | 7.154 | 5.310 | 1.35x |
| 2048 x 2048 x 2048 | 63.289 | 25.384 | 2.5x |
| 4096 x 4096 x 4096 | 986.5 | 506.8 | 1.95x |

Small sizes are bound by the command-buffer round trip of a synchronized single operation, not by the kernel. An earlier version of the simdgroup kernel that staged every K slice through threadgroup memory was only 1.14-1.19x faster at large sizes; loading fragments directly from device memory for interior tiles is what produced the gain. A threadgroup launch-order swizzle was tried and showed no reliable improvement, so it was not kept.



The tables below were recorded for the first version of the runtime, before the performance work, and are kept for reference.

### Method

`benchmarks/benchmark.py` creates float32 inputs once with NumPy's seeded generator. It runs two unmeasured warmups followed by repeated measurements and reports the median. Tensor allocation for each result is included. Input construction and CPU/GPU transfer are excluded from operation timings. Every PyTorch MPS result is synchronized; VectorCore Metal commands synchronized internally in that version (the current harness calls `vc.synchronize()`). Separate transfer rows include VectorCore buffer allocation and copy.

Command used for the recorded run:

```bash
PYTHONPATH=build .venv/bin/python benchmarks/benchmark.py --full --repeats 5
```

Environment: Apple M4, 16 GB unified memory, macOS 15.7.3 arm64, Apple Clang 17, Python 3.9.6, NumPy 2.0.2, PyTorch 2.8.0, VectorCore Release build. Date: 2026-10-02. Values are median milliseconds.

### Addition

| N | NumPy | VC CPU | VC Metal | Torch CPU | Torch MPS |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.003 | 0.071 | 0.641 | 0.006 | 0.894 |
| 256 | 0.015 | 0.475 | 0.457 | 0.035 | 0.370 |
| 512 | 0.050 | 1.319 | 0.368 | 0.067 | 0.647 |
| 1024 | 0.803 | 5.741 | 2.538 | 0.532 | 0.884 |
| 2048 | 1.367 | 18.786 | 6.626 | 1.495 | 1.498 |

### Multiplication

| N | NumPy | VC CPU | VC Metal | Torch CPU | Torch MPS |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.004 | 0.084 | 0.675 | 0.005 | 0.312 |
| 256 | 0.014 | 0.397 | 0.690 | 0.034 | 0.298 |
| 512 | 0.050 | 1.405 | 0.379 | 0.066 | 0.408 |
| 1024 | 0.639 | 5.997 | 1.668 | 0.505 | 0.613 |
| 2048 | 0.970 | 21.517 | 5.485 | 1.744 | 1.413 |

### Matrix multiplication

| N | NumPy | VC CPU | VC Metal | Torch CPU | Torch MPS |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.012 | 0.349 | 0.551 | 0.012 | 0.384 |
| 256 | 0.059 | 1.031 | 0.786 | 0.057 | 0.456 |
| 512 | 0.419 | 9.547 | 3.152 | 0.392 | 0.634 |
| 1024 | 3.934 | 87.988 | 9.504 | 3.697 | 2.586 |
| 2048 | 30.101 | 849.880 | 87.273 | 22.814 | 10.930 |

### VectorCore transfers

| N | CPU to Metal | Metal to CPU |
|---:|---:|---:|
| 128 | 0.042 | 0.009 |
| 256 | 0.100 | 0.023 |
| 512 | 0.242 | 0.116 |
| 1024 | 1.296 | 2.376 |
| 2048 | 4.965 | 5.295 |

### Interpretation

NumPy and PyTorch use heavily tuned vendor kernels, vectorized elementwise loops, packing, cache-aware microkernels, and hardware-specific acceleration. VectorCore CPU performs generic coordinate reconstruction for each broadcast output and uses a straightforward blocked GEMM, so the gaps are expected. Metal dispatch overhead dominates small kernels. At 512 and above, VectorCore Metal is faster than VectorCore CPU, but its simple 16×16 GEMM and synchronous command submission remain well behind PyTorch MPS. These numbers are baselines for engineering work, not performance claims.
