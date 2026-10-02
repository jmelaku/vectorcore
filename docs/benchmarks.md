# Reproducible benchmarks

## Method

`benchmarks/benchmark.py` creates float32 inputs once with NumPy's seeded generator. It runs two unmeasured warmups followed by repeated measurements and reports the median. Tensor allocation for each result is included. Input construction and CPU/GPU transfer are excluded from operation timings. Every PyTorch MPS result is synchronized; VectorCore Metal commands synchronize internally. Separate transfer rows include VectorCore buffer allocation and copy.

Command used for the recorded run:

```bash
PYTHONPATH=build .venv/bin/python benchmarks/benchmark.py --full --repeats 5
```

Environment: Apple M4, 16 GB unified memory, macOS 15.7.3 arm64, Apple Clang 17, Python 3.9.6, NumPy 2.0.2, PyTorch 2.8.0, VectorCore Release build. Date: 2026-10-02. Values are median milliseconds.

## Addition

| N | NumPy | VC CPU | VC Metal | Torch CPU | Torch MPS |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.003 | 0.071 | 0.641 | 0.006 | 0.894 |
| 256 | 0.015 | 0.475 | 0.457 | 0.035 | 0.370 |
| 512 | 0.050 | 1.319 | 0.368 | 0.067 | 0.647 |
| 1024 | 0.803 | 5.741 | 2.538 | 0.532 | 0.884 |
| 2048 | 1.367 | 18.786 | 6.626 | 1.495 | 1.498 |

## Multiplication

| N | NumPy | VC CPU | VC Metal | Torch CPU | Torch MPS |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.004 | 0.084 | 0.675 | 0.005 | 0.312 |
| 256 | 0.014 | 0.397 | 0.690 | 0.034 | 0.298 |
| 512 | 0.050 | 1.405 | 0.379 | 0.066 | 0.408 |
| 1024 | 0.639 | 5.997 | 1.668 | 0.505 | 0.613 |
| 2048 | 0.970 | 21.517 | 5.485 | 1.744 | 1.413 |

## Matrix multiplication

| N | NumPy | VC CPU | VC Metal | Torch CPU | Torch MPS |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.012 | 0.349 | 0.551 | 0.012 | 0.384 |
| 256 | 0.059 | 1.031 | 0.786 | 0.057 | 0.456 |
| 512 | 0.419 | 9.547 | 3.152 | 0.392 | 0.634 |
| 1024 | 3.934 | 87.988 | 9.504 | 3.697 | 2.586 |
| 2048 | 30.101 | 849.880 | 87.273 | 22.814 | 10.930 |

## VectorCore transfers

| N | CPU to Metal | Metal to CPU |
|---:|---:|---:|
| 128 | 0.042 | 0.009 |
| 256 | 0.100 | 0.023 |
| 512 | 0.242 | 0.116 |
| 1024 | 1.296 | 2.376 |
| 2048 | 4.965 | 5.295 |

## Interpretation

NumPy and PyTorch use heavily tuned vendor kernels, vectorized elementwise loops, packing, cache-aware microkernels, and hardware-specific acceleration. VectorCore CPU performs generic coordinate reconstruction for each broadcast output and uses a straightforward blocked GEMM, so the gaps are expected. Metal dispatch overhead dominates small kernels. At 512 and above, VectorCore Metal is faster than VectorCore CPU, but its simple 16×16 GEMM and synchronous command submission remain well behind PyTorch MPS. These numbers are baselines for engineering work, not performance claims.
