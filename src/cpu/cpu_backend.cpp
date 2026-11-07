#include "vectorcore/cpu_backend.hpp"

#include "parallel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define VECTORCORE_NEON 1
#endif

namespace vectorcore::cpu {

namespace {

// Work below these sizes runs on the calling thread: waking the pool costs
// more than it saves.
constexpr std::size_t kElementwiseGrain = 1U << 15;
constexpr std::size_t kTranscendentalGrain = 1U << 13;

struct AddOp { static float apply(float a, float b) { return a + b; } };
struct SubtractOp { static float apply(float a, float b) { return a - b; } };
struct MultiplyOp { static float apply(float a, float b) { return a * b; } };
struct DivideOp { static float apply(float a, float b) { return a / b; } };

template <typename Op>
void binary_kernel(const float* a, const Shape& a_shape, const Strides& a_strides,
                   const float* b, const Shape& b_shape, const Strides& b_strides,
                   float* out, const Shape& out_shape) {
    const auto count = vectorcore::numel(out_shape);
    if (count == 0) return;

    // Fast path 1: identical shapes, all contiguous.
    if (a_shape == out_shape && b_shape == out_shape) {
        parallel_for(count, kElementwiseGrain, [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) out[i] = Op::apply(a[i], b[i]);
        });
        return;
    }
    // Fast path 2: one operand is a single value.
    if (vectorcore::numel(b_shape) == 1 && a_shape == out_shape) {
        const float scalar = b[0];
        parallel_for(count, kElementwiseGrain, [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) out[i] = Op::apply(a[i], scalar);
        });
        return;
    }
    if (vectorcore::numel(a_shape) == 1 && b_shape == out_shape) {
        const float scalar = a[0];
        parallel_for(count, kElementwiseGrain, [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) out[i] = Op::apply(scalar, b[i]);
        });
        return;
    }

    // General broadcast: resolve input offsets once per output row (last
    // axis), then stream along the row with a fixed step of 0 or 1.
    const auto rank = out_shape.size();
    const auto inner = rank == 0 ? 1 : out_shape[rank - 1];
    const auto rows = count / inner;
    auto last_step = [&](const Shape& shape, const Strides& strides) -> std::size_t {
        if (rank == 0 || shape.empty() || shape.back() == 1) return 0;
        return strides.back();
    };
    const auto a_step = last_step(a_shape, a_strides);
    const auto b_step = last_step(b_shape, b_strides);
    const auto row_grain = std::max<std::size_t>(1, kElementwiseGrain / std::max<std::size_t>(inner, 1));
    parallel_for(rows, row_grain, [&](std::size_t row_begin, std::size_t row_end) {
        for (std::size_t row = row_begin; row < row_end; ++row) {
            std::size_t remainder = row;
            std::size_t a_offset = 0;
            std::size_t b_offset = 0;
            for (std::size_t axis = rank == 0 ? 0 : rank - 1; axis-- > 0;) {
                const auto coordinate = remainder % out_shape[axis];
                remainder /= out_shape[axis];
                if (axis >= rank - a_shape.size()) {
                    const auto ia = axis - (rank - a_shape.size());
                    if (a_shape[ia] != 1) a_offset += coordinate * a_strides[ia];
                }
                if (axis >= rank - b_shape.size()) {
                    const auto ib = axis - (rank - b_shape.size());
                    if (b_shape[ib] != 1) b_offset += coordinate * b_strides[ib];
                }
            }
            const float* ar = a + a_offset;
            const float* br = b + b_offset;
            float* orow = out + row * inner;
            if (a_step == 1 && b_step == 1) {
                for (std::size_t j = 0; j < inner; ++j) orow[j] = Op::apply(ar[j], br[j]);
            } else if (a_step == 1 && b_step == 0) {
                const float bv = *br;
                for (std::size_t j = 0; j < inner; ++j) orow[j] = Op::apply(ar[j], bv);
            } else if (a_step == 0 && b_step == 1) {
                const float av = *ar;
                for (std::size_t j = 0; j < inner; ++j) orow[j] = Op::apply(av, br[j]);
            } else {
                for (std::size_t j = 0; j < inner; ++j) {
                    orow[j] = Op::apply(ar[j * a_step], br[j * b_step]);
                }
            }
        }
    });
}

template <typename Function>
void map_unary(const float* input, float* output, std::size_t count, std::size_t grain,
               Function function) {
    parallel_for(count, grain, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) output[i] = function(input[i]);
    });
}

// ---------------------------------------------------------------------------
// Matrix multiplication: packed panels + register-blocked microkernel
// (the BLIS/GotoBLAS structure). C (m x n) = A (m x k) @ B (k x n), row-major.
// ---------------------------------------------------------------------------

constexpr std::size_t MR = 8;    // microkernel rows
constexpr std::size_t NR = 12;   // microkernel columns
constexpr std::size_t KC = 256;  // depth of one packed panel (A sliver 8 KB, B sliver 12 KB)
constexpr std::size_t NC = 4092; // columns per packed B block (multiple of NR)
// One parallel task owns a TILE_ROWS x TILE_COLS block of C and runs the full
// depth for it, so C stays in L1 and tasks never synchronise with each other.
// Small tiles give enough tasks to balance performance and efficiency cores.
constexpr std::size_t TILE_ROWS = 32;  // 4 A slivers
constexpr std::size_t TILE_COLS = 48;  // 4 B slivers

// Strided operand views: element (r, c) lives at data[r * rs + c * cs], which
// covers both a row-major matrix and the transpose of one without copying.
struct MatrixView {
    const float* data;
    std::size_t rs;
    std::size_t cs;
    float at(std::size_t r, std::size_t c) const { return data[r * rs + c * cs]; }
};

// Packed A for every depth panel: panel kb, sliver s holds kc steps of MR
// contiguous row values (zero padded past m).
void pack_a(const MatrixView& a, std::size_t m, std::size_t k, float* packed) {
    const auto slivers = (m + MR - 1) / MR;
    const auto panels = (k + KC - 1) / KC;
    parallel_for(panels * slivers, 16, [&](std::size_t begin, std::size_t end) {
        for (std::size_t item = begin; item < end; ++item) {
            const auto kb = item / slivers;
            const auto s = item % slivers;
            const auto p0 = kb * KC;
            const auto kc = std::min(KC, k - p0);
            float* dst = packed + kb * slivers * KC * MR + s * kc * MR;
            const auto row0 = s * MR;
            const auto rows = std::min(MR, m - row0);
            if (rows < MR) std::fill(dst, dst + kc * MR, 0.0F);
            if (a.cs == 1) {  // rows contiguous in memory: read along p
                for (std::size_t r = 0; r < rows; ++r) {
                    const float* src = a.data + (row0 + r) * a.rs + p0;
                    for (std::size_t p = 0; p < kc; ++p) dst[p * MR + r] = src[p];
                }
            } else {
                for (std::size_t p = 0; p < kc; ++p) {
                    for (std::size_t r = 0; r < rows; ++r) dst[p * MR + r] = a.at(row0 + r, p0 + p);
                }
            }
        }
    });
}

// Packed B for every depth panel of columns [j0, j0 + nc): panel kb, sliver t
// holds kc steps of NR contiguous column values (zero padded past nc).
void pack_b(const MatrixView& b, std::size_t k, std::size_t j0, std::size_t nc, float* packed) {
    const auto slivers = (nc + NR - 1) / NR;
    const auto panels = (k + KC - 1) / KC;
    parallel_for(panels * slivers, 16, [&](std::size_t begin, std::size_t end) {
        for (std::size_t item = begin; item < end; ++item) {
            const auto kb = item / slivers;
            const auto t = item % slivers;
            const auto p0 = kb * KC;
            const auto kc = std::min(KC, k - p0);
            float* dst = packed + kb * slivers * KC * NR + t * kc * NR;
            const auto col0 = j0 + t * NR;
            const auto cols = std::min(NR, j0 + nc - col0);
            if (cols < NR) std::fill(dst, dst + kc * NR, 0.0F);
            if (b.cs == 1) {  // rows contiguous: copy NR values per step
                for (std::size_t p = 0; p < kc; ++p) {
                    const float* src = b.data + (p0 + p) * b.rs + col0;
                    std::copy(src, src + cols, dst + p * NR);
                }
            } else {  // transposed storage: columns are contiguous, read along p
                for (std::size_t c = 0; c < cols; ++c) {
                    const float* src = b.data + (col0 + c) * b.cs + p0 * b.rs;
                    for (std::size_t p = 0; p < kc; ++p) dst[p * NR + c] = src[p * b.rs];
                }
            }
        }
    });
}

// C[0:rows, 0:cols] += A_sliver (kc x MR) * B_sliver (kc x NR).
void microkernel(std::size_t kc, const float* a, const float* b, float* c, std::size_t ldc,
                 std::size_t rows, std::size_t cols) {
#if VECTORCORE_NEON
    float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0), c02 = vdupq_n_f32(0);
    float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0), c12 = vdupq_n_f32(0);
    float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0), c22 = vdupq_n_f32(0);
    float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0), c32 = vdupq_n_f32(0);
    float32x4_t c40 = vdupq_n_f32(0), c41 = vdupq_n_f32(0), c42 = vdupq_n_f32(0);
    float32x4_t c50 = vdupq_n_f32(0), c51 = vdupq_n_f32(0), c52 = vdupq_n_f32(0);
    float32x4_t c60 = vdupq_n_f32(0), c61 = vdupq_n_f32(0), c62 = vdupq_n_f32(0);
    float32x4_t c70 = vdupq_n_f32(0), c71 = vdupq_n_f32(0), c72 = vdupq_n_f32(0);
    for (std::size_t p = 0; p < kc; ++p) {
        const float32x4_t a0 = vld1q_f32(a);
        const float32x4_t a1 = vld1q_f32(a + 4);
        const float32x4_t b0 = vld1q_f32(b);
        const float32x4_t b1 = vld1q_f32(b + 4);
        const float32x4_t b2 = vld1q_f32(b + 8);
        c00 = vfmaq_laneq_f32(c00, b0, a0, 0); c01 = vfmaq_laneq_f32(c01, b1, a0, 0); c02 = vfmaq_laneq_f32(c02, b2, a0, 0);
        c10 = vfmaq_laneq_f32(c10, b0, a0, 1); c11 = vfmaq_laneq_f32(c11, b1, a0, 1); c12 = vfmaq_laneq_f32(c12, b2, a0, 1);
        c20 = vfmaq_laneq_f32(c20, b0, a0, 2); c21 = vfmaq_laneq_f32(c21, b1, a0, 2); c22 = vfmaq_laneq_f32(c22, b2, a0, 2);
        c30 = vfmaq_laneq_f32(c30, b0, a0, 3); c31 = vfmaq_laneq_f32(c31, b1, a0, 3); c32 = vfmaq_laneq_f32(c32, b2, a0, 3);
        c40 = vfmaq_laneq_f32(c40, b0, a1, 0); c41 = vfmaq_laneq_f32(c41, b1, a1, 0); c42 = vfmaq_laneq_f32(c42, b2, a1, 0);
        c50 = vfmaq_laneq_f32(c50, b0, a1, 1); c51 = vfmaq_laneq_f32(c51, b1, a1, 1); c52 = vfmaq_laneq_f32(c52, b2, a1, 1);
        c60 = vfmaq_laneq_f32(c60, b0, a1, 2); c61 = vfmaq_laneq_f32(c61, b1, a1, 2); c62 = vfmaq_laneq_f32(c62, b2, a1, 2);
        c70 = vfmaq_laneq_f32(c70, b0, a1, 3); c71 = vfmaq_laneq_f32(c71, b1, a1, 3); c72 = vfmaq_laneq_f32(c72, b2, a1, 3);
        a += MR;
        b += NR;
    }
    const float32x4_t acc[MR][3] = {{c00, c01, c02}, {c10, c11, c12}, {c20, c21, c22},
                                    {c30, c31, c32}, {c40, c41, c42}, {c50, c51, c52},
                                    {c60, c61, c62}, {c70, c71, c72}};
    if (rows == MR && cols == NR) {
        for (std::size_t r = 0; r < MR; ++r) {
            float* row = c + r * ldc;
            vst1q_f32(row, vaddq_f32(vld1q_f32(row), acc[r][0]));
            vst1q_f32(row + 4, vaddq_f32(vld1q_f32(row + 4), acc[r][1]));
            vst1q_f32(row + 8, vaddq_f32(vld1q_f32(row + 8), acc[r][2]));
        }
        return;
    }
    float tile[MR][NR];
    for (std::size_t r = 0; r < MR; ++r) {
        vst1q_f32(&tile[r][0], acc[r][0]);
        vst1q_f32(&tile[r][4], acc[r][1]);
        vst1q_f32(&tile[r][8], acc[r][2]);
    }
#else
    float tile[MR][NR] = {};
    for (std::size_t p = 0; p < kc; ++p) {
        for (std::size_t r = 0; r < MR; ++r) {
            const float av = a[r];
            for (std::size_t j = 0; j < NR; ++j) tile[r][j] += av * b[j];
        }
        a += MR;
        b += NR;
    }
#endif
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t j = 0; j < cols; ++j) c[r * ldc + j] += tile[r][j];
    }
}

void matmul_small(const MatrixView& a, const MatrixView& b, float* out,
                  std::size_t m, std::size_t k, std::size_t n) {
    for (std::size_t i = 0; i < m; ++i) {
        float* row = out + i * n;
        for (std::size_t p = 0; p < k; ++p) {
            const float av = a.at(i, p);
            for (std::size_t j = 0; j < n; ++j) row[j] += av * b.at(p, j);
        }
    }
}

} // namespace

void binary(const float* a, const Shape& a_shape, const Strides& a_strides,
            const float* b, const Shape& b_shape, const Strides& b_strides,
            float* out, const Shape& out_shape, BinaryOp operation) {
    switch (operation) {
        case BinaryOp::Add:
            return binary_kernel<AddOp>(a, a_shape, a_strides, b, b_shape, b_strides, out, out_shape);
        case BinaryOp::Subtract:
            return binary_kernel<SubtractOp>(a, a_shape, a_strides, b, b_shape, b_strides, out, out_shape);
        case BinaryOp::Multiply:
            return binary_kernel<MultiplyOp>(a, a_shape, a_strides, b, b_shape, b_strides, out, out_shape);
        case BinaryOp::Divide:
            return binary_kernel<DivideOp>(a, a_shape, a_strides, b, b_shape, b_strides, out, out_shape);
    }
    throw std::logic_error("unknown binary operation");
}

void unary(const float* input, float* output, std::size_t count, UnaryOp operation) {
    switch (operation) {
        case UnaryOp::Negate:
            return map_unary(input, output, count, kElementwiseGrain, [](float x) { return -x; });
        case UnaryOp::Relu:
            return map_unary(input, output, count, kElementwiseGrain,
                             [](float x) { return x > 0.0F ? x : 0.0F; });
        case UnaryOp::ReluGradient:
            return map_unary(input, output, count, kElementwiseGrain,
                             [](float x) { return x > 0.0F ? 1.0F : 0.0F; });
        case UnaryOp::Sigmoid:
            return map_unary(input, output, count, kTranscendentalGrain, [](float x) {
                // Branch keeps exp() from overflowing for large |x|.
                if (x >= 0.0F) return 1.0F / (1.0F + std::exp(-x));
                const float e = std::exp(x);
                return e / (1.0F + e);
            });
        case UnaryOp::Exp:
            return map_unary(input, output, count, kTranscendentalGrain,
                             [](float x) { return std::exp(x); });
        case UnaryOp::Log:
            return map_unary(input, output, count, kTranscendentalGrain,
                             [](float x) { return std::log(x); });
    }
    throw std::logic_error("unknown unary operation");
}

void matmul(const float* a_data, const float* b_data, float* out,
            std::size_t m, std::size_t k, std::size_t n, bool trans_a, bool trans_b) {
    std::fill(out, out + m * n, 0.0F);
    if (m == 0 || n == 0 || k == 0) return;
    const MatrixView a{a_data, trans_a ? 1 : k, trans_a ? m : 1};
    const MatrixView b{b_data, trans_b ? 1 : n, trans_b ? k : 1};
    // Packing costs O(mk + kn); for tiny products the direct loop wins.
    if (m * n * k <= 32U * 32U * 32U || m < 4) {
        matmul_small(a, b, out, m, k, n);
        return;
    }
    const auto panels = (k + KC - 1) / KC;
    const auto a_slivers = (m + MR - 1) / MR;
    std::vector<float> packed_a(panels * a_slivers * KC * MR);
    pack_a(a, m, k, packed_a.data());
    const auto max_b_slivers = (std::min(n, NC) + NR - 1) / NR;
    std::vector<float> packed_b(panels * max_b_slivers * KC * NR);
    for (std::size_t j0 = 0; j0 < n; j0 += NC) {
        const auto nc = std::min(NC, n - j0);
        const auto b_slivers = (nc + NR - 1) / NR;
        pack_b(b, k, j0, nc, packed_b.data());
        const auto tile_rows = (m + TILE_ROWS - 1) / TILE_ROWS;
        const auto tile_cols = (nc + TILE_COLS - 1) / TILE_COLS;
        const auto tasks = tile_rows * tile_cols;
        // Below ~2 MFLOP the pool wake-up costs more than it saves.
        const auto grain = m * nc * k >= (1U << 20) ? 1 : tasks;
        parallel_for(tasks, grain, [&](std::size_t t_begin, std::size_t t_end) {
            for (std::size_t task = t_begin; task < t_end; ++task) {
                const auto i_begin = (task / tile_cols) * TILE_ROWS;
                const auto j_begin = (task % tile_cols) * TILE_COLS;
                const auto i_end = std::min(i_begin + TILE_ROWS, m);
                const auto j_end = std::min(j_begin + TILE_COLS, nc);
                for (std::size_t kb = 0; kb < panels; ++kb) {
                    const auto kc = std::min(KC, k - kb * KC);
                    const float* a_panel = packed_a.data() + kb * a_slivers * KC * MR;
                    const float* b_panel = packed_b.data() + kb * b_slivers * KC * NR;
                    for (std::size_t j = j_begin; j < j_end; j += NR) {
                        const float* b_sliver = b_panel + (j / NR) * kc * NR;
                        for (std::size_t i = i_begin; i < i_end; i += MR) {
                            microkernel(kc, a_panel + (i / MR) * kc * MR, b_sliver,
                                        out + i * n + j0 + j, n,
                                        std::min(MR, m - i), std::min(NR, nc - j));
                        }
                    }
                }
            }
        });
    }
}

void transpose(const float* input, float* output, std::size_t batch,
               std::size_t rows, std::size_t cols) {
    // 32x32 tiles keep both the reads and the strided writes within cache.
    constexpr std::size_t tile = 32;
    const auto row_tiles = (rows + tile - 1) / tile;
    const auto work = batch * row_tiles;
    const auto grain = std::max<std::size_t>(1, (1U << 15) / std::max<std::size_t>(tile * cols, 1));
    parallel_for(work, grain, [&](std::size_t begin, std::size_t end) {
        for (std::size_t item = begin; item < end; ++item) {
            const auto z = item / row_tiles;
            const auto i0 = (item % row_tiles) * tile;
            const auto i1 = std::min(i0 + tile, rows);
            const float* src = input + z * rows * cols;
            float* dst = output + z * rows * cols;
            for (std::size_t j0 = 0; j0 < cols; j0 += tile) {
                const auto j1 = std::min(j0 + tile, cols);
                for (std::size_t i = i0; i < i1; ++i) {
                    for (std::size_t j = j0; j < j1; ++j) dst[j * rows + i] = src[i * cols + j];
                }
            }
        }
    });
}

float sum(const float* input, std::size_t count) {
    // Fixed-size chunks summed in double and combined in chunk order: the
    // result is identical whatever the thread count.
    constexpr std::size_t chunk = 1U << 16;
    const auto chunks = (count + chunk - 1) / chunk;
    std::vector<double> partial(chunks, 0.0);
    parallel_for(chunks, 1, [&](std::size_t c_begin, std::size_t c_end) {
        for (std::size_t c = c_begin; c < c_end; ++c) {
            const auto begin = c * chunk;
            const auto end = std::min(begin + chunk, count);
            double acc[4]{};
            std::size_t i = begin;
            for (; i + 4 <= end; i += 4) {
                acc[0] += input[i];
                acc[1] += input[i + 1];
                acc[2] += input[i + 2];
                acc[3] += input[i + 3];
            }
            double total = acc[0] + acc[1] + acc[2] + acc[3];
            for (; i < end; ++i) total += input[i];
            partial[c] = total;
        }
    });
    double total = 0.0;
    for (const auto value : partial) total += value;
    return static_cast<float>(total);
}

void reduce_axis(const float* input, float* output, std::size_t outer,
                 std::size_t length, std::size_t inner, ReduceOp operation) {
    if (length == 0 && operation != ReduceOp::Sum) {
        throw std::invalid_argument("max/argmax over an empty axis is undefined");
    }
    const auto grain = std::max<std::size_t>(1, kElementwiseGrain / std::max<std::size_t>(length * inner, 1));
    parallel_for(outer, grain, [&](std::size_t o_begin, std::size_t o_end) {
        std::vector<double> totals;
        std::vector<float> best_values;
        std::vector<std::size_t> best_indices;
        for (std::size_t o = o_begin; o < o_end; ++o) {
            const float* base = input + o * length * inner;
            float* out = output + o * inner;
            // Walk the reduced axis in the outer loop so reads stay contiguous.
            if (operation == ReduceOp::Sum) {
                totals.assign(inner, 0.0);
                for (std::size_t a = 0; a < length; ++a) {
                    const float* row = base + a * inner;
                    for (std::size_t i = 0; i < inner; ++i) totals[i] += row[i];
                }
                for (std::size_t i = 0; i < inner; ++i) out[i] = static_cast<float>(totals[i]);
                continue;
            }
            best_values.assign(base, base + inner);
            best_indices.assign(inner, 0);
            for (std::size_t a = 1; a < length; ++a) {
                const float* row = base + a * inner;
                for (std::size_t i = 0; i < inner; ++i) {
                    if (row[i] > best_values[i]) {
                        best_values[i] = row[i];
                        best_indices[i] = a;
                    }
                }
            }
            for (std::size_t i = 0; i < inner; ++i) {
                out[i] = operation == ReduceOp::Max ? best_values[i]
                                                    : static_cast<float>(best_indices[i]);
            }
        }
    });
}

void scatter_axis(const float* values, const float* indices, float* output,
                  std::size_t outer, std::size_t length, std::size_t inner) {
    std::fill(output, output + outer * length * inner, 0.0F);
    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t i = 0; i < inner; ++i) {
            const auto a = static_cast<std::size_t>(indices[o * inner + i]);
            output[(o * length + a) * inner + i] = values[o * inner + i];
        }
    }
}

} // namespace vectorcore::cpu
