#include "vectorcore/cpu_backend.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>

namespace vectorcore::cpu {

namespace {

float apply_binary(float a, float b, BinaryOp op) {
    switch (op) {
        case BinaryOp::Add: return a + b;
        case BinaryOp::Subtract: return a - b;
        case BinaryOp::Multiply: return a * b;
        case BinaryOp::Divide: return a / b;
    }
    throw std::logic_error("unknown binary operation");
}

} // namespace

void binary(const float* a, const Shape& a_shape, const Strides& a_strides,
            const float* b, const Shape& b_shape, const Strides& b_strides,
            float* out, const Shape& out_shape, BinaryOp operation) {
    const auto count = vectorcore::numel(out_shape);
    const auto rank = out_shape.size();
    for (std::size_t linear = 0; linear < count; ++linear) {
        auto remainder = linear;
        std::size_t a_offset = 0;
        std::size_t b_offset = 0;
        for (std::size_t axis = rank; axis-- > 0;) {
            const auto coordinate = out_shape[axis] == 0 ? 0 : remainder % out_shape[axis];
            if (out_shape[axis] != 0) remainder /= out_shape[axis];
            if (axis >= rank - a_shape.size()) {
                const auto input_axis = axis - (rank - a_shape.size());
                if (a_shape[input_axis] != 1) a_offset += coordinate * a_strides[input_axis];
            }
            if (axis >= rank - b_shape.size()) {
                const auto input_axis = axis - (rank - b_shape.size());
                if (b_shape[input_axis] != 1) b_offset += coordinate * b_strides[input_axis];
            }
        }
        out[linear] = apply_binary(a[a_offset], b[b_offset], operation);
    }
}

void unary(const float* input, float* output, std::size_t count, UnaryOp operation) {
    for (std::size_t i = 0; i < count; ++i) {
        switch (operation) {
            case UnaryOp::Negate: output[i] = -input[i]; break;
            case UnaryOp::Relu: output[i] = std::max(input[i], 0.0F); break;
            case UnaryOp::Sigmoid: {
                const float x = input[i];
                output[i] = x >= 0.0F ? 1.0F / (1.0F + std::exp(-x))
                                      : std::exp(x) / (1.0F + std::exp(x));
                break;
            }
            case UnaryOp::ReluGradient: output[i] = input[i] > 0.0F ? 1.0F : 0.0F; break;
        }
    }
}

void matmul(const float* a, const float* b, float* out,
            std::size_t m, std::size_t k, std::size_t n) {
    std::fill(out, out + m * n, 0.0F);
    constexpr std::size_t block = 32;
    auto worker = [&](std::size_t row_begin, std::size_t row_end) {
        for (std::size_t ii = row_begin; ii < row_end; ii += block) {
            for (std::size_t kk = 0; kk < k; kk += block) {
                for (std::size_t jj = 0; jj < n; jj += block) {
                    const auto i_end = std::min(ii + block, row_end);
                    const auto k_end = std::min(kk + block, k);
                    const auto j_end = std::min(jj + block, n);
                    for (std::size_t i = ii; i < i_end; ++i) {
                        for (std::size_t p = kk; p < k_end; ++p) {
                            const auto av = a[i * k + p];
                            for (std::size_t j = jj; j < j_end; ++j) {
                                out[i * n + j] += av * b[p * n + j];
                            }
                        }
                    }
                }
            }
        }
    };
    const auto hardware_threads = std::max(1U, std::thread::hardware_concurrency());
    const auto thread_count = (m * n * k >= 2'000'000 && m > 1)
        ? std::min<std::size_t>(m, hardware_threads) : 1;
    if (thread_count == 1) {
        worker(0, m);
        return;
    }
    std::vector<std::thread> workers;
    const auto rows_per_thread = (m + thread_count - 1) / thread_count;
    for (std::size_t thread = 0; thread < thread_count; ++thread) {
        const auto begin = thread * rows_per_thread;
        const auto end = std::min(begin + rows_per_thread, m);
        if (begin < end) workers.emplace_back(worker, begin, end);
    }
    for (auto& thread : workers) thread.join();
}

float sum(const float* input, std::size_t count) {
    // Multiple accumulators reduce the dependency chain while retaining a
    // deterministic order independent of the host thread count.
    double accumulators[4]{};
    std::size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        accumulators[0] += input[i];
        accumulators[1] += input[i + 1];
        accumulators[2] += input[i + 2];
        accumulators[3] += input[i + 3];
    }
    double total = accumulators[0] + accumulators[1] + accumulators[2] + accumulators[3];
    for (; i < count; ++i) total += input[i];
    return static_cast<float>(total);
}

} // namespace vectorcore::cpu
