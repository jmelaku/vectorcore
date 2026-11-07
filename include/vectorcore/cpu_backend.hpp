#pragma once

#include "vectorcore/shape.hpp"

#include <cstddef>

namespace vectorcore::cpu {

enum class BinaryOp { Add, Subtract, Multiply, Divide };
enum class UnaryOp { Negate, Relu, Sigmoid, ReluGradient, Exp, Log };
enum class ReduceOp { Sum, Max, Argmax };

void binary(const float* a, const Shape& a_shape, const Strides& a_strides,
            const float* b, const Shape& b_shape, const Strides& b_strides,
            float* out, const Shape& out_shape, BinaryOp operation);
void unary(const float* input, float* output, std::size_t count, UnaryOp operation);
void matmul(const float* a, const float* b, float* out,
            std::size_t m, std::size_t k, std::size_t n);
float sum(const float* input, std::size_t count);

// Reduce the middle axis of a contiguous [outer, length, inner] view into
// [outer, inner]. Argmax writes the winning index as a float (first wins ties).
void reduce_axis(const float* input, float* output, std::size_t outer,
                 std::size_t length, std::size_t inner, ReduceOp operation);

// Inverse of an argmax reduction: output is [outer, length, inner] and is zero
// except output[o, indices[o, i], i] = values[o, i].
void scatter_axis(const float* values, const float* indices, float* output,
                  std::size_t outer, std::size_t length, std::size_t inner);

} // namespace vectorcore::cpu
