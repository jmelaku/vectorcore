#pragma once

#include "vectorcore/shape.hpp"

#include <cstddef>

namespace vectorcore::cpu {

enum class BinaryOp { Add, Subtract, Multiply, Divide };
enum class UnaryOp { Negate, Relu, Sigmoid, ReluGradient };

void binary(const float* a, const Shape& a_shape, const Strides& a_strides,
            const float* b, const Shape& b_shape, const Strides& b_strides,
            float* out, const Shape& out_shape, BinaryOp operation);
void unary(const float* input, float* output, std::size_t count, UnaryOp operation);
void matmul(const float* a, const float* b, float* out,
            std::size_t m, std::size_t k, std::size_t n);
float sum(const float* input, std::size_t count);

} // namespace vectorcore::cpu
