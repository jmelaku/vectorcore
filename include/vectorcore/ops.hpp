#pragma once

#include "vectorcore/tensor.hpp"

#include <cstdint>

namespace vectorcore {

Tensor add(const Tensor& a, const Tensor& b);
Tensor subtract(const Tensor& a, const Tensor& b);
Tensor multiply(const Tensor& a, const Tensor& b);
Tensor divide(const Tensor& a, const Tensor& b);
Tensor negate(const Tensor& value);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor transpose(const Tensor& value);
Tensor sum(const Tensor& value);
Tensor mean(const Tensor& value);
Tensor relu(const Tensor& value);
Tensor sigmoid(const Tensor& value);
Tensor exp(const Tensor& value);
Tensor log(const Tensor& value);

// Reductions along one axis (negative axes count from the end).
Tensor sum(const Tensor& value, std::int64_t axis, bool keepdim = false);
Tensor mean(const Tensor& value, std::int64_t axis, bool keepdim = false);
Tensor max(const Tensor& value);
Tensor max(const Tensor& value, std::int64_t axis, bool keepdim = false);
// Indices of the maximum along an axis, as float32; not differentiable.
Tensor argmax(const Tensor& value, std::int64_t axis, bool keepdim = false);

Tensor softmax(const Tensor& value, std::int64_t axis = -1);
Tensor log_softmax(const Tensor& value, std::int64_t axis = -1);

} // namespace vectorcore
