#pragma once

#include "vectorcore/tensor.hpp"

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

} // namespace vectorcore
