#include "vectorcore/nn.hpp"

#include "vectorcore/autograd.hpp"
#include "vectorcore/ops.hpp"

#include <cmath>
#include <stdexcept>

namespace vectorcore::nn {

Linear::Linear(std::size_t in_features, std::size_t out_features, std::uint64_t seed)
    : weight(Tensor::randn({in_features, out_features}, true, seed) /
             std::sqrt(static_cast<float>(in_features))),
      bias(Tensor::zeros({out_features}, true)) {
    // Scaling creates a non-leaf; detach it so optimizers mutate a leaf parameter.
    weight = weight.detach();
    weight.set_requires_grad(true);
}

Tensor Linear::operator()(const Tensor& input) const { return input.matmul(weight) + bias; }
std::vector<Tensor*> Linear::parameters() { return {&weight, &bias}; }

Tensor mse_loss(const Tensor& prediction, const Tensor& target) {
    const auto error = prediction - target;
    return (error * error).mean();
}

void sgd_step(const std::vector<Tensor*>& parameters, float learning_rate) {
    NoGradGuard guard;
    for (auto* parameter : parameters) {
        if (!parameter || !parameter->has_grad()) continue;
        if (parameter->device() != Device::cpu()) {
            throw std::runtime_error("in-place SGD currently supports CPU parameters only");
        }
        const auto gradient = parameter->grad().to_vector();
        auto* values = parameter->mutable_data();
        for (std::size_t i = 0; i < parameter->numel(); ++i) {
            values[i] -= learning_rate * gradient[i];
        }
    }
}

void zero_grad(const std::vector<Tensor*>& parameters) {
    for (auto* parameter : parameters) if (parameter) parameter->zero_grad();
}

} // namespace vectorcore::nn
