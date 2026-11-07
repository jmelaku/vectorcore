#include "vectorcore/nn.hpp"

#include "vectorcore/autograd.hpp"
#include "vectorcore/detail/tensor_impl.hpp"
#include "vectorcore/metal_backend.hpp"
#include "vectorcore/ops.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

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

Tensor cross_entropy(const Tensor& logits, const std::vector<std::size_t>& targets) {
    if (logits.ndim() != 2) {
        throw std::invalid_argument("cross_entropy expects [batch, classes] logits, got " +
                                    shape_string(logits.shape()));
    }
    const auto batch = logits.shape()[0];
    const auto classes = logits.shape()[1];
    if (targets.size() != batch) {
        throw std::invalid_argument("cross_entropy received " + std::to_string(targets.size()) +
                                    " targets for a batch of " + std::to_string(batch));
    }
    if (batch == 0) throw std::invalid_argument("cross_entropy of an empty batch is undefined");
    std::vector<float> one_hot(batch * classes, 0.0F);
    for (std::size_t row = 0; row < batch; ++row) {
        if (targets[row] >= classes) {
            throw std::out_of_range("target class " + std::to_string(targets[row]) +
                                    " is out of range for " + std::to_string(classes) + " classes");
        }
        one_hot[row * classes + targets[row]] = 1.0F;
    }
    const Tensor mask(std::move(one_hot), {batch, classes}, false, logits.device());
    return -(logits.log_softmax(1) * mask).sum() / static_cast<float>(batch);
}

void sgd_step(const std::vector<Tensor*>& parameters, float learning_rate) {
    NoGradGuard guard;
    for (auto* parameter : parameters) {
        if (!parameter || !parameter->has_grad()) continue;
        const auto gradient = parameter->grad();
        if (gradient.shape() != parameter->shape() || gradient.device() != parameter->device()) {
            throw std::invalid_argument("parameter gradient shape or device does not match");
        }
        if (parameter->device() == Device::metal()) {
            auto& storage = *parameter->impl()->storage;
            MetalBackend::instance().axpy_inplace(storage, *gradient.impl()->storage,
                                                  parameter->numel(), -learning_rate);
            storage.bump_version();  // in-place update, same rule as mutable_data()
            continue;
        }
        const auto values_in = gradient.to_vector();
        auto* values = parameter->mutable_data();
        for (std::size_t i = 0; i < parameter->numel(); ++i) {
            values[i] -= learning_rate * values_in[i];
        }
    }
}

void zero_grad(const std::vector<Tensor*>& parameters) {
    for (auto* parameter : parameters) if (parameter) parameter->zero_grad();
}

} // namespace vectorcore::nn
