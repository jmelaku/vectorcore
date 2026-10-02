#pragma once

#include "vectorcore/tensor.hpp"

#include <cstdint>
#include <vector>

namespace vectorcore::nn {

class Linear {
public:
    Linear(std::size_t in_features, std::size_t out_features, std::uint64_t seed = 0);
    Tensor operator()(const Tensor& input) const;
    std::vector<Tensor*> parameters();
    Tensor weight;
    Tensor bias;
};

Tensor mse_loss(const Tensor& prediction, const Tensor& target);
void sgd_step(const std::vector<Tensor*>& parameters, float learning_rate);
void zero_grad(const std::vector<Tensor*>& parameters);

} // namespace vectorcore::nn
