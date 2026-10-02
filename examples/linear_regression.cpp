#include "vectorcore/vectorcore.hpp"

#include <iostream>

int main() {
    using namespace vectorcore;
    Tensor x({-2, -1, 0, 1, 2}, {5, 1});
    Tensor y({-3, -1, 1, 3, 5}, {5, 1}); // y = 2x + 1
    nn::Linear model(1, 1, 42);
    for (int epoch = 0; epoch <= 150; ++epoch) {
        auto loss = nn::mse_loss(model(x), y);
        if (epoch % 30 == 0) std::cout << "epoch=" << epoch << " loss=" << loss.item() << '\n';
        loss.backward();
        nn::sgd_step(model.parameters(), 0.08F);
        nn::zero_grad(model.parameters());
    }
    std::cout << "weight=" << model.weight.item() << " bias=" << model.bias.item() << '\n';
}
