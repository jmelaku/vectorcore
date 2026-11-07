#include "vectorcore/vectorcore.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vectorcore;

namespace {

int failures = 0;
int tests = 0;

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void close(const Tensor& actual, const std::vector<float>& expected, float tolerance = 1e-5F) {
    const auto values = actual.to_vector();
    check(values.size() == expected.size(), "tensor size mismatch");
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (std::abs(values[i] - expected[i]) > tolerance) {
            throw std::runtime_error("value mismatch at " + std::to_string(i) + ": " +
                                     std::to_string(values[i]) + " vs " + std::to_string(expected[i]));
        }
    }
}

template <typename Function>
void test(const std::string& name, Function function) {
    ++tests;
    try {
        function();
        std::cout << "[PASS] " << name << '\n';
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
    }
}

template <typename Function>
void throws(Function function) {
    bool caught = false;
    try { function(); } catch (const std::exception&) { caught = true; }
    check(caught, "expected an exception");
}

} // namespace

int main() {
    test("shape, strides, scalar, indexing", [] {
        Tensor value({1, 2, 3, 4, 5, 6}, {2, 3});
        check(value.shape() == Shape({2, 3}), "shape mismatch");
        check(value.strides() == Strides({3, 1}), "stride mismatch");
        check(value.at({1, 2}) == 6.0F, "index mismatch");
        value.set_at({0, 1}, 9.0F);
        check(value.at({0, 1}) == 9.0F, "set_at mismatch");
        check(Tensor::scalar(4.0F).shape().empty(), "scalar rank mismatch");
        throws([&] { (void)value.at({3, 0}); });
        throws([] { Tensor({1, 2}, {3}); });
    });

    test("reshape shares storage and flatten", [] {
        Tensor value({1, 2, 3, 4}, {2, 2});
        auto flat = value.flatten();
        flat.set_at({2}, 8.0F);
        check(value.at({1, 0}) == 8.0F, "reshape did not share storage");
        check(flat.shape() == Shape({4}), "flatten shape mismatch");
        throws([&] { (void)value.reshape({3, 2}); });
    });

    test("elementwise and scalar operations", [] {
        Tensor a({1, 2, 3, 4}, {2, 2});
        Tensor b({4, 3, 2, 1}, {2, 2});
        close(a + b, {5, 5, 5, 5});
        close(a - b, {-3, -1, 1, 3});
        close(a * b, {4, 6, 6, 4});
        close(a / b, {0.25F, 2.0F / 3.0F, 1.5F, 4});
        close(-a, {-1, -2, -3, -4});
        close(2.0F + a * 2.0F, {4, 6, 8, 10});
    });

    test("broadcasting", [] {
        Tensor matrix({1, 2, 3, 4, 5, 6}, {2, 3});
        close(matrix + Tensor({10, 20, 30}, {3}), {11, 22, 33, 14, 25, 36});
        close(matrix + Tensor({10, 20, 30}, {1, 3}), {11, 22, 33, 14, 25, 36});
        close(matrix + Tensor::scalar(2), {3, 4, 5, 6, 7, 8});
        Tensor high(std::vector<float>(24, 1.0F), {2, 3, 4});
        close(high * Tensor({1, 2, 3, 4}, {1, 4}),
              {1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4});
        throws([&] { (void)(matrix + Tensor::ones({2, 2})); });
    });

    test("matmul and transpose", [] {
        Tensor a({1, 2, 3, 4, 5, 6}, {2, 3});
        Tensor b({7, 8, 9, 10, 11, 12}, {3, 2});
        close(a.matmul(b), {58, 64, 139, 154});
        close(a.transpose(), {1, 4, 2, 5, 3, 6});
        throws([&] { (void)a.matmul(Tensor::ones({4, 2})); });
    });

    test("reductions and activations", [] {
        Tensor value({-2, -1, 0, 1, 2}, {5});
        check(value.sum().item() == 0.0F, "sum mismatch");
        check(value.mean().item() == 0.0F, "mean mismatch");
        close(value.relu(), {0, 0, 0, 1, 2});
        const auto sigmoid = value.sigmoid().to_vector();
        check(std::abs(sigmoid[2] - 0.5F) < 1e-6F, "sigmoid mismatch");
        check(Tensor::zeros({0}).sum().item() == 0.0F, "empty sum mismatch");
        throws([] { (void)Tensor::zeros({0}).mean(); });
    });

    test("autograd scalar and repeated paths", [] {
        Tensor x({1, 2, 3}, {3}, true);
        ((x * x) + x).sum().backward();
        close(x.grad(), {3, 5, 7});
        x.zero_grad();
        check(!x.has_grad(), "zero_grad failed");
    });

    test("broadcast gradient reduction", [] {
        Tensor x({1, 2, 3, 4, 5, 6}, {2, 3}, true);
        Tensor bias({0.1F, 0.2F, 0.3F}, {3}, true);
        (x + bias).sum().backward();
        close(x.grad(), {1, 1, 1, 1, 1, 1});
        close(bias.grad(), {2, 2, 2});
    });

    test("matmul gradients", [] {
        Tensor a({1, 2, 3, 4, 5, 6}, {2, 3}, true);
        Tensor b({1, 2, 3, 4, 5, 6}, {3, 2}, true);
        a.matmul(b).sum().backward();
        close(a.grad(), {3, 7, 11, 3, 7, 11});
        close(b.grad(), {5, 5, 7, 7, 9, 9});
    });

    test("sigmoid and reshape gradients", [] {
        Tensor x({-1, 0, 1, 2}, {2, 2}, true);
        x.reshape({4}).sigmoid().sum().backward();
        const auto expected_y = x.detach().sigmoid().to_vector();
        std::vector<float> expected;
        for (float y : expected_y) expected.push_back(y * (1.0F - y));
        close(x.grad(), expected, 1e-5F);
    });

    test("finite-difference gradient", [] {
        const std::vector<float> base{0.3F, -0.7F, 1.2F};
        Tensor x(base, {3}, true);
        ((x * x).sigmoid().sum()).backward();
        const auto analytic = x.grad().to_vector();
        const float epsilon = 1e-3F;
        for (std::size_t i = 0; i < base.size(); ++i) {
            auto plus = base; plus[i] += epsilon;
            auto minus = base; minus[i] -= epsilon;
            const float numerical = ((Tensor(plus, {3}) * Tensor(plus, {3})).sigmoid().sum().item() -
                                     (Tensor(minus, {3}) * Tensor(minus, {3})).sigmoid().sum().item()) /
                                    (2.0F * epsilon);
            check(std::abs(analytic[i] - numerical) < 3e-3F, "finite difference mismatch");
        }
    });

    test("gradient accumulation across backward calls", [] {
        Tensor x({1, 2}, {2}, true);
        (x * 3.0F).sum().backward();
        (x * 4.0F).sum().backward();
        close(x.grad(), {7, 7});
    });

    test("linear regression training", [] {
        Tensor input({-2, -1, 0, 1, 2}, {5, 1});
        Tensor target({-3, -1, 1, 3, 5}, {5, 1});
        nn::Linear model(1, 1, 7);
        const float initial = nn::mse_loss(model(input), target).item();
        for (int epoch = 0; epoch < 150; ++epoch) {
            auto loss = nn::mse_loss(model(input), target);
            loss.backward();
            nn::sgd_step(model.parameters(), 0.08F);
            nn::zero_grad(model.parameters());
        }
        const float final = nn::mse_loss(model(input), target).item();
        check(final < initial * 1e-3F && final < 1e-4F, "training did not converge");
    });

    test("device mismatch and unavailable behavior", [] {
        if (!metal_available()) {
            throws([] { (void)Tensor::ones({2}).to(Device::metal()); });
        }
    });

    test("Metal CPU parity", [] {
        if (!metal_available()) {
            std::cout << "       Metal unavailable: " << MetalBackend::instance().last_error() << '\n';
            return;
        }
        Tensor a({1, 2, 3, 4, 5, 6}, {2, 3});
        Tensor b({10, 20, 30}, {3});
        auto ma = a.to(Device::metal());
        auto mb = b.to(Device::metal());
        close((ma + mb).to(Device::cpu()), (a + b).to_vector(), 1e-5F);
        close((ma * mb).relu().to(Device::cpu()), (a * b).relu().to_vector(), 1e-5F);
        check(std::abs(ma.sum().item() - a.sum().item()) < 1e-4F, "Metal sum mismatch");
        Tensor left({1,2,3,4,5,6}, {2,3});
        Tensor right({7,8,9,10,11,12}, {3,2});
        close(left.to(Device::metal()).matmul(right.to(Device::metal())).to(Device::cpu()),
              left.matmul(right).to_vector(), 1e-4F);
        close(ma.transpose().to(Device::cpu()), a.transpose().to_vector(), 1e-5F);

        Tensor gx({-1, 2, 3, -4, 5, 6}, {2, 3}, true, Device::metal());
        Tensor gb({0.5F, 1.0F, -2.0F}, {3}, true, Device::metal());
        ((gx + gb).relu().sum()).backward();
        close(gx.grad().to(Device::cpu()), {0, 1, 1, 0, 1, 1});
        close(gb.grad().to(Device::cpu()), {0, 2, 2});
    });

    test("version counter is shared by views and bumped by writes", [] {
        Tensor value({1, 2, 3, 4}, {2, 2});
        auto view = value.reshape({4});
        const auto start = value.version();
        (void)(value + value);
        check(value.version() == start, "out-of-place op bumped the version");
        view.set_at({0}, 5.0F);
        check(value.version() == start + 1, "write did not bump the version");
        check(view.version() == value.version(), "views do not share a version");
    });

    test("in-place update between forward and backward is rejected", [] {
        Tensor w({2.0F}, {1}, true);
        Tensor x({3.0F}, {1}, true);
        auto loss = (w * x).sum();
        w.set_at({0}, 1.0F);
        throws([&] { loss.backward(); });

        Tensor a({1, 2, 3, 4}, {2, 2}, true);
        Tensor b({1, 0, 0, 1}, {2, 2}, true);
        auto product = a.matmul(b).sum();
        (b * 1.0F).sum().backward();  // give b a gradient so sgd_step writes to it
        nn::sgd_step({&b}, 0.0F);     // any optimizer step counts as a write
        throws([&] { product.backward(); });

        Tensor s({0.5F, -0.5F}, {2}, true);
        auto y = s.sigmoid();
        auto sigmoid_loss = y.sum();
        y.detach().set_at({0}, 0.0F);  // corrupts sigmoid's saved output
        throws([&] { sigmoid_loss.backward(); });
    });

    test("only leaves keep gradients unless retain_grad is used", [] {
        Tensor a({1, 2}, {2}, true);
        auto hidden = a * 2.0F;
        auto kept = a * 3.0F;
        kept.retain_grad();
        ((hidden * hidden).sum() + kept.sum()).backward();
        check(a.is_leaf() && !hidden.is_leaf(), "leaf flags wrong");
        check(!hidden.has_grad(), "intermediate kept a gradient");
        close(kept.grad(), {1, 1});
        close(a.grad(), {8 + 3, 16 + 3});
        throws([] { Tensor({1.0F}, {1}).retain_grad(); });
    });

    test("axis reductions, exp/log and softmax", [] {
        Tensor x({1, 5, 3, 4, 2, 6}, {2, 3});
        close(x.sum(0), {5, 7, 9});
        close(x.sum(-1, true), {9, 12});
        check(x.sum(1, true).shape() == Shape({2, 1}), "keepdim shape");
        close(x.mean(1), {3, 4});
        close(x.max(1), {5, 6});
        close(x.argmax(1), {1, 2});
        close(x.max(), {6});
        close(Tensor({0.0F, 1.0F}, {2}).exp(), {1.0F, std::exp(1.0F)});
        close(Tensor({1.0F, std::exp(2.0F)}, {2}).log(), {0.0F, 2.0F}, 1e-5F);
        const auto probabilities = x.softmax(1);
        close(probabilities.sum(1), {1, 1}, 1e-6F);
        throws([&] { (void)x.sum(2); });
        throws([&] { (void)x.max(-3); });
        throws([] { (void)Tensor::scalar(1.0F).sum(0); });
    });

    test("cross-entropy value and gradient", [] {
        Tensor logits({0, 0, 0, 0}, {2, 2}, true);
        auto loss = nn::cross_entropy(logits, {0, 1});
        check(std::abs(loss.item() - std::log(2.0F)) < 1e-6F, "uniform CE should be log 2");
        loss.backward();
        // d/dlogits = (softmax - one_hot) / batch
        close(logits.grad(), {-0.25F, 0.25F, 0.25F, -0.25F});
        throws([&] { (void)nn::cross_entropy(logits, {0}); });
        throws([&] { (void)nn::cross_entropy(logits, {0, 2}); });
    });

    test("batched and broadcast matmul", [] {
        // Two batches of [1x2] @ shared [2x2].
        Tensor a({1, 2, 3, 4}, {2, 1, 2}, true);
        Tensor w({1, 0, 0, 2}, {2, 2}, true);
        auto y = a.matmul(w);
        check(y.shape() == Shape({2, 1, 2}), "batched output shape");
        close(y, {1, 4, 3, 8});
        y.sum().backward();
        close(w.grad(), {4, 4, 6, 6});  // summed over both batches
        close(a.grad(), {1, 2, 1, 2});
        close(Tensor({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2}).transpose(), {1, 3, 2, 4, 5, 7, 6, 8});
        throws([] { (void)Tensor::ones({2, 2, 3}).matmul(Tensor::ones({3, 3, 1})); });
    });

    test("Metal classifier ops and SGD", [] {
        if (!metal_available()) return;
        Tensor x({1, 5, 3, 4, 2, 6}, {2, 3});
        auto mx = x.to(Device::metal());
        close(mx.sum(1).to(Device::cpu()), {9, 12});
        close(mx.max(0).to(Device::cpu()), {4, 5, 6});
        close(mx.argmax(1).to(Device::cpu()), {1, 2});
        close(mx.softmax(1).to(Device::cpu()), x.softmax(1).to_vector(), 1e-5F);
        Tensor batched({1, 2, 3, 4}, {2, 1, 2});
        Tensor w({1, 0, 0, 2}, {2, 2});
        close(batched.to(Device::metal()).matmul(w.to(Device::metal())).to(Device::cpu()),
              {1, 4, 3, 8});

        Tensor p({1, 2}, {2}, true, Device::metal());
        (p * p).sum().backward();
        nn::sgd_step({&p}, 0.5F);
        close(p.to(Device::cpu()), {0, 0});  // p - 0.5 * 2p
    });

    std::cout << tests - failures << "/" << tests << " tests passed\n";
    return failures == 0 ? 0 : 1;
}
