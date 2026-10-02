#include "vectorcore/vectorcore.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

using Clock = std::chrono::steady_clock;

template <typename Function>
double median_ms(Function function, int warmups = 2, int repeats = 7) {
    for (int i = 0; i < warmups; ++i) function();
    std::vector<double> samples;
    for (int i = 0; i < repeats; ++i) {
        const auto start = Clock::now();
        function();
        samples.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

int main() {
    using namespace vectorcore;
    std::cout << "size,operation,device,median_ms\n";
    for (std::size_t size : {128U, 256U, 512U}) {
        auto a = Tensor::randn({size, size}, false, 1);
        auto b = Tensor::randn({size, size}, false, 2);
        volatile float sink = 0.0F;
        const auto add_ms = median_ms([&] { sink = (a + b).at({0, 0}); });
        const auto mul_ms = median_ms([&] { sink = (a * b).at({0, 0}); });
        const auto mm_ms = median_ms([&] { sink = a.matmul(b).at({0, 0}); });
        std::cout << size << ",add,cpu," << add_ms << '\n';
        std::cout << size << ",multiply,cpu," << mul_ms << '\n';
        std::cout << size << ",matmul,cpu," << mm_ms << '\n';
        if (metal_available()) {
            auto ma = a.to(Device::metal());
            auto mb = b.to(Device::metal());
            const auto gpu_add = median_ms([&] { sink = (ma + mb).at({0, 0}); }, 0, 1);
            (void)gpu_add; // Python harness is the authoritative Metal benchmark.
        }
        (void)sink;
    }
}
