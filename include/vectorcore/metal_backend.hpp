#pragma once

#include "vectorcore/cpu_backend.hpp"
#include "vectorcore/shape.hpp"
#include "vectorcore/storage.hpp"

#include <memory>
#include <string>
#include <vector>

namespace vectorcore {

class MetalBackend {
public:
    static MetalBackend& instance();
    virtual ~MetalBackend() = default;

    [[nodiscard]] virtual bool available() const noexcept = 0;
    [[nodiscard]] virtual std::string device_name() const = 0;
    [[nodiscard]] virtual std::string last_error() const = 0;
    // "simdgroup" (hardware matrix units), "tiled" (portable), or "unavailable".
    [[nodiscard]] virtual std::string gemm_kernel() const = 0;
    [[nodiscard]] virtual std::shared_ptr<Storage> allocate(std::size_t elements) = 0;
    [[nodiscard]] virtual std::shared_ptr<Storage> upload(const float* data,
                                                          std::size_t elements) = 0;
    virtual void download(const Storage& source, float* destination,
                          std::size_t elements) = 0;
    [[nodiscard]] virtual std::shared_ptr<Storage> binary(
        const Storage& a, const Shape& a_shape, const Strides& a_strides,
        const Storage& b, const Shape& b_shape, const Strides& b_strides,
        const Shape& out_shape, cpu::BinaryOp operation) = 0;
    [[nodiscard]] virtual std::shared_ptr<Storage> unary(
        const Storage& input, std::size_t count, cpu::UnaryOp operation) = 0;
    // Batched GEMM: output batch z is A[a_offsets[z] ...] @ B[b_offsets[z] ...],
    // each an (m x k) @ (k x n) product; offsets are in elements.
    [[nodiscard]] virtual std::shared_ptr<Storage> matmul(
        const Storage& a, const Storage& b, std::size_t m, std::size_t k, std::size_t n,
        const std::vector<std::size_t>& a_offsets,
        const std::vector<std::size_t>& b_offsets,
        bool trans_a = false, bool trans_b = false) = 0;
    // Swap the last two axes of a contiguous [batch, rows, cols] tensor.
    [[nodiscard]] virtual std::shared_ptr<Storage> transpose_last2(
        const Storage& input, std::size_t batch, std::size_t rows, std::size_t cols) = 0;
    [[nodiscard]] virtual std::shared_ptr<Storage> reduce_sum(
        const Storage& input, std::size_t count) = 0;
    [[nodiscard]] virtual std::shared_ptr<Storage> reduce_axis(
        const Storage& input, std::size_t outer, std::size_t length, std::size_t inner,
        cpu::ReduceOp operation) = 0;
    [[nodiscard]] virtual std::shared_ptr<Storage> scatter_axis(
        const Storage& values, const Storage& indices, std::size_t outer,
        std::size_t length, std::size_t inner) = 0;
    // target += alpha * source, in place.
    virtual void axpy_inplace(Storage& target, const Storage& source,
                              std::size_t count, float alpha) = 0;
    virtual void synchronize() = 0;
};

[[nodiscard]] bool metal_available();
[[nodiscard]] std::string metal_device_name();
// Block until all queued GPU work has finished and surface any GPU error.
// Metal operations are asynchronous; reading a tensor's values (item(),
// to_vector(), to(cpu)) synchronizes automatically. No-op without Metal.
void synchronize();

} // namespace vectorcore
