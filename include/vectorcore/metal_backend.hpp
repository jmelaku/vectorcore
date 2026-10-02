#pragma once

#include "vectorcore/cpu_backend.hpp"
#include "vectorcore/shape.hpp"
#include "vectorcore/storage.hpp"

#include <memory>
#include <string>

namespace vectorcore {

class MetalBackend {
public:
    static MetalBackend& instance();
    virtual ~MetalBackend() = default;

    [[nodiscard]] virtual bool available() const noexcept = 0;
    [[nodiscard]] virtual std::string device_name() const = 0;
    [[nodiscard]] virtual std::string last_error() const = 0;
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
    [[nodiscard]] virtual std::shared_ptr<Storage> matmul(
        const Storage& a, const Storage& b, std::size_t m,
        std::size_t k, std::size_t n) = 0;
    [[nodiscard]] virtual std::shared_ptr<Storage> transpose2d(
        const Storage& input, std::size_t rows, std::size_t cols) = 0;
    [[nodiscard]] virtual std::shared_ptr<Storage> reduce_sum(
        const Storage& input, std::size_t count) = 0;
    [[nodiscard]] virtual std::shared_ptr<Storage> reduce_to_shape(
        const Storage& input, const Shape& input_shape,
        const Shape& target_shape) = 0;
    virtual void synchronize() = 0;
};

[[nodiscard]] bool metal_available();
[[nodiscard]] std::string metal_device_name();

} // namespace vectorcore
