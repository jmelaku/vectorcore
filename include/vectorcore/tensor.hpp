#pragma once

#include "vectorcore/device.hpp"
#include "vectorcore/shape.hpp"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vectorcore {

namespace detail { struct TensorImpl; }

enum class DType { Float32 };

class Tensor {
public:
    Tensor();
    // Internal handle constructor; public so backend translation units can create
    // tensors without exposing TensorImpl's fields in the public API.
    explicit Tensor(std::shared_ptr<detail::TensorImpl> impl);
    Tensor(std::vector<float> values, Shape shape, bool requires_grad = false,
           Device device = Device::cpu());

    static Tensor scalar(float value, bool requires_grad = false,
                         Device device = Device::cpu());
    static Tensor zeros(const Shape& shape, bool requires_grad = false,
                        Device device = Device::cpu());
    static Tensor ones(const Shape& shape, bool requires_grad = false,
                       Device device = Device::cpu());
    static Tensor randn(const Shape& shape, bool requires_grad = false,
                        std::uint64_t seed = 0, Device device = Device::cpu());

    [[nodiscard]] const Shape& shape() const;
    [[nodiscard]] const Strides& strides() const;
    [[nodiscard]] std::size_t ndim() const;
    [[nodiscard]] std::size_t numel() const;
    [[nodiscard]] Device device() const;
    [[nodiscard]] DType dtype() const;
    [[nodiscard]] bool is_contiguous() const;
    [[nodiscard]] bool requires_grad() const;
    void set_requires_grad(bool value);

    [[nodiscard]] const float* data() const;
    [[nodiscard]] float* mutable_data();
    [[nodiscard]] std::vector<float> to_vector() const;
    [[nodiscard]] float item() const;
    [[nodiscard]] float at(const std::vector<std::size_t>& indices) const;
    void set_at(const std::vector<std::size_t>& indices, float value);

    [[nodiscard]] Tensor to(Device target) const;
    [[nodiscard]] Tensor reshape(const Shape& new_shape) const;
    [[nodiscard]] Tensor flatten() const;
    [[nodiscard]] Tensor transpose() const;
    [[nodiscard]] Tensor sum() const;
    [[nodiscard]] Tensor mean() const;
    [[nodiscard]] Tensor relu() const;
    [[nodiscard]] Tensor sigmoid() const;
    [[nodiscard]] Tensor matmul(const Tensor& other) const;
    [[nodiscard]] Tensor detach() const;

    [[nodiscard]] bool has_grad() const;
    [[nodiscard]] Tensor grad() const;
    void zero_grad();
    void backward() const;
    void backward(const Tensor& gradient) const;

    [[nodiscard]] std::string repr() const;
    [[nodiscard]] std::shared_ptr<detail::TensorImpl> impl() const { return impl_; }

private:
    std::shared_ptr<detail::TensorImpl> impl_;

    friend Tensor operator+(const Tensor&, const Tensor&);
    friend Tensor operator-(const Tensor&, const Tensor&);
    friend Tensor operator*(const Tensor&, const Tensor&);
    friend Tensor operator/(const Tensor&, const Tensor&);
    friend Tensor operator-(const Tensor&);
    friend Tensor operator+(const Tensor&, float);
    friend Tensor operator-(const Tensor&, float);
    friend Tensor operator*(const Tensor&, float);
    friend Tensor operator/(const Tensor&, float);
    friend Tensor operator+(float, const Tensor&);
    friend Tensor operator-(float, const Tensor&);
    friend Tensor operator*(float, const Tensor&);
    friend Tensor operator/(float, const Tensor&);
};

Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& value);
Tensor operator+(const Tensor& a, float b);
Tensor operator-(const Tensor& a, float b);
Tensor operator*(const Tensor& a, float b);
Tensor operator/(const Tensor& a, float b);
Tensor operator+(float a, const Tensor& b);
Tensor operator-(float a, const Tensor& b);
Tensor operator*(float a, const Tensor& b);
Tensor operator/(float a, const Tensor& b);

} // namespace vectorcore
