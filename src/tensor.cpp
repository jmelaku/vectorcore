#include "vectorcore/tensor.hpp"

#include "vectorcore/autograd.hpp"
#include "vectorcore/detail/tensor_impl.hpp"
#include "vectorcore/metal_backend.hpp"
#include "vectorcore/ops.hpp"
#include "vectorcore/storage.hpp"

#include <algorithm>
#include <functional>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace vectorcore {

std::string Device::str() const {
    return type == DeviceType::CPU ? "cpu" : "metal";
}

namespace detail {

Tensor from_impl(std::shared_ptr<TensorImpl> impl) { return Tensor(std::move(impl)); }

Tensor make_tensor(std::shared_ptr<Storage> storage, Shape shape, bool requires_grad) {
    if (!storage) throw std::invalid_argument("tensor storage cannot be null");
    if (storage->size() < vectorcore::numel(shape)) {
        throw std::invalid_argument("storage is smaller than tensor shape");
    }
    auto impl = std::make_shared<TensorImpl>();
    impl->storage = std::move(storage);
    impl->shape = std::move(shape);
    impl->strides = contiguous_strides(impl->shape);
    impl->requires_grad = requires_grad;
    return from_impl(std::move(impl));
}

} // namespace detail

Tensor::Tensor() : Tensor::Tensor(std::vector<float>{0.0F}, Shape{}) {}

Tensor::Tensor(std::shared_ptr<detail::TensorImpl> impl) : impl_(std::move(impl)) {
    if (!impl_) throw std::invalid_argument("TensorImpl cannot be null");
}

Tensor::Tensor(std::vector<float> values, Shape shape, bool requires_grad, Device device) {
    const auto count = vectorcore::numel(shape);
    if (values.size() != count) {
        throw std::invalid_argument("received " + std::to_string(values.size()) +
                                    " values for shape " + shape_string(shape) +
                                    " containing " + std::to_string(count) + " elements");
    }
    std::shared_ptr<Storage> storage;
    if (device == Device::cpu()) {
        storage = make_cpu_storage(values.data(), values.size());
    } else {
        if (!MetalBackend::instance().available()) {
            throw std::runtime_error("Metal is unavailable: " + MetalBackend::instance().last_error());
        }
        storage = MetalBackend::instance().upload(values.data(), values.size());
    }
    impl_ = detail::make_tensor(std::move(storage), std::move(shape), requires_grad).impl();
}

Tensor Tensor::scalar(float value, bool requires_grad, Device device) {
    return Tensor({value}, {}, requires_grad, device);
}

Tensor Tensor::zeros(const Shape& shape, bool requires_grad, Device device) {
    return Tensor(std::vector<float>(vectorcore::numel(shape), 0.0F), shape,
                  requires_grad, device);
}

Tensor Tensor::ones(const Shape& shape, bool requires_grad, Device device) {
    return Tensor(std::vector<float>(vectorcore::numel(shape), 1.0F), shape,
                  requires_grad, device);
}

Tensor Tensor::randn(const Shape& shape, bool requires_grad, std::uint64_t seed, Device device) {
    std::mt19937_64 generator(seed);
    std::normal_distribution<float> distribution(0.0F, 1.0F);
    std::vector<float> values(vectorcore::numel(shape));
    for (auto& value : values) value = distribution(generator);
    return Tensor(std::move(values), shape, requires_grad, device);
}

const Shape& Tensor::shape() const { return impl_->shape; }
const Strides& Tensor::strides() const { return impl_->strides; }
std::size_t Tensor::ndim() const { return impl_->shape.size(); }
std::size_t Tensor::numel() const { return vectorcore::numel(impl_->shape); }
Device Tensor::device() const { return impl_->storage->device(); }
DType Tensor::dtype() const { return impl_->dtype; }
bool Tensor::is_contiguous() const { return vectorcore::is_contiguous(shape(), strides()); }
bool Tensor::requires_grad() const { return impl_->requires_grad; }

void Tensor::set_requires_grad(bool value) {
    if (!impl_->leaf) throw std::runtime_error("requires_grad can only be changed on leaf tensors");
    impl_->requires_grad = value;
    if (!value) impl_->grad.reset();
}

const float* Tensor::data() const {
    if (device() != Device::cpu()) {
        throw std::runtime_error("data() is only valid for CPU tensors; use to(CPU)");
    }
    return impl_->storage->data() + impl_->offset;
}

float* Tensor::mutable_data() {
    if (device() != Device::cpu()) {
        throw std::runtime_error("mutable_data() is only valid for CPU tensors; use to(CPU)");
    }
    return impl_->storage->mutable_data() + impl_->offset;
}

std::vector<float> Tensor::to_vector() const {
    std::vector<float> result(numel());
    if (device() == Device::cpu()) {
        std::copy(data(), data() + numel(), result.begin());
    } else {
        MetalBackend::instance().download(*impl_->storage, result.data(), result.size());
    }
    return result;
}

float Tensor::item() const {
    if (numel() != 1) throw std::runtime_error("item() requires a one-element tensor");
    return to_vector().front();
}

float Tensor::at(const std::vector<std::size_t>& indices) const {
    if (indices.size() != ndim()) throw std::out_of_range("index rank does not match tensor rank");
    std::size_t offset = 0;
    for (std::size_t axis = 0; axis < indices.size(); ++axis) {
        if (indices[axis] >= shape()[axis]) throw std::out_of_range("tensor index out of bounds");
        offset += indices[axis] * strides()[axis];
    }
    if (device() == Device::cpu()) return data()[offset];
    return to_vector()[offset];
}

void Tensor::set_at(const std::vector<std::size_t>& indices, float value) {
    if (device() != Device::cpu()) throw std::runtime_error("set_at is only supported on CPU tensors");
    if (indices.size() != ndim()) throw std::out_of_range("index rank does not match tensor rank");
    std::size_t offset = 0;
    for (std::size_t axis = 0; axis < indices.size(); ++axis) {
        if (indices[axis] >= shape()[axis]) throw std::out_of_range("tensor index out of bounds");
        offset += indices[axis] * strides()[axis];
    }
    mutable_data()[offset] = value;
}

Tensor Tensor::to(Device target) const {
    if (target == device()) return *this;
    std::shared_ptr<Storage> storage;
    if (target == Device::cpu()) {
        const auto values = to_vector();
        storage = make_cpu_storage(values.data(), values.size());
    } else {
        if (!MetalBackend::instance().available()) {
            throw std::runtime_error("Metal is unavailable: " + MetalBackend::instance().last_error());
        }
        const auto values = to_vector();
        storage = MetalBackend::instance().upload(values.data(), values.size());
    }
    auto output = detail::make_tensor(std::move(storage), shape(), false);
    const auto source_device = device();
    detail::attach_grad_fn(output, {*this}, [source_device](const Tensor& gradient) {
        return std::vector<Tensor>{gradient.to(source_device)};
    });
    return output;
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    if (vectorcore::numel(new_shape) != numel()) {
        throw std::invalid_argument("cannot reshape " + shape_string(shape()) + " to " +
                                    shape_string(new_shape));
    }
    if (!is_contiguous()) throw std::runtime_error("reshape requires contiguous storage");
    auto result_impl = std::make_shared<detail::TensorImpl>();
    result_impl->storage = impl_->storage;
    result_impl->shape = new_shape;
    result_impl->strides = contiguous_strides(new_shape);
    result_impl->offset = impl_->offset;
    auto output = detail::from_impl(std::move(result_impl));
    const auto original_shape = shape();
    detail::attach_grad_fn(output, {*this}, [original_shape](const Tensor& gradient) {
        return std::vector<Tensor>{gradient.reshape(original_shape)};
    });
    return output;
}

Tensor Tensor::flatten() const { return reshape({numel()}); }
Tensor Tensor::transpose() const { return vectorcore::transpose(*this); }
Tensor Tensor::sum() const { return vectorcore::sum(*this); }
Tensor Tensor::mean() const { return vectorcore::mean(*this); }
Tensor Tensor::relu() const { return vectorcore::relu(*this); }
Tensor Tensor::sigmoid() const { return vectorcore::sigmoid(*this); }
Tensor Tensor::matmul(const Tensor& other) const { return vectorcore::matmul(*this, other); }

Tensor Tensor::detach() const {
    auto detached = std::make_shared<detail::TensorImpl>();
    detached->storage = impl_->storage;
    detached->shape = impl_->shape;
    detached->strides = impl_->strides;
    detached->offset = impl_->offset;
    detached->dtype = impl_->dtype;
    return detail::from_impl(std::move(detached));
}

bool Tensor::has_grad() const { return static_cast<bool>(impl_->grad); }

Tensor Tensor::grad() const {
    if (!impl_->grad) throw std::runtime_error("tensor has no gradient");
    return detail::from_impl(impl_->grad);
}

void Tensor::zero_grad() { impl_->grad.reset(); }

void Tensor::backward() const {
    if (numel() != 1) {
        throw std::runtime_error("backward() without a gradient requires a scalar tensor");
    }
    backward(Tensor::ones(shape(), false, device()));
}

void Tensor::backward(const Tensor& gradient) const {
    if (!requires_grad()) throw std::runtime_error("cannot call backward on a tensor without gradients");
    if (gradient.shape() != shape()) throw std::invalid_argument("backward gradient shape mismatch");
    if (gradient.device() != device()) throw std::invalid_argument("backward gradient device mismatch");

    std::vector<std::shared_ptr<detail::TensorImpl>> topology;
    std::unordered_set<detail::TensorImpl*> visited;
    std::function<void(const std::shared_ptr<detail::TensorImpl>&)> visit = [&](const auto& node) {
        if (!visited.insert(node.get()).second) return;
        if (node->grad_fn) for (const auto& parent : node->grad_fn->parents) visit(parent);
        topology.push_back(node);
    };
    visit(impl_);

    NoGradGuard guard;
    std::unordered_map<detail::TensorImpl*, Tensor> gradients;
    gradients.emplace(impl_.get(), gradient.detach());
    for (auto iterator = topology.rbegin(); iterator != topology.rend(); ++iterator) {
        const auto& node = *iterator;
        const auto found = gradients.find(node.get());
        if (found == gradients.end()) continue;
        const Tensor upstream = found->second;
        if (node->requires_grad) {
            if (node->grad) {
                node->grad = (detail::from_impl(node->grad) + upstream).detach().impl();
            } else {
                node->grad = upstream.detach().impl();
            }
        }
        if (!node->grad_fn) continue;
        auto parent_gradients = node->grad_fn->backward(upstream);
        if (parent_gradients.size() != node->grad_fn->parents.size()) {
            throw std::logic_error("autograd function returned the wrong number of gradients");
        }
        for (std::size_t i = 0; i < parent_gradients.size(); ++i) {
            const auto& parent = node->grad_fn->parents[i];
            if (!parent->requires_grad) continue;
            auto existing = gradients.find(parent.get());
            if (existing == gradients.end()) {
                gradients.emplace(parent.get(), parent_gradients[i].detach());
            } else {
                existing->second = existing->second + parent_gradients[i];
            }
        }
    }
}

std::string Tensor::repr() const {
    std::ostringstream stream;
    stream << "Tensor(shape=" << shape_string(shape()) << ", dtype='float32', device='" << device().str()
           << "', requires_grad=" << (requires_grad() ? "true" : "false") << ", values=[";
    const auto values = to_vector();
    const auto shown = std::min<std::size_t>(values.size(), 8);
    for (std::size_t i = 0; i < shown; ++i) {
        if (i) stream << ", ";
        stream << std::setprecision(6) << values[i];
    }
    if (values.size() > shown) stream << ", ...";
    stream << "])";
    return stream.str();
}

Tensor operator+(const Tensor& a, const Tensor& b) { return add(a, b); }
Tensor operator-(const Tensor& a, const Tensor& b) { return subtract(a, b); }
Tensor operator*(const Tensor& a, const Tensor& b) { return multiply(a, b); }
Tensor operator/(const Tensor& a, const Tensor& b) { return divide(a, b); }
Tensor operator-(const Tensor& value) { return negate(value); }
Tensor operator+(const Tensor& a, float b) { return a + Tensor::scalar(b, false, a.device()); }
Tensor operator-(const Tensor& a, float b) { return a - Tensor::scalar(b, false, a.device()); }
Tensor operator*(const Tensor& a, float b) { return a * Tensor::scalar(b, false, a.device()); }
Tensor operator/(const Tensor& a, float b) { return a / Tensor::scalar(b, false, a.device()); }
Tensor operator+(float a, const Tensor& b) { return Tensor::scalar(a, false, b.device()) + b; }
Tensor operator-(float a, const Tensor& b) { return Tensor::scalar(a, false, b.device()) - b; }
Tensor operator*(float a, const Tensor& b) { return Tensor::scalar(a, false, b.device()) * b; }
Tensor operator/(float a, const Tensor& b) { return Tensor::scalar(a, false, b.device()) / b; }

} // namespace vectorcore
