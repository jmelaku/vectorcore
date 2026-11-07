#include "vectorcore/autograd.hpp"
#include "vectorcore/cpu_backend.hpp"
#include "vectorcore/detail/tensor_impl.hpp"
#include "vectorcore/metal_backend.hpp"
#include "vectorcore/ops.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace vectorcore {

namespace { thread_local bool enabled = true; }

NoGradGuard::NoGradGuard() : previous_(enabled) { enabled = false; }
NoGradGuard::~NoGradGuard() { enabled = previous_; }
bool grad_enabled() { return enabled; }

namespace detail {

SavedTensor::SavedTensor(const Tensor& value)
    : tensor_(value.detach()), version_(value.version()) {}

Tensor SavedTensor::unpack() const {
    if (tensor_.version() != version_) {
        throw std::runtime_error(
            "a tensor needed for gradient computation was modified in place after it was "
            "used (saved version " + std::to_string(version_) + ", current version " +
            std::to_string(tensor_.version()) + "); run backward() before mutating it");
    }
    return tensor_;
}

void attach_grad_fn(Tensor& output, const std::vector<Tensor>& parents,
                    std::function<std::vector<Tensor>(const Tensor&)> backward_fn) {
    if (!grad_enabled()) return;
    bool needed = false;
    for (const auto& parent : parents) needed |= parent.requires_grad();
    if (!needed) return;
    auto fn = std::make_shared<GradFn>();
    for (const auto& parent : parents) fn->parents.push_back(parent.impl());
    fn->backward = std::move(backward_fn);
    output.impl()->requires_grad = true;
    output.impl()->leaf = false;
    output.impl()->grad_fn = std::move(fn);
}

Tensor sum_to_shape(const Tensor& gradient, const Shape& target_shape) {
    if (gradient.shape() == target_shape) return gradient;
    const auto& source = gradient.shape();
    if (target_shape.size() > source.size()) {
        throw std::invalid_argument("cannot reduce " + shape_string(source) + " to larger rank " +
                                    shape_string(target_shape));
    }
    // Align the target to the source rank with leading 1s, then sum every axis
    // where the target is 1 but the source is not, using the axis-reduction
    // kernel (one contiguous pass per axis on either device).
    const auto lead = source.size() - target_shape.size();
    Tensor current = gradient;
    for (std::size_t axis = 0; axis < source.size(); ++axis) {
        const auto target_dim = axis < lead ? 1 : target_shape[axis - lead];
        if (target_dim == source[axis]) continue;
        if (target_dim != 1) {
            throw std::invalid_argument("cannot reduce " + shape_string(source) + " to " +
                                        shape_string(target_shape));
        }
        current = vectorcore::sum(current, static_cast<std::int64_t>(axis), true);
    }
    return current.reshape(target_shape);
}

Tensor broadcast_to(const Tensor& value, const Shape& target_shape) {
    if (value.shape() == target_shape) return value;
    // Adding a scalar zero with an explicit output shape reuses the
    // broadcasting kernels without materialising a ones/zeros tensor.
    (void)broadcast_shape(value.shape(), target_shape);  // validates compatibility
    const auto zero = Tensor::scalar(0.0F, false, value.device());
    std::shared_ptr<Storage> storage;
    if (value.device() == Device::cpu()) {
        storage = make_cpu_storage(vectorcore::numel(target_shape));
        cpu::binary(value.data(), value.shape(), value.strides(), zero.data(), {}, {},
                    storage->mutable_data(), target_shape, cpu::BinaryOp::Add);
    } else {
        storage = MetalBackend::instance().binary(*value.impl()->storage, value.shape(),
                                                  value.strides(), *zero.impl()->storage, {}, {},
                                                  target_shape, cpu::BinaryOp::Add);
    }
    return make_tensor(std::move(storage), target_shape, false);
}

} // namespace detail

} // namespace vectorcore
