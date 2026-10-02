#include "vectorcore/autograd.hpp"
#include "vectorcore/detail/tensor_impl.hpp"
#include "vectorcore/metal_backend.hpp"
#include "vectorcore/ops.hpp"

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace vectorcore {

namespace { thread_local bool enabled = true; }

NoGradGuard::NoGradGuard() : previous_(enabled) { enabled = false; }
NoGradGuard::~NoGradGuard() { enabled = previous_; }
bool grad_enabled() { return enabled; }

namespace detail {

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
    if (gradient.device() == Device::metal()) {
        auto storage = MetalBackend::instance().reduce_to_shape(
            *gradient.impl()->storage, gradient.shape(), target_shape);
        return make_tensor(std::move(storage), target_shape, false);
    }
    const auto source = gradient.to_vector();
    std::vector<float> reduced(vectorcore::numel(target_shape), 0.0F);
    const auto target_strides = contiguous_strides(target_shape);
    const auto& source_shape = gradient.shape();
    const auto rank = source_shape.size();
    for (std::size_t linear = 0; linear < source.size(); ++linear) {
        auto remainder = linear;
        std::size_t target_offset = 0;
        for (std::size_t axis = rank; axis-- > 0;) {
            const auto coordinate = source_shape[axis] == 0 ? 0 : remainder % source_shape[axis];
            if (source_shape[axis] != 0) remainder /= source_shape[axis];
            if (axis >= rank - target_shape.size()) {
                const auto target_axis = axis - (rank - target_shape.size());
                if (target_shape[target_axis] != 1) {
                    target_offset += coordinate * target_strides[target_axis];
                }
            }
        }
        reduced[target_offset] += source[linear];
    }
    return Tensor(std::move(reduced), target_shape, false, Device::cpu());
}

} // namespace detail

} // namespace vectorcore
