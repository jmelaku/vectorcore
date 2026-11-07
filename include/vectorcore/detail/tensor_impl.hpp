#pragma once

#include "vectorcore/shape.hpp"
#include "vectorcore/storage.hpp"
#include "vectorcore/tensor.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace vectorcore::detail {

struct TensorImpl;

struct GradFn {
    std::vector<std::shared_ptr<TensorImpl>> parents;
    std::function<std::vector<Tensor>(const Tensor&)> backward;
};

struct TensorImpl {
    std::shared_ptr<Storage> storage;
    Shape shape;
    Strides strides;
    std::size_t offset{0};
    DType dtype{DType::Float32};
    bool requires_grad{false};
    bool leaf{true};
    bool retains_grad{false};
    std::shared_ptr<GradFn> grad_fn;
    std::shared_ptr<TensorImpl> grad;
};

// A tensor captured for use in a backward rule, together with the storage
// version at capture time. unpack() throws if the storage was modified in place
// afterwards, since the gradient would otherwise be computed from new values.
class SavedTensor {
public:
    explicit SavedTensor(const Tensor& value);
    [[nodiscard]] Tensor unpack() const;
private:
    Tensor tensor_;
    std::uint64_t version_;
};

Tensor from_impl(std::shared_ptr<TensorImpl> impl);
Tensor make_tensor(std::shared_ptr<Storage> storage, Shape shape,
                   bool requires_grad = false);
void attach_grad_fn(Tensor& output, const std::vector<Tensor>& parents,
                    std::function<std::vector<Tensor>(const Tensor&)> backward);
Tensor sum_to_shape(const Tensor& gradient, const Shape& target_shape);
// Expand a tensor to a broadcast-compatible larger shape (not differentiable).
Tensor broadcast_to(const Tensor& value, const Shape& target_shape);

} // namespace vectorcore::detail
