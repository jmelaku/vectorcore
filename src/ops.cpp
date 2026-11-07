#include "vectorcore/ops.hpp"

#include "vectorcore/autograd.hpp"
#include "vectorcore/cpu_backend.hpp"
#include "vectorcore/detail/tensor_impl.hpp"
#include "vectorcore/metal_backend.hpp"
#include "vectorcore/storage.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vectorcore {

namespace {

void require_same_device(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) {
        throw std::invalid_argument("tensor device mismatch: " + a.device().str() + " vs " +
                                    b.device().str());
    }
}

Tensor binary_forward(const Tensor& a, const Tensor& b, cpu::BinaryOp operation) {
    require_same_device(a, b);
    const auto output_shape = broadcast_shape(a.shape(), b.shape());
    std::shared_ptr<Storage> storage;
    if (a.device() == Device::cpu()) {
        storage = make_cpu_storage(vectorcore::numel(output_shape));
        cpu::binary(a.data(), a.shape(), a.strides(), b.data(), b.shape(), b.strides(),
                    storage->mutable_data(), output_shape, operation);
    } else {
        storage = MetalBackend::instance().binary(*a.impl()->storage, a.shape(), a.strides(),
                                                   *b.impl()->storage, b.shape(), b.strides(),
                                                   output_shape, operation);
    }
    return detail::make_tensor(std::move(storage), output_shape, false);
}

Tensor unary_forward(const Tensor& value, cpu::UnaryOp operation) {
    std::shared_ptr<Storage> storage;
    if (value.device() == Device::cpu()) {
        storage = make_cpu_storage(value.numel());
        cpu::unary(value.data(), storage->mutable_data(), value.numel(), operation);
    } else {
        storage = MetalBackend::instance().unary(*value.impl()->storage, value.numel(), operation);
    }
    return detail::make_tensor(std::move(storage), value.shape(), false);
}

struct AxisSplit {
    std::size_t axis;
    std::size_t outer;
    std::size_t length;
    std::size_t inner;
    Shape kept;     // shape with the reduced axis set to 1
    Shape removed;  // shape with the reduced axis dropped
};

AxisSplit split_axis(const Shape& shape, std::int64_t axis) {
    const auto rank = static_cast<std::int64_t>(shape.size());
    if (rank == 0) throw std::invalid_argument("cannot reduce a scalar along an axis");
    if (axis < -rank || axis >= rank) {
        throw std::out_of_range("axis " + std::to_string(axis) + " is out of range for shape " +
                                shape_string(shape));
    }
    AxisSplit split{};
    split.axis = static_cast<std::size_t>(axis < 0 ? axis + rank : axis);
    split.outer = vectorcore::numel(Shape(shape.begin(), shape.begin() + static_cast<std::ptrdiff_t>(split.axis)));
    split.length = shape[split.axis];
    split.inner = vectorcore::numel(Shape(shape.begin() + static_cast<std::ptrdiff_t>(split.axis) + 1, shape.end()));
    split.kept = shape;
    split.kept[split.axis] = 1;
    split.removed = shape;
    split.removed.erase(split.removed.begin() + static_cast<std::ptrdiff_t>(split.axis));
    return split;
}

std::shared_ptr<Storage> reduce_storage(const Tensor& value, const AxisSplit& split,
                                        cpu::ReduceOp operation) {
    if (value.device() == Device::cpu()) {
        auto storage = make_cpu_storage(split.outer * split.inner);
        cpu::reduce_axis(value.data(), storage->mutable_data(), split.outer, split.length,
                         split.inner, operation);
        return storage;
    }
    return MetalBackend::instance().reduce_axis(*value.impl()->storage, split.outer,
                                                split.length, split.inner, operation);
}

} // namespace

// Backward rules compute only the gradients their inputs need (requires_grad
// is captured at forward time); a skipped slot is ignored by the engine.
Tensor add(const Tensor& a, const Tensor& b) {
    auto output = binary_forward(a, b, cpu::BinaryOp::Add);
    const auto a_shape = a.shape();
    const auto b_shape = b.shape();
    const bool need_a = a.requires_grad();
    const bool need_b = b.requires_grad();
    detail::attach_grad_fn(output, {a, b}, [=](const Tensor& gradient) {
        return std::vector<Tensor>{need_a ? detail::sum_to_shape(gradient, a_shape) : Tensor(),
                                   need_b ? detail::sum_to_shape(gradient, b_shape) : Tensor()};
    });
    return output;
}

Tensor subtract(const Tensor& a, const Tensor& b) {
    auto output = binary_forward(a, b, cpu::BinaryOp::Subtract);
    const auto a_shape = a.shape();
    const auto b_shape = b.shape();
    const bool need_a = a.requires_grad();
    const bool need_b = b.requires_grad();
    detail::attach_grad_fn(output, {a, b}, [=](const Tensor& gradient) {
        return std::vector<Tensor>{need_a ? detail::sum_to_shape(gradient, a_shape) : Tensor(),
                                   need_b ? detail::sum_to_shape(-gradient, b_shape) : Tensor()};
    });
    return output;
}

Tensor multiply(const Tensor& a, const Tensor& b) {
    auto output = binary_forward(a, b, cpu::BinaryOp::Multiply);
    const auto a_shape = a.shape();
    const auto b_shape = b.shape();
    const bool need_a = a.requires_grad();
    const bool need_b = b.requires_grad();
    const detail::SavedTensor saved_a(a);
    const detail::SavedTensor saved_b(b);
    detail::attach_grad_fn(output, {a, b}, [=](const Tensor& gradient) {
        return std::vector<Tensor>{
            need_a ? detail::sum_to_shape(gradient * saved_b.unpack(), a_shape) : Tensor(),
            need_b ? detail::sum_to_shape(gradient * saved_a.unpack(), b_shape) : Tensor()};
    });
    return output;
}

Tensor divide(const Tensor& a, const Tensor& b) {
    auto output = binary_forward(a, b, cpu::BinaryOp::Divide);
    const auto a_shape = a.shape();
    const auto b_shape = b.shape();
    const bool need_a = a.requires_grad();
    const bool need_b = b.requires_grad();
    const detail::SavedTensor saved_a(a);
    const detail::SavedTensor saved_b(b);
    detail::attach_grad_fn(output, {a, b}, [=](const Tensor& gradient) {
        std::vector<Tensor> grads{Tensor(), Tensor()};
        const auto bv = saved_b.unpack();
        if (need_a) grads[0] = detail::sum_to_shape(gradient / bv, a_shape);
        if (need_b) grads[1] = detail::sum_to_shape(-(gradient * saved_a.unpack()) / (bv * bv), b_shape);
        return grads;
    });
    return output;
}

Tensor negate(const Tensor& value) {
    auto output = unary_forward(value, cpu::UnaryOp::Negate);
    detail::attach_grad_fn(output, {value}, [](const Tensor& gradient) {
        return std::vector<Tensor>{-gradient};
    });
    return output;
}

namespace {

// op(A) @ op(B) where op transposes the last two axes when requested. The
// transposes are folded into the kernels' indexing, so backward passes never
// materialise A^T or B^T. Not differentiable by itself.
Tensor matmul_kernel(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b) {
    require_same_device(a, b);
    if (a.ndim() < 2 || b.ndim() < 2) {
        throw std::invalid_argument("matmul requires tensors of rank 2 or more, got " +
                                    shape_string(a.shape()) + " and " + shape_string(b.shape()));
    }
    const auto& as = a.shape();
    const auto& bs = b.shape();
    const auto a_rows = as[as.size() - 2], a_cols = as[as.size() - 1];
    const auto b_rows = bs[bs.size() - 2], b_cols = bs[bs.size() - 1];
    const auto m = trans_a ? a_cols : a_rows;
    const auto k = trans_a ? a_rows : a_cols;
    const auto n = trans_b ? b_rows : b_cols;
    if ((trans_b ? b_cols : b_rows) != k) {
        throw std::invalid_argument("matmul shape mismatch: " + shape_string(as) +
                                    " and " + shape_string(bs));
    }
    // Leading (batch) dimensions broadcast like elementwise operations.
    const Shape a_batch(as.begin(), as.end() - 2);
    const Shape b_batch(bs.begin(), bs.end() - 2);
    Shape out_batch;
    try {
        out_batch = broadcast_shape(a_batch, b_batch);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument("matmul batch dimensions do not broadcast: " +
                                    shape_string(as) + " and " + shape_string(bs));
    }
    const auto batch = vectorcore::numel(out_batch);
    const auto a_strides = contiguous_strides(a_batch);
    const auto b_strides = contiguous_strides(b_batch);
    std::vector<std::size_t> a_offsets(batch);
    std::vector<std::size_t> b_offsets(batch);
    const auto rank = out_batch.size();
    for (std::size_t z = 0; z < batch; ++z) {
        auto remainder = z;
        std::size_t a_index = 0;
        std::size_t b_index = 0;
        for (std::size_t axis = rank; axis-- > 0;) {
            const auto coordinate = remainder % out_batch[axis];
            remainder /= out_batch[axis];
            if (axis >= rank - a_batch.size()) {
                const auto ia = axis - (rank - a_batch.size());
                if (a_batch[ia] != 1) a_index += coordinate * a_strides[ia];
            }
            if (axis >= rank - b_batch.size()) {
                const auto ib = axis - (rank - b_batch.size());
                if (b_batch[ib] != 1) b_index += coordinate * b_strides[ib];
            }
        }
        a_offsets[z] = a_index * m * k;
        b_offsets[z] = b_index * k * n;
    }

    std::shared_ptr<Storage> storage;
    if (a.device() == Device::cpu()) {
        storage = make_cpu_storage(batch * m * n);
        auto* out = storage->mutable_data();
        for (std::size_t z = 0; z < batch; ++z) {
            cpu::matmul(a.data() + a_offsets[z], b.data() + b_offsets[z], out + z * m * n,
                        m, k, n, trans_a, trans_b);
        }
    } else {
        storage = MetalBackend::instance().matmul(*a.impl()->storage, *b.impl()->storage,
                                                  m, k, n, a_offsets, b_offsets, trans_a, trans_b);
    }
    Shape out_shape = out_batch;
    out_shape.push_back(m);
    out_shape.push_back(n);
    return detail::make_tensor(std::move(storage), out_shape, false);
}

} // namespace

Tensor matmul(const Tensor& a, const Tensor& b) {
    auto output = matmul_kernel(a, b, false, false);
    const detail::SavedTensor saved_a(a);
    const detail::SavedTensor saved_b(b);
    const auto as = a.shape();
    const auto bs = b.shape();
    const bool need_a = a.requires_grad();
    const bool need_b = b.requires_grad();
    detail::attach_grad_fn(output, {a, b}, [=](const Tensor& gradient) {
        // dA = dY @ B^T, dB = A^T @ dY; batch-broadcast operands receive the
        // sum over the broadcast batches.
        return std::vector<Tensor>{
            need_a ? detail::sum_to_shape(matmul_kernel(gradient, saved_b.unpack(), false, true), as)
                   : Tensor(),
            need_b ? detail::sum_to_shape(matmul_kernel(saved_a.unpack(), gradient, true, false), bs)
                   : Tensor()};
    });
    return output;
}

Tensor transpose(const Tensor& value) {
    if (value.ndim() < 2) {
        throw std::invalid_argument("transpose requires a tensor of rank 2 or more");
    }
    const auto& shape = value.shape();
    const auto rows = shape[shape.size() - 2];
    const auto cols = shape[shape.size() - 1];
    const auto batch = vectorcore::numel(Shape(shape.begin(), shape.end() - 2));
    std::shared_ptr<Storage> storage;
    if (value.device() == Device::cpu()) {
        storage = make_cpu_storage(value.numel());
        cpu::transpose(value.data(), storage->mutable_data(), batch, rows, cols);
    } else {
        storage = MetalBackend::instance().transpose_last2(*value.impl()->storage, batch, rows, cols);
    }
    Shape out_shape = shape;
    std::swap(out_shape[out_shape.size() - 2], out_shape[out_shape.size() - 1]);
    auto output = detail::make_tensor(std::move(storage), out_shape, false);
    detail::attach_grad_fn(output, {value}, [](const Tensor& gradient) {
        return std::vector<Tensor>{gradient.transpose()};
    });
    return output;
}

Tensor sum(const Tensor& value) {
    std::shared_ptr<Storage> storage;
    if (value.device() == Device::cpu()) {
        const float result = cpu::sum(value.data(), value.numel());
        storage = make_cpu_storage(&result, 1);
    } else {
        storage = MetalBackend::instance().reduce_sum(*value.impl()->storage, value.numel());
    }
    auto output = detail::make_tensor(std::move(storage), {}, false);
    const auto input_shape = value.shape();
    detail::attach_grad_fn(output, {value}, [input_shape](const Tensor& gradient) {
        return std::vector<Tensor>{detail::broadcast_to(gradient, input_shape)};
    });
    return output;
}

Tensor sum(const Tensor& value, std::int64_t axis, bool keepdim) {
    const auto split = split_axis(value.shape(), axis);
    auto output = detail::make_tensor(reduce_storage(value, split, cpu::ReduceOp::Sum),
                                      keepdim ? split.kept : split.removed, false);
    const auto input_shape = value.shape();
    const auto kept = split.kept;
    detail::attach_grad_fn(output, {value}, [input_shape, kept](const Tensor& gradient) {
        // Every element along the axis contributed once: broadcast back.
        return std::vector<Tensor>{detail::broadcast_to(gradient.reshape(kept), input_shape)};
    });
    return output;
}

Tensor mean(const Tensor& value, std::int64_t axis, bool keepdim) {
    const auto split = split_axis(value.shape(), axis);
    if (split.length == 0) throw std::invalid_argument("mean over an empty axis is undefined");
    return sum(value, axis, keepdim) / static_cast<float>(split.length);
}

Tensor max(const Tensor& value, std::int64_t axis, bool keepdim) {
    const auto split = split_axis(value.shape(), axis);
    const auto out_shape = keepdim ? split.kept : split.removed;
    auto output = detail::make_tensor(reduce_storage(value, split, cpu::ReduceOp::Max),
                                      out_shape, false);
    if (!grad_enabled() || !value.requires_grad()) return output;
    // The gradient flows only to the (first) maximal element along the axis.
    const auto indices = detail::make_tensor(reduce_storage(value, split, cpu::ReduceOp::Argmax),
                                             out_shape, false);
    const auto input_shape = value.shape();
    detail::attach_grad_fn(output, {value}, [indices, split, input_shape](const Tensor& gradient) {
        std::shared_ptr<Storage> storage;
        if (gradient.device() == Device::cpu()) {
            storage = make_cpu_storage(vectorcore::numel(input_shape));
            cpu::scatter_axis(gradient.data(), indices.data(), storage->mutable_data(),
                              split.outer, split.length, split.inner);
        } else {
            storage = MetalBackend::instance().scatter_axis(
                *gradient.impl()->storage, *indices.impl()->storage,
                split.outer, split.length, split.inner);
        }
        return std::vector<Tensor>{detail::make_tensor(std::move(storage), input_shape, false)};
    });
    return output;
}

Tensor max(const Tensor& value) {
    if (value.numel() == 0) throw std::invalid_argument("max of an empty tensor is undefined");
    return max(value.flatten(), 0, false);
}

Tensor argmax(const Tensor& value, std::int64_t axis, bool keepdim) {
    const auto split = split_axis(value.shape(), axis);
    return detail::make_tensor(reduce_storage(value, split, cpu::ReduceOp::Argmax),
                               keepdim ? split.kept : split.removed, false);
}

Tensor exp(const Tensor& value) {
    auto output = unary_forward(value, cpu::UnaryOp::Exp);
    const detail::SavedTensor saved(output);
    detail::attach_grad_fn(output, {value}, [saved](const Tensor& gradient) {
        return std::vector<Tensor>{gradient * saved.unpack()};
    });
    return output;
}

Tensor log(const Tensor& value) {
    auto output = unary_forward(value, cpu::UnaryOp::Log);
    const detail::SavedTensor saved(value);
    detail::attach_grad_fn(output, {value}, [saved](const Tensor& gradient) {
        return std::vector<Tensor>{gradient / saved.unpack()};
    });
    return output;
}

Tensor log_softmax(const Tensor& value, std::int64_t axis) {
    // Subtracting the per-row max keeps exp() from overflowing. log_softmax is
    // invariant to that shift, so treating it as a constant (detached) leaves
    // the gradient exact.
    const auto shifted = value - max(value.detach(), axis, true);
    return shifted - log(sum(exp(shifted), axis, true));
}

Tensor softmax(const Tensor& value, std::int64_t axis) {
    return exp(log_softmax(value, axis));
}

Tensor mean(const Tensor& value) {
    if (value.numel() == 0) throw std::invalid_argument("mean of an empty tensor is undefined");
    return value.sum() / static_cast<float>(value.numel());
}

Tensor relu(const Tensor& value) {
    auto output = unary_forward(value, cpu::UnaryOp::Relu);
    const detail::SavedTensor saved(value);
    detail::attach_grad_fn(output, {value}, [saved](const Tensor& gradient) {
        return std::vector<Tensor>{
            gradient * unary_forward(saved.unpack(), cpu::UnaryOp::ReluGradient)};
    });
    return output;
}

Tensor sigmoid(const Tensor& value) {
    auto output = unary_forward(value, cpu::UnaryOp::Sigmoid);
    const detail::SavedTensor saved(output);
    detail::attach_grad_fn(output, {value}, [saved](const Tensor& gradient) {
        const auto y = saved.unpack();
        return std::vector<Tensor>{gradient * y * (1.0F - y)};
    });
    return output;
}

} // namespace vectorcore
