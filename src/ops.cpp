#include "vectorcore/ops.hpp"

#include "vectorcore/cpu_backend.hpp"
#include "vectorcore/detail/tensor_impl.hpp"
#include "vectorcore/metal_backend.hpp"
#include "vectorcore/storage.hpp"

#include <cmath>
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

} // namespace

Tensor add(const Tensor& a, const Tensor& b) {
    auto output = binary_forward(a, b, cpu::BinaryOp::Add);
    const auto a_shape = a.shape();
    const auto b_shape = b.shape();
    detail::attach_grad_fn(output, {a, b}, [a_shape, b_shape](const Tensor& gradient) {
        return std::vector<Tensor>{detail::sum_to_shape(gradient, a_shape),
                                   detail::sum_to_shape(gradient, b_shape)};
    });
    return output;
}

Tensor subtract(const Tensor& a, const Tensor& b) {
    auto output = binary_forward(a, b, cpu::BinaryOp::Subtract);
    const auto a_shape = a.shape();
    const auto b_shape = b.shape();
    detail::attach_grad_fn(output, {a, b}, [a_shape, b_shape](const Tensor& gradient) {
        return std::vector<Tensor>{detail::sum_to_shape(gradient, a_shape),
                                   detail::sum_to_shape(-gradient, b_shape)};
    });
    return output;
}

Tensor multiply(const Tensor& a, const Tensor& b) {
    auto output = binary_forward(a, b, cpu::BinaryOp::Multiply);
    const auto a_shape = a.shape();
    const auto b_shape = b.shape();
    detail::attach_grad_fn(output, {a, b}, [a, b, a_shape, b_shape](const Tensor& gradient) {
        return std::vector<Tensor>{detail::sum_to_shape(gradient * b, a_shape),
                                   detail::sum_to_shape(gradient * a, b_shape)};
    });
    return output;
}

Tensor divide(const Tensor& a, const Tensor& b) {
    auto output = binary_forward(a, b, cpu::BinaryOp::Divide);
    const auto a_shape = a.shape();
    const auto b_shape = b.shape();
    detail::attach_grad_fn(output, {a, b}, [a, b, a_shape, b_shape](const Tensor& gradient) {
        return std::vector<Tensor>{detail::sum_to_shape(gradient / b, a_shape),
            detail::sum_to_shape(-(gradient * a) / (b * b), b_shape)};
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

Tensor matmul(const Tensor& a, const Tensor& b) {
    require_same_device(a, b);
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("matmul currently requires rank-2 tensors");
    }
    const auto m = a.shape()[0];
    const auto k = a.shape()[1];
    if (b.shape()[0] != k) {
        throw std::invalid_argument("matmul shape mismatch: " + shape_string(a.shape()) +
                                    " and " + shape_string(b.shape()));
    }
    const auto n = b.shape()[1];
    std::shared_ptr<Storage> storage;
    if (a.device() == Device::cpu()) {
        storage = make_cpu_storage(m * n);
        cpu::matmul(a.data(), b.data(), storage->mutable_data(), m, k, n);
    } else {
        storage = MetalBackend::instance().matmul(*a.impl()->storage, *b.impl()->storage, m, k, n);
    }
    auto output = detail::make_tensor(std::move(storage), {m, n}, false);
    detail::attach_grad_fn(output, {a, b}, [a, b](const Tensor& gradient) {
        return std::vector<Tensor>{gradient.matmul(b.transpose()),
                                   a.transpose().matmul(gradient)};
    });
    return output;
}

Tensor transpose(const Tensor& value) {
    if (value.ndim() != 2) throw std::invalid_argument("transpose currently requires a rank-2 tensor");
    const auto rows = value.shape()[0];
    const auto cols = value.shape()[1];
    std::shared_ptr<Storage> storage;
    if (value.device() == Device::cpu()) {
        storage = make_cpu_storage(value.numel());
        auto* destination = storage->mutable_data();
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) destination[j * rows + i] = value.data()[i * cols + j];
        }
    } else {
        storage = MetalBackend::instance().transpose2d(*value.impl()->storage, rows, cols);
    }
    auto output = detail::make_tensor(std::move(storage), {cols, rows}, false);
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
    const auto input_device = value.device();
    detail::attach_grad_fn(output, {value}, [input_shape, input_device](const Tensor& gradient) {
        return std::vector<Tensor>{Tensor::ones(input_shape, false, input_device) * gradient};
    });
    return output;
}

Tensor mean(const Tensor& value) {
    if (value.numel() == 0) throw std::invalid_argument("mean of an empty tensor is undefined");
    return value.sum() / static_cast<float>(value.numel());
}

Tensor relu(const Tensor& value) {
    auto output = unary_forward(value, cpu::UnaryOp::Relu);
    const Tensor saved = value.detach();
    detail::attach_grad_fn(output, {value}, [saved](const Tensor& gradient) {
        return std::vector<Tensor>{gradient * unary_forward(saved, cpu::UnaryOp::ReluGradient)};
    });
    return output;
}

Tensor sigmoid(const Tensor& value) {
    auto output = unary_forward(value, cpu::UnaryOp::Sigmoid);
    const Tensor saved = output.detach();
    detail::attach_grad_fn(output, {value}, [saved](const Tensor& gradient) {
        return std::vector<Tensor>{gradient * saved * (1.0F - saved)};
    });
    return output;
}

} // namespace vectorcore
