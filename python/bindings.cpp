#include "vectorcore/vectorcore.hpp"

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using namespace vectorcore;

namespace {

Device parse_device(const std::string& name) {
    if (name == "cpu") return Device::cpu();
    if (name == "metal" || name == "mps") return Device::metal();
    throw std::invalid_argument("device must be 'cpu' or 'metal'");
}

Tensor tensor_from_python(const py::object& object, bool requires_grad,
                          const std::string& device_name) {
    const auto device = parse_device(device_name);
    if (py::isinstance<py::float_>(object) || py::isinstance<py::int_>(object)) {
        return Tensor::scalar(py::cast<float>(object), requires_grad, device);
    }
    auto array = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(object);
    if (!array) throw std::invalid_argument("Tensor values must be a scalar or rectangular array-like object");
    Shape shape;
    for (py::ssize_t axis = 0; axis < array.ndim(); ++axis) {
        shape.push_back(static_cast<std::size_t>(array.shape(axis)));
    }
    std::vector<float> values(static_cast<std::size_t>(array.size()));
    std::memcpy(values.data(), array.data(), values.size() * sizeof(float));
    return Tensor(std::move(values), std::move(shape), requires_grad, device);
}

py::array_t<float> tensor_numpy(const Tensor& tensor) {
    const auto values = tensor.to_vector();
    std::vector<py::ssize_t> shape;
    for (const auto dim : tensor.shape()) shape.push_back(static_cast<py::ssize_t>(dim));
    py::array_t<float> output(shape);
    std::memcpy(output.mutable_data(), values.data(), values.size() * sizeof(float));
    return output;
}

} // namespace

PYBIND11_MODULE(vectorcore, module) {
    module.doc() = "VectorCore C++20 tensor and automatic differentiation runtime";

    py::class_<Tensor>(module, "Tensor")
        .def(py::init(&tensor_from_python), py::arg("values"),
             py::arg("requires_grad") = false, py::arg("device") = "cpu")
        .def_static("zeros", [](const Shape& shape, bool requires_grad, const std::string& device) {
            return Tensor::zeros(shape, requires_grad, parse_device(device));
        }, py::arg("shape"), py::arg("requires_grad") = false, py::arg("device") = "cpu")
        .def_static("ones", [](const Shape& shape, bool requires_grad, const std::string& device) {
            return Tensor::ones(shape, requires_grad, parse_device(device));
        }, py::arg("shape"), py::arg("requires_grad") = false, py::arg("device") = "cpu")
        .def_static("randn", [](const Shape& shape, bool requires_grad, std::uint64_t seed,
                                 const std::string& device) {
            return Tensor::randn(shape, requires_grad, seed, parse_device(device));
        }, py::arg("shape"), py::arg("requires_grad") = false,
           py::arg("seed") = 0, py::arg("device") = "cpu")
        .def_property_readonly("shape", &Tensor::shape)
        .def_property_readonly("strides", &Tensor::strides)
        .def_property_readonly("ndim", &Tensor::ndim)
        .def_property_readonly("numel", &Tensor::numel)
        .def_property_readonly("device", [](const Tensor& value) { return value.device().str(); })
        .def_property_readonly("dtype", [](const Tensor&) { return "float32"; })
        .def_property("requires_grad", &Tensor::requires_grad, &Tensor::set_requires_grad)
        .def_property_readonly("grad", [](const Tensor& value) -> py::object {
            return value.has_grad() ? py::cast(value.grad()) : py::none();
        })
        .def("numpy", &tensor_numpy)
        .def("item", &Tensor::item)
        .def("at", &Tensor::at)
        .def("reshape", &Tensor::reshape)
        .def("flatten", &Tensor::flatten)
        .def("transpose", &Tensor::transpose)
        .def_property_readonly("T", &Tensor::transpose)
        .def("sum", &Tensor::sum)
        .def("mean", &Tensor::mean)
        .def("relu", &Tensor::relu)
        .def("sigmoid", &Tensor::sigmoid)
        .def("matmul", &Tensor::matmul)
        .def("to", [](const Tensor& value, const std::string& device) {
            return value.to(parse_device(device));
        })
        .def("detach", &Tensor::detach)
        .def("backward", py::overload_cast<>(&Tensor::backward, py::const_))
        .def("backward", py::overload_cast<const Tensor&>(&Tensor::backward, py::const_))
        .def("zero_grad", &Tensor::zero_grad)
        .def("__repr__", &Tensor::repr)
        .def("__matmul__", &Tensor::matmul, py::is_operator())
        .def("__add__", [](const Tensor& a, const Tensor& b) { return a + b; }, py::is_operator())
        .def("__sub__", [](const Tensor& a, const Tensor& b) { return a - b; }, py::is_operator())
        .def("__mul__", [](const Tensor& a, const Tensor& b) { return a * b; }, py::is_operator())
        .def("__truediv__", [](const Tensor& a, const Tensor& b) { return a / b; }, py::is_operator())
        .def("__add__", [](const Tensor& a, float b) { return a + b; }, py::is_operator())
        .def("__sub__", [](const Tensor& a, float b) { return a - b; }, py::is_operator())
        .def("__mul__", [](const Tensor& a, float b) { return a * b; }, py::is_operator())
        .def("__truediv__", [](const Tensor& a, float b) { return a / b; }, py::is_operator())
        .def("__radd__", [](const Tensor& b, float a) { return a + b; }, py::is_operator())
        .def("__rsub__", [](const Tensor& b, float a) { return a - b; }, py::is_operator())
        .def("__rmul__", [](const Tensor& b, float a) { return a * b; }, py::is_operator())
        .def("__rtruediv__", [](const Tensor& b, float a) { return a / b; }, py::is_operator())
        .def("__neg__", [](const Tensor& value) { return -value; }, py::is_operator());

    py::class_<nn::Linear>(module, "Linear")
        .def(py::init<std::size_t, std::size_t, std::uint64_t>(),
             py::arg("in_features"), py::arg("out_features"), py::arg("seed") = 0)
        .def("__call__", &nn::Linear::operator())
        .def_readwrite("weight", &nn::Linear::weight)
        .def_readwrite("bias", &nn::Linear::bias);

    module.def("mse_loss", &nn::mse_loss);
    module.def("sgd_step", [](std::vector<Tensor> parameters, float learning_rate) {
        std::vector<Tensor*> pointers;
        for (auto& parameter : parameters) pointers.push_back(&parameter);
        nn::sgd_step(pointers, learning_rate);
    });
    module.def("zero_grad", [](std::vector<Tensor> parameters) {
        for (auto& parameter : parameters) parameter.zero_grad();
    });
    module.def("metal_available", &metal_available);
    module.def("metal_device_name", &metal_device_name);
}
