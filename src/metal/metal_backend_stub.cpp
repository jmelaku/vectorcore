#include "vectorcore/metal_backend.hpp"

#include <stdexcept>

namespace vectorcore {
namespace {
class MetalBackendStub final : public MetalBackend {
public:
    bool available() const noexcept override { return false; }
    std::string device_name() const override { return "unavailable"; }
    std::string last_error() const override { return "VectorCore was built without Apple Metal"; }
    std::shared_ptr<Storage> allocate(std::size_t) override { fail(); }
    std::shared_ptr<Storage> upload(const float*, std::size_t) override { fail(); }
    void download(const Storage&, float*, std::size_t) override { fail(); }
    std::shared_ptr<Storage> binary(const Storage&, const Shape&, const Strides&,
        const Storage&, const Shape&, const Strides&, const Shape&, cpu::BinaryOp) override { fail(); }
    std::shared_ptr<Storage> unary(const Storage&, std::size_t, cpu::UnaryOp) override { fail(); }
    std::shared_ptr<Storage> matmul(const Storage&, const Storage&, std::size_t,
        std::size_t, std::size_t) override { fail(); }
    std::shared_ptr<Storage> transpose2d(const Storage&, std::size_t, std::size_t) override { fail(); }
    std::shared_ptr<Storage> reduce_sum(const Storage&, std::size_t) override { fail(); }
    std::shared_ptr<Storage> reduce_to_shape(const Storage&, const Shape&, const Shape&) override { fail(); }
    void synchronize() override {}
private:
    [[noreturn]] static void fail() { throw std::runtime_error("Metal backend is unavailable"); }
};
}
MetalBackend& MetalBackend::instance() { static MetalBackendStub backend; return backend; }
bool metal_available() { return false; }
std::string metal_device_name() { return "unavailable"; }
} // namespace vectorcore
