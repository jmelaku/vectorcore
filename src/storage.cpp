#include "vectorcore/storage.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace vectorcore {

const float* Storage::data() const {
    throw std::runtime_error("storage is not directly CPU-addressable");
}

float* Storage::mutable_data() {
    throw std::runtime_error("storage is not directly CPU-addressable");
}

namespace {

class CPUStorage final : public Storage {
public:
    explicit CPUStorage(std::size_t elements) : values_(elements) {}
    CPUStorage(const float* values, std::size_t elements) : values_(values, values + elements) {}
    [[nodiscard]] Device device() const noexcept override { return Device::cpu(); }
    [[nodiscard]] std::size_t size() const noexcept override { return values_.size(); }
    [[nodiscard]] const float* data() const override { return values_.data(); }
    [[nodiscard]] float* mutable_data() override { return values_.data(); }
private:
    std::vector<float> values_;
};

} // namespace

std::shared_ptr<Storage> make_cpu_storage(std::size_t elements) {
    return std::make_shared<CPUStorage>(elements);
}

std::shared_ptr<Storage> make_cpu_storage(const float* values, std::size_t elements) {
    if (!values && elements != 0) throw std::invalid_argument("null storage source");
    if (elements == 0) return std::make_shared<CPUStorage>(0);
    return std::make_shared<CPUStorage>(values, elements);
}

} // namespace vectorcore
