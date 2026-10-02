#pragma once

#include "vectorcore/device.hpp"

#include <cstddef>
#include <memory>

namespace vectorcore {

class Storage {
public:
    virtual ~Storage() = default;
    [[nodiscard]] virtual Device device() const noexcept = 0;
    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    [[nodiscard]] virtual const float* data() const;
    [[nodiscard]] virtual float* mutable_data();
};

std::shared_ptr<Storage> make_cpu_storage(std::size_t elements);
std::shared_ptr<Storage> make_cpu_storage(const float* values, std::size_t elements);

} // namespace vectorcore
