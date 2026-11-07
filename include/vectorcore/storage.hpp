#pragma once

#include "vectorcore/device.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace vectorcore {

class Storage {
public:
    virtual ~Storage() = default;
    [[nodiscard]] virtual Device device() const noexcept = 0;
    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    [[nodiscard]] virtual const float* data() const;
    [[nodiscard]] virtual float* mutable_data();

    // In-place modification counter shared by every view of this storage.
    // Autograd records it when saving a tensor for backward and rejects the
    // backward pass if the value changed in between.
    [[nodiscard]] std::uint64_t version() const noexcept { return version_; }
    void bump_version() noexcept { ++version_; }

private:
    std::uint64_t version_{0};
};

std::shared_ptr<Storage> make_cpu_storage(std::size_t elements);
std::shared_ptr<Storage> make_cpu_storage(const float* values, std::size_t elements);

} // namespace vectorcore
