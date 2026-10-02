#pragma once

#include <string>

namespace vectorcore {

enum class DeviceType { CPU, METAL };

struct Device {
    DeviceType type{DeviceType::CPU};

    constexpr Device() = default;
    constexpr explicit Device(DeviceType value) : type(value) {}
    [[nodiscard]] std::string str() const;

    friend constexpr bool operator==(Device a, Device b) { return a.type == b.type; }
    friend constexpr bool operator!=(Device a, Device b) { return !(a == b); }

    static constexpr Device cpu() { return Device(DeviceType::CPU); }
    static constexpr Device metal() { return Device(DeviceType::METAL); }
};

} // namespace vectorcore
