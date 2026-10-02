#pragma once

namespace vectorcore::detail {
inline constexpr const char* embedded_metal_source = R"VCMETAL(
@METAL_SOURCE@
)VCMETAL";
} // namespace vectorcore::detail
