#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace vectorcore {

using Shape = std::vector<std::size_t>;
using Strides = std::vector<std::size_t>;

[[nodiscard]] std::size_t numel(const Shape& shape);
[[nodiscard]] Strides contiguous_strides(const Shape& shape);
[[nodiscard]] bool is_contiguous(const Shape& shape, const Strides& strides);
[[nodiscard]] Shape broadcast_shape(const Shape& a, const Shape& b);
[[nodiscard]] std::string shape_string(const Shape& shape);

} // namespace vectorcore
