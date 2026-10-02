#include "vectorcore/shape.hpp"

#include <limits>
#include <sstream>
#include <stdexcept>

namespace vectorcore {

std::size_t numel(const Shape& shape) {
    std::size_t total = 1;
    for (const auto dim : shape) {
        if (dim != 0 && total > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::overflow_error("tensor element count overflows size_t");
        }
        total *= dim;
    }
    return total;
}

Strides contiguous_strides(const Shape& shape) {
    Strides result(shape.size(), 1);
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        result[i] = stride;
        stride *= shape[i];
    }
    return result;
}

bool is_contiguous(const Shape& shape, const Strides& strides) {
    return strides == contiguous_strides(shape);
}

Shape broadcast_shape(const Shape& a, const Shape& b) {
    const auto rank = std::max(a.size(), b.size());
    Shape result(rank, 1);
    for (std::size_t i = 0; i < rank; ++i) {
        const auto ai = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
        const auto bi = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
        if (ai != bi && ai != 1 && bi != 1) {
            throw std::invalid_argument("cannot broadcast shapes " + shape_string(a) +
                                        " and " + shape_string(b));
        }
        result[i] = std::max(ai, bi);
    }
    return result;
}

std::string shape_string(const Shape& shape) {
    std::ostringstream stream;
    stream << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) stream << ", ";
        stream << shape[i];
    }
    stream << ']';
    return stream.str();
}

} // namespace vectorcore
