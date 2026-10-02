#pragma once

#include "vectorcore/tensor.hpp"

namespace vectorcore {

class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;
private:
    bool previous_;
};

[[nodiscard]] bool grad_enabled();

} // namespace vectorcore
