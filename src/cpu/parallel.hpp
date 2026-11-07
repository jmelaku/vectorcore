#pragma once

#include <cstddef>
#include <functional>

namespace vectorcore::cpu {

// Runs body(begin, end) over [0, count) split into chunks of `grain` items.
// Chunks are claimed dynamically by a persistent worker pool plus the calling
// thread, so faster (performance) cores take more chunks than slower
// (efficiency) cores. Small ranges and nested calls run inline on the caller.
// The partition into chunks depends only on count and grain, never on the
// thread count, so callers that combine per-chunk results in chunk order get
// deterministic output.
void parallel_for(std::size_t count, std::size_t grain,
                  const std::function<void(std::size_t, std::size_t)>& body);

// Number of threads parallel_for may use, including the caller.
std::size_t parallel_threads();

} // namespace vectorcore::cpu
