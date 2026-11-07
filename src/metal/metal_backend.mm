#include "vectorcore/metal_backend.hpp"
#include "metal_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace vectorcore {

namespace {

std::string utf8(NSString* value) {
    return value ? std::string(value.UTF8String) : std::string{};
}

// Defined after MetalBackendImpl; hands a dead tensor's buffer back to the
// pool once the GPU work that may still use it has completed.
void release_buffer(id<MTLBuffer> buffer);

class MetalStorage final : public Storage {
public:
    MetalStorage(id<MTLBuffer> buffer, std::size_t elements)
        : buffer_(buffer), elements_(elements) {}
    ~MetalStorage() override { release_buffer(buffer_); }
    MetalStorage(const MetalStorage&) = delete;
    MetalStorage& operator=(const MetalStorage&) = delete;
    [[nodiscard]] Device device() const noexcept override { return Device::metal(); }
    [[nodiscard]] std::size_t size() const noexcept override { return elements_; }
    [[nodiscard]] id<MTLBuffer> buffer() const { return buffer_; }
private:
    id<MTLBuffer> buffer_;
    std::size_t elements_;
};

const MetalStorage& as_metal(const Storage& storage) {
    const auto* result = dynamic_cast<const MetalStorage*>(&storage);
    if (!result) throw std::invalid_argument("expected Metal storage");
    return *result;
}

struct BroadcastMeta {
    std::uint32_t rank{};
    std::uint32_t a_rank{};
    std::uint32_t b_rank{};
    std::uint32_t operation{};
    std::uint32_t out_shape[8]{};
    std::uint32_t a_shape[8]{};
    std::uint32_t a_stride[8]{};
    std::uint32_t b_shape[8]{};
    std::uint32_t b_stride[8]{};
};

struct AxisMeta {
    std::uint32_t outer;
    std::uint32_t length;
    std::uint32_t inner;
    std::uint32_t operation;
};

std::uint32_t checked_u32(std::size_t value, const char* label) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(std::string(label) + " exceeds Metal uint range");
    }
    return static_cast<std::uint32_t>(value);
}

class MetalBackendImpl final : public MetalBackend {
public:
    MetalBackendImpl() {
        @autoreleasepool {
            device_ = MTLCreateSystemDefaultDevice();
            if (!device_) {
                error_ = "MTLCreateSystemDefaultDevice returned nil";
                return;
            }
            queue_ = [device_ newCommandQueue];
            if (!queue_) {
                error_ = "failed to create Metal command queue";
                return;
            }
            NSError* error = nil;
            NSString* source = [NSString stringWithUTF8String:detail::embedded_metal_source];
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            library_ = [device_ newLibraryWithSource:source options:options error:&error];
            if (!library_) {
                error_ = "Metal shader compilation failed: " + utf8(error.localizedDescription);
                return;
            }
            binary_pipeline_ = pipeline(@"binary_broadcast");
            unary_pipeline_ = pipeline(@"unary");
            for (int variant = 0; variant < 4; ++variant) {
                matmul_pipelines_[variant] = matmul_pipeline(variant & 1, (variant & 2) != 0);
            }
            transpose_pipeline_ = pipeline(@"transpose_2d");
            reduce_pipeline_ = pipeline(@"reduce_sum_pass");
            reduce_axis_pipeline_ = pipeline(@"reduce_axis");
            scatter_axis_pipeline_ = pipeline(@"scatter_axis");
            axpy_pipeline_ = pipeline(@"axpy_inplace");
            ready_ = binary_pipeline_ && unary_pipeline_ && matmul_pipelines_[0] &&
                     matmul_pipelines_[1] && matmul_pipelines_[2] && matmul_pipelines_[3] &&
                     transpose_pipeline_ && reduce_pipeline_ &&
                     reduce_axis_pipeline_ && scatter_axis_pipeline_ && axpy_pipeline_;
            if (!ready_ && error_.empty()) error_ = "one or more Metal pipelines failed to initialize";
            backend_alive().store(true);
        }
    }

    bool available() const noexcept override { return ready_; }
    std::string device_name() const override { return device_ ? utf8(device_.name) : "unavailable"; }
    std::string last_error() const override { return error_; }

    ~MetalBackendImpl() override {
        // Tensors that outlive the backend (e.g. at interpreter shutdown) must
        // not touch the pool afterwards.
        backend_alive().store(false);
    }

    std::shared_ptr<Storage> allocate(std::size_t elements) override {
        require_ready();
        const auto bytes = bucket_bytes(std::max<std::size_t>(elements * sizeof(float), sizeof(float)));
        id<MTLBuffer> buffer = nil;
        {
            // Every pooled buffer is idle: it only enters the free list after
            // the last command buffer that could reference it has completed,
            // so the CPU may write it immediately (as upload() does).
            std::lock_guard<std::mutex> lock(pool_mutex_);
            auto found = free_buffers_.find(bytes);
            if (found != free_buffers_.end() && !found->second.empty()) {
                buffer = found->second.back();
                found->second.pop_back();
                pooled_bytes_ -= bytes;
            }
        }
        if (!buffer) {
            buffer = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (!buffer) throw std::runtime_error("Metal buffer allocation failed");
        }
        return std::make_shared<MetalStorage>(buffer, elements);
    }

    // Called from ~MetalStorage. The buffer may still be referenced by queued
    // or in-flight work, so it waits in pending_release_ until the next
    // command buffer commits, and reaches the free list from that command
    // buffer's completion handler (the queue completes in submission order).
    void release(id<MTLBuffer> buffer) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        pending_release_.push_back(buffer);
    }

    static std::atomic<bool>& backend_alive() {
        static std::atomic<bool> alive{false};
        return alive;
    }

    std::shared_ptr<Storage> upload(const float* data, std::size_t elements) override {
        auto storage = allocate(elements);
        if (elements) std::copy(data, data + elements,
                                static_cast<float*>(as_metal(*storage).buffer().contents));
        return storage;
    }

    void download(const Storage& source, float* destination, std::size_t elements) override {
        if (elements > source.size()) throw std::out_of_range("Metal download exceeds storage");
        synchronize();  // the host read must see every queued write
        if (elements) std::copy(static_cast<const float*>(as_metal(source).buffer().contents),
                                static_cast<const float*>(as_metal(source).buffer().contents) + elements,
                                destination);
    }

    std::shared_ptr<Storage> binary(const Storage& a, const Shape& a_shape,
                                    const Strides& a_strides, const Storage& b,
                                    const Shape& b_shape, const Strides& b_strides,
                                    const Shape& out_shape, cpu::BinaryOp operation) override {
        if (out_shape.size() > 8 || a_shape.size() > 8 || b_shape.size() > 8) {
            throw std::invalid_argument("Metal broadcasting supports at most 8 dimensions");
        }
        auto output = allocate(vectorcore::numel(out_shape));
        BroadcastMeta meta;
        meta.rank = checked_u32(out_shape.size(), "rank");
        meta.a_rank = checked_u32(a_shape.size(), "rank");
        meta.b_rank = checked_u32(b_shape.size(), "rank");
        meta.operation = static_cast<std::uint32_t>(operation);
        for (std::size_t i = 0; i < out_shape.size(); ++i) meta.out_shape[i] = checked_u32(out_shape[i], "dimension");
        for (std::size_t i = 0; i < a_shape.size(); ++i) {
            meta.a_shape[i] = checked_u32(a_shape[i], "dimension");
            meta.a_stride[i] = checked_u32(a_strides[i], "stride");
        }
        for (std::size_t i = 0; i < b_shape.size(); ++i) {
            meta.b_shape[i] = checked_u32(b_shape[i], "dimension");
            meta.b_stride[i] = checked_u32(b_strides[i], "stride");
        }
        encode_1d(binary_pipeline_, vectorcore::numel(out_shape), [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:as_metal(a).buffer() offset:0 atIndex:0];
            [encoder setBuffer:as_metal(b).buffer() offset:0 atIndex:1];
            [encoder setBuffer:as_metal(*output).buffer() offset:0 atIndex:2];
            [encoder setBytes:&meta length:sizeof(meta) atIndex:3];
        });
        return output;
    }

    std::shared_ptr<Storage> unary(const Storage& input, std::size_t count,
                                   cpu::UnaryOp operation) override {
        auto output = allocate(count);
        const auto op = static_cast<std::uint32_t>(operation);
        const auto size = checked_u32(count, "element count");
        encode_1d(unary_pipeline_, count, [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:as_metal(input).buffer() offset:0 atIndex:0];
            [encoder setBuffer:as_metal(*output).buffer() offset:0 atIndex:1];
            [encoder setBytes:&op length:sizeof(op) atIndex:2];
            [encoder setBytes:&size length:sizeof(size) atIndex:3];
        });
        return output;
    }

    std::shared_ptr<Storage> matmul(const Storage& a, const Storage& b,
                                    std::size_t m_size, std::size_t k_size, std::size_t n_size,
                                    const std::vector<std::size_t>& a_offsets,
                                    const std::vector<std::size_t>& b_offsets,
                                    bool trans_a, bool trans_b) override {
        if (a_offsets.size() != b_offsets.size() || a_offsets.empty()) {
            throw std::invalid_argument("Metal matmul needs one A and B offset per batch");
        }
        const auto batch = a_offsets.size();
        auto output = allocate(batch * m_size * n_size);
        if (batch * m_size * n_size == 0) return output;
        (void)checked_u32(batch * m_size * n_size, "matmul output size");
        const auto m = checked_u32(m_size, "matrix dimension");
        const auto k = checked_u32(k_size, "matrix dimension");
        const auto n = checked_u32(n_size, "matrix dimension");
        std::vector<std::uint32_t> offsets(batch * 2);
        for (std::size_t z = 0; z < batch; ++z) {
            offsets[2 * z] = checked_u32(a_offsets[z], "matmul offset");
            offsets[2 * z + 1] = checked_u32(b_offsets[z], "matmul offset");
        }
        id<MTLBuffer> offset_buffer = [device_ newBufferWithBytes:offsets.data()
                                                           length:offsets.size() * sizeof(std::uint32_t)
                                                          options:MTLResourceStorageModeShared];
        if (!offset_buffer) throw std::runtime_error("Metal buffer allocation failed");
        submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:matmul_pipelines_[(trans_a ? 1 : 0) | (trans_b ? 2 : 0)]];
            [encoder setBuffer:as_metal(a).buffer() offset:0 atIndex:0];
            [encoder setBuffer:as_metal(b).buffer() offset:0 atIndex:1];
            [encoder setBuffer:as_metal(*output).buffer() offset:0 atIndex:2];
            [encoder setBytes:&m length:sizeof(m) atIndex:3];
            [encoder setBytes:&k length:sizeof(k) atIndex:4];
            [encoder setBytes:&n length:sizeof(n) atIndex:5];
            [encoder setBuffer:offset_buffer offset:0 atIndex:6];
            // Launch complete tiles: even edge outputs need all 16 lanes to
            // cooperatively load K-dimension values into threadgroup memory.
            [encoder dispatchThreadgroups:MTLSizeMake((n_size + 15) / 16,
                                                       (m_size + 15) / 16, batch)
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        });
        return output;
    }

    std::shared_ptr<Storage> transpose_last2(const Storage& input, std::size_t batch,
                                             std::size_t rows_size,
                                             std::size_t cols_size) override {
        auto output = allocate(batch * rows_size * cols_size);
        if (batch * rows_size * cols_size == 0) return output;
        (void)checked_u32(batch * rows_size * cols_size, "transpose size");
        const auto rows = checked_u32(rows_size, "matrix dimension");
        const auto cols = checked_u32(cols_size, "matrix dimension");
        submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:transpose_pipeline_];
            [encoder setBuffer:as_metal(input).buffer() offset:0 atIndex:0];
            [encoder setBuffer:as_metal(*output).buffer() offset:0 atIndex:1];
            [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
            [encoder setBytes:&cols length:sizeof(cols) atIndex:3];
            [encoder dispatchThreads:MTLSizeMake(cols_size, rows_size, batch)
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        });
        return output;
    }

    std::shared_ptr<Storage> reduce_axis(const Storage& input, std::size_t outer,
                                         std::size_t length, std::size_t inner,
                                         cpu::ReduceOp operation) override {
        if (length == 0 && operation != cpu::ReduceOp::Sum) {
            throw std::invalid_argument("max/argmax over an empty axis is undefined");
        }
        auto output = allocate(outer * inner);
        AxisMeta meta{checked_u32(outer, "outer size"), checked_u32(length, "axis length"),
                      checked_u32(inner, "inner size"), static_cast<std::uint32_t>(operation)};
        (void)checked_u32(outer * length * inner, "reduction size");
        encode_1d(reduce_axis_pipeline_, outer * inner, [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:as_metal(input).buffer() offset:0 atIndex:0];
            [encoder setBuffer:as_metal(*output).buffer() offset:0 atIndex:1];
            [encoder setBytes:&meta length:sizeof(meta) atIndex:2];
        });
        return output;
    }

    std::shared_ptr<Storage> scatter_axis(const Storage& values, const Storage& indices,
                                          std::size_t outer, std::size_t length,
                                          std::size_t inner) override {
        const auto total = outer * length * inner;
        auto output = allocate(total);
        AxisMeta meta{checked_u32(outer, "outer size"), checked_u32(length, "axis length"),
                      checked_u32(inner, "inner size"), 0};
        (void)checked_u32(total, "scatter size");
        encode_1d(scatter_axis_pipeline_, total, [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:as_metal(values).buffer() offset:0 atIndex:0];
            [encoder setBuffer:as_metal(indices).buffer() offset:0 atIndex:1];
            [encoder setBuffer:as_metal(*output).buffer() offset:0 atIndex:2];
            [encoder setBytes:&meta length:sizeof(meta) atIndex:3];
        });
        return output;
    }

    void axpy_inplace(Storage& target, const Storage& source, std::size_t count,
                      float alpha) override {
        if (count > target.size() || count > source.size()) {
            throw std::out_of_range("Metal axpy exceeds storage");
        }
        const auto size = checked_u32(count, "element count");
        encode_1d(axpy_pipeline_, count, [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:as_metal(target).buffer() offset:0 atIndex:0];
            [encoder setBuffer:as_metal(source).buffer() offset:0 atIndex:1];
            [encoder setBytes:&alpha length:sizeof(alpha) atIndex:2];
            [encoder setBytes:&size length:sizeof(size) atIndex:3];
        });
    }

    std::shared_ptr<Storage> reduce_sum(const Storage& input, std::size_t count) override {
        if (count == 0) {
            const float zero = 0.0F;
            return upload(&zero, 1);
        }
        const Storage* current = &input;
        std::shared_ptr<Storage> owned;
        std::size_t current_count = count;
        constexpr std::size_t threads = 256;
        while (current_count > 1) {
            const auto groups = (current_count + threads - 1) / threads;
            auto next = allocate(groups);
            const auto size = checked_u32(current_count, "reduction element count");
            submit([&](id<MTLComputeCommandEncoder> encoder) {
                [encoder setComputePipelineState:reduce_pipeline_];
                [encoder setBuffer:as_metal(*current).buffer() offset:0 atIndex:0];
                [encoder setBuffer:as_metal(*next).buffer() offset:0 atIndex:1];
                [encoder setBytes:&size length:sizeof(size) atIndex:2];
                [encoder setThreadgroupMemoryLength:threads * sizeof(float) atIndex:0];
                [encoder dispatchThreads:MTLSizeMake(groups * threads, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            });
            owned = std::move(next);
            current = owned.get();
            current_count = groups;
        }
        if (!owned) {
            synchronize();  // the input may be the pending output of queued work
            auto result = allocate(1);
            std::copy(static_cast<const float*>(as_metal(input).buffer().contents),
                      static_cast<const float*>(as_metal(input).buffer().contents) + 1,
                      static_cast<float*>(as_metal(*result).buffer().contents));
            return result;
        }
        return owned;
    }

    void synchronize() override {
        if (!ready_) return;
        id<MTLCommandBuffer> last = nil;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            commit_locked();
            last = last_committed_;
        }
        // Command buffers on one queue complete in order, so waiting for the
        // most recent one waits for all of them.
        if (last) [last waitUntilCompleted];
        // Completion handlers may run slightly after the wait returns, so
        // check the last command buffer's status directly as well.
        if (last && last.status == MTLCommandBufferStatusError) {
            throw std::runtime_error("Metal command failed: " + utf8(last.error.localizedDescription));
        }
        {
            // Everything submitted so far has finished: buffers released since
            // the last commit can go straight to the free list.
            std::lock_guard<std::mutex> lock(pool_mutex_);
            recycle_locked(pending_release_);
            pending_release_.clear();
        }
        std::string error;
        {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            error.swap(async_error_);
        }
        if (!error.empty()) throw std::runtime_error("Metal command failed: " + error);
    }

private:
    id<MTLComputePipelineState> matmul_pipeline(bool trans_a, bool trans_b) {
        MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];
        [constants setConstantValue:&trans_a type:MTLDataTypeBool atIndex:0];
        [constants setConstantValue:&trans_b type:MTLDataTypeBool atIndex:1];
        NSError* error = nil;
        id<MTLFunction> function = [library_ newFunctionWithName:@"matmul_tiled"
                                                  constantValues:constants
                                                           error:&error];
        if (!function) {
            error_ = "Metal matmul specialisation failed: " + utf8(error.localizedDescription);
            return nil;
        }
        auto result = [device_ newComputePipelineStateWithFunction:function error:&error];
        if (!result) error_ = "pipeline creation failed for matmul_tiled: " +
                              utf8(error.localizedDescription);
        return result;
    }

    id<MTLComputePipelineState> pipeline(NSString* name) {
        NSError* error = nil;
        id<MTLFunction> function = [library_ newFunctionWithName:name];
        if (!function) {
            error_ = "Metal function not found: " + utf8(name);
            return nil;
        }
        auto result = [device_ newComputePipelineStateWithFunction:function error:&error];
        if (!result) error_ = "pipeline creation failed for " + utf8(name) + ": " +
                              utf8(error.localizedDescription);
        return result;
    }

    void require_ready() const {
        if (!ready_) throw std::runtime_error("Metal backend unavailable: " + error_);
    }

    // Size classes: powers of two up to 4 MiB, then 1 MiB steps, so freed
    // buffers are reusable by similar-sized tensors with bounded waste.
    static std::size_t bucket_bytes(std::size_t bytes) {
        constexpr std::size_t min_bucket = 256;
        constexpr std::size_t pow2_limit = 4U << 20;
        constexpr std::size_t step = 1U << 20;
        if (bytes <= min_bucket) return min_bucket;
        if (bytes <= pow2_limit) {
            std::size_t bucket = min_bucket;
            while (bucket < bytes) bucket <<= 1;
            return bucket;
        }
        return (bytes + step - 1) / step * step;
    }

    void recycle_locked(const std::vector<id<MTLBuffer>>& buffers) {
        constexpr std::size_t max_pooled_bytes = 1ULL << 30;  // beyond this, let ARC free
        for (id<MTLBuffer> buffer : buffers) {
            const auto bytes = static_cast<std::size_t>(buffer.length);
            if (pooled_bytes_ + bytes > max_pooled_bytes) continue;
            free_buffers_[bytes].push_back(buffer);
            pooled_bytes_ += bytes;
        }
    }

    // Encode into the shared command buffer. Work is committed every
    // kOpsPerCommandBuffer operations (so the GPU starts early) and otherwise
    // only when the host needs results (synchronize/download). A serial
    // compute encoder with tracked resources runs dispatches in order and
    // makes each one's writes visible to the next.
    template <typename Configure>
    void submit(Configure configure) {
        require_ready();
        @autoreleasepool {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (!current_) {
                current_ = [queue_ commandBuffer];
                encoder_ = [current_ computeCommandEncoder];
            }
            configure(encoder_);
            if (++encoded_ops_ >= kOpsPerCommandBuffer) commit_locked();
        }
    }

    void commit_locked() {
        if (!current_) return;
        [encoder_ endEncoding];
        // Buffers freed before this commit may be used by this or earlier
        // command buffers; once this one completes, all of those have too.
        // The list stays in the backend under pool_mutex_ (not copied into the
        // block) so the hand-off to Metal's callback thread is mutex-ordered.
        std::uint64_t batch = 0;
        {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            batch = next_batch_++;
            in_flight_[batch].swap(pending_release_);
        }
        auto* self = this;
        [current_ addCompletedHandler:^(id<MTLCommandBuffer> command) {
            std::lock_guard<std::mutex> lock(self->pool_mutex_);
            if (command.status == MTLCommandBufferStatusError && self->async_error_.empty()) {
                self->async_error_ = utf8(command.error.localizedDescription);
            }
            auto found = self->in_flight_.find(batch);
            if (found == self->in_flight_.end()) return;
            if (backend_alive().load()) self->recycle_locked(found->second);
            self->in_flight_.erase(found);
        }];
        [current_ commit];
        last_committed_ = current_;
        current_ = nil;
        encoder_ = nil;
        encoded_ops_ = 0;
    }

    template <typename Configure>
    void encode_1d(id<MTLComputePipelineState> pipeline_state, std::size_t count,
                   Configure configure) {
        if (count == 0) return;
        submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline_state];
            configure(encoder);
            const auto width = std::min<std::size_t>(256, pipeline_state.maxTotalThreadsPerThreadgroup);
            [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        });
    }

    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLLibrary> library_;
    id<MTLComputePipelineState> binary_pipeline_;
    id<MTLComputePipelineState> unary_pipeline_;
    id<MTLComputePipelineState> matmul_pipelines_[4];  // index: trans_a | trans_b << 1
    id<MTLComputePipelineState> transpose_pipeline_;
    id<MTLComputePipelineState> reduce_pipeline_;
    id<MTLComputePipelineState> reduce_axis_pipeline_;
    id<MTLComputePipelineState> scatter_axis_pipeline_;
    id<MTLComputePipelineState> axpy_pipeline_;
    static constexpr std::size_t kOpsPerCommandBuffer = 64;
    id<MTLCommandBuffer> current_{nil};
    id<MTLComputeCommandEncoder> encoder_{nil};
    id<MTLCommandBuffer> last_committed_{nil};
    std::size_t encoded_ops_{0};
    std::mutex pool_mutex_;
    std::unordered_map<std::size_t, std::vector<id<MTLBuffer>>> free_buffers_;
    std::vector<id<MTLBuffer>> pending_release_;
    std::unordered_map<std::uint64_t, std::vector<id<MTLBuffer>>> in_flight_;
    std::uint64_t next_batch_{0};
    std::size_t pooled_bytes_{0};
    std::string async_error_;
    bool ready_{false};
    std::string error_;
    std::mutex queue_mutex_;
};

MetalBackendImpl& backend_impl() {
    static MetalBackendImpl backend;
    return backend;
}

void release_buffer(id<MTLBuffer> buffer) {
    if (buffer && MetalBackendImpl::backend_alive().load()) backend_impl().release(buffer);
}

} // namespace

MetalBackend& MetalBackend::instance() { return backend_impl(); }

void synchronize() {
    if (MetalBackend::instance().available()) MetalBackend::instance().synchronize();
}

bool metal_available() { return MetalBackend::instance().available(); }
std::string metal_device_name() { return MetalBackend::instance().device_name(); }

} // namespace vectorcore
