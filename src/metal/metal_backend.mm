#include "vectorcore/metal_backend.hpp"
#include "metal_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

namespace vectorcore {

namespace {

std::string utf8(NSString* value) {
    return value ? std::string(value.UTF8String) : std::string{};
}

class MetalStorage final : public Storage {
public:
    MetalStorage(id<MTLBuffer> buffer, std::size_t elements)
        : buffer_(buffer), elements_(elements) {}
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

struct ReduceShapeMeta {
    std::uint32_t rank{};
    std::uint32_t target_rank{};
    std::uint32_t source_count{};
    std::uint32_t target_count{};
    std::uint32_t source_shape[8]{};
    std::uint32_t target_shape[8]{};
    std::uint32_t target_stride[8]{};
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
            matmul_pipeline_ = pipeline(@"matmul_tiled");
            transpose_pipeline_ = pipeline(@"transpose_2d");
            reduce_pipeline_ = pipeline(@"reduce_sum_pass");
            reduce_shape_pipeline_ = pipeline(@"reduce_to_shape");
            ready_ = binary_pipeline_ && unary_pipeline_ && matmul_pipeline_ &&
                     transpose_pipeline_ && reduce_pipeline_ && reduce_shape_pipeline_;
            if (!ready_ && error_.empty()) error_ = "one or more Metal pipelines failed to initialize";
        }
    }

    bool available() const noexcept override { return ready_; }
    std::string device_name() const override { return device_ ? utf8(device_.name) : "unavailable"; }
    std::string last_error() const override { return error_; }

    std::shared_ptr<Storage> allocate(std::size_t elements) override {
        require_ready();
        const auto bytes = std::max<std::size_t>(elements * sizeof(float), sizeof(float));
        id<MTLBuffer> buffer = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buffer) throw std::runtime_error("Metal buffer allocation failed");
        return std::make_shared<MetalStorage>(buffer, elements);
    }

    std::shared_ptr<Storage> upload(const float* data, std::size_t elements) override {
        auto storage = allocate(elements);
        if (elements) std::copy(data, data + elements,
                                static_cast<float*>(as_metal(*storage).buffer().contents));
        return storage;
    }

    void download(const Storage& source, float* destination, std::size_t elements) override {
        if (elements > source.size()) throw std::out_of_range("Metal download exceeds storage");
        synchronize();
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
                                    std::size_t m_size, std::size_t k_size,
                                    std::size_t n_size) override {
        auto output = allocate(m_size * n_size);
        const auto m = checked_u32(m_size, "matrix dimension");
        const auto k = checked_u32(k_size, "matrix dimension");
        const auto n = checked_u32(n_size, "matrix dimension");
        submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:matmul_pipeline_];
            [encoder setBuffer:as_metal(a).buffer() offset:0 atIndex:0];
            [encoder setBuffer:as_metal(b).buffer() offset:0 atIndex:1];
            [encoder setBuffer:as_metal(*output).buffer() offset:0 atIndex:2];
            [encoder setBytes:&m length:sizeof(m) atIndex:3];
            [encoder setBytes:&k length:sizeof(k) atIndex:4];
            [encoder setBytes:&n length:sizeof(n) atIndex:5];
            // Launch complete tiles: even edge outputs need all 16 lanes to
            // cooperatively load K-dimension values into threadgroup memory.
            [encoder dispatchThreadgroups:MTLSizeMake((n_size + 15) / 16,
                                                       (m_size + 15) / 16, 1)
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        });
        return output;
    }

    std::shared_ptr<Storage> transpose2d(const Storage& input, std::size_t rows_size,
                                         std::size_t cols_size) override {
        auto output = allocate(rows_size * cols_size);
        const auto rows = checked_u32(rows_size, "matrix dimension");
        const auto cols = checked_u32(cols_size, "matrix dimension");
        submit([&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:transpose_pipeline_];
            [encoder setBuffer:as_metal(input).buffer() offset:0 atIndex:0];
            [encoder setBuffer:as_metal(*output).buffer() offset:0 atIndex:1];
            [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
            [encoder setBytes:&cols length:sizeof(cols) atIndex:3];
            [encoder dispatchThreads:MTLSizeMake(cols_size, rows_size, 1)
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        });
        return output;
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
            auto result = allocate(1);
            std::copy(static_cast<const float*>(as_metal(input).buffer().contents),
                      static_cast<const float*>(as_metal(input).buffer().contents) + 1,
                      static_cast<float*>(as_metal(*result).buffer().contents));
            return result;
        }
        return owned;
    }

    std::shared_ptr<Storage> reduce_to_shape(const Storage& input,
                                             const Shape& input_shape,
                                             const Shape& target_shape) override {
        if (input_shape.size() > 8 || target_shape.size() > 8 ||
            target_shape.size() > input_shape.size()) {
            throw std::invalid_argument("Metal shape reduction supports compatible ranks up to 8");
        }
        const auto target_count = vectorcore::numel(target_shape);
        auto output = allocate(target_count);
        ReduceShapeMeta meta;
        meta.rank = checked_u32(input_shape.size(), "rank");
        meta.target_rank = checked_u32(target_shape.size(), "rank");
        meta.source_count = checked_u32(vectorcore::numel(input_shape), "element count");
        meta.target_count = checked_u32(target_count, "element count");
        const auto target_strides = contiguous_strides(target_shape);
        for (std::size_t i = 0; i < input_shape.size(); ++i) {
            meta.source_shape[i] = checked_u32(input_shape[i], "dimension");
        }
        for (std::size_t i = 0; i < target_shape.size(); ++i) {
            meta.target_shape[i] = checked_u32(target_shape[i], "dimension");
            meta.target_stride[i] = checked_u32(target_strides[i], "stride");
        }
        encode_1d(reduce_shape_pipeline_, target_count,
                  [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:as_metal(input).buffer() offset:0 atIndex:0];
            [encoder setBuffer:as_metal(*output).buffer() offset:0 atIndex:1];
            [encoder setBytes:&meta length:sizeof(meta) atIndex:2];
        });
        return output;
    }

    void synchronize() override {
        // Operations currently wait at submission boundaries, so this method is
        // intentionally a no-op and remains part of the public timing contract.
    }

private:
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

    template <typename Configure>
    void submit(Configure configure) {
        require_ready();
        @autoreleasepool {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            id<MTLCommandBuffer> command = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            configure(encoder);
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status == MTLCommandBufferStatusError) {
                throw std::runtime_error("Metal command failed: " + utf8(command.error.localizedDescription));
            }
        }
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
    id<MTLComputePipelineState> matmul_pipeline_;
    id<MTLComputePipelineState> transpose_pipeline_;
    id<MTLComputePipelineState> reduce_pipeline_;
    id<MTLComputePipelineState> reduce_shape_pipeline_;
    bool ready_{false};
    std::string error_;
    std::mutex queue_mutex_;
};

} // namespace

MetalBackend& MetalBackend::instance() {
    static MetalBackendImpl backend;
    return backend;
}

bool metal_available() { return MetalBackend::instance().available(); }
std::string metal_device_name() { return MetalBackend::instance().device_name(); }

} // namespace vectorcore
