#include <metal_stdlib>
using namespace metal;

struct BroadcastMeta {
    uint rank;
    uint a_rank;
    uint b_rank;
    uint operation;
    uint out_shape[8];
    uint a_shape[8];
    uint a_stride[8];
    uint b_shape[8];
    uint b_stride[8];
};

kernel void binary_broadcast(const device float* a [[buffer(0)]],
                             const device float* b [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant BroadcastMeta& meta [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    uint total = 1;
    for (uint axis = 0; axis < meta.rank; ++axis) total *= meta.out_shape[axis];
    if (gid >= total) return;
    uint remainder = gid;
    uint a_offset = 0;
    uint b_offset = 0;
    for (uint reverse_axis = 0; reverse_axis < meta.rank; ++reverse_axis) {
        uint axis = meta.rank - 1 - reverse_axis;
        uint coordinate = remainder % meta.out_shape[axis];
        remainder /= meta.out_shape[axis];
        if (axis >= meta.rank - meta.a_rank) {
            uint input_axis = axis - (meta.rank - meta.a_rank);
            if (meta.a_shape[input_axis] != 1) a_offset += coordinate * meta.a_stride[input_axis];
        }
        if (axis >= meta.rank - meta.b_rank) {
            uint input_axis = axis - (meta.rank - meta.b_rank);
            if (meta.b_shape[input_axis] != 1) b_offset += coordinate * meta.b_stride[input_axis];
        }
    }
    float av = a[a_offset];
    float bv = b[b_offset];
    switch (meta.operation) {
        case 0: output[gid] = av + bv; break;
        case 1: output[gid] = av - bv; break;
        case 2: output[gid] = av * bv; break;
        default: output[gid] = av / bv; break;
    }
}

kernel void unary(const device float* input [[buffer(0)]],
                  device float* output [[buffer(1)]],
                  constant uint& operation [[buffer(2)]],
                  constant uint& count [[buffer(3)]],
                  uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float value = input[gid];
    switch (operation) {
        case 0: output[gid] = -value; break;
        case 1: output[gid] = max(value, 0.0f); break;
        case 2: output[gid] = 1.0f / (1.0f + exp(-value)); break;
        case 3: output[gid] = value > 0.0f ? 1.0f : 0.0f; break;
        case 4: output[gid] = exp(value); break;
        default: output[gid] = log(value); break;
    }
}

struct ReduceShapeMeta {
    uint rank;
    uint target_rank;
    uint source_count;
    uint target_count;
    uint source_shape[8];
    uint target_shape[8];
    uint target_stride[8];
};

// One thread owns each target element, so no floating-point atomics are needed.
// This path is used for broadcast-gradient reduction; full sum uses the faster
// parallel tree kernel below.
kernel void reduce_to_shape(const device float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant ReduceShapeMeta& meta [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= meta.target_count) return;
    float accumulator = 0.0f;
    for (uint source_linear = 0; source_linear < meta.source_count; ++source_linear) {
        uint remainder = source_linear;
        uint target_offset = 0;
        for (uint reverse_axis = 0; reverse_axis < meta.rank; ++reverse_axis) {
            uint axis = meta.rank - 1 - reverse_axis;
            uint coordinate = remainder % meta.source_shape[axis];
            remainder /= meta.source_shape[axis];
            if (axis >= meta.rank - meta.target_rank) {
                uint target_axis = axis - (meta.rank - meta.target_rank);
                if (meta.target_shape[target_axis] != 1)
                    target_offset += coordinate * meta.target_stride[target_axis];
            }
        }
        if (target_offset == gid) accumulator += input[source_linear];
    }
    output[gid] = accumulator;
}

constant uint TILE = 16;

// Batched tiled GEMM. Threadgroup z selects the batch; offsets[z] holds the
// element offsets of that batch's A and B matrices, which lets broadcast batch
// dimensions reuse one operand without copying it.
kernel void matmul_tiled(const device float* a [[buffer(0)]],
                         const device float* b [[buffer(1)]],
                         device float* output [[buffer(2)]],
                         constant uint& m [[buffer(3)]],
                         constant uint& k [[buffer(4)]],
                         constant uint& n [[buffer(5)]],
                         const device uint2* offsets [[buffer(6)]],
                         uint3 tid [[thread_position_in_threadgroup]],
                         uint3 gid [[thread_position_in_grid]]) {
    threadgroup float tile_a[16][16];
    threadgroup float tile_b[16][16];
    const device float* batch_a = a + offsets[gid.z].x;
    const device float* batch_b = b + offsets[gid.z].y;
    float accumulator = 0.0f;
    for (uint tile = 0; tile < (k + TILE - 1) / TILE; ++tile) {
        uint a_col = tile * TILE + tid.x;
        uint b_row = tile * TILE + tid.y;
        tile_a[tid.y][tid.x] = gid.y < m && a_col < k ? batch_a[gid.y * k + a_col] : 0.0f;
        tile_b[tid.y][tid.x] = b_row < k && gid.x < n ? batch_b[b_row * n + gid.x] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < TILE; ++inner) {
            accumulator += tile_a[tid.y][inner] * tile_b[inner][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (gid.y < m && gid.x < n) output[gid.z * m * n + gid.y * n + gid.x] = accumulator;
}

// Swaps the last two axes of a contiguous [batch, rows, cols] tensor.
kernel void transpose_2d(const device float* input [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         constant uint& rows [[buffer(2)]],
                         constant uint& cols [[buffer(3)]],
                         uint3 gid [[thread_position_in_grid]]) {
    if (gid.y < rows && gid.x < cols) {
        uint base = gid.z * rows * cols;
        output[base + gid.x * rows + gid.y] = input[base + gid.y * cols + gid.x];
    }
}

kernel void reduce_sum_pass(const device float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            threadgroup float* scratch [[threadgroup(0)]],
                            uint gid [[thread_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint group [[threadgroup_position_in_grid]]) {
    scratch[tid] = gid < count ? input[gid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[group] = scratch[0];
}

struct AxisMeta {
    uint outer;
    uint length;
    uint inner;
    uint operation;  // 0 sum, 1 max, 2 argmax
};

// One thread per [outer, inner] output; walks the reduced axis sequentially.
kernel void reduce_axis(const device float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant AxisMeta& meta [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= meta.outer * meta.inner) return;
    uint o = gid / meta.inner;
    uint i = gid % meta.inner;
    const device float* base = input + o * meta.length * meta.inner + i;
    if (meta.operation == 0) {
        float total = 0.0f;
        for (uint a = 0; a < meta.length; ++a) total += base[a * meta.inner];
        output[gid] = total;
        return;
    }
    uint best = 0;
    float best_value = base[0];
    for (uint a = 1; a < meta.length; ++a) {
        float value = base[a * meta.inner];
        if (value > best_value) {
            best_value = value;
            best = a;
        }
    }
    output[gid] = meta.operation == 1 ? best_value : float(best);
}

// One thread per full-size output element: it receives the upstream value only
// if its position along the axis is the recorded argmax.
kernel void scatter_axis(const device float* values [[buffer(0)]],
                         const device float* indices [[buffer(1)]],
                         device float* output [[buffer(2)]],
                         constant AxisMeta& meta [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= meta.outer * meta.length * meta.inner) return;
    uint i = gid % meta.inner;
    uint a = (gid / meta.inner) % meta.length;
    uint o = gid / (meta.inner * meta.length);
    uint reduced = o * meta.inner + i;
    output[gid] = uint(indices[reduced]) == a ? values[reduced] : 0.0f;
}

// target += alpha * source, in place (used by the optimizer).
kernel void axpy_inplace(device float* target [[buffer(0)]],
                         const device float* source [[buffer(1)]],
                         constant float& alpha [[buffer(2)]],
                         constant uint& count [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid < count) target[gid] += alpha * source[gid];
}
