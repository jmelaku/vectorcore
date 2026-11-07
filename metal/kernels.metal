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

constant uint TILE = 16;
// Specialisation constants: the host builds one matmul pipeline per
// (TRANS_A, TRANS_B) pair, so the transpose choice costs nothing at run time.
constant bool TRANS_A [[function_constant(0)]];
constant bool TRANS_B [[function_constant(1)]];

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
    // TRANS_A: A is stored (k x m); TRANS_B: B is stored (n x k). Folding the
    // transpose into the loads lets backward passes skip materialising it.
    threadgroup float tile_a[16][16];
    threadgroup float tile_b[16][16];
    const device float* batch_a = a + offsets[gid.z].x;
    const device float* batch_b = b + offsets[gid.z].y;
    float accumulator = 0.0f;
    for (uint tile = 0; tile < (k + TILE - 1) / TILE; ++tile) {
        uint a_col = tile * TILE + tid.x;
        uint b_row = tile * TILE + tid.y;
        tile_a[tid.y][tid.x] = gid.y < m && a_col < k
            ? (TRANS_A ? batch_a[a_col * m + gid.y] : batch_a[gid.y * k + a_col]) : 0.0f;
        tile_b[tid.y][tid.x] = b_row < k && gid.x < n
            ? (TRANS_B ? batch_b[gid.x * k + b_row] : batch_b[b_row * n + gid.x]) : 0.0f;
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

#ifdef VECTORCORE_SIMDGROUP_MATRIX
#include <metal_simdgroup_matrix>

// GEMM on Apple-family GPUs using the hardware 8x8 simdgroup matrix
// multiply-accumulate. A threadgroup of 4 simdgroups (128 threads) computes a
// 64x64 tile of C; the simdgroups form a 2x2 grid and each accumulates a 32x32
// block as 4x4 simdgroup_float8x8 fragments. Each K step stages a 64x16 slice
// of A and a 16x64 slice of B in threadgroup memory, double-buffered so the
// next slice loads while the current one is multiplied.
constant uint SG_BM = 64;
constant uint SG_BN = 64;
constant uint SG_BK = 16;

// Each of the 128 threads loads 8 elements of A and 8 of B per K step. The
// thread-to-element mapping follows each operand's storage order so that
// neighbouring threads read neighbouring addresses (coalesced loads).
inline void load_gemm_tiles(threadgroup float* tile_a, threadgroup float* tile_b,
                            const device float* a, const device float* b,
                            uint m, uint k, uint n, uint row0, uint col0, uint k0, uint tid) {
    for (uint i = 0; i < 8; ++i) {
        const uint t = tid + i * 128;
        uint r, c;
        if (TRANS_A) { r = t % SG_BM; c = t / SG_BM; }
        else         { r = t / SG_BK; c = t % SG_BK; }
        const uint gr = row0 + r, gc = k0 + c;
        float v = 0.0f;
        if (gr < m && gc < k) v = TRANS_A ? a[gc * m + gr] : a[gr * k + gc];
        tile_a[r * SG_BK + c] = v;

        uint kr, cc;
        if (TRANS_B) { kr = t % SG_BK; cc = t / SG_BK; }
        else         { kr = t / SG_BN; cc = t % SG_BN; }
        const uint gk = k0 + kr, gn = col0 + cc;
        float w = 0.0f;
        if (gk < k && gn < n) w = TRANS_B ? b[gn * k + gk] : b[gk * n + gn];
        tile_b[kr * SG_BN + cc] = w;
    }
}

kernel void matmul_simdgroup(const device float* a [[buffer(0)]],
                             const device float* b [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant uint& m [[buffer(3)]],
                             constant uint& k [[buffer(4)]],
                             constant uint& n [[buffer(5)]],
                             const device uint2* offsets [[buffer(6)]],
                             uint3 group [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tile_a[2][SG_BM * SG_BK];
    threadgroup float tile_b[2][SG_BK * SG_BN];

    const device float* batch_a = a + offsets[group.z].x;
    const device float* batch_b = b + offsets[group.z].y;
    const uint row0 = group.y * SG_BM;
    const uint col0 = group.x * SG_BN;
    const uint sg_row = (sg / 2) * 32;
    const uint sg_col = (sg % 2) * 32;

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0.0f);

    // Interior fast path: when the whole 64x64 tile is in bounds and k is a
    // multiple of 8, every 8x8 fragment can be loaded straight from device
    // memory (the GPU caches provide the reuse that threadgroup staging would),
    // with no bounds checks and no barriers. simdgroup_load's transpose flag
    // handles transposed operands. The decision is uniform per threadgroup.
    if (row0 + SG_BM <= m && col0 + SG_BN <= n && (k % 8) == 0) {
        const uint a_row = row0 + sg_row;
        const uint b_col = col0 + sg_col;
        for (uint k0 = 0; k0 < k; k0 += 8) {
            simdgroup_float8x8 fa[4];
            simdgroup_float8x8 fb[4];
            for (uint i = 0; i < 4; ++i) {
                if (TRANS_A) simdgroup_load(fa[i], batch_a + ulong(k0) * m + a_row + i * 8, m, ulong2(0), true);
                else         simdgroup_load(fa[i], batch_a + ulong(a_row + i * 8) * k + k0, k);
                if (TRANS_B) simdgroup_load(fb[i], batch_b + ulong(b_col + i * 8) * k + k0, k, ulong2(0), true);
                else         simdgroup_load(fb[i], batch_b + ulong(k0) * n + b_col + i * 8, n);
            }
            for (uint i = 0; i < 4; ++i)
                for (uint j = 0; j < 4; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        device float* batch_out = output + group.z * m * n;
        for (uint i = 0; i < 4; ++i)
            for (uint j = 0; j < 4; ++j)
                simdgroup_store(acc[i][j], batch_out + ulong(a_row + i * 8) * n + b_col + j * 8, n);
        return;
    }

    const uint steps = (k + SG_BK - 1) / SG_BK;
    load_gemm_tiles(tile_a[0], tile_b[0], batch_a, batch_b, m, k, n, row0, col0, 0, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 0; step < steps; ++step) {
        const uint current = step & 1;
        if (step + 1 < steps) {
            load_gemm_tiles(tile_a[current ^ 1], tile_b[current ^ 1], batch_a, batch_b,
                            m, k, n, row0, col0, (step + 1) * SG_BK, tid);
        }
        for (uint kk = 0; kk < SG_BK; kk += 8) {
            simdgroup_float8x8 fa[4];
            simdgroup_float8x8 fb[4];
            for (uint i = 0; i < 4; ++i) {
                simdgroup_load(fa[i], &tile_a[current][(sg_row + i * 8) * SG_BK + kk], SG_BK);
                simdgroup_load(fb[i], &tile_b[current][kk * SG_BN + sg_col + i * 8], SG_BN);
            }
            for (uint i = 0; i < 4; ++i)
                for (uint j = 0; j < 4; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        // One barrier per step: it both publishes the next buffer and stops
        // the following load from overwriting the buffer just consumed.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* batch_out = output + group.z * m * n;
    const uint out_row = row0 + sg_row;
    const uint out_col = col0 + sg_col;
    if (out_row + 32 <= m && out_col + 32 <= n) {
        for (uint i = 0; i < 4; ++i)
            for (uint j = 0; j < 4; ++j)
                simdgroup_store(acc[i][j], batch_out + (out_row + i * 8) * n + out_col + j * 8, n);
        return;
    }
    // Edge block: stage through threadgroup memory (reusing the A tiles, which
    // are free after the final barrier) and write only the valid elements.
    // The A tiles hold 2048 floats: 512 per simdgroup, i.e. one 32x16 half of
    // its 32x32 block at a time.
    threadgroup float* stage = &tile_a[0][0] + sg * 512;
    for (uint part = 0; part < 2; ++part) {
        for (uint i = 0; i < 4; ++i)
            for (uint j = 0; j < 2; ++j)
                simdgroup_store(acc[i][part * 2 + j], stage + (i * 8) * 16 + j * 8, 16);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 32 * 16; e += 32) {
            const uint r = e / 16, c = e % 16;
            const uint gr = out_row + r, gc = out_col + part * 16 + c;
            if (gr < m && gc < n) batch_out[gr * n + gc] = stage[r * 16 + c];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
#endif
