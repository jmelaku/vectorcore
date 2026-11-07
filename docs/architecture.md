# VectorCore architecture

## Tensor and storage

`Tensor` is a value-like handle containing `shared_ptr<TensorImpl>`. `TensorImpl` stores a `shared_ptr<Storage>`, `Shape`, contiguous `Strides`, storage offset, float32 dtype, and autograd state. Copying a `Tensor` is cheap and preserves identity, which is important for gradient accumulation. CPU storage owns a `std::vector<float>`. Metal storage owns an `MTLBuffer`; Objective-C ARC and the C++ virtual destructor jointly provide deterministic resource release.

A scalar uses an empty shape and has one element. A dimension may be zero; products containing it have zero elements. `sum` of an empty tensor is zero and `mean` rejects it. Shape products use checked multiplication. `reshape` validates equal element counts and shares contiguous storage; operations and transpose return newly allocated contiguous tensors.

## Broadcasting

Shapes are aligned from the trailing dimension. At each axis, dimensions must match or one must equal one. CPU kernels iterate the output linear index, reconstruct coordinates, and map singleton axes to a zero input offset. Metal passes fixed-size rank/shape/stride metadata to `binary_broadcast`, which performs the same mapping in each GPU thread. Neither backend expands broadcast operands.

Backward uses `sum_to_shape`: the target shape is aligned to the gradient's rank, and every axis that was introduced or stretched by broadcasting is summed with the axis-reduction kernel (keepdim), followed by a reshape. Each pass is a contiguous reduction on the gradient's own device, so no host transfer is involved. The reverse expansion, `broadcast_to`, reuses the binary broadcasting kernel with a scalar zero and an explicit output shape instead of multiplying by a materialised tensor of ones.

## CPU execution

A persistent thread pool (`src/cpu/parallel.cpp`) runs `parallel_for(count, grain, body)`: the range is split into fixed chunks that workers and the calling thread claim from an atomic counter, so faster performance cores naturally take more chunks than efficiency cores. Ranges at or below one grain and nested calls run inline. Because the chunk partition depends only on the range and grain, reductions that combine per-chunk partials in chunk order (full sum, for example) are deterministic regardless of thread count.

Elementwise kernels dispatch the operation once (templated functors) rather than per element. Same-shape and scalar-operand cases are straight loops; general broadcasting resolves the two input offsets once per output row and then streams the last axis with a step of 0 or 1. Output buffers are allocated without zero-filling because every kernel writes its whole output.

GEMM follows the BLIS/GotoBLAS structure. Operands are described by (row stride, column stride), so a transposed operand is read in place. A is packed once into 8-row slivers and B into 12-column slivers for every 256-deep panel (zero-padded at the edges). C is divided into 32x48 tiles; each parallel task owns one tile and runs all depth panels for it, so tasks never synchronise and C stays in L1. The inner 8x12 microkernel keeps 24 NEON accumulators in registers and performs 24 fused multiply-adds per depth step from two A and three B vector loads; a scalar version is used on non-ARM targets, and products below 32^3 skip packing. Matmul backward calls the kernel with transpose flags (`dA = dY @ B^T`, `dB = A^T @ dY`) and skips gradients for inputs that do not require them.

## Autograd

Every differentiable operation creates a result and conditionally attaches `GradFn` when global gradient recording is enabled and at least one parent requires gradients. A node owns its parents and a local vector-Jacobian-product callback. Parents do not own children, so the graph has no ownership cycle.

`backward` performs a visited-set depth-first topological traversal. It walks the order in reverse, accumulates contributions in a map keyed by `TensorImpl*`, stores gradients on participating tensors, and invokes each local backward rule. `NoGradGuard` disables graph construction during this work. Only leaves (tensors created directly by the user, such as parameters) store `.grad`; a non-leaf stores one only after `retain_grad()`. Each node's accumulated gradient is erased from the working map as soon as it has been propagated, so intermediate gradients do not all stay alive until the pass ends. Existing gradients are added, matching repeated-backward accumulation semantics. `zero_grad` explicitly releases a stored gradient.

Every `Storage` carries a version counter shared by all views of it. `Tensor::mutable_data()` (used by `set_at` and `sgd_step`) bumps it. Backward rules never capture tensors directly; they capture a `SavedTensor`, which records the version at forward time and throws from `unpack()` if it has changed. This turns a silently wrong gradient—e.g. an optimizer step between forward and backward—into a descriptive error, the same guarantee PyTorch provides.

Backward rules are built from native VectorCore operations: matrix multiplication uses `dA = dY @ B.T` and `dB = A.T @ dY`; sigmoid saves a detached output; ReLU saves a detached input and computes its mask on the active device; reshape restores the source shape; device transfer moves the gradient back to its source device.

## Reductions, softmax and classification

Axis reductions view a contiguous tensor as `[outer, length, inner]` around the reduced axis and produce `[outer, inner]`; `keepdim` only changes the reported shape. CPU kernels accumulate sums in double precision; Metal runs one thread per output element. `max` also records an argmax (first index wins ties) and its backward rule scatters the upstream gradient to that single position, matching PyTorch's `max(dim)`. `sum` backward broadcasts the gradient back with a keepdim-shaped reshape.

`log_softmax(x) = (x - m) - log(sum(exp(x - m)))` with `m = max(x.detach())`. Because log-softmax is invariant to subtracting a per-row constant, treating `m` as a constant keeps the gradient exact while preventing `exp` overflow. `softmax` is `exp(log_softmax)`. `cross_entropy` builds a one-hot mask from the integer targets on the logits' device and returns `-(log_softmax(logits) * mask).sum() / batch`, so its gradient is composed entirely from verified primitives.

Matrix multiplication treats all but the last two axes as batch dimensions that broadcast like elementwise operands. The forward pass computes, per output batch, the element offsets of the A and B matrices it reads, so a broadcast operand is reused rather than copied; Metal receives these offsets in a buffer indexed by the threadgroup's z coordinate. In backward, `dA = dY @ B^T` and `dB = A^T @ dY` are batched products whose results are reduced with `sum_to_shape` back to each operand's original (possibly un-batched) shape.

`sgd_step` updates CPU parameters through `mutable_data()` and Metal parameters through an in-place `axpy` kernel; both bump the storage version, so stale graphs are still rejected.

## Metal execution

The build embeds `metal/kernels.metal` into the library. `MetalBackend` obtains the default device, creates one command queue, compiles the MSL source with `newLibraryWithSource`, and caches all compute pipeline states. Failure leaves the backend unavailable with a diagnostic. There is no CPU fallback inside Metal operations.

Buffers use shared storage, appropriate for Apple unified memory. Upload/download still copy deliberately so tensor ownership and timing boundaries remain unambiguous.

Execution is asynchronous. Operations are encoded under a mutex into one shared compute command buffer; a serial compute encoder with hazard-tracked resources runs dispatches in order and makes each dispatch's writes visible to the next, including in-place optimizer updates. The command buffer is committed (without waiting) every 64 operations so the GPU starts early, and otherwise when the host needs data: `download`, `item()`, `to('cpu')` and `vc.synchronize()` commit and wait for the last command buffer, which on a single queue implies all earlier ones. Errors are recorded by completion handlers and by checking the last command buffer, and thrown at that synchronization point.

Freed `MTLBuffer`s go through a pool. When a Metal tensor dies its buffer may still be referenced by queued or running work, so it is parked until the next commit and moves to the free list from that command buffer's completion handler; the hand-off is keyed by batch id under the pool mutex. Every buffer in the free list is therefore idle, so `allocate` may hand it to `upload` for an immediate CPU write. Size classes are powers of two up to 4 MiB and then 1 MiB steps, and the pool is capped at 1 GiB.

GEMM has two kernels, both taking the batch index from threadgroup z and per-batch operand offsets from a buffer, and both specialised into four pipelines with function constants for the (transpose A, transpose B) combinations.

On Apple7-family GPUs (M1/A14 and newer) the library is compiled with `VECTORCORE_SIMDGROUP_MATRIX` and `matmul_simdgroup` is used. A threadgroup of four simdgroups computes a 64x64 tile of C; each simdgroup owns a 32x32 block held as 4x4 `simdgroup_float8x8` accumulators and issues 16 hardware multiply-accumulates per 8-deep step. When the whole tile is in bounds and k is a multiple of 8, fragments are loaded directly from device memory with `simdgroup_load` (whose transpose flag handles transposed operands), with no threadgroup staging or barriers; edge tiles stage double-buffered 64x16 / 16x64 slices through threadgroup memory with bounds checks and write only valid outputs. If that compilation or pipeline creation fails, the backend keeps the portable kernel and `vc.metal_gemm_kernel()` reports the reason; `VECTORCORE_METAL_GEMM=tiled` forces the portable kernel.

The portable tiled kernel launches complete 16×16 threadgroups. Threads cooperatively load A and B tiles into threadgroup memory, synchronize, and accumulate one output. Complete edge threadgroups are required even when the output is smaller than a tile because otherwise K-dimension loader lanes do not exist. Sum repeatedly dispatches a 256-thread tree-reduction pass until one value remains. Axis reductions use one thread per output element walking the reduced axis.

## Python boundary

pybind11 converts scalar or rectangular array-like input into contiguous float32 host values, then constructs a native tensor. `.numpy()` always returns an owning copy, preventing dangling views across C++ lifetimes or asynchronous device state. Python operators call the same C++ semantics and backends used by the native tests; Python never performs tensor math for VectorCore.

## Failure behavior

Invalid shapes, indices, reshape sizes, matmul ranks/dimensions, device mixtures, non-scalar implicit backward, missing gradients, and unavailable Metal devices throw descriptive C++ exceptions, which pybind11 translates to Python exceptions. Metal command-buffer errors are checked synchronously. Device mixing is rejected rather than triggering an implicit transfer.
