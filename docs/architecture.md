# VectorCore architecture

## Tensor and storage

`Tensor` is a value-like handle containing `shared_ptr<TensorImpl>`. `TensorImpl` stores a `shared_ptr<Storage>`, `Shape`, contiguous `Strides`, storage offset, float32 dtype, and autograd state. Copying a `Tensor` is cheap and preserves identity, which is important for gradient accumulation. CPU storage owns a `std::vector<float>`. Metal storage owns an `MTLBuffer`; Objective-C ARC and the C++ virtual destructor jointly provide deterministic resource release.

A scalar uses an empty shape and has one element. A dimension may be zero; products containing it have zero elements. `sum` of an empty tensor is zero and `mean` rejects it. Shape products use checked multiplication. `reshape` validates equal element counts and shares contiguous storage; operations and transpose return newly allocated contiguous tensors.

## Broadcasting

Shapes are aligned from the trailing dimension. At each axis, dimensions must match or one must equal one. CPU kernels iterate the output linear index, reconstruct coordinates, and map singleton axes to a zero input offset. Metal passes fixed-size rank/shape/stride metadata to `binary_broadcast`, which performs the same mapping in each GPU thread. Neither backend expands broadcast operands.

Backward uses `sum_to_shape`: dimensions introduced by broadcasting and singleton target axes are summed. CPU performs a direct coordinate mapping. Metal executes `reduce_to_shape` on the GPU, avoiding an implicit host transfer.

## CPU execution

Elementwise operations are direct float32 loops. Sum uses four double-precision accumulators before returning float32. Matrix multiplication is an `i-k-j` blocked kernel with 32-element tiles. For sufficiently large products, output rows are partitioned among hardware threads; no two workers write the same result element, so locks are unnecessary. This is educational systems code, not a replacement for a packed BLAS microkernel.

## Autograd

Every differentiable operation creates a result and conditionally attaches `GradFn` when global gradient recording is enabled and at least one parent requires gradients. A node owns its parents and a local vector-Jacobian-product callback. Parents do not own children, so the graph has no ownership cycle.

`backward` performs a visited-set depth-first topological traversal. It walks the order in reverse, accumulates contributions in a map keyed by `TensorImpl*`, stores gradients on participating tensors, and invokes each local backward rule. `NoGradGuard` disables graph construction during this work. Existing leaf gradients are added, matching repeated-backward accumulation semantics. `zero_grad` explicitly releases a stored gradient.

Backward rules are built from native VectorCore operations: matrix multiplication uses `dA = dY @ B.T` and `dB = A.T @ dY`; sigmoid saves a detached output; ReLU saves a detached input and computes its mask on the active device; reshape restores the source shape; device transfer moves the gradient back to its source device.

## Metal execution

The build embeds `metal/kernels.metal` into the library. `MetalBackend` obtains the default device, creates one command queue, compiles the MSL source with `newLibraryWithSource`, and caches all compute pipeline states. Failure leaves the backend unavailable with a diagnostic. There is no CPU fallback inside Metal operations.

Buffers use shared storage, appropriate for Apple unified memory. Upload/download still copy deliberately so tensor ownership and timing boundaries remain unambiguous. Commands are serialized through a mutex, committed, and checked after `waitUntilCompleted`.

The tiled GEMM kernel launches complete 16×16 threadgroups. Threads cooperatively load A and B tiles into threadgroup memory, synchronize, and accumulate one output. Complete edge threadgroups are required even when the output is smaller than a tile because otherwise K-dimension loader lanes do not exist. Sum repeatedly dispatches a 256-thread tree-reduction pass until one value remains. The current synchronous model makes correctness and timing simple, at the cost of dispatch overhead and lost overlap.

## Python boundary

pybind11 converts scalar or rectangular array-like input into contiguous float32 host values, then constructs a native tensor. `.numpy()` always returns an owning copy, preventing dangling views across C++ lifetimes or asynchronous device state. Python operators call the same C++ semantics and backends used by the native tests; Python never performs tensor math for VectorCore.

## Failure behavior

Invalid shapes, indices, reshape sizes, matmul ranks/dimensions, device mixtures, non-scalar implicit backward, missing gradients, and unavailable Metal devices throw descriptive C++ exceptions, which pybind11 translates to Python exceptions. Metal command-buffer errors are checked synchronously. Device mixing is rejected rather than triggering an implicit transfer.
