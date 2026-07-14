
---

```markdown
# Case Study: Fused GELU + Bias Kernel for NVIDIA T4 (sm_75)

**Author:** Mustafa-cuda-dev  
**Repository:** [cuda-gelu-bias-fusion](https://github.com/Mustafa-cuda-dev/cuda-gelu-bias-fusion)  
**Hardware:** NVIDIA T4 (sm_75)  
**Goal:** Build a production‑ready fused activation kernel for transformer inference that computes GELU + bias in a single pass, with zero register spills and near‑peak memory bandwidth.

---

## 1. Project Overview

GELU (Gaussian Error Linear Unit) is the activation function used in modern transformer models (BERT, GPT, LLaMA). In real‑world inference pipelines, bias addition is almost always applied immediately after GELU. Separating them into two kernels requires an extra global memory write and read, wasting bandwidth and increasing latency.

This project fuses both operations into a **single kernel**:
```

y[i] = GELU(x[i]) + bias[i % C]

```
The kernel is optimised for NVIDIA T4 (sm_75) with two memory paths:
- **Shared memory path** – caches the bias vector for small channel sizes (`C ≤ 1024`)
- **Global memory path** – reads bias via `__ldg()` for large channel sizes (`C > 1024`)

---

## 2. Technical Challenges & Solutions

### 2.1. Register Spills (Compiler Pressure)
- **Problem:** First compilation revealed register spills due to high register pressure from `#pragma unroll` and local variables.
- **Solution:** Applied `__launch_bounds__(256, 4)` to enforce a 64‑register cap per thread. Used explicit unrolling only where beneficial. Result: **38–43 registers per thread, 0 spills**.

### 2.2. Shared Memory Bank Conflicts
- **Problem:** Strided access to shared memory caused **4‑way bank conflicts** – threads 0, 8, 16, 24 all mapped to the same bank.
- **Solution:** Added padding: `phys_idx = logical_idx + (logical_idx >> 5)`. This inserts one padding slot every 32 elements, ensuring all 32 threads in a warp access unique banks. Conflicts eliminated.

### 2.3. Expensive Modulo Operations
- **Problem:** `% C` is computed 4 times per vectorised iteration for non‑power‑of‑2 `C`. Integer division is multi‑cycle and serialises execution.
- **Solution:** Implemented **additive tracking**:
  - Compute `b0 = (tid * 4) % C` once per thread.
  - Inside loop: `b0 += stride_mod; if (b0 >= C) b0 -= C;`
  - Derive `b1 = b0+1, b2 = b1+1, b3 = b2+1` with conditional wraps.
  - **Replaced 4 modulo operations** with additions + conditional branches.

### 2.4. Power‑of‑2 Specialisation
- **Problem:** For power‑of‑2 `C`, modulo can be replaced with bitwise AND for a 20× speedup.
- **Solution:** Template parameter `IS_POW2` dispatches to specialised kernels that use `& (C-1)`.

### 2.5. Hardcoded Shared Memory Size
- **Problem:** `__shared__ float s_bias[1024]` is fragile – if host‑side dispatch changes to `C ≤ 2048`, the kernel would overflow.
- **Solution:** Switched to dynamic shared memory with `extern __shared__ float s_bias[]` and passed `C * sizeof(float)` at launch. Allocation size is now `C + (C >> 5) + 1` to accommodate padding.

### 2.6. Integer Overflow in Index Arithmetic
- **Problem:** `int i = vectorized_end + tid` can overflow for large `N`.
- **Solution:** Changed all index variables to `size_t` – 64‑bit safe.

### 2.7. Missing CUDA Error Checking
- **Problem:** No validation of `cudaMalloc`, `cudaMemcpy`, or kernel launches – errors would surface unpredictably.
- **Solution:** Added `CUDA_CHECK` macro for all runtime APIs and `cudaGetLastError()` after kernel launches.

### 2.8. Host Memory Bottleneck
- **Problem:** `std::vector` is pageable – transfers require staging buffers. Synchronous `cudaMemcpy` blocks the CPU.
- **Solution:** Used `cudaHostAlloc` for pinned memory and `cudaMemcpyAsync` with streams. This overlaps transfers with computation and increases benchmark accuracy.

---

## 3. Final Architecture

### Kernel Variant 1 – Shared Memory Path (`C ≤ 1024`)
1. Cooperative load of bias into shared memory with padding.
2. `__syncthreads()` – ensure bias is ready.
3. Vectorised `float4` load of input (checking 16‑byte alignment).
4. Compute GELU for 4 elements simultaneously.
5. Read bias from shared memory using padded index.
6. Store `float4` output.
7. Scalar remainder loop for trailing elements.

### Kernel Variant 2 – Global Memory Path (`C > 1024`)
1. Direct `__ldg()` reads from global bias (read‑only cache).
2. Optional vectorised bias load when `C % 4 == 0`.
3. Same GELU computation and writeback as shared path.

### Template Dispatch Matrix

| `IS_POW2` | `BIAS_VEC` | Kernel | Description |
|-----------|------------|--------|-------------|
| true | true | `fused_gelu_bias_global<true, true>` | Power‑of‑2, bias aligned |
| true | false | `fused_gelu_bias_global<true, false>` | Power‑of‑2, bias unaligned |
| false | true | `fused_gelu_bias_global<false, true>` | Non‑power‑of‑2, bias aligned |
| false | false | `fused_gelu_bias_global<false, false>` | Non‑power‑of‑2, bias unaligned |

---

## 4. Benchmark Results

**Setup:** Google Colab T4 (sm_75), CUDA 11.8, 100 iterations per configuration, FP32 data.

| N (elements) | C | GPU Time (ms) | Bandwidth (GB/s) | Registers | Spills |
|--------------|---|---------------|------------------|-----------|--------|
| 1,000,000    | 256 | 0.0375 | 213.4 | 43 | 0 |
| 1,000,000    | 2048 | 0.0362 | 220.9 | 41 | 0 |
| 10,000,000   | 256 | 0.3510 | 227.9 | 43 | 0 |
| 10,000,000   | 2048 | 0.3505 | 228.3 | 41 | 0 |
| 100,000,000  | 256 | 3.5585 | 224.8 | 43 | 0 |
| 100,000,000  | 2048 | 3.5595 | 224.8 | 41 | 0 |

**Compiler Report:**
```

0 bytes stack frame
0 bytes spill stores
0 bytes spill loads
Used 38–43 registers

```

**Correctness:** PASS (relative error ≤ 1e-5 for all configurations)

---

## 5. What This Demonstrates

1. **Fusion Strategy** – single‑pass GELU + bias eliminates redundant memory traffic.
2. **Memory Hierarchy Mastery** – shared memory caching for small bias, `__ldg()` for large.
3. **Bank‑Conflict Elimination** – mathematical padding removes 4‑way conflicts.
4. **Modulo Elimination** – additive tracking replaces expensive division.
5. **Power‑of‑2 Specialisation** – compile‑time dispatch for optimal code.
6. **Vectorisation** – `float4` for 4× instruction reduction.
7. **Production‑Grade Code** – error handling, pinned memory, async transfers.
8. **Zero Spills** – fully register‑resident, no local memory penalty.

---

## 6. Lessons Learned

- **Alignment is Non‑Negotiable:** `float4` requires 16‑byte alignment. Without a fallback, unaligned pointers crash.
- **Bank Conflicts Are Subtle:** A 4‑way conflict may not crash, but it costs 4× latency. Always verify with profiling.
- **Modulo is Expensive:** GPU integer division is not free – avoiding it in hot loops yields measurable gains.
- **Template Specialisation Works:** Compile‑time dispatch for power‑of‑2 and vectorised paths keeps code efficient without runtime overhead.
- **Pinned Memory Matters:** For accurate bandwidth measurement, `cudaHostAlloc` + async transfers are essential.

---

## 7. Future Work

- **Mixed Precision:** Extend to FP16/BF16 with FP32 accumulation.
- **Strided Layouts:** Support for non‑contiguous tensors.
- **Multi‑Stream Concurrency:** Overlap data transfer and kernel execution for even higher throughput.

---

## 8. Conclusion

This project delivered a **highly optimised, production‑ready fused GELU + Bias kernel** for NVIDIA T4, achieving **~225 GB/s** sustained throughput with **zero register spills** and perfect correctness. The design demonstrates deep understanding of GPU memory hierarchy, vectorisation, bank conflicts, and compile‑time optimisation – making it a strong portfolio piece for CUDA engineering roles.
```

---

