


```markdown
# Fused GELU + Bias Kernel for NVIDIA T4

A high‑performance, fused activation kernel that computes **GELU(x) + bias** in a single GPU pass – eliminating separate kernel launches and redundant memory traffic. Optimised for NVIDIA T4 (sm_75) with **zero register spills**, **bank‑conflict‑free shared memory**, and **~225 GB/s** sustained memory bandwidth.

---

## Repository Structure

```

cuda-gelu-bias-fusion/
├── README.md
├── gelu_bias.cu          # Complete kernel + benchmark + CPU reference
├── LICENSE
└── docs/
└── case-study.md     # Detailed optimisation journey

```

---

## Key Features

- **Fused GELU + bias** – single‑pass computation, no intermediate global writes.
- **Two memory paths** – shared memory caching for small bias (`C ≤ 1024`), global `__ldg()` for large bias.
- **Zero register spills** – 38–43 registers per thread, 0 stack, 0 spills.
- **Bank‑conflict‑free shared memory** – padding formula `phys_idx = logical_idx + (logical_idx >> 5)` eliminates 4‑way conflicts.
- **Vectorised `float4` loads** – 128‑bit memory accesses maximise bus utilisation.
- **Power‑of‑2 specialisation** – replaces `% C` with `& (C-1)` for speed.
- **Modulo‑free indexing** – additive tracking avoids expensive integer division.
- **Full error handling** – `CUDA_CHECK` macro for all runtime APIs.
- **Host pinned memory + async transfers** – accurate bandwidth measurement.

---

## Performance

**Hardware:** NVIDIA T4 (sm_75) on Google Colab  
**Tested sizes:** 1M, 10M, 100M elements, bias channels `C = 256` and `C = 2048`

| N (elements) | C | GPU Time (ms) | Bandwidth (GB/s) | Registers | Spills |
|--------------|---|---------------|------------------|-----------|--------|
| 1M           | 256 | 0.0375 | 213.4 | 43 | 0 |
| 1M           | 2048 | 0.0362 | 220.9 | 41 | 0 |
| 10M          | 256 | 0.3510 | 227.9 | 43 | 0 |
| 10M          | 2048 | 0.3505 | 228.3 | 41 | 0 |
| 100M         | 256 | 3.5585 | 224.8 | 43 | 0 |
| 100M         | 2048 | 3.5595 | 224.8 | 41 | 0 |

**Compiler Report (`nvcc -O3 -arch=sm_75`):**
```

0 bytes stack frame
0 bytes spill stores
0 bytes spill loads
Used 38–43 registers, 0 barriers

```

---

## Compilation & Usage

### Prerequisites
- CUDA Toolkit 11.8 or later
- NVIDIA driver supporting sm_75 (T4)

### Build
```bash
nvcc -O3 -arch=sm_75 -lineinfo --ptxas-options=-v -o gelu_bias gelu_bias.cu
```

Run

```bash
./gelu_bias
```

The benchmark runs tests for N = 1M, 10M, 100M with C = 256 and C = 2048, verifies against a CPU reference, and reports execution time and bandwidth.

---

How It Works

1. Input: Flattened tensor x of size N, bias vector of size C (broadcasted).
2. Kernel Selection:
   · If C ≤ 1024: load bias into shared memory once per block → fast repeated access.
   · If C > 1024: read bias from global memory using __ldg() (read‑only cache).
3. Vectorisation: Each thread loads/stores 4 elements using float4 for peak bandwidth.
4. GELU Computation: y = x * 0.5 * (1 + erf(x / sqrt(2))) + bias
5. Writeback: Output stored in global memory.
6. Trailing Elements: Scalar loop handles any leftover elements not divisible by 4.

---

Optimisation Journey

Issue Fix
uintptr_t undefined Added #include <cstdint>
Integer overflow in index arithmetic Changed int to size_t for all indices
Division by zero when C ≤ 0 Added validation guard in solution()
Hardcoded shared memory size Switched to dynamic shared memory (extern __shared__)
4‑way shared memory bank conflicts Applied padding phys_idx = idx + (idx >> 5)
4 expensive modulo operations per thread Replaced with additive tracking and conditional wraps
Missing error checking Added CUDA_CHECK and cudaGetLastError()
Pageable memory + synchronous copies Used cudaHostAlloc + cudaMemcpyAsync

---

License

MIT License – see LICENSE file.

---

Author

Mustafa-cuda-dev
GitHub: https://github.com/Mustafa-cuda-dev

---

Acknowledgements

· NVIDIA CUDA Toolkit
· Google Colab for free T4 GPU access
· The open‑source CUDA community

```

---



---


