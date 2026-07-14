#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstdint>

constexpr float INV_SQRT_2 = 0.7071067811865475f;

#define CUDA_CHECK(err) do { cudaError_t err_ = (err); if (err_ != cudaSuccess) { fprintf(stderr, "CUDA error %d: %s\n", err_, cudaGetErrorString(err_)); exit(EXIT_FAILURE); } } while(0)

template <bool IS_POW2>
__global__ __launch_bounds__(256, 4)
void fused_gelu_bias_shared(const float* __restrict__ input, const float* __restrict__ bias, float* __restrict__ output, size_t N, size_t C) {
    extern __shared__ float s_bias[];
    for (size_t i = threadIdx.x; i < C; i += blockDim.x) s_bias[i + (i >> 5)] = bias[i];
    __syncthreads();
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t v_end = (N / 4) * 4;
    bool aligned = (reinterpret_cast<uintptr_t>(input) % 16 == 0 && reinterpret_cast<uintptr_t>(output) % 16 == 0);
    if (aligned) {
        const float4* in_v = (const float4*)input;
        float4* out_v = (float4*)output;
        if (IS_POW2) {
            for (size_t i = tid; i < v_end / 4; i += stride) {
                float4 in = in_v[i], out;
                size_t base = i * 4;
                size_t b0 = base & (C - 1), b1 = (base + 1) & (C - 1), b2 = (base + 2) & (C - 1), b3 = (base + 3) & (C - 1);
                float x = in.x; out.x = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + s_bias[b0 + (b0 >> 5)];
                x = in.y; out.y = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + s_bias[b1 + (b1 >> 5)];
                x = in.z; out.z = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + s_bias[b2 + (b2 >> 5)];
                x = in.w; out.w = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + s_bias[b3 + (b3 >> 5)];
                out_v[i] = out;
            }
        } else {
            size_t stride_mod = (stride * 4) % C;
            size_t b0 = (tid * 4) % C;
            for (size_t i = tid; i < v_end / 4; i += stride) {
                float4 in = in_v[i], out;
                size_t b1 = b0 + 1; if (b1 >= C) b1 -= C;
                size_t b2 = b1 + 1; if (b2 >= C) b2 -= C;
                size_t b3 = b2 + 1; if (b3 >= C) b3 -= C;
                float x = in.x; out.x = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + s_bias[b0 + (b0 >> 5)];
                x = in.y; out.y = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + s_bias[b1 + (b1 >> 5)];
                x = in.z; out.z = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + s_bias[b2 + (b2 >> 5)];
                x = in.w; out.w = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + s_bias[b3 + (b3 >> 5)];
                out_v[i] = out;
                b0 += stride_mod; if (b0 >= C) b0 -= C;
            }
        }
    } else {
        if (IS_POW2) {
            for (size_t i = tid; i < v_end / 4; i += stride) {
                size_t base = i * 4;
                #pragma unroll
                for (size_t j = 0; j < 4; ++j) {
                    size_t idx = base + j;
                    float x = input[idx];
                    size_t b = idx & (C - 1);
                    output[idx] = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + s_bias[b + (b >> 5)];
                }
            }
        } else {
            size_t stride_mod = (stride * 4) % C;
            size_t b0 = (tid * 4) % C;
            for (size_t i = tid; i < v_end / 4; i += stride) {
                size_t b1 = b0 + 1; if (b1 >= C) b1 -= C;
                size_t b2 = b1 + 1; if (b2 >= C) b2 -= C;
                size_t b3 = b2 + 1; if (b3 >= C) b3 -= C;
                float x0 = input[i*4]; output[i*4] = x0 * 0.5f * (1.0f + erff(x0 * INV_SQRT_2)) + s_bias[b0 + (b0 >> 5)];
                float x1 = input[i*4+1]; output[i*4+1] = x1 * 0.5f * (1.0f + erff(x1 * INV_SQRT_2)) + s_bias[b1 + (b1 >> 5)];
                float x2 = input[i*4+2]; output[i*4+2] = x2 * 0.5f * (1.0f + erff(x2 * INV_SQRT_2)) + s_bias[b2 + (b2 >> 5)];
                float x3 = input[i*4+3]; output[i*4+3] = x3 * 0.5f * (1.0f + erff(x3 * INV_SQRT_2)) + s_bias[b3 + (b3 >> 5)];
                b0 += stride_mod; if (b0 >= C) b0 -= C;
            }
        }
    }
    for (size_t i = v_end + tid; i < N; i += stride) {
        float x = input[i];
        size_t b = IS_POW2 ? (i & (C - 1)) : (i % C);
        output[i] = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + s_bias[b + (b >> 5)];
    }
}

template <bool IS_POW2, bool BIAS_VEC>
__global__ __launch_bounds__(256, 4)
void fused_gelu_bias_global(const float* __restrict__ input, const float* __restrict__ bias, float* __restrict__ output, size_t N, size_t C) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t v_end = (N / 4) * 4;
    bool aligned = (reinterpret_cast<uintptr_t>(input) % 16 == 0 && reinterpret_cast<uintptr_t>(output) % 16 == 0 && (!BIAS_VEC || (reinterpret_cast<uintptr_t>(bias) % 16 == 0)));
    if (aligned) {
        const float4* in_v = (const float4*)input;
        float4* out_v = (float4*)output;
        const float4* bias_v = (const float4*)bias;
        if (IS_POW2) {
            for (size_t i = tid; i < v_end / 4; i += stride) {
                float4 in = in_v[i], out;
                size_t base = i * 4;
                if (BIAS_VEC) {
                    size_t b_idx = (base & (C - 1)) >> 2;
                    float4 bv = __ldg(&bias_v[b_idx]);
                    float x = in.x; out.x = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + bv.x;
                    x = in.y; out.y = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + bv.y;
                    x = in.z; out.z = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + bv.z;
                    x = in.w; out.w = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + bv.w;
                } else {
                    size_t b0 = base & (C - 1), b1 = (base + 1) & (C - 1), b2 = (base + 2) & (C - 1), b3 = (base + 3) & (C - 1);
                    float x = in.x; out.x = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + __ldg(&bias[b0]);
                    x = in.y; out.y = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + __ldg(&bias[b1]);
                    x = in.z; out.z = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + __ldg(&bias[b2]);
                    x = in.w; out.w = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + __ldg(&bias[b3]);
                }
                out_v[i] = out;
            }
        } else {
            size_t stride_mod = (stride * 4) % C;
            size_t b0 = (tid * 4) % C;
            for (size_t i = tid; i < v_end / 4; i += stride) {
                float4 in = in_v[i], out;
                if (BIAS_VEC) {
                    size_t b_idx = b0 >> 2;
                    float4 bv = __ldg(&bias_v[b_idx]);
                    float x = in.x; out.x = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + bv.x;
                    x = in.y; out.y = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + bv.y;
                    x = in.z; out.z = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + bv.z;
                    x = in.w; out.w = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + bv.w;
                } else {
                    size_t b1 = b0 + 1; if (b1 >= C) b1 -= C;
                    size_t b2 = b1 + 1; if (b2 >= C) b2 -= C;
                    size_t b3 = b2 + 1; if (b3 >= C) b3 -= C;
                    float x = in.x; out.x = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + __ldg(&bias[b0]);
                    x = in.y; out.y = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + __ldg(&bias[b1]);
                    x = in.z; out.z = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + __ldg(&bias[b2]);
                    x = in.w; out.w = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + __ldg(&bias[b3]);
                }
                out_v[i] = out;
                b0 += stride_mod; if (b0 >= C) b0 -= C;
            }
        }
    } else {
        if (IS_POW2) {
            for (size_t i = tid; i < v_end / 4; i += stride) {
                size_t base = i * 4;
                #pragma unroll
                for (size_t j = 0; j < 4; ++j) {
                    size_t idx = base + j;
                    float x = input[idx];
                    size_t b = idx & (C - 1);
                    output[idx] = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + __ldg(&bias[b]);
                }
            }
        } else {
            size_t stride_mod = (stride * 4) % C;
            size_t b0 = (tid * 4) % C;
            for (size_t i = tid; i < v_end / 4; i += stride) {
                size_t b1 = b0 + 1; if (b1 >= C) b1 -= C;
                size_t b2 = b1 + 1; if (b2 >= C) b2 -= C;
                size_t b3 = b2 + 1; if (b3 >= C) b3 -= C;
                float x0 = input[i*4]; output[i*4] = x0 * 0.5f * (1.0f + erff(x0 * INV_SQRT_2)) + __ldg(&bias[b0]);
                float x1 = input[i*4+1]; output[i*4+1] = x1 * 0.5f * (1.0f + erff(x1 * INV_SQRT_2)) + __ldg(&bias[b1]);
                float x2 = input[i*4+2]; output[i*4+2] = x2 * 0.5f * (1.0f + erff(x2 * INV_SQRT_2)) + __ldg(&bias[b2]);
                float x3 = input[i*4+3]; output[i*4+3] = x3 * 0.5f * (1.0f + erff(x3 * INV_SQRT_2)) + __ldg(&bias[b3]);
                b0 += stride_mod; if (b0 >= C) b0 -= C;
            }
        }
    }
    for (size_t i = v_end + tid; i < N; i += stride) {
        float x = input[i];
        size_t b = IS_POW2 ? (i & (C - 1)) : (i % C);
        output[i] = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + __ldg(&bias[b]);
    }
}

extern "C" void solution(const float* input, const float* bias, float* output, int N, int C) {
    if (N <= 0 || C <= 0 || !input || !bias || !output) return;
    int num_sm; cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    int grid = num_sm * 4;
    bool is_pow2 = (C & (C-1)) == 0;
    bool bias_vec = (C % 4 == 0);
    if (C <= 1024) {
        size_t smem = (C + (C >> 5) + 1) * sizeof(float);
        if (is_pow2) fused_gelu_bias_shared<true><<<grid, 256, smem>>>(input, bias, output, (size_t)N, (size_t)C);
        else fused_gelu_bias_shared<false><<<grid, 256, smem>>>(input, bias, output, (size_t)N, (size_t)C);
    } else {
        if (is_pow2) {
            if (bias_vec) fused_gelu_bias_global<true, true><<<grid, 256>>>(input, bias, output, (size_t)N, (size_t)C);
            else fused_gelu_bias_global<true, false><<<grid, 256>>>(input, bias, output, (size_t)N, (size_t)C);
        } else {
            if (bias_vec) fused_gelu_bias_global<false, true><<<grid, 256>>>(input, bias, output, (size_t)N, (size_t)C);
            else fused_gelu_bias_global<false, false><<<grid, 256>>>(input, bias, output, (size_t)N, (size_t)C);
        }
    }
    cudaGetLastError();
}

void cpu_reference(const float* input, const float* bias, float* output, int N, int C) {
    for (int i = 0; i < N; ++i) {
        float x = input[i];
        output[i] = x * 0.5f * (1.0f + erff(x * INV_SQRT_2)) + bias[i % C];
    }
}

bool verify(const float* gpu, const float* cpu, int N, float tol = 1e-5f) {
    for (int i = 0; i < N; ++i) {
        float diff = fabsf(gpu[i] - cpu[i]);
        if (diff > tol && diff / fmaxf(1e-9f, fabsf(cpu[i])) > tol) {
            printf("Mismatch at %d: %.6f vs %.6f\n", i, gpu[i], cpu[i]);
            return false;
        }
    }
    return true;
}

void run_benchmark(int N, int C) {
    printf("N=%d C=%d\n", N, C);
    size_t in_bytes = N * sizeof(float), bias_bytes = C * sizeof(float), out_bytes = N * sizeof(float);
    float *h_in, *h_bias, *h_gpu, *h_cpu;
    cudaHostAlloc(&h_in, in_bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_bias, bias_bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_gpu, out_bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_cpu, out_bytes, cudaHostAllocDefault);
    for (int i = 0; i < N; ++i) h_in[i] = -2.0f + 4.0f * (float)rand() / RAND_MAX;
    for (int i = 0; i < C; ++i) h_bias[i] = -0.5f + (float)rand() / RAND_MAX;

    float *d_in, *d_bias, *d_out;
    cudaMalloc(&d_in, in_bytes); cudaMalloc(&d_bias, bias_bytes); cudaMalloc(&d_out, out_bytes);
    cudaStream_t stream; cudaStreamCreate(&stream);
    cudaMemcpyAsync(d_in, h_in, in_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_bias, h_bias, bias_bytes, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    solution(d_in, d_bias, d_out, N, C); cudaDeviceSynchronize();
    cudaMemcpy(h_gpu, d_out, out_bytes, cudaMemcpyDeviceToHost);
    cpu_reference(h_in, h_bias, h_cpu, N, C);
    printf("Verification: %s\n", verify(h_gpu, h_cpu, N) ? "PASS" : "FAIL");

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    const int iters = 100;
    cudaEventRecord(start, stream);
    for (int i = 0; i < iters; ++i) solution(d_in, d_bias, d_out, N, C);
    cudaEventRecord(stop, stream); cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop); ms /= iters;
    double gb = (double)(in_bytes + out_bytes) / (ms * 1e-3) / 1e9;
    printf("Time: %.4f ms, Bandwidth: %.2f GB/s\n\n", ms, gb);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(d_in); cudaFree(d_bias); cudaFree(d_out);
    cudaFreeHost(h_in); cudaFreeHost(h_bias); cudaFreeHost(h_gpu); cudaFreeHost(h_cpu);
}

int main() {
    srand(42);
    for (int N : {1000000, 10000000, 100000000}) {
        run_benchmark(N, 256);
        run_benchmark(N, 2048);
    }
    return 0;
}
