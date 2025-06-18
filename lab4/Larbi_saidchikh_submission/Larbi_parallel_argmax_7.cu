#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include <chrono>

#define THREADS_PER_BLOCK 512

// CUDA kernel for optimized partial argmax
__global__ void argmax_partial_kernel(float *input, int n, float *max_vals_partial, int *max_idxs_partial)
{
    __shared__ float s_vals[THREADS_PER_BLOCK];
    __shared__ int s_idxs[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Pre-reduce locally using grid-stride loop
    float local_max = -FLT_MAX;
    int local_idx = -1;
    for (int i = gidx; i < n; i += stride)
    {
        if (input[i] > local_max)
        {
            local_max = input[i];
            local_idx = i;
        }
    }

    // Load into shared memory
    s_vals[tid] = local_max;
    s_idxs[tid] = local_idx;

    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            if (s_vals[tid + stride] > s_vals[tid])
            {
                s_vals[tid] = s_vals[tid + stride];
                s_idxs[tid] = s_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (tid == 0)
    {
        max_vals_partial[blockIdx.x] = s_vals[0];
        max_idxs_partial[blockIdx.x] = s_idxs[0];
    }
}

// CPU implementation for argmax (unchanged)
int cpu_argmax(float *input, int n)
{
    float max_val = -FLT_MAX;
    int max_idx = -1;
    for (int i = 0; i < n; i++)
    {
        if (input[i] > max_val)
        {
            max_val = input[i];
            max_idx = i;
        }
    }
    return max_idx;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <vector_size>\n", argv[0]);
        return 1;
    }
    const int n = atoi(argv[1]); // Vector size
    const int threadsPerBlock = THREADS_PER_BLOCK;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock / 2; // Reduced due to grid-stride

    float *h_input, *h_max_vals_partial;
    int *h_max_idxs_partial;
    float *d_input, *d_max_vals_partial;
    int *d_max_idxs_partial;

    // Allocate pinned host memory
    cudaHostAlloc(&h_input, n * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_max_vals_partial, blocksPerGrid * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_max_idxs_partial, blocksPerGrid * sizeof(int), cudaHostAllocDefault);

    // Initialize input array
    for (int i = 0; i < n; i++)
    {
        h_input[i] = (float)(n - i); // Decreasing values
    }
    h_input[n - 1] = n; // Maximum at the end

    // Allocate device memory
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_max_vals_partial, blocksPerGrid * sizeof(float));
    cudaMalloc(&d_max_idxs_partial, blocksPerGrid * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    // GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    argmax_partial_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, n, d_max_vals_partial, d_max_idxs_partial);

    // Copy partial results back to host
    cudaMemcpy(h_max_vals_partial, d_max_vals_partial, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max_idxs_partial, d_max_idxs_partial, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // Final reduction on host
    float max_val = -FLT_MAX;
    int max_idx = -1;
    for (int i = 0; i < blocksPerGrid; i++)
    {
        if (h_max_vals_partial[i] > max_val)
        {
            max_val = h_max_vals_partial[i];
            max_idx = h_max_idxs_partial[i];
        }
    }

    // CPU timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    int cpu_max_idx = cpu_argmax(h_input, n);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> cpu_duration = cpu_end - cpu_start;
    float cpu_time = cpu_duration.count() * 1000.0f; // Convert to milliseconds

    // Calculate speedup
    float speedup = cpu_time / gpu_time;

    // Print results
    printf("GPU Max Index: %d\n", max_idx);
    printf("CPU Max Index: %d\n", cpu_max_idx);
    printf("GPU Time: %f ms\n", gpu_time);
    printf("CPU Time: %f ms\n", cpu_time);
    printf("Speedup: %f x\n", speedup);

    // Verify results
    if (max_idx == cpu_max_idx)
    {
        printf("Results match\n");
    }
    else
    {
        printf("Results do not match\n");
    }

    // Free memory
    cudaFreeHost(h_input);
    cudaFreeHost(h_max_vals_partial);
    cudaFreeHost(h_max_idxs_partial);
    cudaFree(d_input);
    cudaFree(d_max_vals_partial);
    cudaFree(d_max_idxs_partial);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

// GPU Max Index : 0 CPU Max Index : 0 
// GPU Time : 0.177376 ms
// CPU Time : 2.681462 ms
// Speedup : 15.117389 x
// Results match