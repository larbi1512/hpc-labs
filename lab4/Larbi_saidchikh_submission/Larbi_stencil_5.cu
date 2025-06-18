#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

#define RADIUS 3
#define BLOCK_SIZE 512

// CUDA kernel for optimized 1D stencil computation
__global__ void stencil_1d(float *in, float *out, int n)
{
    __shared__ float temp[BLOCK_SIZE + 2 * RADIUS];
    int t = threadIdx.x;
    int block_start = blockIdx.x * BLOCK_SIZE;
    int stride = blockDim.x * gridDim.x;

    // Each thread processes multiple elements using grid-stride loop
    for (int gindex = block_start + t; gindex < n; gindex += stride)
    {
        // Load shared memory: primary and halo elements
        int smem_idx = t + RADIUS;
        // Primary element
        temp[smem_idx] = (gindex < n) ? in[gindex] : 0.0f;

        // Load halo elements using all threads
        if (t < 2 * RADIUS)
        {
            int offset = t - RADIUS;
            int global_idx = block_start + offset;
            int smem_halo_idx = (offset < 0) ? RADIUS + offset : BLOCK_SIZE + RADIUS + offset;
            temp[smem_halo_idx] = (global_idx >= 0 && global_idx < n) ? in[global_idx] : 0.0f;
        }

        __syncthreads();

        // Compute stencil
        float sum = 0.0f;
        for (int offset = -RADIUS; offset <= RADIUS; offset++)
        {
            sum += temp[smem_idx + offset];
        }
        out[gindex] = sum;

        __syncthreads(); // Ensure shared memory is reusable for next iteration
    }
}

// CPU implementation of 1D stencil (unchanged for fair comparison)
void cpu_stencil(float *in, float *out, int n)
{
    for (int i = 0; i < n; i++)
    {
        float sum = 0.0f;
        for (int offset = -RADIUS; offset <= RADIUS; offset++)
        {
            int j = i + offset;
            if (j >= 0 && j < n)
            {
                sum += in[j];
            }
        }
        out[i] = sum;
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <vector_size>\n", argv[0]);
        return 1;
    }
    const int n = atoi(argv[1]); 
    const int threadsPerBlock = BLOCK_SIZE;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    float *h_in, *h_out_gpu, *h_out_cpu;
    float *d_in, *d_out;

    // Allocate pinned host memory
    cudaHostAlloc(&h_in, n * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_out_gpu, n * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_out_cpu, n * sizeof(float), cudaHostAllocDefault);

    // Initialize input array
    for (int i = 0; i < n; i++)
    {
        h_in[i] = i;
    }

    // Allocate device memory
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    // GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch CUDA kernel
    stencil_1d<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // Copy result back to host
    cudaMemcpy(h_out_gpu, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_stencil(h_in, h_out_cpu, n);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> cpu_duration = cpu_end - cpu_start;
    float cpu_time = cpu_duration.count() * 1000.0f; // Convert to milliseconds

    // Calculate speedup
    float speedup = cpu_time / gpu_time;

    // Print results
    printf("GPU Time: %f ms\n", gpu_time);
    printf("CPU Time: %f ms\n", cpu_time);
    printf("Speedup: %f x\n", speedup);

    // Verify results
    bool correct = true;
    for (int i = 0; i < n; i++)
    {
        if (abs(h_out_gpu[i] - h_out_cpu[i]) > 1e-5)
        {
            correct = false;
            break;
        }
    }
    if (correct)
    {
        printf("Results match\n");
    }
    else
    {
        printf("Results do not match\n");
    }

    // Free memory
    cudaFreeHost(h_in);
    cudaFreeHost(h_out_gpu);
    cudaFreeHost(h_out_cpu);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

// GPU Time : 0.144384 ms
// CPU Time : 22.558172 ms
// Speedup : 156.237350 x
