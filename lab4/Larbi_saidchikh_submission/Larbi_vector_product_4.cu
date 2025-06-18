#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

#define THREADS_PER_BLOCK 256

// CUDA kernel for vector dot product using shared memory
__global__ void vectorDotProduct_partial(float *a, float *b, float *partial_c, int n)
{
    __shared__ float cache[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.0f;

    // Grid-stride loop to handle large vectors
    while (gidx < n)
    {
        temp += a[gidx] * b[gidx];
        gidx += blockDim.x * gridDim.x;
    }

    cache[tid] = temp;

    __syncthreads();

    // Parallel reduction
    int i = blockDim.x / 2;
    while (i > 0)
    {
        if (tid < i)
        {
            cache[tid] += cache[tid + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (tid == 0)
    {
        partial_c[blockIdx.x] = cache[0];
    }
}

// CPU dot product function
float cpuDotProduct(float *a, float *b, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        sum += a[i] * b[i];
    }
    return sum;
}

int main()
{
    const int n = 1000000; // Vector size
    const int threadsPerBlock = THREADS_PER_BLOCK;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    float *h_a, *h_b, *h_partial_c;
    float *d_a, *d_b, *d_partial_c;

    // Allocate host memory
    h_a = (float *)malloc(n * sizeof(float));
    h_b = (float *)malloc(n * sizeof(float));
    h_partial_c = (float *)malloc(blocksPerGrid * sizeof(float));

    // Initialize host arrays
    for (int i = 0; i < n; i++)
    {
        h_a[i] = i;
        h_b[i] = i;
    }

    // Allocate device memory
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_partial_c, blocksPerGrid * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    // GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    vectorDotProduct_partial<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_partial_c, n);

    // Copy partial sums back to host
    cudaMemcpy(h_partial_c, d_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // Sum partial sums on host
    float gpu_result = 0.0f;
    for (int i = 0; i < blocksPerGrid; i++)
    {
        gpu_result += h_partial_c[i];
    }

    // CPU timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_result = cpuDotProduct(h_a, h_b, n);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> cpu_duration = cpu_end - cpu_start;
    float cpu_time = cpu_duration.count() * 1000.0f; // Convert to milliseconds

    // Calculate speedup
    float speedup = cpu_time / gpu_time;

    // Print results
    printf("GPU Result: %f\n", gpu_result);
    printf("CPU Result: %f\n", cpu_result);
    printf("GPU Time: %f ms\n", gpu_time);
    printf("CPU Time: %f ms\n", cpu_time);
    printf("Speedup: %f x\n", speedup);

    // Free memory
    free(h_a);
    free(h_b);
    free(h_partial_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_partial_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

// GPU Result : 333332789799682048.000000 CPU Result : 333380996512612352.000000 
// GPU Time : 0.158752 ms
// CPU Time : 3.125603 ms
// Speedup : 19.688591 x