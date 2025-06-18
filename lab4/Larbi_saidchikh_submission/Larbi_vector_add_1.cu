#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop for better efficiency with large arrays
    for (int i = index; i < n; i += stride)
    {
        c[i] = a[i] + b[i];
    }
}

void cpuVectorAdd(const float *a, const float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <exponent_for_vector_size>\n", argv[0]);
        return 1;
    }
    const int N = 1 << atoi(argv[1]);
    size_t size = N * sizeof(float);

    // Host vectors
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);
    float *h_cpu_result = (float *)malloc(size);

    // Initialize host vectors
    for (int i = 0; i < N; i++)
    {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // Device vectors
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Threads per block - optimized for most GPUs
    const int THREADS_PER_BLOCK = 256;

    // Calculate number of blocks needed - using more blocks than minimum for better occupancy
    int numBlocks = min((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 65535);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up kernel run (to initialize CUDA context)
    vectorAdd<<<numBlocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    // Time GPU execution
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
    { // Run multiple times for more accurate timing
        vectorAdd<<<numBlocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);
    gpuTime /= 100.0f; // Average time per iteration

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Time CPU execution
    clock_t cpuStart = clock();
    for (int i = 0; i < 100; i++)
    {
        cpuVectorAdd(h_a, h_b, h_cpu_result, N);
    }
    clock_t cpuEnd = clock();
    float cpuTime = (float)(cpuEnd - cpuStart) / (CLOCKS_PER_SEC / 1000.0f) / 100.0f;

    // Verify results
    int errors = 0;
    for (int i = 0; i < N; i++)
    {
        if (fabs(h_c[i] - h_cpu_result[i]) > 1e-5)
        {
            errors++;
            if (errors < 10)
            {
                printf("Error at index %d: GPU %f != CPU %f\n", i, h_c[i], h_cpu_result[i]);
            }
        }
    }

    // Print results
    printf("Array size: %d elements (%zu MB)\n", N, size / (1024 * 1024));
    printf("CPU time: %.3f ms\n", cpuTime);
    printf("GPU time: %.3f ms\n", gpuTime);
    printf("Speedup: %.2fx\n", cpuTime / gpuTime);
    if (errors == 0)
    {
        printf("Results verified successfully!\n");
    }
    else
    {
        printf("Completed with %d errors\n", errors);
    }

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_cpu_result);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}