#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

// Naive intra-block scan kernel (from previous implementation)
__global__ void naiveScanKernel(float *input, float *output, int n)
{
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int blockOffset = blockIdx.x * blockDim.x;

    if (blockOffset + tid < n)
    {
        temp[tid] = input[blockOffset + tid];
    }
    else
    {
        temp[tid] = 0.0f;
    }
    __syncthreads();

    for (int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        float val = 0.0f;
        if (tid >= stride)
        {
            val = temp[tid - stride];
        }
        __syncthreads();
        if (tid >= stride)
        {
            temp[tid] += val;
        }
        __syncthreads();
    }

    if (blockOffset + tid < n)
    {
        output[blockOffset + tid] = temp[tid];
    }
}

// Optimized intra-block scan kernel (work-efficient, two-phase)
__global__ void optimizedScanKernel(float *input, float *output, int n)
{
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int blockOffset = blockIdx.x * blockDim.x;
    int offset = 1;

    // Load input to shared memory (two elements per thread for efficiency)
    if (blockOffset + 2 * tid < n)
    {
        temp[2 * tid] = input[blockOffset + 2 * tid];
    }
    else
    {
        temp[2 * tid] = 0.0f;
    }
    if (blockOffset + 2 * tid + 1 < n)
    {
        temp[2 * tid + 1] = input[blockOffset + 2 * tid + 1];
    }
    else
    {
        temp[2 * tid + 1] = 0.0f;
    }
    __syncthreads();

    // Phase 1: Up-sweep (reduce)
    for (int d = blockDim.x; d > 0; d >>= 1)
    {
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
        __syncthreads();
    }

    // Clear the last element
    if (tid == 0)
    {
        temp[2 * blockDim.x - 1] = 0.0f;
    }
    __syncthreads();

    // Phase 2: Down-sweep
    for (int d = 1; d < 2 * blockDim.x; d *= 2)
    {
        offset >>= 1;
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
        __syncthreads();
    }

    // Write results to output
    if (blockOffset + 2 * tid < n)
    {
        output[blockOffset + 2 * tid] = temp[2 * tid];
    }
    if (blockOffset + 2 * tid + 1 < n)
    {
        output[blockOffset + 2 * tid + 1] = temp[2 * tid + 1];
    }
}

// Sequential CPU scan for verification
void sequentialScan(float *input, float *output, int n)
{
    output[0] = input[0];
    for (int i = 1; i < n; i++)
    {
        output[i] = output[i - 1] + input[i];
    }
}

int main(int argc, char *argv[])
{
    // Array sizes to test (2^10 to 2^20)
    int sizes[] = {1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const int THREADS_PER_BLOCK = 256;     // For naive kernel
    const int OPT_THREADS_PER_BLOCK = 128; // For optimized kernel (handles 2 elements per thread)

    // Timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int s = 0; s < num_sizes; s++)
    {
        int N = sizes[s];
        size_t SIZE = N * sizeof(float);
        printf("\nTesting array size: %d\n", N);

        // Host memory allocation
        float *h_input = (float *)malloc(SIZE);
        float *h_output_naive = (float *)malloc(SIZE);
        float *h_output_opt = (float *)malloc(SIZE);
        float *h_verify = (float *)malloc(SIZE);

        // Initialize input data
        for (int i = 0; i < N; i++)
        {
            h_input[i] = 1.0f; // Simple test case: all 1s
        }

        // Device memory allocation
        float *d_input, *d_output_naive, *d_output_opt;
        cudaMalloc(&d_input, SIZE);
        cudaMalloc(&d_output_naive, SIZE);
        cudaMalloc(&d_output_opt, SIZE);

        // Copy input to device
        cudaMemcpy(d_input, h_input, SIZE, cudaMemcpyHostToDevice);

        // Timing variables
        float cpu_time, naive_time, opt_time;

        // CPU execution and timing
        clock_t cpu_start = clock();
        sequentialScan(h_input, h_verify, N);
        clock_t cpu_end = clock();
        cpu_time = ((float)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0f;

        // Naive kernel execution and timing
        int naive_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        cudaEventRecord(start);
        naiveScanKernel<<<naive_blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float)>>>(d_input, d_output_naive, N);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&naive_time, start, stop);
        cudaMemcpy(h_output_naive, d_output_naive, SIZE, cudaMemcpyDeviceToHost);

        // Optimized kernel execution and timing
        int opt_blocks = (N + 2 * OPT_THREADS_PER_BLOCK - 1) / (2 * OPT_THREADS_PER_BLOCK);
        cudaEventRecord(start);
        optimizedScanKernel<<<opt_blocks, OPT_THREADS_PER_BLOCK, 2 * OPT_THREADS_PER_BLOCK * sizeof(float)>>>(d_input, d_output_opt, N);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&opt_time, start, stop);
        cudaMemcpy(h_output_opt, d_output_opt, SIZE, cudaMemcpyDeviceToHost);

        // Verify results (optimized kernel)
        bool correct = true;
        for (int i = 0; i < N; i++)
        {
            if (fabs(h_output_opt[i] - h_verify[i]) > 1e-5)
            {
                printf("Optimized kernel verification failed at index %d: GPU = %f, CPU = %f\n",
                       i, h_output_opt[i], h_verify[i]);
                correct = false;
                break;
            }
        }
        if (correct)
        {
            printf("Optimized kernel verification passed!\n");
        }

        // Calculate and report speedups
        float speedup_opt_vs_naive = naive_time / opt_time;
        float speedup_opt_vs_cpu = cpu_time / opt_time;
        printf("CPU time: %.2f ms\n", cpu_time);
        printf("Naive GPU time: %.2f ms\n", naive_time);
        printf("Optimized GPU time: %.2f ms\n", opt_time);
        printf("Speedup (Optimized vs Naive): %.2fx\n", speedup_opt_vs_naive);
        printf("Speedup (Optimized vs CPU): %.2fx\n", speedup_opt_vs_cpu);

        // Free memory
        free(h_input);
        free(h_output_naive);
        free(h_output_opt);
        free(h_verify);
        cudaFree(d_input);
        cudaFree(d_output_naive);
        cudaFree(d_output_opt);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
