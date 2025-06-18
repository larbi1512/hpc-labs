#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// CUDA kernel with Kahan summation for improved numerical stability
__global__ void matVecMul_kernel(const float *M, const float *x, float *y,
                                 int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows)
    {
        // Kahan summation algorithm
        float sum = 0.0f;
        float c = 0.0f; // A running compensation for lost low-order bits

        for (int col = 0; col < cols; col++)
        {
            float product = M[row * cols + col] * x[col];
            float y = product - c; // So far, so good: c is zero
            float t = sum + y;     
            c = (t - sum) - y;     // (t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
            sum = t;               
        }
        y[row] = sum;
    }
}

// CPU reference implementation with same summation method
void cpuMatVecMul(const float *M, const float *x, float *y, int rows, int cols)
{
    for (int row = 0; row < rows; row++)
    {
        float sum = 0.0f;
        float c = 0.0f;
        for (int col = 0; col < cols; col++)
        {
            float product = M[row * cols + col] * x[col];
            float y = product - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        y[row] = sum;
    }
}

void verify_results(const float *cpu_y, const float *gpu_y, int n)
{
    int errors = 0;
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    const float abs_threshold = 1e-6f;
    const float rel_threshold = 1e-6f;

    for (int i = 0; i < n; i++)
    {
        if (fabs(cpu_y[i]) < 1e-6f)
            continue; // Skip very small values

        float abs_error = fabs(cpu_y[i] - gpu_y[i]);
        float rel_error = abs_error / fabs(cpu_y[i]);

        if (abs_error > abs_threshold && rel_error > rel_threshold)
        {
            errors++;
            if (abs_error > max_abs_error)
                max_abs_error = abs_error;
            if (rel_error > max_rel_error)
                max_rel_error = rel_error;
            if (errors <= 5)
            {
                printf("Mismatch at %d: CPU=%.8f, GPU=%.8f (abs=%.2e, rel=%.2e)\n",
                       i, cpu_y[i], gpu_y[i], abs_error, rel_error);
            }
        }
    }

    if (errors == 0)
    {
        printf("All results verified successfully!\n");
    }
    else
    {
        printf("Found %d significant errors (max abs=%.2e, max rel=%.2e)\n",
               errors, max_abs_error, max_rel_error);
    }
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Usage: %s <rows> <cols>\n", argv[0]);
        return 1;
    }
    const int rows = atoi(argv[1]);
    const int cols = atoi(argv[2]);
    size_t matrix_size = rows * cols * sizeof(float);
    size_t vec_size = cols * sizeof(float);
    size_t result_size = rows * sizeof(float);

    // Host memory allocation
    float *h_M = (float *)malloc(matrix_size);
    float *h_x = (float *)malloc(vec_size);
    float *h_y_gpu = (float *)malloc(result_size);
    float *h_y_cpu = (float *)malloc(result_size);

    // Initialize with random values (-1 to 1)
    for (int i = 0; i < rows * cols; i++)
    {
        h_M[i] = (2.0f * rand() / RAND_MAX) - 1.0f;
    }
    for (int i = 0; i < cols; i++)
    {
        h_x[i] = (2.0f * rand() / RAND_MAX) - 1.0f;
    }

    // Device memory allocation
    float *d_M, *d_x, *d_y;
    cudaMalloc(&d_M, matrix_size);
    cudaMalloc(&d_x, vec_size);
    cudaMalloc(&d_y, result_size);

    // Copy data to device
    cudaMemcpy(d_M, h_M, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, vec_size, cudaMemcpyHostToDevice);

    // Kernel configuration
    const int threads_per_block = 256;
    int blocks_per_grid = (rows + threads_per_block - 1) / threads_per_block;

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up run
    matVecMul_kernel<<<blocks_per_grid, threads_per_block>>>(d_M, d_x, d_y, rows, cols);
    cudaDeviceSynchronize();

    // Time kernel execution (average over 100 runs)
    int timing_runs = 100;
    cudaEventRecord(start);
    for (int i = 0; i < timing_runs; i++)
    {
        matVecMul_kernel<<<blocks_per_grid, threads_per_block>>>(d_M, d_x, d_y, rows, cols);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    gpu_ms /= timing_runs;

    // Copy result back
    cudaMemcpy(h_y_gpu, d_y, result_size, cudaMemcpyDeviceToHost);

    // Compute reference on CPU
    clock_t cpu_start = clock();
    cpuMatVecMul(h_M, h_x, h_y_cpu, rows, cols);
    clock_t cpu_end = clock();
    float cpu_ms = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // Calculate FLOPs: 2*rows*cols FLOPs per matrix-vector multiply
    double gflops = (2.0 * rows * cols) / (gpu_ms * 1e6);

    // Print results
    printf("\nMatrix-vector multiplication (%d x %d)\n", rows, cols);
    printf("GPU time: %.3f ms (avg over %d runs)\n", gpu_ms, timing_runs);
    printf("CPU time: %.3f ms\n", cpu_ms);
    printf("Speedup: %.2fx\n", cpu_ms / gpu_ms);
    printf("GPU throughput: %.2f GFLOP/s\n", gflops);

    // Verify results
    verify_results(h_y_cpu, h_y_gpu, rows);

    // Cleanup
    free(h_M);
    free(h_x);
    free(h_y_gpu);
    free(h_y_cpu);
    cudaFree(d_M);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}