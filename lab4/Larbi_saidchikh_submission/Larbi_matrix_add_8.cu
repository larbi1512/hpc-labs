#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

// 2D Matrix Addition Kernel
__global__ void matrixAdd(const float *A, const float *B, float *C,
                          int width, int height)
{
    // Calculate 2D global indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (col < width && row < height)
    {
        int index = row * width + col;
        C[index] = A[index] + B[index];
    }
}

// CPU reference implementation
void cpuMatrixAdd(const float *A, const float *B, float *C, int width, int height)
{
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            int index = row * width + col;
            C[index] = A[index] + B[index];
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Usage: %s <width> <height>\n", argv[0]);
        printf("Example: %s 4096 4096\n", argv[0]);
        return 1;
    }

    const int width = atoi(argv[1]);
    const int height = atoi(argv[2]);
    if (width <= 0 || height <= 0)
    {
        printf("Invalid dimensions! Width and height must be positive integers.\n");
        return 1;
    }

    size_t size = width * height * sizeof(float);

    // Host memory allocation
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_C_ref = (float *)malloc(size);

    // Initialize matrices with random values
    for (int i = 0; i < width * height; i++)
    {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // Compute reference on CPU
    clock_t cpu_start = clock();
    cpuMatrixAdd(h_A, h_B, h_C_ref, width, height);
    clock_t cpu_end = clock();
    float cpu_time = (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0f;

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Configure 2D kernel launch parameters
    dim3 blockSize(16, 16); // 256 threads per block (16x16)
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up run
    matrixAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, width, height);
    cudaDeviceSynchronize();

    // Time kernel execution (average over 100 runs)
    int runs = 100;
    cudaEventRecord(start);
    for (int i = 0; i < runs; i++)
    {
        matrixAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, width, height);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    gpu_time /= runs; // Average time per run

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify results
    int errors = 0;
    for (int i = 0; i < width * height; i++)
    {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-5f)
        {
            errors++;
            if (errors < 5)
            {
                printf("Mismatch at %d: CPU=%f, GPU=%f\n", i, h_C_ref[i], h_C[i]);
            }
        }
    }

    // Print performance results
    printf("\nMatrix Addition (%d x %d)\n", width, height);
    printf("Block size: %d x %d (%d threads)\n", blockSize.x, blockSize.y,
           blockSize.x * blockSize.y);
    printf("Grid size: %d x %d\n", gridSize.x, gridSize.y);
    printf("CPU time: %.3f ms\n", cpu_time);
    printf("GPU time: %.3f ms\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("Throughput: %.2f GB/s\n",
           (3.0 * size / (gpu_time * 1e6))); // 2 reads + 1 write

    if (errors == 0)
    {
        printf("Results verified successfully!\n");
    }
    else
    {
        printf("Completed with %d errors\n", errors);
    }
    printf("\nMatrix Addition (%d x %d)\n", width, height);

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
