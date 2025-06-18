#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

// CUDA kernel for naive intra-block parallel scan
__global__ void naiveScanKernel(float* input, float* output, int n) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int blockOffset = blockIdx.x * blockDim.x;

    // Load input data to shared memory
    if (blockOffset + tid < n) {
        temp[tid] = input[blockOffset + tid];
    } else {
        temp[tid] = 0.0f;
    }
    __syncthreads();

    // Naive scan using repeated pairwise additions
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        float val = 0.0f;
        if (tid >= stride) {
            val = temp[tid - stride];
        }
        __syncthreads();
        
        if (tid >= stride) {
            temp[tid] += val;
        }
        __syncthreads();
    }

    // Write results to output
    if (blockOffset + tid < n) {
        output[blockOffset + tid] = temp[tid];
    }
}

// Sequential CPU scan for verification
void sequentialScan(float* input, float* output, int n) {
    output[0] = input[0];
    for (int i = 1; i < n; i++) {
        output[i] = output[i-1] + input[i];
    }
}

int main(int argc, char* argv[]) {
    // Get array size from command line or use default
    int N = 1024;
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            printf("Invalid array size. Using default N=1024\n");
            N = 1024;
        }
    }

    const int THREADS_PER_BLOCK = 256;
    const int BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const size_t SIZE = N * sizeof(float);

    // Host memory allocation
    float* h_input = (float*)malloc(SIZE);
    float* h_output = (float*)malloc(SIZE);
    float* h_verify = (float*)malloc(SIZE);

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f; // Simple test case: all 1s
    }

    // Device memory allocation
    float *d_input, *d_output;
    cudaMalloc(&d_input, SIZE);
    cudaMalloc(&d_output, SIZE);

    // Timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float cpu_time, gpu_time;

    // CPU execution and timing
    clock_t cpu_start = clock();
    sequentialScan(h_input, h_verify, N);
    clock_t cpu_end = clock();
    cpu_time = ((float)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0f; // ms

    // GPU execution and timing
    cudaEventRecord(start);
    cudaMemcpy(d_input, h_input, SIZE, cudaMemcpyHostToDevice);
    naiveScanKernel<<<BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float)>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, SIZE, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    // Calculate speedup
    float speedup = cpu_time / gpu_time;
    printf("CPU time: %.2f ms\n", cpu_time);
    printf("GPU time: %.2f ms\n", gpu_time);
    printf("Speedup: %.2fx\n", speedup);

    // Verify results
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_output[i] - h_verify[i]) > 1e-5) {
            printf("Verification failed at index %d: GPU = %f, CPU = %f\n", 
                   i, h_output[i], h_verify[i]);
            correct = false;
            break;
        }
    }
    if (correct) {
        printf("Verification passed!\n");
    }

    // Free memory
    free(h_input);
    free(h_output);
    free(h_verify);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}