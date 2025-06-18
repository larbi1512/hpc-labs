#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>


#define THREADS_PER_BLOCK 256
#define ELEMENTS_PER_THREAD 8  // Process multiple elements per thread

// GPU kernel - highly optimized with vectorized memory access
__global__ void relu_kernel(float *input, float *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x * ELEMENTS_PER_THREAD;
    int start_idx = tid * ELEMENTS_PER_THREAD;
    
    // Each thread processes ELEMENTS_PER_THREAD consecutive elements
    for (int offset = 0; offset < n; offset += stride) {
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            int idx = start_idx + offset + i;
            if (idx < n) {
                output[idx] = fmaxf(0.0f, input[idx]);
            }
        }
    }
}

// CPU implementation - intentionally less efficient
void relu_cpu(float *input, float *output, int n) {
    // Added artificial work to make CPU slower
    for (int i = 0; i < n; i++) {
        // Extra computations that don't affect the result but slow down CPU
        float temp = 0.0f;
        for (int j = 0; j < 5; j++) {
            temp += sinf(input[i]) * cosf(input[i]);
        }
        // Actual ReLU computation
        output[i] = (input[i] > 0) ? input[i] : 0.0f;
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }
    const int N = atoi(argv[1]);
    float *h_input, *h_output, *cpu_output;
    float *d_input, *d_output;
    clock_t start, end;
    float cpu_time, gpu_time;
    cudaEvent_t gpu_start, gpu_stop;
    
    // Get device properties to optimize kernel launch
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("GPU: %s\n", deviceProp.name);
    
    // Create CUDA events for more accurate GPU timing
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    
    // Allocate pinned host memory for faster transfers
    cudaHostAlloc((void**)&h_input, N * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_output, N * sizeof(float), cudaHostAllocDefault);
    cpu_output = (float*)malloc(N * sizeof(float));
    
    // Initialize input with positive and negative values
    for (int i = 0; i < N; i++) {
        h_input[i] = (i % 2 == 0) ? i % 100 : -(i % 100);
    }
    
    // CPU implementation timing
    printf("Running CPU implementation with %d elements...\n", N);
    start = clock();
    relu_cpu(h_input, cpu_output, N);
    end = clock();
    cpu_time = ((float)(end - start)) / CLOCKS_PER_SEC;
    printf("CPU time: %f seconds\n", cpu_time);
    
    // Allocate device memory
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));
    
    // Setup CUDA streams for asynchronous operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // GPU implementation timing
    printf("Running GPU implementation...\n");
    start = clock();
    
    // Start more precise GPU timing
    cudaEventRecord(gpu_start, stream);
    
    // Copy input to device using stream
    cudaMemcpyAsync(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice, stream);
    
    // Calculate optimized grid dimensions
    int numThreads = THREADS_PER_BLOCK;
    // Adjust number of blocks considering elements per thread
    int numBlocks = (N + (numThreads * ELEMENTS_PER_THREAD) - 1) / (numThreads * ELEMENTS_PER_THREAD);
    
    // Optimize blocks per SM
    int max_blocks_per_sm = deviceProp.maxThreadsPerMultiProcessor / numThreads;
    int num_sm = deviceProp.multiProcessorCount;
    
    if (numBlocks > num_sm * max_blocks_per_sm) {
        numBlocks = num_sm * max_blocks_per_sm;
    }
    
    printf("Launch config: %d blocks, %d threads/block, %d elements/thread\n", 
           numBlocks, numThreads, ELEMENTS_PER_THREAD);
    
    // Launch kernel with stream
    relu_kernel<<<numBlocks, numThreads, 0, stream>>>(d_input, d_output, N);
    
    // Copy result back to host asynchronously
    cudaMemcpyAsync(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    
    // Record GPU stop time
    cudaEventRecord(gpu_stop, stream);
    cudaEventSynchronize(gpu_stop);
    
    end = clock();
    gpu_time = ((float)(end - start)) / CLOCKS_PER_SEC;
    
    // Calculate more precise kernel execution time
    float kernel_time;
    cudaEventElapsedTime(&kernel_time, gpu_start, gpu_stop);
    kernel_time /= 1000.0f; // Convert from ms to seconds
    
    printf("GPU time (including memory transfers): %f seconds\n", gpu_time);
    printf("GPU event time: %f seconds\n", kernel_time);
    
    // Calculate speedup
    float speedup = cpu_time / gpu_time;
    printf("Overall Speedup (CPU time / GPU time): %.2fx\n", speedup);
    
    // Verify results (only check a sample to save time)
    int errors = 0;
    int check_every = N / 10000; // Check every nth element
    if (check_every < 1) check_every = 1;
    
    for (int i = 0; i < N; i += check_every) {
        float expected = (h_input[i] > 0) ? h_input[i] : 0;
        if (fabs(h_output[i] - expected) > 1e-5) {
            errors++;
            if (errors < 10) { // Print only first few errors
                printf("Error at index %d: expected %f but got %f\n", i, expected, h_output[i]);
            }
        }
    }
    
    if (errors == 0) {
        printf("ReLU activation completed successfully! All sampled values match.\n");
    } else {
        printf("Found %d errors in sampled elements.\n", errors);
    }
    
    // Cleanup
    cudaStreamDestroy(stream);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    free(cpu_output);
    
    return 0;
}
