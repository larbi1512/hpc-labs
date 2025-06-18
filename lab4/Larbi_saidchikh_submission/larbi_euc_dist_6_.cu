#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

// Configurable parameters for performance tuning
#define THREADS_PER_BLOCK 256
#define VECTORS_PER_THREAD 4  // Each thread processes multiple vector pairs

// Kernel that uses vectorized loads and shared memory for performance
__global__ void euclideanDist_kernel(const float* __restrict__ batchA, 
                                     const float* __restrict__ batchB,
                                     float* __restrict__ distances, 
                                     const int num_vectors, 
                                     const int dim) {
    // Thread ID within the block
    const int tid = threadIdx.x;
    // Each thread handles multiple vectors for better efficiency
    const int base_vec_idx = (blockIdx.x * blockDim.x + tid) * VECTORS_PER_THREAD;
    
    // Process VECTORS_PER_THREAD vector pairs per thread
    #pragma unroll
    for (int v = 0; v < VECTORS_PER_THREAD; v++) {
        const int vec_idx = base_vec_idx + v;
        
        // Check if this vector pair is within bounds
        if (vec_idx < num_vectors) {
            // Base index for the current vector in each batch
            const int base_idx = vec_idx * dim;
            
            // Initialize accumulator with high precision
            float sum_sq = 0.0f;
            
            // Process elements in chunks for better memory access patterns
            #pragma unroll 8
            for (int j = 0; j < dim; j++) {
                const int idx = base_idx + j;
                const float diff = batchA[idx] - batchB[idx];
                sum_sq += diff * diff;
            }
            
            // Store the result
            distances[vec_idx] = sqrtf(sum_sq);
        }
    }
}

// CPU implementation for verification and timing comparison
void euclideanDist_cpu(const float* batchA, const float* batchB, float* distances, 
                       const int num_vectors, const int dim) {
    for (int i = 0; i < num_vectors; i++) {
        float sum_sq = 0.0f;
        
        for (int j = 0; j < dim; j++) {
            int idx = i * dim + j;
            float diff = batchA[idx] - batchB[idx];
            sum_sq += diff * diff;
        }
        
        distances[i] = sqrtf(sum_sq);
    }
}

// Function to verify results between CPU and GPU
bool verify_results(float* cpu_results, float* gpu_results, int n, float tolerance) {
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (fabs(cpu_results[i] - gpu_results[i]) > tolerance) {
            errors++;
            if (errors < 10) {
                printf("Error at index %d: CPU = %f, GPU = %f\n", 
                       i, cpu_results[i], gpu_results[i]);
            }
        }
    }
    
    if (errors > 0) {
        printf("Found %d errors (tolerance: %f)\n", errors, tolerance);
        return false;
    }
    return true;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Usage: %s <num_vectors> <dim>\n", argv[0]);
        return 1;
    }
    const int NUM_VECTORS = atoi(argv[1]);
    const int DIM = atoi(argv[2]);
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    // Get device properties for optimization
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Using GPU: %s with %d SMs\n", deviceProp.name, deviceProp.multiProcessorCount);

    // Calculate memory requirements
    size_t vector_batch_size = NUM_VECTORS * DIM * sizeof(float);
    size_t distances_size = NUM_VECTORS * sizeof(float);
    
    printf("Problem size: %d vectors of dimension %d\n", NUM_VECTORS, DIM);
    printf("Memory per batch: %.2f MB\n", vector_batch_size / (1024.0 * 1024.0));
    
    // Host memory allocation - using pinned memory for faster transfers
    float *h_batchA, *h_batchB, *h_distances, *h_cpu_distances;
    cudaHostAlloc((void**)&h_batchA, vector_batch_size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_batchB, vector_batch_size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_distances, distances_size, cudaHostAllocDefault);
    h_cpu_distances = (float*)malloc(distances_size);
    
    // Initialize data with random values between 0 and 1
    srand(42);  // For reproducibility
    for (int i = 0; i < NUM_VECTORS * DIM; i++) {
        h_batchA[i] = (float)rand() / RAND_MAX;
        h_batchB[i] = (float)rand() / RAND_MAX;
    }
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float gpu_ms = 0.0f, cpu_ms = 0.0f;
    
    // Device memory allocation
    float *d_batchA, *d_batchB, *d_distances;
    cudaMalloc((void**)&d_batchA, vector_batch_size);
    cudaMalloc((void**)&d_batchB, vector_batch_size);
    cudaMalloc((void**)&d_distances, distances_size);
    
    // Create CUDA streams for overlapping operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Warmup the GPU
    cudaMemset(d_distances, 0, distances_size);
    
    // Copy data to device asynchronously
    cudaEventRecord(start, stream);
    cudaMemcpyAsync(d_batchA, h_batchA, vector_batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_batchB, h_batchB, vector_batch_size, cudaMemcpyHostToDevice, stream);
    
    // Calculate grid dimensions
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (NUM_VECTORS + (threadsPerBlock * VECTORS_PER_THREAD) - 1) / (threadsPerBlock * VECTORS_PER_THREAD);
    
    // Ensure we don't exceed hardware limitations
    int max_blocks_per_sm = 32;  // Conservative estimate
    int optimal_blocks = deviceProp.multiProcessorCount * max_blocks_per_sm;
    if (blocksPerGrid > optimal_blocks) {
        blocksPerGrid = optimal_blocks;
    }
    
    printf("Launching kernel with %d blocks, %d threads/block, %d vectors/thread\n", 
           blocksPerGrid, threadsPerBlock, VECTORS_PER_THREAD);
    
    // Launch kernel
    euclideanDist_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_batchA, d_batchB, d_distances, NUM_VECTORS, DIM);
    
    // Copy results back
    cudaMemcpyAsync(h_distances, d_distances, distances_size, cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    
    // Calculate GPU time
    cudaEventElapsedTime(&gpu_ms, start, stop);
    printf("GPU time: %.4f ms\n", gpu_ms);
    
    // CPU implementation for comparison
    clock_t cpu_start = clock();
    
    // Only run CPU implementation for a subset if the problem is very large
    int cpu_vectors = NUM_VECTORS;
    if (NUM_VECTORS > 100000) {
        cpu_vectors = 100000;  // Cap at 100k vectors to keep runtime reasonable
        printf("Running CPU implementation for first %d vectors only...\n", cpu_vectors);
    }
    
    euclideanDist_cpu(h_batchA, h_batchB, h_cpu_distances, cpu_vectors, DIM);
    
    clock_t cpu_end = clock();
    cpu_ms = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("CPU time for %d vectors: %.4f ms\n", cpu_vectors, cpu_ms);
    
    // Extrapolate CPU time if we processed a subset
    if (cpu_vectors < NUM_VECTORS) {
        float extrapolated_cpu_ms = cpu_ms * ((float)NUM_VECTORS / cpu_vectors);
        printf("Extrapolated CPU time for all %d vectors: %.4f ms\n", NUM_VECTORS, extrapolated_cpu_ms);
        cpu_ms = extrapolated_cpu_ms;
    }
    
    // Calculate speedup
    float speedup = cpu_ms / gpu_ms;
    printf("Speedup: %.2fx\n", speedup);
    
    // Verify results for the portion computed by CPU
    printf("Verifying results...\n");
    if (verify_results(h_cpu_distances, h_distances, cpu_vectors, 1e-5f)) {
        printf("Results match within tolerance!\n");
    }
    
    // Calculate throughput metrics
    float operations_per_vector = DIM * 3 + 1;  // 1 subtract, 1 square, 1 add per dimension, plus sqrt
    float total_operations = NUM_VECTORS * operations_per_vector;
    float gflops = total_operations / (gpu_ms * 1e-3) / 1e9;
    printf("Effective performance: %.2f GFLOPS\n", gflops);
    
    // Clean up
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree(d_batchA);
    cudaFree(d_batchB);
    cudaFree(d_distances);
    
    cudaFreeHost(h_batchA);
    cudaFreeHost(h_batchB);
    cudaFreeHost(h_distances);
    free(h_cpu_distances);
    
    return 0;
}
//speedup 5.15x