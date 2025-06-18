# High Performance Computing Labs

This repository contains a collection of lab exercises and assignments completed for a High Performance Computing (HPC) module. The labs explore parallel and distributed computing concepts using technologies such as OpenMP, MPI, and CUDA. Each lab focuses on different aspects of HPC, including parallel programming with shared memory, distributed memory, and GPU acceleration.

---

## Table of Contents

- [Lab Structure](#lab-structure)
- [Technologies Used](#technologies-used)
- [Setup and Prerequisites](#setup-and-prerequisites)
- [Lab Summaries](#lab-summaries)
  - [Lab 2: Distributed Regression with MPI](#lab-2-distributed-regression-with-mpi)
  - [Lab 3: Parallel Programming with OpenMP](#lab-3-parallel-programming-with-openmp)
  - [Lab 4: GPU Programming with CUDA](#lab-4-gpu-programming-with-cuda)
  - [Lab 5: Parallel Scan Algorithms on GPU](#lab-5-parallel-scan-algorithms-on-gpu)
- [Performance Results](#performance-results)
- [How to Run](#how-to-run)
- [Contact](#contact)

---

## Lab Structure

The repository is organized as follows:

- `lab2/`: Distributed regression tasks using MPI and Python.
- `lab3/`: Parallel computation tasks using OpenMP and C.
- `lab4/`: CUDA programming assignments for vector/matrix operations and performance analysis.
- `lab5/`: Advanced CUDA implementations of parallel scan (prefix sum) algorithms.

---

## Technologies Used

- **C/C++** with OpenMP for shared-memory parallelism.
- **Python** with `mpi4py` and `scikit-learn` for distributed learning algorithms.
- **CUDA (C/C++)** for GPU-accelerated computing.
- **NVIDIA GPUs** (for CUDA labs).

---

## Setup and Prerequisites

- **OpenMP**: GCC or Clang compiler with OpenMP support.
- **MPI**: Install [Open MPI](https://www.open-mpi.org/) or [MPICH](https://www.mpich.org/) and `mpi4py` for Python tasks.
- **CUDA Toolkit**: Required for compiling and running CUDA code ([NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)).
- **Python Packages**: `numpy`, `scikit-learn`, `mpi4py`.

To install Python requirements:
```bash
pip install numpy scikit-learn mpi4py
```

To compile C/CUDA code, use:
```bash
# For OpenMP
gcc -fopenmp -o task2 lab3/task2.c

# For CUDA
nvcc -o optimized_parallel_scan lab5/optimized_parallel_scan.cu
```

---

## Lab Summaries

### Lab 2: Distributed Regression with MPI

- Implements distributed linear regression using MPI in Python.
- Each process receives a partition of the data, computes local gradients, and synchronizes with the master process for parameter updates.
- Demonstrates collective communication, broadcasting, and distributed computation.
- Uses `scikit-learn` for data generation and evaluation.

### Lab 3: Parallel Programming with OpenMP

- **Task 1**: Demonstrates OpenMP parallel regions and thread identification.
- **Task 2**: Vector addition using OpenMP parallel for-loops.
- **Task 3**: Matrix multiplication parallelized with OpenMP.
- **Task 4**: Illustrates the importance of synchronization in multi-threaded increments with and without atomic operations.

### Lab 4: GPU Programming with CUDA

- Implements and benchmarks various CUDA kernels:
  - Vector addition, ReLU activation, matrix-vector multiplication, dot product with shared memory, and 1D stencil computation.
  - Batched Euclidean distance computation with device queries and optimization.
- Measures speedup of GPU implementations over CPU baselines and reports verification outcomes.

### Lab 5: Parallel Scan Algorithms on GPU

- Implements both naive and optimized versions of parallel scan (prefix sum) on the GPU.
- Benchmarks speedups against CPU and naive GPU implementations.
- Includes detailed result output and kernel verification.

---

## Performance Results

Sample results from lab submissions:

- **Vector Addition**: Speedup of 117x (GPU vs CPU)
- **Element-wise Activation Function**: Speedup of 112x (GPU vs CPU)
- **Dot Product**: Speedup of 101x (GPU vs CPU)
- **1D Stencil Computation**: Speedup of 141x (GPU vs CPU)
- **Parallel Scan**: Optimized GPU scan achieves up to 130x speedup over CPU baseline

See `lab4/Larbi_saidchikh_submission/speedup_results.txt` and `lab5/submission.txt` for detailed benchmarks and verification logs.

---

## How to Run

### OpenMP C Programs

```bash
gcc -fopenmp -o task3 lab3/task3.c
./task3
```

### MPI Python Programs

```bash
mpiexec -n 4 python lab2/exo3.py
```

### CUDA Programs

```bash
nvcc -o optimized_parallel_scan lab5/optimized_parallel_scan.cu
./optimized_parallel_scan
```

---

## Contact

For questions or collaboration, please contact [larbi1512](https://github.com/larbi1512).

---
