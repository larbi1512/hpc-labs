from mpi4py import MPI
import numpy as np
from sklearn.datasets import make_regression

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Hyperparameters
learning_rate = 0.01
num_iterations = 100

if rank == 0:
    X, y = make_regression(n_samples=1000, n_features=2, noise=10)
    y = y.reshape(-1, 1)
    
    # Initialize model parameters
    w = np.random.randn(2, 1)
    b = np.random.randn(1, 1)
    
    # Split data into partitions for workers
    partitions = np.array_split(X, size - 1)
    target_partitions = np.array_split(y, size - 1)
    
    # Send partitions to workers
    for i in range(1, size):
        comm.send((partitions[i-1], target_partitions[i-1]), dest=i)
    
    for _ in range(num_iterations):
        # Broadcast current weights and bias
        comm.bcast((w, b), root=0)
        
        # Aggregate gradients from workers
        total_dw = np.zeros_like(w)
        total_db = np.zeros_like(b)
        
        for i in range(1, size):
            dw, db = comm.recv(source=i)
            total_dw += dw
            total_db += db
        
        # Update parameters
        w -= learning_rate * (total_dw / (size - 1))
        b -= learning_rate * (total_db / (size - 1))
    
    print("Final parameters:", w, b)
else:
    # Receive data partition
    X_local, y_local = comm.recv(source=0)
    
    for _ in range(num_iterations):
        # Receive current weights and bias
        w, b = comm.bcast(None, root=0)
        
        # Compute predictions and gradients
        y_pred = X_local @ w + b
        error = y_pred - y_local
        dw = (X_local.T @ error) / len(y_local)
        db = np.sum(error) / len(y_local)
        
        # Send gradients to master
        comm.send((dw, db), dest=0)
