from mpi4py import MPI
import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

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
    
    # Generate test data
    X_test, y_test = make_regression(n_samples=200, n_features=2, noise=10)
    y_test = y_test.reshape(-1, 1)
    
    # Split test data into partitions
    test_partitions = np.array_split(X_test, size - 1)
    test_target_partitions = np.array_split(y_test, size - 1)
    
    # Send test data partitions to workers
    for i in range(1, size):
        comm.send((test_partitions[i-1], test_target_partitions[i-1]), dest=i)
    
    # Collect local MSEs
    total_mse = 0
    for i in range(1, size):
        local_mse = comm.recv(source=i)
        total_mse += local_mse
    
    # Compute average MSE
    avg_mse = total_mse / (size - 1)
    print("Final average MSE:", avg_mse)
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
    
    # Receive test data partition
    X_test_local, y_test_local = comm.recv(source=0)
    
    # Evaluate the model 
    y_test_pred = X_test_local @ w + b
    local_mse = mean_squared_error(y_test_local, y_test_pred)
    
    comm.send(local_mse, dest=0)
