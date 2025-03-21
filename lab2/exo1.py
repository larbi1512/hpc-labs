from mpi4py import MPI
import numpy as np

#initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    print("Master process (rank 0) is loading data...")
    data = np.random.rand(100, 10)  
    print("Data loaded on master. Broadcasting to workers...")
    
else:
    print("Worker process (rank %d) is waiting for data..." % rank)
    data = None
    

data = comm.bcast(data, root=0)

if rank == 0:
    print("Master process (rank 0) has finished broadcasting data to workers.")
else:
    print("Worker process (rank %d) has received data." % rank)
    
print(f"Process {rank} received data with shape: {data.shape}")
    
