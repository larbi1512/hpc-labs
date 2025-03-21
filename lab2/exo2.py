from mpi4py import MPI
import numpy as np

#initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# comm.barrier()

if rank == 0:
    k = 3
    data = np.random.rand(100, 2)
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    print("Master process (rank 0) is loading data...")
    
else:
    print("Worker process (rank %d) is waiting for data..." % rank)
    data = None
    centroids = None
    
# comm.Barrier()

data = comm.bcast(data, root=0)
comm.Barrier()
centroids = comm.bcast(centroids, root=0)


# comm.Barrier()

if rank == 0:
    print("Master process (rank 0) has finished broadcasting data to workers.")
else:
    print("Worker process (rank %d) has finished receiving data from master." % rank)
    

# print(f"Process {rank} received data with shape: {data.shape}")
# print(f"Process {rank} received centroids with shape: {centroids.shape}")
    