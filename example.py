import numpy as np

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

except ImportError:
    print("mpi4py import failed.")
    rank = 0
    size = 1

from mpi_utils import mpi_split_array, mpi_combine_array

# Generate a test array
test_array = np.arange(1253)

# Split the array into sections - different bits go to different cores
core_test_array = mpi_split_array(test_array)

# Demonstrate that the array has been split into equal parts
print(rank, size, core_test_array.shape[0])

# Do something to the array sections
core_test_array_processed = core_test_array**2

# Re-combined the processed sections on the rank zero core
test_array_processed = mpi_combine_array(core_test_array_processed,
                                         test_array.shape[0])

# Save the processed array only on the rank zero core
if rank == 0:
    np.savetxt("test_processed.txt", test_array_processed)
