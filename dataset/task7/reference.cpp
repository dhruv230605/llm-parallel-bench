#include <mpi.h>

#include <numeric>
#include <vector>

int mpi_sum_reduce(const std::vector<int> &local_values, MPI_Comm comm) {
    const int local_sum = std::accumulate(local_values.begin(), local_values.end(), 0);
    int global_sum = 0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, comm);
    return global_sum;
}
