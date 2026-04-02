#include <cassert>
#include <vector>

#include <mpi.h>

#include "solution.cpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> local_values = {rank + 1, rank + 2};
    const int actual = mpi_sum_reduce(local_values, MPI_COMM_WORLD);

    int expected = 0;
    for (int r = 0; r < size; ++r) {
        expected += (r + 1) + (r + 2);
    }
    assert(actual == expected);

    MPI_Finalize();
    return 0;
}
