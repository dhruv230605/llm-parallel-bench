#include <stdexcept>
#include <vector>

std::vector<std::vector<int>> parallel_matrix_multiply(
    const std::vector<std::vector<int>> &a,
    const std::vector<std::vector<int>> &b) {
    if (a.empty() || b.empty() || a[0].empty() || b[0].empty()) {
        throw std::invalid_argument("matrices must be non-empty");
    }

    const std::size_t a_rows = a.size();
    const std::size_t a_cols = a[0].size();
    const std::size_t b_rows = b.size();
    const std::size_t b_cols = b[0].size();

    if (a_cols != b_rows) {
        throw std::invalid_argument("incompatible matrix dimensions");
    }

    for (const auto &row : a) {
        if (row.size() != a_cols) {
            throw std::invalid_argument("matrix a is ragged");
        }
    }
    for (const auto &row : b) {
        if (row.size() != b_cols) {
            throw std::invalid_argument("matrix b is ragged");
        }
    }

    std::vector<std::vector<int>> result(a_rows, std::vector<int>(b_cols, 0));

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < static_cast<int>(a_rows); ++i) {
        for (int j = 0; j < static_cast<int>(b_cols); ++j) {
            int sum = 0;
            for (int k = 0; k < static_cast<int>(a_cols); ++k) {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }

    return result;
}
