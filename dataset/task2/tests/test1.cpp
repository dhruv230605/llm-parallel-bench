#include <cassert>
#include <stdexcept>
#include <vector>

#include "solution.cpp"

int main() {
    const std::vector<std::vector<int>> a = {
        {1, 2, 3},
        {4, 5, 6}
    };
    const std::vector<std::vector<int>> b = {
        {7, 8},
        {9, 10},
        {11, 12}
    };

    const auto result = parallel_matrix_multiply(a, b);
    assert(result.size() == 2);
    assert(result[0][0] == 58);
    assert(result[0][1] == 64);
    assert(result[1][0] == 139);
    assert(result[1][1] == 154);

    bool threw = false;
    try {
        (void)parallel_matrix_multiply({{1, 2}}, {{1, 2}});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);
    return 0;
}
