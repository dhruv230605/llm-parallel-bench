#include <algorithm>
#include <cassert>
#include <random>
#include <vector>

#include "solution.cpp"

int main() {
    std::vector<int> values = {5, 1, 9, 3, 3, 7, 2, 8, 6, 4, 0};
    parallel_quicksort(values);
    assert(std::is_sorted(values.begin(), values.end()));

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-1000, 1000);
    std::vector<int> large(5000);
    for (int &value : large) {
        value = dist(rng);
    }
    auto expected = large;
    std::sort(expected.begin(), expected.end());
    parallel_quicksort(large);
    assert(large == expected);
    return 0;
}
