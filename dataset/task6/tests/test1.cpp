#include <cassert>
#include <vector>

#include "solution.cpp"

int main() {
    std::vector<int> input = {1, 2, 3, 4, 5};
    std::vector<long long> expected = {1, 3, 6, 10, 15};
    assert(parallel_prefix_sum(input) == expected);

    std::vector<int> larger(100, 1);
    auto result = parallel_prefix_sum(larger);
    assert(result.size() == larger.size());
    for (std::size_t i = 0; i < result.size(); ++i) {
        assert(result[i] == static_cast<long long>(i + 1));
    }

    assert(parallel_prefix_sum({}).empty());
    return 0;
}
