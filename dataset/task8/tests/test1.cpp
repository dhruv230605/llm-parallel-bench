#include <cassert>
#include <vector>

#include "solution.cpp"

int main() {
    const std::vector<std::vector<int>> graph = {
        {1, 2},
        {3, 4},
        {4},
        {5},
        {5},
        {},
        {7},
        {}
    };

    const auto distances = parallel_bfs_levels(graph, 0);
    const std::vector<int> expected = {0, 1, 1, 2, 2, 3, -1, -1};
    assert(distances == expected);

    const auto invalid_source = parallel_bfs_levels(graph, 100);
    assert(invalid_source.size() == graph.size());
    for (int distance : invalid_source) {
        assert(distance == -1);
    }
    return 0;
}
