#include <algorithm>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <thread>
#include <vector>

std::vector<int> parallel_bfs_levels(const std::vector<std::vector<int>> &graph, int source) {
    const std::size_t n = graph.size();
    if (source < 0 || static_cast<std::size_t>(source) >= n) {
        return std::vector<int>(n, -1);
    }

    std::vector<std::atomic<int>> distances(n);
    for (std::size_t i = 0; i < n; ++i) {
        distances[i].store(-1, std::memory_order_relaxed);
    }
    distances[static_cast<std::size_t>(source)].store(0, std::memory_order_relaxed);

    std::vector<int> frontier = {source};
    int level = 0;

    while (!frontier.empty()) {
        const std::size_t frontier_size = frontier.size();
        const std::size_t thread_count = std::max<std::size_t>(
            1,
            std::min<std::size_t>(
                std::thread::hardware_concurrency() == 0 ? 4 : std::thread::hardware_concurrency(),
                frontier_size));
        const std::size_t block_size = (frontier_size + thread_count - 1) / thread_count;

        std::vector<int> next_frontier;
        std::mutex next_mutex;
        std::vector<std::thread> threads;

        for (std::size_t block = 0; block < thread_count; ++block) {
            const std::size_t begin = block * block_size;
            const std::size_t end = std::min(frontier_size, begin + block_size);
            threads.emplace_back([&, begin, end]() {
                std::vector<int> local_next;
                for (std::size_t index = begin; index < end; ++index) {
                    const int node = frontier[index];
                    for (int neighbor : graph[static_cast<std::size_t>(node)]) {
                        int expected = -1;
                        if (distances[static_cast<std::size_t>(neighbor)].compare_exchange_strong(
                                expected, level + 1, std::memory_order_relaxed)) {
                            local_next.push_back(neighbor);
                        }
                    }
                }
                std::lock_guard<std::mutex> lock(next_mutex);
                next_frontier.insert(next_frontier.end(), local_next.begin(), local_next.end());
            });
        }
        for (std::thread &thread : threads) {
            thread.join();
        }

        frontier.swap(next_frontier);
        ++level;
    }

    std::vector<int> result(n, -1);
    for (std::size_t i = 0; i < n; ++i) {
        result[i] = distances[i].load(std::memory_order_relaxed);
    }
    return result;
}
