#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

std::vector<long long> parallel_prefix_sum(const std::vector<int> &input) {
    if (input.empty()) {
        return {};
    }

    const std::size_t n = input.size();
    const std::size_t thread_count = std::max<std::size_t>(1, std::min<std::size_t>(std::thread::hardware_concurrency() == 0 ? 4 : std::thread::hardware_concurrency(), n));
    const std::size_t block_size = (n + thread_count - 1) / thread_count;

    std::vector<long long> result(n, 0);
    std::vector<long long> block_totals(thread_count, 0);
    std::vector<std::thread> threads;

    for (std::size_t block = 0; block < thread_count; ++block) {
        const std::size_t begin = block * block_size;
        const std::size_t end = std::min(n, begin + block_size);
        threads.emplace_back([&, block, begin, end]() {
            long long running = 0;
            for (std::size_t i = begin; i < end; ++i) {
                running += input[i];
                result[i] = running;
            }
            block_totals[block] = running;
        });
    }
    for (std::thread &thread : threads) {
        thread.join();
    }

    long long offset = 0;
    for (std::size_t block = 0; block < thread_count; ++block) {
        const long long current_total = block_totals[block];
        block_totals[block] = offset;
        offset += current_total;
    }

    threads.clear();
    for (std::size_t block = 0; block < thread_count; ++block) {
        const std::size_t begin = block * block_size;
        const std::size_t end = std::min(n, begin + block_size);
        threads.emplace_back([&, block, begin, end]() {
            const long long block_offset = block_totals[block];
            for (std::size_t i = begin; i < end; ++i) {
                result[i] += block_offset;
            }
        });
    }
    for (std::thread &thread : threads) {
        thread.join();
    }

    return result;
}
