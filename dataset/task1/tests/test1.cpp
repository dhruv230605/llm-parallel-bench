#include <atomic>
#include <cassert>
#include <chrono>
#include <future>
#include <thread>
#include <vector>

#include "solution.cpp"

int main() {
    ThreadPool pool(4);

    std::vector<std::future<int>> futures;
    int expected_sum = 0;
    for (int i = 1; i <= 32; ++i) {
        expected_sum += i * i;
        futures.push_back(pool.enqueue([i]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            return i * i;
        }));
    }

    int actual_sum = 0;
    for (std::future<int> &future : futures) {
        actual_sum += future.get();
    }
    assert(actual_sum == expected_sum);

    std::atomic<int> counter{0};
    for (int i = 0; i < 100; ++i) {
        pool.enqueue([&counter]() {
            counter.fetch_add(1, std::memory_order_relaxed);
            return 1;
        });
    }
    pool.wait_for_tasks();
    assert(counter.load() == 100);
    return 0;
}
