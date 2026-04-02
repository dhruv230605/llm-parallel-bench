#include <atomic>
#include <cassert>
#include <chrono>
#include <thread>

#include "solution.cpp"

int main() {
    WorkStealingScheduler scheduler(4);
    std::atomic<int> completed{0};
    std::atomic<int> sum{0};

    for (int i = 1; i <= 200; ++i) {
        scheduler.submit([i, &completed, &sum]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            sum.fetch_add(i, std::memory_order_relaxed);
            completed.fetch_add(1, std::memory_order_relaxed);
        });
    }

    scheduler.wait();
    assert(completed.load() == 200);
    assert(sum.load() == (200 * 201) / 2);
    return 0;
}
