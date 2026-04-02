#include <cassert>
#include <thread>
#include <vector>

#include "solution.cpp"

int main() {
    ConcurrentHashMap map(32);
    map.put(100, 10);

    int value = 0;
    assert(map.get(100, value));
    assert(value == 10);

    constexpr int threads = 8;
    constexpr int keys = 20;
    constexpr int increments_per_thread = 100;
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&map]() {
            for (int i = 0; i < increments_per_thread; ++i) {
                for (int key = 0; key < keys; ++key) {
                    map.increment(key, 1);
                }
            }
        });
    }
    for (std::thread &worker : workers) {
        worker.join();
    }

    for (int key = 0; key < keys; ++key) {
        int current = 0;
        assert(map.get(key, current));
        assert(current == threads * increments_per_thread);
    }
    assert(map.size() >= static_cast<std::size_t>(keys));
    return 0;
}
