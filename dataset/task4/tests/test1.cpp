#include <algorithm>
#include <cassert>
#include <mutex>
#include <thread>
#include <vector>

#include "solution.cpp"

int main() {
    constexpr int producers = 4;
    constexpr int consumers = 4;
    constexpr int items_per_producer = 100;
    constexpr int total_items = producers * items_per_producer;

    LockFreeQueue queue(256);
    std::vector<int> seen(total_items, 0);
    std::mutex seen_mutex;
    int consumed_total = 0;

    std::vector<std::thread> producer_threads;
    for (int producer = 0; producer < producers; ++producer) {
        producer_threads.emplace_back([producer, &queue]() {
            for (int i = 0; i < items_per_producer; ++i) {
                const int value = producer * items_per_producer + i;
                while (!queue.enqueue(value)) {
                    std::this_thread::yield();
                }
            }
        });
    }

    std::vector<std::thread> consumer_threads;
    for (int consumer = 0; consumer < consumers; ++consumer) {
        consumer_threads.emplace_back([&]() {
            while (true) {
                int value = -1;
                if (queue.dequeue(value)) {
                    {
                        std::lock_guard<std::mutex> lock(seen_mutex);
                        ++seen[value];
                        ++consumed_total;
                        if (consumed_total == total_items) {
                            return;
                        }
                    }
                } else {
                    std::lock_guard<std::mutex> lock(seen_mutex);
                    if (consumed_total == total_items) {
                        return;
                    }
                    std::this_thread::yield();
                }
            }
        });
    }

    for (std::thread &producer : producer_threads) {
        producer.join();
    }
    for (std::thread &consumer : consumer_threads) {
        consumer.join();
    }

    assert(consumed_total == total_items);
    for (int count : seen) {
        assert(count == 1);
    }
    return 0;
}
