#include <algorithm>
#include <cassert>
#include <mutex>
#include <thread>
#include <vector>

#include "solution.cpp"

int main() {
    ProducerConsumerQueue queue(8);
    std::vector<int> consumed;
    std::mutex consumed_mutex;

    auto producer = [&queue](int base) {
        for (int i = 0; i < 50; ++i) {
            queue.push(base + i);
        }
    };

    auto consumer = [&]() {
        int value = 0;
        while (queue.pop(value)) {
            std::lock_guard<std::mutex> lock(consumed_mutex);
            consumed.push_back(value);
        }
    };

    std::thread p1(producer, 0);
    std::thread p2(producer, 100);
    std::thread c1(consumer);
    std::thread c2(consumer);

    p1.join();
    p2.join();
    queue.close();
    c1.join();
    c2.join();

    std::sort(consumed.begin(), consumed.end());
    assert(consumed.size() == 100);
    for (int i = 0; i < 50; ++i) {
        assert(consumed[i] == i);
        assert(consumed[i + 50] == 100 + i);
    }
    return 0;
}
