#include <atomic>
#include <cstddef>
#include <list>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

class ConcurrentHashMap {
public:
    explicit ConcurrentHashMap(std::size_t bucket_count)
        : buckets_(bucket_count), locks_(bucket_count), size_(0) {
        if (bucket_count == 0) {
            throw std::invalid_argument("bucket_count must be positive");
        }
    }

    void put(int key, int value) {
        Bucket &bucket = buckets_[bucket_index(key)];
        std::lock_guard<std::mutex> lock(locks_[bucket_index(key)]);
        for (auto &entry : bucket) {
            if (entry.first == key) {
                entry.second = value;
                return;
            }
        }
        bucket.emplace_back(key, value);
        size_.fetch_add(1, std::memory_order_relaxed);
    }

    bool get(int key, int &value) const {
        const std::size_t index = bucket_index(key);
        std::lock_guard<std::mutex> lock(locks_[index]);
        const Bucket &bucket = buckets_[index];
        for (const auto &entry : bucket) {
            if (entry.first == key) {
                value = entry.second;
                return true;
            }
        }
        return false;
    }

    void increment(int key, int delta) {
        Bucket &bucket = buckets_[bucket_index(key)];
        std::lock_guard<std::mutex> lock(locks_[bucket_index(key)]);
        for (auto &entry : bucket) {
            if (entry.first == key) {
                entry.second += delta;
                return;
            }
        }
        bucket.emplace_back(key, delta);
        size_.fetch_add(1, std::memory_order_relaxed);
    }

    std::size_t size() const {
        return size_.load(std::memory_order_relaxed);
    }

private:
    using Bucket = std::list<std::pair<int, int>>;

    std::size_t bucket_index(int key) const {
        return static_cast<std::size_t>(key >= 0 ? key : -key) % buckets_.size();
    }

    std::vector<Bucket> buckets_;
    mutable std::vector<std::mutex> locks_;
    std::atomic<std::size_t> size_;
};
