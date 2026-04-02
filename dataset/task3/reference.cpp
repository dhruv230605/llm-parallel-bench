#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <stdexcept>

class ProducerConsumerQueue {
public:
    explicit ProducerConsumerQueue(std::size_t capacity) : capacity_(capacity), closed_(false) {
        if (capacity_ == 0) {
            throw std::invalid_argument("capacity must be positive");
        }
    }

    void push(int value) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this]() { return queue_.size() < capacity_ || closed_; });
        if (closed_) {
            throw std::runtime_error("queue is closed");
        }
        queue_.push(value);
        not_empty_.notify_one();
    }

    bool pop(int &value) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this]() { return !queue_.empty() || closed_; });
        if (queue_.empty()) {
            return false;
        }
        value = queue_.front();
        queue_.pop();
        not_full_.notify_one();
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    std::size_t capacity_;
    std::queue<int> queue_;
    bool closed_;
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
};
