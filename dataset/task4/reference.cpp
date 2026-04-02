#include <atomic>
#include <cstddef>
#include <memory>
#include <stdexcept>

class LockFreeQueue {
public:
    explicit LockFreeQueue(std::size_t capacity) {
        if (capacity < 2) {
            throw std::invalid_argument("capacity must be at least 2");
        }
        capacity_ = 1;
        while (capacity_ < capacity) {
            capacity_ <<= 1U;
        }
        mask_ = capacity_ - 1;
        buffer_ = std::make_unique<Cell[]>(capacity_);
        for (std::size_t i = 0; i < capacity_; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
            buffer_[i].data = 0;
        }
        enqueue_pos_.store(0, std::memory_order_relaxed);
        dequeue_pos_.store(0, std::memory_order_relaxed);
    }

    bool enqueue(int value) {
        Cell *cell = nullptr;
        std::size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
        while (true) {
            cell = &buffer_[pos & mask_];
            const std::size_t sequence = cell->sequence.load(std::memory_order_acquire);
            const intptr_t diff = static_cast<intptr_t>(sequence) - static_cast<intptr_t>(pos);
            if (diff == 0) {
                if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false;
            } else {
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }

        cell->data = value;
        cell->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    bool dequeue(int &value) {
        Cell *cell = nullptr;
        std::size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        while (true) {
            cell = &buffer_[pos & mask_];
            const std::size_t sequence = cell->sequence.load(std::memory_order_acquire);
            const intptr_t diff = static_cast<intptr_t>(sequence) - static_cast<intptr_t>(pos + 1);
            if (diff == 0) {
                if (dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false;
            } else {
                pos = dequeue_pos_.load(std::memory_order_relaxed);
            }
        }

        value = cell->data;
        cell->sequence.store(pos + capacity_, std::memory_order_release);
        return true;
    }

private:
    struct Cell {
        std::atomic<std::size_t> sequence;
        int data;
    };

    std::unique_ptr<Cell[]> buffer_;
    std::size_t capacity_{0};
    std::size_t mask_{0};
    std::atomic<std::size_t> enqueue_pos_{0};
    std::atomic<std::size_t> dequeue_pos_{0};
};
