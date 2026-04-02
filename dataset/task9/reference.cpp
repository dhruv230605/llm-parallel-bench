#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

class WorkStealingScheduler {
public:
    explicit WorkStealingScheduler(std::size_t worker_count)
        : next_queue_(0), pending_tasks_(0), stopping_(false) {
        if (worker_count == 0) {
            throw std::invalid_argument("worker_count must be positive");
        }

        queues_.reserve(worker_count);
        for (std::size_t i = 0; i < worker_count; ++i) {
            queues_.push_back(std::make_unique<WorkerQueue>());
        }
        for (std::size_t i = 0; i < worker_count; ++i) {
            workers_.emplace_back([this, i]() { worker_loop(i); });
        }
    }

    ~WorkStealingScheduler() {
        {
            std::lock_guard<std::mutex> lock(global_mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
        for (std::thread &worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    void submit(std::function<void()> task) {
        const std::size_t index = next_queue_.fetch_add(1, std::memory_order_relaxed) % queues_.size();
        {
            std::lock_guard<std::mutex> lock(queues_[index]->mutex);
            queues_[index]->tasks.push_front(std::move(task));
        }
        pending_tasks_.fetch_add(1, std::memory_order_relaxed);
        cv_.notify_all();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(global_mutex_);
        idle_cv_.wait(lock, [this]() { return pending_tasks_.load(std::memory_order_relaxed) == 0; });
    }

private:
    struct WorkerQueue {
        std::deque<std::function<void()>> tasks;
        std::mutex mutex;
    };

    bool pop_local(std::size_t index, std::function<void()> &task) {
        std::lock_guard<std::mutex> lock(queues_[index]->mutex);
        if (queues_[index]->tasks.empty()) {
            return false;
        }
        task = std::move(queues_[index]->tasks.front());
        queues_[index]->tasks.pop_front();
        return true;
    }

    bool steal_task(std::size_t thief_index, std::function<void()> &task) {
        for (std::size_t victim = 0; victim < queues_.size(); ++victim) {
            if (victim == thief_index) {
                continue;
            }
            std::lock_guard<std::mutex> lock(queues_[victim]->mutex);
            if (!queues_[victim]->tasks.empty()) {
                task = std::move(queues_[victim]->tasks.back());
                queues_[victim]->tasks.pop_back();
                return true;
            }
        }
        return false;
    }

    void worker_loop(std::size_t index) {
        while (true) {
            std::function<void()> task;
            if (pop_local(index, task) || steal_task(index, task)) {
                task();
                const std::size_t remaining = pending_tasks_.fetch_sub(1, std::memory_order_relaxed) - 1;
                if (remaining == 0) {
                    std::lock_guard<std::mutex> lock(global_mutex_);
                    idle_cv_.notify_all();
                }
                continue;
            }

            std::unique_lock<std::mutex> lock(global_mutex_);
            cv_.wait(lock, [this]() {
                return stopping_ || pending_tasks_.load(std::memory_order_relaxed) > 0;
            });
            if (stopping_ && pending_tasks_.load(std::memory_order_relaxed) == 0) {
                return;
            }
        }
    }

    std::vector<std::unique_ptr<WorkerQueue>> queues_;
    std::vector<std::thread> workers_;
    std::atomic<std::size_t> next_queue_;
    std::atomic<std::size_t> pending_tasks_;
    std::mutex global_mutex_;
    std::condition_variable cv_;
    std::condition_variable idle_cv_;
    bool stopping_;
};
