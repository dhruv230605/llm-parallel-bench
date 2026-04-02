#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

class ThreadPool {
public:
    explicit ThreadPool(std::size_t num_threads) : stopping_(false), active_tasks_(0) {
        if (num_threads == 0) {
            throw std::invalid_argument("num_threads must be positive");
        }
        for (std::size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this]() { worker_loop(); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
        for (std::thread &worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    std::future<int> enqueue(std::function<int()> task) {
        auto packaged = std::make_shared<std::packaged_task<int()>>(std::move(task));
        std::future<int> result = packaged->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                throw std::runtime_error("cannot enqueue on stopped thread pool");
            }
            tasks_.push([packaged]() { (*packaged)(); });
        }
        cv_.notify_one();
        return result;
    }

    void wait_for_tasks() {
        std::unique_lock<std::mutex> lock(mutex_);
        completed_cv_.wait(lock, [this]() { return tasks_.empty() && active_tasks_ == 0; });
    }

private:
    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.front());
                tasks_.pop();
                ++active_tasks_;
            }

            task();

            {
                std::lock_guard<std::mutex> lock(mutex_);
                --active_tasks_;
                if (tasks_.empty() && active_tasks_ == 0) {
                    completed_cv_.notify_all();
                }
            }
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable completed_cv_;
    bool stopping_;
    std::size_t active_tasks_;
};
