#include "parallel.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace vectorcore::cpu {

namespace {

thread_local bool inside_parallel_region = false;

class ThreadPool {
public:
    ThreadPool() {
        const auto hardware = std::max(1U, std::thread::hardware_concurrency());
        for (unsigned i = 1; i < hardware; ++i) workers_.emplace_back([this] { worker_loop(); });
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        wake_.notify_all();
        for (auto& worker : workers_) worker.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    std::size_t threads() const { return workers_.size() + 1; }

    void run(std::size_t count, std::size_t grain,
             const std::function<void(std::size_t, std::size_t)>& body) {
        // One job at a time; concurrent callers from different user threads queue here.
        std::lock_guard<std::mutex> job_lock(job_mutex_);
        const auto chunks = (count + grain - 1) / grain;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            body_ = &body;
            count_ = count;
            grain_ = grain;
            chunks_ = chunks;
            next_chunk_.store(0, std::memory_order_relaxed);
            active_workers_ = workers_.size();
            error_ = nullptr;
            ++generation_;
        }
        wake_.notify_all();
        execute_chunks();  // the caller works too
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [this] { return active_workers_ == 0; });
        body_ = nullptr;
        if (error_) std::rethrow_exception(error_);
    }

private:
    void execute_chunks() {
        const bool was_inside = inside_parallel_region;
        inside_parallel_region = true;
        for (;;) {
            const auto chunk = next_chunk_.fetch_add(1, std::memory_order_relaxed);
            if (chunk >= chunks_) break;
            const auto begin = chunk * grain_;
            const auto end = std::min(begin + grain_, count_);
            try {
                (*body_)(begin, end);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!error_) error_ = std::current_exception();
                next_chunk_.store(chunks_, std::memory_order_relaxed);  // stop handing out work
            }
        }
        inside_parallel_region = was_inside;
    }

    void worker_loop() {
        std::size_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [&] { return stopping_ || generation_ != seen; });
                if (stopping_) return;
                seen = generation_;
            }
            execute_chunks();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                --active_workers_;
            }
            done_.notify_one();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex job_mutex_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;
    const std::function<void(std::size_t, std::size_t)>* body_{nullptr};
    std::size_t count_{0};
    std::size_t grain_{1};
    std::size_t chunks_{0};
    std::atomic<std::size_t> next_chunk_{0};
    std::size_t active_workers_{0};
    std::size_t generation_{0};
    bool stopping_{false};
    std::exception_ptr error_;
};

ThreadPool& pool() {
    static ThreadPool instance;
    return instance;
}

} // namespace

void parallel_for(std::size_t count, std::size_t grain,
                  const std::function<void(std::size_t, std::size_t)>& body) {
    if (count == 0) return;
    grain = std::max<std::size_t>(grain, 1);
    if (count <= grain || inside_parallel_region || pool().threads() == 1) {
        // Still honour the chunk partition so results stay identical to the
        // parallel path for callers that reduce per chunk.
        for (std::size_t begin = 0; begin < count; begin += grain) {
            body(begin, std::min(begin + grain, count));
        }
        return;
    }
    pool().run(count, grain, body);
}

std::size_t parallel_threads() { return pool().threads(); }

} // namespace vectorcore::cpu
