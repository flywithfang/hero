// core.cpp — the one non-inline seam: par_for's backing thread pool (M5).
//
// A persistent pool of N-1 worker threads plus the calling thread splits [0,n)
// via an atomic work-stealing counter. Every participant (main + workers) runs
// exactly one work_loop per generation and decrements the same `remaining_`,
// so the barrier count is always exactly N — regardless of how small n is.
//
// Reentrancy: par_for called from inside another par_for (e.g. a matvec inside
// a per-head map) runs SERIALLY — a thread_local guard prevents the classic
// nested-pool deadlock where every worker is already blocked in work_loop.
#include "core.hpp"
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>

namespace {

thread_local bool t_in_parallel = false;

class Pool {
public:
    static Pool& get() {
        static Pool p;
        return p;
    }
    size_t workers() const { return nthreads_; }

    void run(size_t n, const std::function<void(size_t)>& f) {
        if (n == 0) return;
        if (nthreads_ <= 1 || n == 1 || t_in_parallel) {  // serial / nested
            for (size_t i = 0; i < n; ++i) f(i);
            return;
        }
        const std::unique_lock<std::mutex> job(job_m_);  // one job at a time
        fn_ = &f;
        n_ = n;
        // Grab work in CHUNKS, not one index at a time: for the big matvecs
        // (Out up to 128256) per-index atomics dominate. ~16 chunks/worker keeps
        // load balanced while cutting atomic ops by the chunk factor.
        chunk_ = std::max<size_t>(1, n / (nthreads_ * 16));
        next_.store(0, std::memory_order_relaxed);
        remaining_.store(nthreads_, std::memory_order_relaxed);
        {
            const std::lock_guard<std::mutex> lk(m_);
            gen_++;
        }
        cv_.notify_all();
        work_loop();  // main participates
        std::unique_lock<std::mutex> lk(done_m_);
        done_cv_.wait(lk, [&] { return remaining_.load(std::memory_order_acquire) == 0; });
        fn_ = nullptr;
    }

    ~Pool() {
        {
            const std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            gen_++;
        }
        cv_.notify_all();
        for (auto& t : threads_) t.join();
    }

private:
    Pool() {
        const unsigned hc = std::thread::hardware_concurrency();
        nthreads_ = hc ? hc : 1;
        if (const char* e = std::getenv("NM_THREADS")) {
            const long v = std::atol(e);
            if (v > 0) nthreads_ = size_t(v);
        }
        for (size_t i = 1; i < nthreads_; ++i) threads_.emplace_back([this] { worker(); });
    }

    void work_loop() {
        t_in_parallel = true;
        for (;;) {
            const size_t b = next_.fetch_add(chunk_, std::memory_order_relaxed);
            if (b >= n_) break;
            const size_t e = std::min(b + chunk_, n_);
            for (size_t i = b; i < e; ++i) (*fn_)(i);
        }
        t_in_parallel = false;
        if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            const std::lock_guard<std::mutex> lk(done_m_);
            done_cv_.notify_all();
        }
    }

    void worker() {
        uint64_t seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
            if (stop_) return;
            seen = gen_;
            lk.unlock();
            work_loop();
        }
    }

    size_t nthreads_ = 1;
    std::vector<std::thread> threads_;
    std::mutex m_, done_m_, job_m_;
    std::condition_variable cv_, done_cv_;
    const std::function<void(size_t)>* fn_ = nullptr;
    size_t n_ = 0, chunk_ = 1;
    std::atomic<size_t> next_{0}, remaining_{0};
    uint64_t gen_ = 0;
    bool stop_ = false;
};

}  // namespace

void par_for(size_t n, const std::function<void(size_t)>& f) { Pool::get().run(n, f); }
size_t nm_num_threads() { return Pool::get().workers(); }
