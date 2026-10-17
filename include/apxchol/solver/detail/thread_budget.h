#pragma once
#include <omp.h>
#include <stdexcept>

namespace apxchol::detail {

// omp_set_num_threads changes the calling task's ICV, not a process-global setting.
class thread_budget {
public:
    explicit thread_budget(int threads) {
        if (threads < 0) throw std::invalid_argument("num_threads must be nonnegative");
        if (threads > 0 && threads != omp_get_max_threads()) {
            previous_ = omp_get_max_threads();
            omp_set_num_threads(threads);
        }
    }
    ~thread_budget() {
        if (previous_ > 0) omp_set_num_threads(previous_);
    }
    thread_budget(const thread_budget&) = delete;
    thread_budget& operator=(const thread_budget&) = delete;
private:
    int previous_ = 0;
};

} // namespace apxchol::detail
