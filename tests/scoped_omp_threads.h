#pragma once

#include <omp.h>

// Override the requested team size without changing dynamic adjustment.
class scoped_omp_threads {
public:
    explicit scoped_omp_threads(int threads) {
        previous_ = omp_get_max_threads();
        omp_set_num_threads(threads);
    }

    ~scoped_omp_threads() {
        omp_set_num_threads(previous_);
    }

    scoped_omp_threads(const scoped_omp_threads&) = delete;
    scoped_omp_threads& operator=(const scoped_omp_threads&) = delete;

private:
    int previous_ = 1;
};
