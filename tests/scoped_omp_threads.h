#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

// Override the requested team size without changing dynamic adjustment.
class scoped_omp_threads {
public:
    explicit scoped_omp_threads([[maybe_unused]] int threads) {
#ifdef _OPENMP
        previous_ = omp_get_max_threads();
        omp_set_num_threads(threads);
#endif
    }

    ~scoped_omp_threads() {
#ifdef _OPENMP
        omp_set_num_threads(previous_);
#endif
    }

    scoped_omp_threads(const scoped_omp_threads&) = delete;
    scoped_omp_threads& operator=(const scoped_omp_threads&) = delete;

private:
#ifdef _OPENMP
    int previous_ = 1;
#endif
};
