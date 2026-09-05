#include "cmg_setup_packed.h"
#include <chrono>
// The standalone driver is single-call, serial; no global instrumentation API.
double cmg_setup_seconds = 0.0;
int cmg_setup_calls = 0;
bool cmg_hierarchy_valid = false;
unsigned cmg_levels = 0;
void cmg_setup_packed_timed(const coder::sparse *A, struct0_T *H, int *flag) {
    const auto begin = std::chrono::steady_clock::now();
    cmg_setup_packed(A, H, flag);
    cmg_setup_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
    cmg_hierarchy_valid = H->valid;
    cmg_levels = H->nlevels;
    ++cmg_setup_calls;
}
