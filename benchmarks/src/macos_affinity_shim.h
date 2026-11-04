// Force-included into RCHOL's sources on macOS only. rchol_lap.cpp pins its
// recursion threads with Linux's sched_setaffinity/sched_getcpu, which macOS
// lacks (it has no thread-to-core pinning at all). These stand-ins keep the
// upstream source unchanged; pinning becomes a no-op and the CPU number it
// prints in its debug output is 0. Nothing in RCHOL's numerics depends on it.
#pragma once
#include <sched.h>
#include <sys/types.h>

typedef struct { unsigned long bits; } cpu_set_t;
#define CPU_ZERO(set) ((set)->bits = 0)
#define CPU_SET(cpu, set) ((set)->bits |= 1ul << ((cpu) % 64))
static inline int sched_setaffinity(pid_t, size_t, const cpu_set_t*) { return 0; }
static inline int sched_getcpu(void) { return 0; }
