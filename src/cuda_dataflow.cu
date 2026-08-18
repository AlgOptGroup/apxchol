#include "apxchol/solver/sptrsv/cuda_dataflow.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>

// Sync-free (dataflow) GPU triangular solve with EPOCH-TAGGED values. See
// cuda_dataflow.h for the scheme, the deadlock-freedom argument, the state
// contract and the determinism claim; this file is the kernel.
//
// Tuning macros (compile-time). The defaults are the measured best on the
// RTX 4090 Laptop over grid_500 / grid_2000 / iter0040 (pair time, ms:
// kPre 4 / 6 / 8 / 16 = 0.28,1.59,0.38 / 0.30,1.36,0.38 / 0.28,1.34,0.42 /
// 0.27,-,- ; spin 100..800 ns beats fixed 100 / 200 / 400 ns and no spin
// (0.31,1.41,0.54); __nanosleep is unusable: its granularity is ~1 us here).
#ifndef APXCHOL_DF_SLEEP
#define APXCHOL_DF_SLEEP 3    // 0 = __nanosleep exponential 32..1024 ns; 1 = no sleep; 3 = clock64 spin, APXCHOL_DF_SPIN_NS doubling to APXCHOL_DF_SPIN_MAX_NS
#endif
#ifndef APXCHOL_DF_SPIN_NS
#define APXCHOL_DF_SPIN_NS 100
#endif
#ifndef APXCHOL_DF_SPIN_MAX_NS
#define APXCHOL_DF_SPIN_MAX_NS 800
#endif
#ifndef APXCHOL_DF_PRE
#define APXCHOL_DF_PRE 8      // (colidx, value) pairs prefetched into registers per lane before the wait
#endif
// -DAPXCHOL_DF_TRACE: per-row globaltimer stamps (claimed / ready / published)
// into a caller-supplied device buffer (dataflow_set_trace) -- a tuning aid.

namespace apxchol {

#ifdef APXCHOL_DF_TRACE
__device__ unsigned long long* g_df_trace = nullptr;   // 5 x m: claimed / ready / published / gathered / fenced (globaltimer ns)
void dataflow_set_trace(unsigned long long* buf) { cudaMemcpyToSymbol(g_df_trace, &buf, sizeof(buf)); }
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#endif

namespace {

using VAL = sptrsv_gpu_value_t;
constexpr int kPre = APXCHOL_DF_PRE;
static_assert(kPre >= 1 && kPre <= 32, "the pending mask is 32 bits");

#ifdef APXCHOL_SPTRSV_FP32
static_assert(sizeof(VAL) == 4, "the tagged word packs a 4-byte value with a 4-byte epoch");
constexpr unsigned kFull = 0xffffffffu;

// ── the tagged word: {value, epoch} in one 8-byte word ──────────────────────
// A finished row publishes its value and its "done" flag in ONE naturally
// aligned 64-bit store (single-copy atomic), and a consumer reads both with
// ONE 64-bit load: if the epoch matches the current sweep the value is the
// value -- no fence, no counter, no atomics anywhere on the data path. Both
// sides use gpu-scope relaxed (L2, coherent) accesses; the asm is also a
// compiler barrier, so the poll cannot be hoisted.
__device__ __forceinline__ unsigned long long ld_tag(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_tag(unsigned long long* p, unsigned long long v) {
    asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long pack_tag(float v, unsigned epoch) {
    return (static_cast<unsigned long long>(epoch) << 32) | __float_as_uint(v);
}
__device__ __forceinline__ unsigned tag_epoch(unsigned long long w) { return static_cast<unsigned>(w >> 32); }
__device__ __forceinline__ float    tag_value(unsigned long long w) { return __uint_as_float(static_cast<unsigned>(w)); }

__device__ __forceinline__ void backoff(unsigned& sleep_ns) {
#if APXCHOL_DF_SLEEP == 0
    __nanosleep(sleep_ns);
    if (sleep_ns < 1024u) sleep_ns <<= 1;
#elif APXCHOL_DF_SLEEP == 3
    // Busy-wait on the SM clock: __nanosleep's granularity is ~1 us on this
    // GPU (measured), far above the ~0.3 us signal latency; a clock spin
    // costs no memory traffic and returns on time. ~1.8 cycles/ns.
    {
        const long long t0 = clock64();
        const long long cyc = static_cast<long long>(sleep_ns) * 18 / 10;
        while (clock64() - t0 < cyc) { }
        if (sleep_ns < APXCHOL_DF_SPIN_MAX_NS) sleep_ns = min(2u * sleep_ns, unsigned(APXCHOL_DF_SPIN_MAX_NS));
    }
#else
    (void)sleep_ns;
#endif
}

template <bool FP16>
__device__ __forceinline__ VAL load_val(const void* __restrict__ vals_raw, int p) {
    if constexpr (FP16) return static_cast<VAL>(__half2float(static_cast<const __half*>(vals_raw)[p]));
    else                return static_cast<const VAL*>(vals_raw)[p];
}

// Group-wide AND of a lane predicate (G aligned power of two: the xor
// butterfly stays inside the group). Every lane must participate.
__device__ __forceinline__ bool group_all(bool pred, int G) {
    int v = pred ? 1 : 0;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const int v2 = __shfl_xor_sync(kFull, v, o);
        if (o < G) v &= v2;
    }
    return v != 0;
}

// One chunk of this lane's slice of a row: entries p0, p0 + G, ..., kPre of
// them (or fewer at the end). Loaded once (structure + value), then the tags
// of the off-diagonal neighbours are polled until every one carries the
// current epoch. `pending` = the entries still missing; y[t] the values.
struct chunk {
    int jj[kPre]; VAL vv[kPre]; VAL y[kPre];
    unsigned pending;
    template <bool FP16>
    __device__ __forceinline__ void load(bool have, int i, int p0, int G, int end,
                                         const int* __restrict__ colidx, const void* __restrict__ vals_raw) {
        pending = 0u;
        #pragma unroll
        for (int t = 0; t < kPre; ++t) {
            const int p = p0 + t * G;
            const bool ok = have && p < end;
            jj[t] = ok ? colidx[p] : -1;
            vv[t] = ok ? load_val<FP16>(vals_raw, p) : VAL(0);
            y[t]  = VAL(0);
            if (ok && jj[t] != i) pending |= 1u << t;
        }
    }
    // Poll the pending entries once (all loads in flight together).
    __device__ __forceinline__ void poll(const unsigned long long* __restrict__ tag, unsigned epoch) {
        if (!pending) return;
        unsigned long long w[kPre];
        #pragma unroll
        for (int t = 0; t < kPre; ++t) w[t] = (pending >> t) & 1u ? ld_tag(tag + jj[t]) : 0ull;
        #pragma unroll
        for (int t = 0; t < kPre; ++t)
            if (((pending >> t) & 1u) && tag_epoch(w[t]) == epoch) { y[t] = tag_value(w[t]); pending &= ~(1u << t); }
    }
    // Accumulate in entry order (fixed -> deterministic); the fp32 diagonal
    // slot is captured, the fp16 one skipped (its diagonal comes from diag[]).
    template <bool FP16>
    __device__ __forceinline__ void accumulate(int i, VAL& sum, VAL& d) const {
        #pragma unroll
        for (int t = 0; t < kPre; ++t) {
            if (jj[t] < 0) continue;
            if (jj[t] == i) { if constexpr (!FP16) d = vv[t]; }
            else sum += vv[t] * y[t];
        }
    }
};

// ── the persistent kernel ───────────────────────────────────────────────────
// Every warp claims one BATCH per ticket: a run of consecutive rows (in sweep
// order) packed by the host into 32 lanes -- a row of at most kPre entries
// gets one lane, longer rows a group of G = 2, 4, ..., 32 lanes (G aligned,
// dataflow_batches in cuda_host.h). The G lanes of a row stride its CSR
// entries (lane sub takes entries sub, sub+G, ...), so every row's structure
// is prefetched into registers before the wait and, once its neighbours are
// published, its gather is ONE memory round trip whatever its length (rows
// beyond G*kPre entries continue in G*kPre chunks). The dense tail rows of an
// elimination factor (dozens of off-diagonals) are why: with a lane-per-row-
// plus-serial-fallback scheme they cost several round trips each, serialised
// inside the warp, and the tail is the critical path.
template <bool FP16>
__global__ void __launch_bounds__(kDataflowBlock)
dataflow_kernel(int m, bool reverse,
                const int* __restrict__ rowptr, const int* __restrict__ colidx,
                const void* __restrict__ vals_raw,
                const float* __restrict__ diag, const float* __restrict__ in_scale,
                const int* __restrict__ batch_start, int n_batches,
                unsigned long long* tag, unsigned epoch,
                int* ctrl, int total_warps,
                const VAL* __restrict__ rhs, VAL* __restrict__ out) {
    const int lane = threadIdx.x & 31;
    int* ticket   = ctrl;
    int* finished = ctrl + 1;
    for (;;) {
        int b = 0;
        if (lane == 0) b = atomicAdd(ticket, 1);
        b = __shfl_sync(kFull, b, 0);
        if (b >= n_batches) break;
        const int q0 = batch_start[b], q1 = batch_start[b + 1];   // sweep positions [q0, q1)
        const int k  = q1 - q0;                                    // rows in the batch (<= 32)
        // Lane r < k describes row r of the batch: its id, its CSR range and
        // its lane-group size G (the host's rule, restated).
        int i = 0, beg = 0, end = 0, G = 1;
        if (lane < k) {
            const int q = q0 + lane;
            i   = reverse ? m - 1 - q : q;
            beg = rowptr[i]; end = rowptr[i + 1];
            G = 1;
            while (G < 32 && G * kPre < end - beg) G <<= 1;
        }
        // Lane assignment: rows are laid out in order, each aligned to its G
        // (the host packed them so they fit); lane l belongs to the row whose
        // [start, start + G) contains it. Serial over the k rows (shuffles).
        int myrow = -1, mystart = 0, myG = 1;
        {
            int pos = 0;
            for (int r = 0; r < k; ++r) {
                const int g = __shfl_sync(kFull, G, r);
                pos = (pos + g - 1) & ~(g - 1);
                if (lane >= pos && lane < pos + g) { myrow = r; mystart = pos; myG = g; }
                pos += g;
            }
        }
        const bool have = myrow >= 0;
        const int  src  = have ? myrow : 0;
        i   = __shfl_sync(kFull, i,   src);
        beg = __shfl_sync(kFull, beg, src);
        end = __shfl_sync(kFull, end, src);
        const int  sub    = lane - mystart;               // this lane's slot inside its group
        const bool leader = have && sub == 0;
        // Prefetch: this lane's first chunk of the row, plus rhs / diag / scale
        // on the leader -- everything that does not depend on other rows.
        chunk c;
        c.load<FP16>(have, i, beg + sub, myG, end, colidx, vals_raw);
        VAL bval = VAL(0), dval = VAL(0);
        if (leader) {
            if constexpr (FP16) { bval = in_scale ? rhs[i] * static_cast<VAL>(in_scale[i]) : rhs[i]; dval = static_cast<VAL>(diag[i]); }
            else                { bval = rhs[i]; }
        }
#ifdef APXCHOL_DF_TRACE
        if (leader) g_df_trace[i] = gtimer();
#endif
        bool     done     = !have;
        unsigned sleep_ns = APXCHOL_DF_SPIN_NS;
        for (;;) {
            // Poll this lane's missing neighbours; a row is ready when every
            // lane of its group has its first chunk complete.
            if (!done) c.poll(tag, epoch);
            const bool complete = group_all(c.pending == 0u, myG);   // every lane takes part in the shuffles
            const bool ready    = !done && complete;
            if (__ballot_sync(kFull, ready) == 0u) {              // nothing to do: back off
                backoff(sleep_ns);
                continue;
            }
            sleep_ns = APXCHOL_DF_SPIN_NS;
#ifdef APXCHOL_DF_TRACE
            if (ready && leader) g_df_trace[m + i] = gtimer();
#endif
            // ── gather: first chunk, then any further chunks (rows longer than
            // G * kPre entries; each chunk is loaded, polled to completion by the
            // whole group, and accumulated in order) ──
            VAL sum = VAL(0), d = VAL(0);
            if (ready) c.accumulate<FP16>(i, sum, d);
            {
                // The lanes of ready rows walk their extra chunks in lock-step
                // with their group; other lanes idle through the shuffles.
                int p0 = ready ? beg + sub + kPre * myG : end;
                bool more = ready && p0 < end;
                while (__ballot_sync(kFull, more)) {
                    chunk c2;
                    c2.load<FP16>(more, i, p0, myG, end, colidx, vals_raw);
                    for (;;) {
                        if (more) c2.poll(tag, epoch);
                        const bool ok = group_all(!more || c2.pending == 0u, myG);
                        if (__all_sync(kFull, ok)) break;
                        backoff(sleep_ns);
                    }
                    sleep_ns = APXCHOL_DF_SPIN_NS;
                    if (more) { c2.accumulate<FP16>(i, sum, d); p0 += kPre * myG; more = p0 < end; }
                }
            }
            // Group reduction (G aligned, power of two): the xor butterfly stays
            // inside the group. Lanes of a not-ready group contribute nothing
            // and ignore the result. Fixed order -> deterministic.
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const VAL s2 = __shfl_xor_sync(kFull, sum, o);
                const VAL d2 = __shfl_xor_sync(kFull, d, o);
                if (o < myG) { sum += s2; d += d2; }
            }
            if (ready && leader) {
                VAL yi;
                if constexpr (FP16) yi = (bval - sum) / dval;
                else                yi = (bval - sum) / d;
                out[i] = yi;                                   // the plain result vector
                st_tag(tag + i, pack_tag(yi, epoch));          // and the tagged word the consumers poll
#ifdef APXCHOL_DF_TRACE
                g_df_trace[2ull * m + i] = g_df_trace[3ull * m + i] = g_df_trace[4ull * m + i] = gtimer();
#endif
            }
            if (ready) done = true;
            if (__all_sync(kFull, done)) break;
        }
    }
    // Last warp out resets the control words. The fence orders this warp's
    // final ticket atomicAdd before its finished-count increment (relaxed
    // atomics to different addresses are otherwise unordered for observers),
    // so when the last warp sees total_warps-1 no ticket increment is in
    // flight any more.
    __syncwarp();
    if (lane == 0) {
        __threadfence();
        const int f = atomicAdd(finished, 1);
        if (f == total_warps - 1) {
            __threadfence();
            atomicExch(ticket, 0);
            atomicExch(finished, 0);
        }
    }
}

int resident_blocks(const void* kernel) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 1;
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || sms < 1) return 1;
    int per_sm = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kDataflowBlock, 0) != cudaSuccess || per_sm < 1)
        per_sm = 1;
    return std::max(1, per_sm * sms);
}
#endif // APXCHOL_SPTRSV_FP32

} // namespace

int dataflow_prefetch_depth() { return kPre; }

bool dataflow_supported() {
#ifdef APXCHOL_SPTRSV_FP32
    return true;
#else
    return false;
#endif
}

int dataflow_grid_size(bool fp16) {
    if (const char* e = std::getenv("APXCHOL_GPU_DATAFLOW_BLOCKS")) {
        const int v = std::atoi(e);
        if (v > 0) return v;
    }
#ifdef APXCHOL_SPTRSV_FP32
    return fp16 ? resident_blocks(reinterpret_cast<const void*>(&dataflow_kernel<true>))
                : resident_blocks(reinterpret_cast<const void*>(&dataflow_kernel<false>));
#else
    (void)fp16;
    return 1;
#endif
}

void dataflow_solve(cudaStream_t stream, int m, bool reverse,
                    const int* rowptr, const int* colidx, const VAL* vals,
                    const int* batch_start, int n_batches,
                    unsigned long long* tag, unsigned epoch, int* ctrl, int grid,
                    const VAL* rhs, VAL* out) {
    if (m <= 0 || n_batches <= 0) return;
#ifdef APXCHOL_SPTRSV_FP32
    const int total_warps = grid * (kDataflowBlock / 32);
    dataflow_kernel<false><<<grid, kDataflowBlock, 0, stream>>>(
        m, reverse, rowptr, colidx, vals, nullptr, nullptr,
        batch_start, n_batches, tag, epoch, ctrl, total_warps, rhs, out);
#else
    (void)stream; (void)reverse; (void)rowptr; (void)colidx; (void)vals; (void)batch_start;
    (void)tag; (void)epoch; (void)ctrl; (void)grid; (void)rhs; (void)out;
    throw std::runtime_error("apxchol: the dataflow GPU SpTRSV backend needs the fp32 SpTRSV build (APXCHOL_SPTRSV_FP32)");
#endif
}

void dataflow_solve_fp16(cudaStream_t stream, int m, bool reverse,
                         const int* rowptr, const int* colidx, const unsigned short* vals16,
                         const float* diag, const float* in_scale,
                         const int* batch_start, int n_batches,
                         unsigned long long* tag, unsigned epoch, int* ctrl, int grid,
                         const VAL* rhs, VAL* out) {
    if (m <= 0 || n_batches <= 0) return;
#ifdef APXCHOL_SPTRSV_FP32
    const int total_warps = grid * (kDataflowBlock / 32);
    dataflow_kernel<true><<<grid, kDataflowBlock, 0, stream>>>(
        m, reverse, rowptr, colidx, vals16, diag, in_scale,
        batch_start, n_batches, tag, epoch, ctrl, total_warps, rhs, out);
#else
    (void)stream; (void)reverse; (void)rowptr; (void)colidx; (void)vals16; (void)diag; (void)in_scale;
    (void)batch_start; (void)tag; (void)epoch; (void)ctrl; (void)grid; (void)rhs; (void)out;
    throw std::runtime_error("apxchol: the dataflow GPU SpTRSV backend needs the fp32 SpTRSV build (APXCHOL_SPTRSV_FP32)");
#endif
}

} // namespace apxchol
