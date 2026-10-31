#pragma once
#include "apxchol/solver/detail/metal_host.h"
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "apxchol/solver/detail/residual_policy.h"
#define APXCHOL_GPU_CODE(...) __VA_ARGS__
#define APXCHOL_GPU_FUNCTION inline
#define APXCHOL_GPU_UINT64 std::uint64_t
#include "gks_star.inc"
namespace sparse_round { using std::isfinite; using std::sqrt; }
#include "math.inc"
#undef APXCHOL_GPU_FUNCTION
#undef APXCHOL_GPU_CODE
#include <cstddef>
#include <limits>
#include <stdexcept>
#define APXCHOL_GPU_CODE(...) __VA_ARGS__
#define APXCHOL_GPU_FUNCTION inline
#define APXCHOL_GPU_BUFFER(T) T*
#include "setup.inc"
#include "sparsify.inc"
#include "view.inc"
#undef APXCHOL_GPU_BUFFER
#undef APXCHOL_GPU_UINT64
#undef APXCHOL_GPU_FUNCTION
#undef APXCHOL_GPU_CODE
namespace resident_setup {
inline Layout layout(unsigned n,unsigned e) {
    using enum Field;
    Layout l{};
    l.bytes=sizeof(State);
    auto add=[&]<class T>(Field field,std::size_t count) {
        l.bytes=(l.bytes+15)&~std::size_t(15);
        l.offsets[unsigned(field)]=l.bytes;
        if (count>std::numeric_limits<std::size_t>::max()/sizeof(T) ||
            count*sizeof(T)>std::numeric_limits<std::size_t>::max()-l.bytes)
            throw std::overflow_error("Metal setup buffer size overflow");
        l.bytes+=count*sizeof(T);
    };
    add.operator()<Edge>(graph0,e);add.operator()<Edge>(graph1,e);
    for (auto f:{begin,active_ids,entry_offsets,fill_offsets,update_ptr,round_ptr}) add.operator()<unsigned>(f,std::size_t(n)+1);
    for (auto f:{active,candidates,winners,selected,cursor,update_cursor}) add.operator()<unsigned>(f,n);
    using Wide=apxchol::detail::metal_host::df;
    add.operator()<Wide>(excess,n);
    add.operator()<Column>(columns,n);
    // Selected vertices are independent; sparsification stores each edge pair once.
    const auto pairs = std::size_t(e) / 2 + (e & 1u);
    add.operator()<Fill>(fills,pairs);
    add.operator()<Update<Wide>>(updates,pairs);
    add.operator()<sparse_round::row_result>(rows,n);
    add.operator()<Counts>(counts,std::size_t(n)+1);
    add.operator()<sparse_round::neighbor<Wide>>(records,e);
    // Star records are dead before update IDs are written; sparsification only uses IDs.
    l.offsets[unsigned(update_ids)]=l.offsets[unsigned(records)];
    add.operator()<unsigned>(histogram,33);
    add.operator()<unsigned>(degree_bound,n);
    return l;
}
}
