APXCHOL_GPU_CODE(
struct Params {
    unsigned n;         // operator rows
    unsigned groups;    // reduction groups, ceil(n / kTreeRows)
    unsigned offset;    // first workgroup in this dependency stage
    unsigned count;     // number of dependency words to initialize
    unsigned mode;      // finalize mode
    unsigned iter;      // 1-based PCG iteration
    unsigned window;    // stagnation window, 0 = off
    unsigned m;         // factor dimension
};
enum : unsigned { kTreeRows = 256, kTreeLanes = 16 };
)
