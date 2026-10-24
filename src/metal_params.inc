APXCHOL_GPU_CODE(
struct Params {
    unsigned n;         // operator rows
    unsigned groups;    // reduction groups, ceil(n / kTreeRows)
    unsigned offset;    // first workgroup in this dependency stage
    unsigned count;     // number of dependency words to initialize
    unsigned mode;      // finalize mode
    unsigned iter;      // 1-based PCG iteration
    unsigned m;         // factor dimension
    unsigned spmv_lanes; // threads per operator row, 1 or kTreeLanes
    float preconditioner_scale;
    unsigned mapped_rows; // operator rows retain input order
};
enum : unsigned { kTreeRows = 256, kTreeLanes = 16 };
)
