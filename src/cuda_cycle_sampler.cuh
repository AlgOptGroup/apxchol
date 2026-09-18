// Included inside cuda_round_shadow.cu after its canonical-record/RNG helpers.
// Each pivot owns its plan and shuffle, including actual rejection consumption.
// Large trace parents use an indexed one-draw CDF; other rows retain their streams.
// Materialize standard-library constants outside device functions for NVCC.
constexpr double kCyclePoolMax = std::numeric_limits<pool_value_t>::max();
constexpr double kCycleEpsilon = std::numeric_limits<double>::epsilon();
constexpr double kCycleInfinity = std::numeric_limits<double>::infinity();
constexpr std::uint32_t kCycleCooperativeDegree = 128;
__device__ std::uint32_t cycle_uniform_index(unsigned long long& state,
                                            std::uint32_t bound) {
    const auto b = static_cast<unsigned long long>(bound), threshold = -b % b;
    unsigned long long value;
    do { value = next_random(state); } while (value < threshold);
    return static_cast<std::uint32_t>(value % b);
}
__device__ bool cycle_pool_edge(double value) {
    // Match pool insertion: positive subnormals survive; zero/overflow do not.
    return value > 0 && value <= kCyclePoolMax &&
           static_cast<pool_value_t>(value) > 0;
}
// 0: eligible, 1: input-only numerical fallback, 2: invalid/overflow.
__device__ int cycle_pool_range(double value) {
    if (!(value >= 0) || !(value <= kCyclePoolMax)) return 2;
    return static_cast<pool_value_t>(value) > 0 ? 0 : 1;
}
__device__ double cycle_edge_weight(const work_record* a, double D,
        std::uint32_t h, std::uint32_t i, std::uint32_t j) {
    return a[i].value * a[j].value / D * (h - 1) * .5;
}
__device__ bool cycle_nonnegative(double value, double magnitude, double& out) {
    const double tolerance = 512 * kCycleEpsilon * fmax(1., magnitude);
    if (!isfinite(value) || value < -tolerance) return false;
    out = fmax(0., value); return true;
}
__device__ std::uint32_t cycle_trace_cut(const work_record* a, std::uint32_t d,
        double* suffix, double* square) {
    const double scale = a[d - 1].value;
    if (!(scale > 0) || !isfinite(scale)) return d;
    double sum = 0, sq = 0;
    for (std::uint32_t i = d; i-- > 0;) {
        const double x = a[i].value / scale;
        if (!(x > 0) || !isfinite(x) || (i && a[i].value < a[i - 1].value)) return d;
        sum = x + sum; sq = x * x + sq;
        if (!isfinite(sum) || !isfinite(sq)) return d;
        suffix[i] = sum; square[i] = sq;
    }
    if (!(sum > 0) || !isfinite(sum * sum)) return d;
    double parents = 0, best = kCycleInfinity;
    std::uint32_t cut = 0;
    for (std::uint32_t i = 0; i + 2 < d; ++i) {
        const auto h = d - i;
        const double cycle = h == 3 ? 0. : double(h - 3) * double(h - 1) * .5 * square[i];
        // The finite positive total^2 is common to every cutoff. Compare the
        // numerators directly; no normalized objective escapes this routine.
        const double score = parents + cycle;
        if (!(score >= 0) || !isfinite(score)) return d;
        if (score < best) { best = score; cut = i; }
        if (i + 3 < d) {
            const double x = a[i].value / scale, m = d - i - 1;
            parents += (m - 1) * x * (2 * suffix[i + 1] + m * x);
        }
    }
    return cut;
}
// Large-row planning keeps the same positive moment objective, but folds each
// 32-item tile cooperatively. Floating-point sums/cutoff ties can differ from
// the scalar path; the estimator, earliest exact tie, and input-only fallback
// remain unchanged. Every call/return below is uniform across the full warp.
__device__ std::uint32_t cycle_trace_cut_warp(const work_record* a, std::uint32_t d,
        double* suffix, double* square) {
    constexpr unsigned mask=0xffffffffu;
    const unsigned lane=threadIdx.x%32;
    const double scale=a[d-1].value;
    if(!(scale>0)||!isfinite(scale))return d;
    double carry=0,carry_square=0;
    for(std::uint32_t end=d;end;) {
        const auto begin=end>32?end-32:0;
        const auto i=begin+lane;
        double x=i<end?a[i].value/scale:0;
        const bool invalid=i<end && (!(x>0)||!isfinite(x)||
            (i && a[i].value<a[i-1].value));
        if(__any_sync(mask,invalid))return d;
        double sum=x,sq=x*x;
        for(unsigned offset=1;offset<32;offset*=2) {
            const double right=__shfl_down_sync(mask,sum,offset);
            const double right_square=__shfl_down_sync(mask,sq,offset);
            if(lane+offset<32) {sum+=right;sq+=right_square;}
        }
        sum+=carry;sq+=carry_square;
        // Different positive folds can reverse adjacent suffixes by roundoff.
        // The parent inverse CDF requires monotonicity, including tile seams.
        sum=fmax(sum,carry);
        for(unsigned offset=1;offset<32;offset*=2) {
            const double right=__shfl_down_sync(mask,sum,offset);
            if(lane+offset<32)sum=fmax(sum,right);
        }
        if(__any_sync(mask,!isfinite(sum)||!isfinite(sq)))return d;
        if(i<end) {suffix[i]=sum;square[i]=sq;}
        carry=__shfl_sync(mask,sum,0);
        carry_square=__shfl_sync(mask,sq,0);
        end=begin;
    }
    if(!(carry>0)||!isfinite(carry*carry))return d;
    __syncwarp(); // all positive suffix moments precede score/parent reads
    double parents=0,best=kCycleInfinity;
    std::uint32_t cut=d;
    for(std::uint32_t begin=0;begin<d-2;begin+=32) {
        const auto i=begin+lane;
        double increment=0;
        if(i<d-3) {
            const double x=a[i].value/scale,m=d-i-1;
            increment=(m-1)*x*(2*suffix[i+1]+m*x);
        }
        double inclusive=increment;
        for(unsigned offset=1;offset<32;offset*=2) {
            const double left=__shfl_up_sync(mask,inclusive,offset);
            if(lane>=offset)inclusive+=left;
        }
        const double previous=__shfl_up_sync(mask,inclusive,1);
        const double before=parents+(lane?previous:0.);
        double score=kCycleInfinity;
        if(i<d-2) {
            const auto h=d-i;
            const double cycle=h==3?0.:double(h-3)*double(h-1)*.5*square[i];
            score=before+cycle;
        }
        if(__any_sync(mask,i<d-2 && (!(score>=0)||!isfinite(score))))return d;
        if(score<best || (score==best && i<cut)) {best=score;cut=i;}
        parents+=__shfl_sync(mask,inclusive,31);
    }
    for(unsigned offset=16;offset;offset/=2) {
        const double other=__shfl_down_sync(mask,best,offset);
        const auto other_cut=__shfl_down_sync(mask,cut,offset);
        if(lane+offset<32 && (other<best || (other==best && other_cut<cut))) {
            best=other;cut=other_cut;
        }
    }
    return __shfl_sync(mask,cut,0);
}
__device__ double cycle_parent_weight(const work_record* a, std::uint32_t d,
        double D, const double* suffix, std::uint32_t i, std::uint32_t j) {
    const double scale=a[d-1].value, ai=a[i].value/scale, aj=a[j].value/scale;
    const double mass=suffix[i+1]+double(d-i-1)*ai;
    return (a[i].value*(scale/D))*(mass/(1+ai/aj));
}
// Input-only numerical fallback uses the established GKS law and original seed.
__device__ std::uint32_t cycle_gks_row(const work_record* a, std::uint32_t d,
        double D, unsigned long long state, double* prefix,
        work_record* fill, std::uint8_t* flags) {
    if (!d) return 0;
    prefix[0]=a[0].value;
    for (std::uint32_t i=1;i<d;++i) prefix[i]=prefix[i-1]+a[i].value;
    std::uint32_t emitted=0;
    for (std::uint32_t i=0;i+1<d;++i) {
        const double suffix=prefix[d-1]-prefix[i];
        if (suffix<=0) continue;
        const double target=prefix[i]+next_unit(state)*suffix;
        std::uint32_t lo=i+1,hi=d;
        while(lo<hi) {const auto mid=lo+(hi-lo)/2;if(prefix[mid]<=target)lo=mid+1;else hi=mid;}
        const auto j=lo<d?lo:d-1;
        fill[i]={min(a[i].b,a[j].b),max(a[i].b,a[j].b),a[i].value*suffix/D};
        flags[i]=1;++emitted;
    }
    return emitted;
}
__device__ std::uint32_t cycle_gks_fallback_row(const work_record* a, std::uint32_t d,
        double D, unsigned long long state, double* prefix,
        work_record* fill, std::uint8_t* flags, std::uint32_t* status) {
    double total=0,partial=0;
    for(std::uint32_t i=0;i<d;++i) total+=a[i].value;
    if(!isfinite(total)) {atomicOr(status,2u);return 0;}
    for(std::uint32_t i=0;i+1<d;++i) {
        partial+=a[i].value;
        const double suffix=total-partial;
        if(suffix>0 && cycle_pool_range(a[i].value*suffix/D)==2) {
            atomicOr(status,2u);return 0;
        }
    }
    atomicAdd(status+1,1u);
    return cycle_gks_row(a,d,D,state,prefix,fill,flags);
}
__global__ void sample_cycle_rows(const work_record* canonical,
        const std::uint32_t* offsets, const device_pivot* pivots, std::size_t p,
        const double* total_degree, double* prefix, double* workspace,
        work_record* fills, std::uint8_t* fill_flags,
        gpu_round_shadow_pivot_counter* counters, clique_sampler sampler,
        std::uint32_t* status) {
    const std::size_t ordinal=blockIdx.x*blockDim.x+threadIdx.x;
    if(ordinal>=p)return;
    const auto begin=offsets[ordinal],d=offsets[ordinal+1]-begin;
    if(sampler==clique_sampler::trace_cycle && d>kCycleCooperativeDegree)return;
    if(!d){if(counters)counters[ordinal].emitted_edges=0;return;}
    const auto* a=canonical+begin;auto* fill=fills+begin;auto* flags=fill_flags+begin;
    auto* sums=prefix+begin;
    for(std::uint32_t i=0;i<d;++i)flags[i]=0;
    const double D=total_degree[ordinal];
    unsigned long long state=pivots[ordinal].seed;
    if(d<3) {
        const auto emitted=cycle_gks_row(a,d,D,state,sums,fill,flags);
        if(counters)counters[ordinal].emitted_edges=emitted;
        return;
    }
    int range = 0;
    if (!(D > 0) || !isfinite(D)) {atomicOr(status,2u);return;}
    for (std::uint32_t i=0;i<d;++i) {
        if (!(a[i].value >= 0) || !isfinite(a[i].value)) {atomicOr(status,2u);return;}
        if (!(a[i].value > 0)) range = 1;
    }
    if (!range && !(a[0].value/a[d-1].value > 0)) range=1;
    if (range) {
        const auto emitted=cycle_gks_fallback_row(a,d,D,state,sums,fill,flags,status);
        if(counters)counters[ordinal].emitted_edges=emitted;
        return;
    }
    const auto cut=cycle_trace_cut(a,d,sums,workspace+begin);
    if(cut==d) {
        const auto emitted=cycle_gks_fallback_row(a,d,D,state,sums,fill,flags,status);
        if(counters)counters[ordinal].emitted_edges=emitted;
        return;
    }
    const auto h=d-cut;
    range=max(cycle_pool_range(cycle_edge_weight(a,D,h,cut,cut+1)),
              cycle_pool_range(cycle_edge_weight(a,D,h,d-2,d-1)));
    // The final output slots double as temporary parent/permutation state.
    // Light and core ranges are disjoint; a cycle slot is overwritten only
    // after both endpoint indices are read, and its first endpoint is saved.
    for(std::uint32_t i=0;i<cut;++i) {
        const double ai=a[i].value/a[d-1].value,m=d-i-1;
        const double probability=m*ai/(sums[i+1]+m*ai);
        if(!(probability>=0 && probability<=.5) || !isfinite(probability)) {atomicOr(status,2u);return;}
        if (!(probability>0)) range=max(range,1);
        range=max(range,cycle_pool_range(cycle_parent_weight(a,d,D,sums,i,i+1)));
        range=max(range,cycle_pool_range(cycle_parent_weight(a,d,D,sums,i,d-1)));
    }
    if (range==2) {atomicOr(status,2u);return;}
    if (range==1) {
        const auto emitted=cycle_gks_fallback_row(a,d,D,state,sums,fill,flags,status);
        if(counters)counters[ordinal].emitted_edges=emitted;
        return;
    }
    for(std::uint32_t k=0;k<h;++k)fill[cut+k].a=cut+k;
    for(std::uint32_t k=h;k>1;--k) {
        const auto j=cycle_uniform_index(state,k),x=fill[cut+k-1].a;
        fill[cut+k-1].a=fill[cut+j].a;fill[cut+j].a=x;
    }
    const auto first=fill[cut].a;
    for(std::uint32_t k=0;k<h;++k) {
        const auto i=fill[cut+k].a,j=k+1<h?fill[cut+k+1].a:first;
        const auto value=cycle_edge_weight(a,D,h,i,j);
        if(!cycle_pool_edge(value)){atomicOr(status,2u);return;}
        fill[cut+k]={min(a[i].b,a[j].b),max(a[i].b,a[j].b),value};flags[cut+k]=1;
    }
    for(std::uint32_t i=0;i<cut;++i) {
        std::uint32_t j;
        const double ai=a[i].value/a[d-1].value,m=d-i-1;
        if(next_unit(state)<m*ai/(sums[i+1]+m*ai))j=i+1+cycle_uniform_index(state,d-i-1);
        else {
            const double sum=sums[i+1];double target=next_unit(state)*sum;
            if(target>=sum)target=nextafter(sum,0.);
            std::uint32_t lo=i+1,hi=d;
            while(lo+1<hi){const auto mid=lo+(hi-lo)/2;if(target<sums[mid])lo=mid;else hi=mid;}
            j=lo;
        }
        const double value=cycle_parent_weight(a,d,D,sums,i,j);
        if(!cycle_pool_edge(value)){atomicOr(status,2u);return;}
        fill[i]={min(a[i].b,a[j].b),max(a[i].b,a[j].b),value};flags[i]=1;
    }
    if(counters)counters[ordinal].emitted_edges=d;
}

// PERF: the plan kernel's validity scan, suffix moments and cutoff search are
// three passes over the same star in the original row kernel. This variant
// folds the validity scan into the moment pass and reports the reason, so the
// caller keeps the established semantics (invalid input is an error, a zero or
// degenerate ratio is a GKS fallback) at one pass instead of two.
// reason: 0 eligible, 1 numerical fallback, 2 invalid input.
__device__ std::uint32_t cycle_trace_cut_checked(const work_record* a, std::uint32_t d,
        double D, double* suffix, double* square, int& reason) {
    reason = 0;
    if (!(D > 0) || !isfinite(D)) { reason = 2; return d; }
    const double scale = a[d - 1].value;
    double sum = 0, sq = 0;
    for (std::uint32_t i = d; i-- > 0;) {
        const double w = a[i].value;
        if (!(w >= 0) || !isfinite(w)) { reason = 2; return d; }
        if (!(w > 0)) { reason = 1; return d; }
        if (i && w < a[i - 1].value) { reason = 1; return d; }
        const double x = w / scale;
        if (!(x > 0) || !isfinite(x)) { reason = 1; return d; }
        sum = x + sum; sq = x * x + sq;
        if (!isfinite(sum) || !isfinite(sq)) { reason = 1; return d; }
        suffix[i] = sum; square[i] = sq;
    }
    if (!(sum > 0) || !isfinite(sum * sum)) { reason = 1; return d; }
    if (!(a[0].value / scale > 0)) { reason = 1; return d; }
    double parents = 0, best = kCycleInfinity;
    std::uint32_t cut = 0;
    for (std::uint32_t i = 0; i + 2 < d; ++i) {
        const auto h = d - i;
        const double cycle = h == 3 ? 0. : double(h - 3) * double(h - 1) * .5 * square[i];
        const double score = parents + cycle;
        if (!(score >= 0) || !isfinite(score)) { reason = 1; return d; }
        if (score < best) { best = score; cut = i; }
        if (i + 3 < d) {
            const double x = a[i].value / scale, m = d - i - 1;
            parents += (m - 1) * x * (2 * suffix[i + 1] + m * x);
        }
    }
    return cut;
}

// PERF: two-kernel small-degree trace path, mirroring the GKS
// prepare_gks_prefix / sample_gks_items split. The plan kernel keeps one
// thread per pivot for the inherently serial work (suffix moments, cutoff,
// input-only range checks, Fisher-Yates); the item kernel then emits one
// edge per neighbour slot in parallel. Light parents use the same decoupled
// SplitMix draw and inverted CDF that sample_large_trace_rows already uses,
// so the law is unchanged while the serial per-pivot emission disappears.
// cut_out[ordinal] >= d marks a row the plan kernel completed by itself
// (empty, degree < 3, numerical fallback) or left to the cooperative kernel.
__global__ void prepare_trace_plan(const work_record* canonical,
        const std::uint32_t* offsets, const device_pivot* pivots, std::size_t p,
        const double* total_degree, double* prefix, double* workspace,
        work_record* fills, std::uint8_t* fill_flags,
        gpu_round_shadow_pivot_counter* counters,
        std::uint32_t* cut_out, unsigned long long* state_out,
        std::uint32_t* status) {
    const std::size_t ordinal=blockIdx.x*blockDim.x+threadIdx.x;
    if(ordinal>=p)return;
    const auto begin=offsets[ordinal],d=offsets[ordinal+1]-begin;
    cut_out[ordinal]=d;   // default: nothing left for the item kernel
    if(d>kCycleCooperativeDegree)return;            // cooperative kernel owns this row
    if(!d){if(counters)counters[ordinal].emitted_edges=0;return;}
    const auto* a=canonical+begin;auto* fill=fills+begin;auto* flags=fill_flags+begin;
    auto* sums=prefix+begin;
    const double D=total_degree[ordinal];
    unsigned long long state=pivots[ordinal].seed;
    // Every trace slot is written by the item kernel, so only the fallback
    // paths below need the zeroed flags the row kernel used to clear up front.
    auto fallback=[&](bool gks_law)->void{
        for(std::uint32_t i=0;i<d;++i)flags[i]=0;
        const auto emitted=gks_law?cycle_gks_row(a,d,D,state,sums,fill,flags)
                                  :cycle_gks_fallback_row(a,d,D,state,sums,fill,flags,status);
        if(counters)counters[ordinal].emitted_edges=emitted;
    };
    if(d<3){fallback(true);return;}
    int reason=0;
    const auto cut=cycle_trace_cut_checked(a,d,D,sums,workspace+begin,reason);
    if(reason==2){for(std::uint32_t i=0;i<d;++i)flags[i]=0;atomicOr(status,2u);return;}
    if(cut==d){fallback(false);return;}
    const auto h=d-cut;
    const double inv_scale=1./a[d-1].value;
    int range=max(cycle_pool_range(cycle_edge_weight(a,D,h,cut,cut+1)),
                  cycle_pool_range(cycle_edge_weight(a,D,h,d-2,d-1)));
    for(std::uint32_t i=0;i<cut;++i){
        const double ai=a[i].value*inv_scale,m=d-i-1;
        const double probability=m*ai/(sums[i+1]+m*ai);
        if(!(probability>=0 && probability<=.5) || !isfinite(probability)){
            for(std::uint32_t k=0;k<d;++k)flags[k]=0;atomicOr(status,2u);return;}
        if(!(probability>0))range=max(range,1);
        range=max(range,cycle_pool_range(cycle_parent_weight(a,d,D,sums,i,i+1)));
        range=max(range,cycle_pool_range(cycle_parent_weight(a,d,D,sums,i,d-1)));
    }
    if(range==2){for(std::uint32_t i=0;i<d;++i)flags[i]=0;atomicOr(status,2u);return;}
    if(range==1){fallback(false);return;}
    // The squared moments are dead once the cutoff is known; reuse that scratch
    // for the core permutation so the item kernel never overwrites an index it
    // or a sibling still has to read.
    double* perm=workspace+begin;
    for(std::uint32_t k=0;k<h;++k)perm[cut+k]=cut+k;
    for(std::uint32_t k=h;k>1;--k){
        const auto j=cycle_uniform_index(state,k);
        const double x=perm[cut+k-1];perm[cut+k-1]=perm[cut+j];perm[cut+j]=x;
    }
    state_out[ordinal]=state;   // post-shuffle stream: the parents draw beyond it
    cut_out[ordinal]=cut;
    if(counters)counters[ordinal].emitted_edges=d;
}

__global__ void sample_trace_items(const work_record* canonical, std::size_t count,
        const std::uint32_t* offsets, const double* total_degree,
        const double* prefix, const double* workspace,
        const std::uint32_t* cut_in, const unsigned long long* state_in,
        work_record* fills, std::uint8_t* fill_flags, std::uint32_t* status) {
    const std::size_t item=blockIdx.x*blockDim.x+threadIdx.x;
    if(item>=count)return;
    const auto i=static_cast<std::uint32_t>(item);
    const std::uint32_t ordinal=canonical[i].a;
    const auto begin=offsets[ordinal],d=offsets[ordinal+1]-begin;
    const std::uint32_t cut=cut_in[ordinal];
    if(cut>=d)return;                       // completed by the plan or cooperative kernel
    const auto* a=canonical+begin;
    const auto* sums=prefix+begin;
    const double D=total_degree[ordinal];
    const std::uint32_t k=i-begin;
    if(k>=cut) {
        const auto h=d-cut;
        const auto* perm=workspace+begin;
        const std::uint32_t slot=k-cut;
        const auto x=static_cast<std::uint32_t>(perm[cut+slot]);
        const auto y=static_cast<std::uint32_t>(perm[cut+(slot+1==h?0u:slot+1)]);
        const double value=cycle_edge_weight(a,D,h,x,y);
        if(!cycle_pool_edge(value)){atomicOr(status,2u);return;}
        fills[i]={min(a[x].b,a[y].b),max(a[x].b,a[y].b),value};
        fill_flags[i]=1;
        return;
    }
    auto draw_state=state_in[ordinal]+0x9E3779B97F4A7C15ULL*k;
    const double ai=a[k].value/a[d-1].value;
    const double mass=sums[k+1]+double(d-k-1)*ai;
    double target=next_unit(draw_state)*mass;
    if(target>=mass)target=nextafter(mass,0.);
    // R_k(j)=sum_{l>=j}a_l+(d-j)*a_k is decreasing; its adjacent interval has
    // width a_k+a_j, so the selected j keeps q_kj=(a_k+a_j)/mass.
    std::uint32_t lo=k+1,hi=d;
    while(lo+1<hi){const auto mid=lo+(hi-lo)/2;if(target<sums[mid]+double(d-mid)*ai)lo=mid;else hi=mid;}
    const auto j=lo;
    const double value=cycle_parent_weight(a,d,D,sums,k,j);
    if(!cycle_pool_edge(value)){atomicOr(status,2u);return;}
    fills[i]={min(a[k].b,a[j].b),max(a[k].b,a[j].b),value};
    fill_flags[i]=1;
}

__device__ int cycle_warp_range(int range) {
    for(int distance=16;distance;distance/=2)
        range=max(range,__shfl_down_sync(0xffffffffu,range,distance));
    return __shfl_sync(0xffffffffu,range,0);
}

// Large trace rows retain ordered moment folds and Fisher-Yates on lane zero.
// Light parents use one indexed draw each after the shuffle's actual stream
// consumption. Inverting q directly preserves its law without serial draw
// assignment; seeded parents intentionally differ from the two-draw mixture.
__global__ void sample_large_trace_rows(const work_record* canonical,
        const std::uint32_t* offsets, const device_pivot* pivots, std::size_t p,
        const double* total_degree, double* prefix, double* workspace,
        work_record* fills, std::uint8_t* fill_flags,
        gpu_round_shadow_pivot_counter* counters, std::uint32_t* status) {
    const unsigned lane=threadIdx.x%32;
    const std::size_t ordinal=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(ordinal>=p)return;
    const auto begin=offsets[ordinal],d=offsets[ordinal+1]-begin;
    if(d<=kCycleCooperativeDegree)return;
    const auto* a=canonical+begin;auto* fill=fills+begin;auto* flags=fill_flags+begin;
    auto* sums=prefix+begin;
    for(std::uint32_t i=lane;i<d;i+=32)flags[i]=0;
    __syncwarp();
    const double D=total_degree[ordinal];
    unsigned long long state=pivots[ordinal].seed;
    int range=(!(D>0)||!isfinite(D))?2:0;
    for(std::uint32_t i=lane;i<d;i+=32) {
        if(!(a[i].value>=0)||!isfinite(a[i].value))range=2;
        else if(!(a[i].value>0))range=max(range,1);
    }
    if(!lane && !(a[0].value/a[d-1].value>0))range=max(range,1);
    range=cycle_warp_range(range);
    if(range==2) {if(!lane)atomicOr(status,2u);return;}
    if(range==1) {
        if(!lane) {
            const auto emitted=cycle_gks_fallback_row(a,d,D,state,sums,fill,flags,status);
            if(counters)counters[ordinal].emitted_edges=emitted;
        }
        return;
    }
    const auto cut=cycle_trace_cut_warp(a,d,sums,workspace+begin);
    if(cut==d) {
        if(!lane) {
            const auto emitted=cycle_gks_fallback_row(a,d,D,state,sums,fill,flags,status);
            if(counters)counters[ordinal].emitted_edges=emitted;
        }
        return;
    }
    const auto h=d-cut;
    range=0;
    if(!lane)range=max(cycle_pool_range(cycle_edge_weight(a,D,h,cut,cut+1)),
                      cycle_pool_range(cycle_edge_weight(a,D,h,d-2,d-1)));
    for(std::uint32_t i=lane;i<cut;i+=32) {
        const double ai=a[i].value/a[d-1].value,m=d-i-1;
        const double probability=m*ai/(sums[i+1]+m*ai);
        if(!(probability>=0 && probability<=.5)||!isfinite(probability))range=2;
        else if(!(probability>0))range=max(range,1);
        range=max(range,cycle_pool_range(cycle_parent_weight(a,d,D,sums,i,i+1)));
        range=max(range,cycle_pool_range(cycle_parent_weight(a,d,D,sums,i,d-1)));
    }
    range=cycle_warp_range(range);
    if(range==2) {if(!lane)atomicOr(status,2u);return;}
    if(range==1) {
        if(!lane) {
            const auto emitted=cycle_gks_fallback_row(a,d,D,state,sums,fill,flags,status);
            if(counters)counters[ordinal].emitted_edges=emitted;
        }
        return;
    }
    for(std::uint32_t k=lane;k<h;k+=32)fill[cut+k].a=cut+k;
    __syncwarp();
    if(!lane)for(std::uint32_t k=h;k>1;--k) {
        const auto j=cycle_uniform_index(state,k),x=fill[cut+k-1].a;
        fill[cut+k-1].a=fill[cut+j].a;fill[cut+j].a=x;
    }
    __syncwarp();
    const auto first=fill[cut].a;
    for(std::uint32_t base=0;base<h;base+=32) {
        const auto k=base+lane;
        std::uint32_t i=0,j=0;
        if(k<h) {i=fill[cut+k].a;j=k+1<h?fill[cut+k+1].a:first;}
        __syncwarp(); // all endpoint indices precede overwriting permutation slots
        if(k<h) {
            const double value=cycle_edge_weight(a,D,h,i,j);
            if(!cycle_pool_edge(value))atomicOr(status,2u);
            else {fill[cut+k]={min(a[i].b,a[j].b),max(a[i].b,a[j].b),value};flags[cut+k]=1;}
        }
        __syncwarp();
    }
    const auto parent_state=__shfl_sync(0xffffffffu,state,0);
    for(std::uint32_t i=lane;i<cut;i+=32) {
        auto draw_state=parent_state+0x9E3779B97F4A7C15ULL*i;
        const double ai=a[i].value/a[d-1].value;
        const double mass=sums[i+1]+double(d-i-1)*ai;
        double target=next_unit(draw_state)*mass;
        if(target>=mass)target=nextafter(mass,0.);
        // R_i(j)=sum_{l>=j}a_l+(d-j)*a_i is decreasing. Its adjacent
        // interval has width a_i+a_j, hence probability q_ij=(a_i+a_j)/mass.
        std::uint32_t lo=i+1,hi=d;
        while(lo+1<hi) {
            const auto mid=lo+(hi-lo)/2;
            if(target<sums[mid]+double(d-mid)*ai)lo=mid;else hi=mid;
        }
        const auto j=lo;
        const double value=cycle_parent_weight(a,d,D,sums,i,j);
        if(!cycle_pool_edge(value))atomicOr(status,2u);
        else {fill[i]={min(a[i].b,a[j].b),max(a[i].b,a[j].b),value};flags[i]=1;}
    }
    if(!lane && counters)counters[ordinal].emitted_edges=d;
}
