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
__device__ std::uint32_t cycle_heavy_cut(const work_record* a, std::uint32_t d,
                                        double* tail) {
    if (d == 3) return 0;
    const double scale = a[d - 1].value;
    double D = 0, other = 0;
    for (std::uint32_t i = 0; i < d; ++i) {
        const double x = a[i].value / scale;
        if (!(x > 0) || !isfinite(x)) return d;
        D += x; if (i + 1 < d) other += x;
    }
    double sum[8] = {};
    for (std::uint32_t i = d; i-- > 0;) {
        const double x = a[i].value / scale;
        const double q = x * (i + 1 == d ? other : D - x) / D;
        if (!(q > 0)) return d;
        const double x2 = x*x, q2 = q*q;
        const double atom[8] = {x,x/q,x/q2,x2/q2,x2/q,x2,x2*x/q2,x2*x2/q2};
        for (int k = 0; k < 8; ++k) {
            sum[k] = sum[k] + atom[k];
            if (!isfinite(sum[k])) return d;
            tail[std::size_t(i)*8+k] = sum[k];
        }
    }
    double parents=0, best=kCycleInfinity;
    std::uint32_t cut=0;
    for (std::uint32_t i=0; i+2<d; ++i) {
        const auto h=d-i; const double* m=tail+std::size_t(i)*8;
        double cycle=0;
        if (h>3) {
            const double N=h-1, first=m[5]*m[3]-m[7];
            const double second=m[0]*m[0]*m[3]-2*m[0]*m[6]+m[7];
            const double diagonal=N*(N-2)/(2*(N-1))*(first-second/N);
            const double off=(h-3)*.5*(m[4]*m[4]-m[7]);
            const double magnitude=double(h)*h*(fabs(m[5]*m[3])+fabs(m[7])+fabs(m[0]*m[0]*m[3])+fabs(2*m[0]*m[6]))+h*(fabs(m[4]*m[4])+fabs(m[7]));
            double x,y;
            if (!cycle_nonnegative(diagonal,magnitude,x) || !cycle_nonnegative(off,magnitude,y)) return d;
            cycle=x+y;
        }
        const double value=(parents+cycle)/(D*D);
        if (!isfinite(value)) return d;
        if (value<best) { best=value;cut=i; }
        const double* b=tail+std::size_t(i+1)*8;
        const double a_i=a[i].value/scale, s_i=a_i*(D-a_i)/D;
        const double diagonal=a_i*a_i*(b[0]*b[2]-b[3]);
        const double off=2*a_i*a_i/s_i*(b[0]*b[1]-b[4]);
        const double magnitude=a_i*a_i*(fabs(b[0]*b[2])+fabs(b[3]))+2*a_i*a_i/s_i*(fabs(b[0]*b[1])+fabs(b[4]));
        double x,y;
        if (!cycle_nonnegative(diagonal,magnitude,x) || !cycle_nonnegative(off,magnitude,y)) return d;
        parents += x+y;
    }
    return cut;
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
    const bool trace=sampler==clique_sampler::trace_cycle;
    const auto cut=trace?cycle_trace_cut(a,d,sums,workspace+begin)
                        :cycle_heavy_cut(a,d,workspace+std::size_t(begin)*8);
    if(cut==d) {
        const auto emitted=cycle_gks_fallback_row(a,d,D,state,sums,fill,flags,status);
        if(counters)counters[ordinal].emitted_edges=emitted;
        return;
    }
    const auto h=d-cut;
    range=max(cycle_pool_range(cycle_edge_weight(a,D,h,cut,cut+1)),
              cycle_pool_range(cycle_edge_weight(a,D,h,d-2,d-1)));
    if(!trace) {
        sums[0]=a[0].value;
        for(std::uint32_t i=1;i<d;++i)sums[i]=sums[i-1]+a[i].value;
    }
    // The final output slots double as temporary parent/permutation state.
    // Light and core ranges are disjoint; a cycle slot is overwritten only
    // after both endpoint indices are read, and its first endpoint is saved.
    for(std::uint32_t i=0;i<cut;++i) {
        fill[i].a=d;
        if(trace) {
            const double ai=a[i].value/a[d-1].value,m=d-i-1;
            const double probability=m*ai/(sums[i+1]+m*ai);
            if(!(probability>=0 && probability<=.5) || !isfinite(probability)) {atomicOr(status,2u);return;}
            if (!(probability>0)) range=max(range,1);
            range=max(range,cycle_pool_range(cycle_parent_weight(a,d,D,sums,i,i+1)));
            range=max(range,cycle_pool_range(cycle_parent_weight(a,d,D,sums,i,d-1)));
        } else {
            fill[i].value=sums[d-1]-sums[i];
            range=max(range,cycle_pool_range(a[i].value*fill[i].value/D));
        }
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
    if(!trace)for(std::uint32_t k=0;k<2;++k) {
        const auto receiver=d-1-k;bool open=false;double cumulative=0;
        const double phase=next_unit(state);
        for(std::uint32_t i=0;i<cut;++i)if(fill[i].a==d) {
            open=true;const double prob=fmin(1.,fmax(0.,a[receiver].value/fill[i].value));
            if(prob>=1 || phase+ceil(cumulative-phase)<cumulative+prob)fill[i].a=receiver;
            else fill[i].value-=a[receiver].value;
            cumulative+=prob;
        }
        if(!open)break;
    }
    for(std::uint32_t i=0;i<cut;++i) {
        std::uint32_t j;double value;
        if(trace) {
            const double ai=a[i].value/a[d-1].value,m=d-i-1;
            if(next_unit(state)<m*ai/(sums[i+1]+m*ai))j=i+1+cycle_uniform_index(state,d-i-1);
            else {
                const double sum=sums[i+1];double target=next_unit(state)*sum;
                if(target>=sum)target=nextafter(sum,0.);
                std::uint32_t lo=i+1,hi=d;
                while(lo+1<hi){const auto mid=lo+(hi-lo)/2;if(target<sums[mid])lo=mid;else hi=mid;}
                j=lo;
            }
            value=cycle_parent_weight(a,d,D,sums,i,j);
        } else {
            j=fill[i].a;
            if(j==d) {
                const auto last=d-3;
                if(last<=i)j=d-2;
                else {
                    const double target=sums[i]+next_unit(state)*(sums[last]-sums[i]);
                    std::uint32_t lo=i+1,hi=last+1;
                    while(lo<hi){const auto mid=lo+(hi-lo)/2;if(sums[mid]<=target)lo=mid+1;else hi=mid;}
                    j=lo<last?lo:last;
                }
            }
            value=a[i].value*(sums[d-1]-sums[i])/D;
        }
        if(!cycle_pool_edge(value)){atomicOr(status,2u);return;}
        fill[i]={min(a[i].b,a[j].b),max(a[i].b,a[j].b),value};flags[i]=1;
    }
    if(counters)counters[ordinal].emitted_edges=d;
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
