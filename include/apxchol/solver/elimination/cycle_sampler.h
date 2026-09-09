#pragma once
// Builtin cycle-core samplers. Included after the public eliminator types.
#include "apxchol/solver/elimination/elimination.h"
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
namespace apxchol::detail {
inline void require_pool_edge(double value) {
    if (!(value >= std::numeric_limits<pool_value_t>::min()) ||
        !(value <= std::numeric_limits<pool_value_t>::max()) || !std::isfinite(value))
        throw std::domain_error("cycle sampler: edge outside normal residual-pool range");
}
inline std::size_t uniform_index(random_stream& rng, std::size_t bound) {
    const auto b = static_cast<std::uint64_t>(bound), threshold = -b % b;
    std::uint64_t value;
    do { value = rng.next(); } while (value < threshold);
    return value % b;
}
struct trace_cycle_plan {
    enum slot : std::size_t { mass_sum=0, square_sum=1 };
    std::vector<double> a;
    std::vector<std::array<double,2>> suffix;
    std::size_t cut=0;
    double scale=0,total=0,best_score=0;
    void prepare(std::span<const weighted_neighbor> n) {
        const auto d=n.size();
        if(d<3)throw std::invalid_argument("relative trace plan requires degree>=3");
        scale=n.back().weight;
        if(!(scale>0) || !std::isfinite(scale))
            throw std::domain_error("relative trace: invalid scale");
        a.resize(d);suffix.resize(d+1);suffix[d]={0,0};
        for(std::size_t i=d;i-->0;) {
            if(!(n[i].weight>0) || !std::isfinite(n[i].weight) ||
               (i && n[i].weight<n[i-1].weight))
                throw std::domain_error("relative trace: invalid sorted weights");
            a[i]=n[i].weight/scale;
            if(!(a[i]>0) || !std::isfinite(a[i]))
                throw std::domain_error("relative trace: normalized weight underflow");
            suffix[i]={a[i]+suffix[i+1][mass_sum],
                       a[i]*a[i]+suffix[i+1][square_sum]};
            if(!std::isfinite(suffix[i][mass_sum]) || !std::isfinite(suffix[i][square_sum]))
                throw std::domain_error("relative trace: suffix overflow");
        }
        total=suffix[0][mass_sum];
        if(!(total>0) || !std::isfinite(total*total))
            throw std::domain_error("relative trace: total overflow");
        double parents=0;
        best_score=std::numeric_limits<double>::infinity();cut=0;
        for(std::size_t i=0;i+2<d;++i) {
            const double score=(parents+cycle_numerator(i))/(total*total);
            if(!(score>=0) || !std::isfinite(score))
                throw std::domain_error("relative trace: invalid score");
            // Ascending light-prefix order: an exact tie keeps the larger core.
            if(score<best_score){best_score=score;cut=i;}
            if(i+3<d)parents+=parent_numerator(i);
        }
    }
    std::size_t receiver_count(std::size_t i) const {return a.size()-i-1;}
    double mass(std::size_t i) const {
        return suffix[i+1][mass_sum]+double(receiver_count(i))*a[i];
    }
    double probability(std::size_t i,std::size_t j) const {
        if(!(i<j && j<a.size()))throw std::out_of_range("relative trace parent index");
        return (a[i]+a[j])/mass(i);
    }
    double uniform_probability(std::size_t i) const {
        // Sorted weights imply this lies in (0,1/2]; compute the small
        // component directly rather than subtracting a probability from one.
        return (double(receiver_count(i))*a[i])/mass(i);
    }
    double parent_numerator(std::size_t i) const {
        const double m=double(receiver_count(i));
        return (m-1)*a[i]*(2*suffix[i+1][mass_sum]+m*a[i]);
    }
    double cycle_numerator(std::size_t light) const {
        const auto h=a.size()-light;
        if(h<3)throw std::out_of_range("relative trace core size");
        return h==3?0.:double(h-3)*double(h-1)*.5*suffix[light][square_sum];
    }
    double parent_weight(std::span<const weighted_neighbor> n,double pivot,
                         std::size_t i,std::size_t j) const {
        // Algebraically c_ij/q_ij, with actual pivot (also valid for D>A).
        // Sorted a_i<=a_j makes the denominator lie in [1,2].
        return (n[i].weight*(scale/pivot))*(mass(i)/(1+a[i]/a[j]));
    }
    std::size_t weighted_parent(std::size_t i,double u) const {
        if(!(u>=0 && u<1) || !std::isfinite(u))
            throw std::domain_error("relative trace: invalid uniform");
        const double sum=suffix[i+1][mass_sum];
        double target=u*sum;
        // Multiplication can round to the half-open endpoint.
        if(target>=sum)target=std::nextafter(sum,0.);
        std::size_t lo=i+1,hi=a.size();
        while(lo+1<hi) {
            const auto mid=lo+(hi-lo)/2;
            if(target<suffix[mid][mass_sum])lo=mid;else hi=mid;
        }
        return lo;
    }
};
inline double cycle_weight(std::span<const weighted_neighbor> n,double pivot,
                           std::size_t h,std::size_t i,std::size_t j) {
    return n[i].weight*n[j].weight/pivot*(h-1)*.5;
}
using moments=std::array<double,8>;
inline double checked_nonnegative(double value,double scale) {
    const double tolerance=512*std::numeric_limits<double>::epsilon()*std::max(1.,scale);
    if(!std::isfinite(value) || value < -tolerance)throw std::domain_error("invalid variance moment");
    return std::max(0.,value);
}

// Scores the independent-parent law. The same cutoff is deliberately frozen
// for all coordination arms, so their comparison isolates dependence, not a
// retrospective change in topology selection. O(d) work/storage after sorting.
inline std::size_t heavy_core_cut(std::span<const weighted_neighbor> neighbors) {
    const auto d=neighbors.size();
    if(d==3) return 0;
    static thread_local std::vector<double> a,s;
    static thread_local std::vector<moments> tail;
    a.resize(d);s.resize(d);tail.resize(d+1);
    const double scale=neighbors.back().weight;
    double D=0,other=0;
    for(std::size_t i=0;i<d;++i) {
        a[i]=neighbors[i].weight/scale;
        if(!(a[i]>0) || !std::isfinite(a[i])) throw std::domain_error("invalid weights");
        D+=a[i];if(i+1<d)other+=a[i];
    }
    tail[d]={};
    for(std::size_t i=d;i-->0;) {
        s[i]=a[i]*(i+1==d?other:D-a[i])/D;
        if(!(s[i]>0)) throw std::domain_error("invalid normalization");
        const double x=a[i],q=s[i],x2=x*x,q2=q*q;
        const moments atom={x,x/q,x/q2,x2/q2,x2/q,x2,x2*x/q2,x2*x2/q2};
        for(std::size_t k=0;k<8;++k) {
            tail[i][k]=tail[i+1][k]+atom[k];
            if(!std::isfinite(tail[i][k])) throw std::domain_error("moment overflow");
        }
    }
    double parents=0,best=std::numeric_limits<double>::infinity();
    std::size_t cut=0;
    for(std::size_t i=0;i+2<d;++i) {
        const auto h=d-i;const auto& m=tail[i];
        double cycle=0;
        if(h>3) {
            const double N=h-1,first=m[5]*m[3]-m[7];
            const double second=m[0]*m[0]*m[3]-2*m[0]*m[6]+m[7];
            const double diagonal=N*(N-2)/(2*(N-1))*(first-second/N);
            const double off=(h-3)*.5*(m[4]*m[4]-m[7]);
            const double magnitude=h*h*(std::abs(m[5]*m[3])+std::abs(m[7])+std::abs(m[0]*m[0]*m[3])+std::abs(2*m[0]*m[6]))+h*(std::abs(m[4]*m[4])+std::abs(m[7]));
            cycle=checked_nonnegative(diagonal,magnitude)+checked_nonnegative(off,magnitude);
        }
        const double value=(parents+cycle)/(D*D);
        if(!std::isfinite(value)) throw std::domain_error("score overflow");
        if(value<best) {best=value;cut=i;}
        const auto& b=tail[i+1];
        const double diagonal=a[i]*a[i]*(b[0]*b[2]-b[3]);
        const double off=2*a[i]*a[i]/s[i]*(b[0]*b[1]-b[4]);
        const double magnitude=a[i]*a[i]*(std::abs(b[0]*b[2])+std::abs(b[3]))+2*a[i]*a[i]/s[i]*(std::abs(b[0]*b[1])+std::abs(b[4]));
        parents+=checked_nonnegative(diagonal,magnitude)+checked_nonnegative(off,magnitude);
    }
    return cut;
}
inline void sample_trace_cycle(std::span<weighted_neighbor> n, double pivot,
                               std::uint64_t seed, edge_emitter out) {
    static thread_local trace_cycle_plan workspace;
    static thread_local std::vector<std::size_t> cycle;
    static thread_local std::vector<deferred_edge> staged;
    staged.clear(); staged.reserve(n.size());
    workspace.prepare(n);
    const auto cut=workspace.cut,h=n.size()-cut,d=n.size();
    // Input-only all-outcome checks before any RNG or publication. Positive
    // operations make cycle weights and each source's HT weight monotone in
    // the receiver weight; endpoint checks cover every possible edge.
    require_pool_edge(cycle_weight(n,pivot,h,cut,cut+1));
    require_pool_edge(cycle_weight(n,pivot,h,d-2,d-1));
    for(std::size_t i=0;i<cut;++i) {
        const double probability=workspace.uniform_probability(i);
        if(!(probability>0 && probability<=.5) || !std::isfinite(probability))
            throw std::domain_error("relative trace: unrepresentable mixture");
        require_pool_edge(workspace.parent_weight(n,pivot,i,i+1));
        require_pool_edge(workspace.parent_weight(n,pivot,i,d-1));
    }
    random_stream rng{seed};cycle.resize(h);std::iota(cycle.begin(),cycle.end(),cut);
    for(std::size_t k=h;k>1;--k)
        std::swap(cycle[k-1],cycle[uniform_index(rng,k)]);
    for(std::size_t k=0;k<h;++k) {
        const auto i=cycle[k],j=cycle[(k+1)%h];
        const double weight=cycle_weight(n,pivot,h,i,j);
        require_pool_edge(weight);
        staged.push_back({n[i].vertex,n[j].vertex,weight});
    }
    for(std::size_t i=0;i<cut;++i) {
        // One component coin, then one fresh component draw. Uniform_index
        // can reject integer residues, as in the unchanged cycle shuffle.
        std::size_t j;
        if(rng.next_unit()<workspace.uniform_probability(i)) {
            j=i+1+uniform_index(rng,workspace.receiver_count(i));
        } else {
            j=workspace.weighted_parent(i,rng.next_unit());
        }
        const double weight=workspace.parent_weight(n,pivot,i,j);
        require_pool_edge(weight);
        staged.push_back({n[i].vertex,n[j].vertex,weight});
    }
    // Every sampled edge was checked while staging; publish only the complete
    // validated batch. Invalid outcomes never trigger fallback or redrawing.
    out(std::span<const deferred_edge>(staged));
}
// The heavy-core reference coordinates GKS parents at the two largest
// receivers using systematic rounding. Their individual marginals remain GKS.
inline void sample_heavy_core_k2(std::span<weighted_neighbor> n, double pivot,
                                 std::uint64_t seed, edge_emitter out) {
    const auto d = n.size();
    std::size_t cut;
    try { cut = heavy_core_cut(n); }
    catch (const std::domain_error&) {
        tree_elimination{}.sample_clique(n, pivot, seed, out);
        return;
    }
    const auto h = d - cut;
    static thread_local std::vector<double> prefix, mass, remaining;
    static thread_local std::vector<std::size_t> cycle, partner;
    static thread_local std::vector<deferred_edge> staged;
    staged.clear(); staged.reserve(d);
    prefix.resize(d + 1); prefix[0] = 0;
    for (std::size_t i = 0; i < d; ++i) prefix[i+1] = prefix[i] + n[i].weight;
    mass.resize(cut); remaining.resize(cut); partner.assign(cut, d);
    for (std::size_t i = 0; i < cut; ++i) {
        remaining[i] = prefix[d] - prefix[i+1];
        mass[i] = n[i].weight * remaining[i] / pivot;
        require_pool_edge(mass[i]);
    }
    require_pool_edge(cycle_weight(n, pivot, h, cut, cut+1));
    require_pool_edge(cycle_weight(n, pivot, h, d-2, d-1));
    random_stream rng{seed}; cycle.resize(h); std::iota(cycle.begin(), cycle.end(), cut);
    for (std::size_t k = h; k > 1; --k)
        std::swap(cycle[k-1], cycle[uniform_index(rng, k)]);
    for (std::size_t k = 0; k < h; ++k) {
        const auto i = cycle[k], j = cycle[(k+1)%h];
        const auto weight = cycle_weight(n, pivot, h, i, j);
        require_pool_edge(weight);
        staged.push_back({n[i].vertex, n[j].vertex, weight});
    }
    for (std::size_t k = 0; k < 2; ++k) {
        const auto receiver = d-1-k;
        bool open = false; double cumulative = 0;
        const double phase = rng.next_unit();
        for (std::size_t i = 0; i < cut; ++i) {
            if (partner[i] != d) continue;
            open = true;
            const double p = std::clamp(n[receiver].weight / remaining[i], 0., 1.);
            if (p >= 1 || phase + std::ceil(cumulative-phase) < cumulative+p)
                partner[i] = receiver;
            else remaining[i] -= n[receiver].weight;
            cumulative += p;
        }
        if (!open) break;
    }
    for (std::size_t i = 0; i < cut; ++i) {
        if (partner[i] == d) {
            const auto last = d-3;
            const double mass_left = prefix[last+1] - prefix[i+1];
            const double target = prefix[i+1] + rng.next_unit() * mass_left;
            const auto it = std::upper_bound(prefix.begin()+i+2, prefix.begin()+last+2, target);
            partner[i] = std::min(last, static_cast<std::size_t>(it-prefix.begin()-1));
        }
        staged.push_back({n[i].vertex, n[partner[i]].vertex, mass[i]});
    }
    out(std::span<const deferred_edge>(staged));
}
inline void sample_cycle_clique(std::span<weighted_neighbor> n, double pivot,
                               std::uint64_t seed, edge_emitter out, clique_sampler sampler) {
    if (!(pivot > 0) || !std::isfinite(pivot))
        throw std::domain_error("cycle sampler: invalid pivot");
    for (const auto& edge : n)
        if (!(edge.weight > 0) || !std::isfinite(edge.weight))
            throw std::domain_error("cycle sampler: invalid neighbor");
    if (sampler == clique_sampler::trace_cycle) sample_trace_cycle(n, pivot, seed, out);
    else if (sampler == clique_sampler::heavy_core_k2) sample_heavy_core_k2(n, pivot, seed, out);
    else throw std::invalid_argument("unknown clique sampler");
}
} // namespace apxchol::detail
