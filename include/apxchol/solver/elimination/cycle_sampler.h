#pragma once
// Builtin cycle-core samplers. Included after the public eliminator types.
#include "apxchol/solver/elimination/elimination.h"
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
namespace apxchol::detail {
// Input-only cycle eligibility. GKS remains the established law when pool
// rounding or normalized probabilities cannot retain a positive cycle update.
inline bool cycle_pool_eligible(double value) {
    if (!(value >= 0) || !(value <= std::numeric_limits<pool_value_t>::max()))
        throw std::domain_error("cycle sampler: invalid or overflowing residual-pool edge");
    return static_cast<pool_value_t>(value) > 0;
}
inline void require_pool_edge(double value) {
    if (!cycle_pool_eligible(value))
        throw std::domain_error("cycle sampler: edge outside positive residual-pool range");
}
inline bool cycle_inputs_eligible(std::span<const weighted_neighbor> n, double pivot) {
    if (!(pivot > 0) || !std::isfinite(pivot))
        throw std::domain_error("cycle sampler: invalid pivot");
    bool eligible = true;
    for (const auto& edge : n) {
        if (!(edge.weight >= 0) || !std::isfinite(edge.weight))
            throw std::domain_error("cycle sampler: invalid neighbor weight");
        eligible &= edge.weight > 0;
    }
    if (eligible) eligible = n.front().weight / n.back().weight > 0;
    return eligible;
}
inline void sample_cycle_gks_fallback(std::span<weighted_neighbor> n, double pivot,
                                       std::uint64_t seed, edge_emitter out) {
    // GKS parent weights depend only on the input star, not the chosen parent.
    // Allow its established zero rounding, but never hide an overflowing update.
    double total=0,prefix=0;
    for(const auto& edge:n) total+=edge.weight;
    if(!std::isfinite(total))
        throw std::domain_error("cycle sampler: overflowing GKS fallback sum");
    for(std::size_t i=0;i+1<n.size();++i) {
        prefix+=n[i].weight;
        const double suffix=total-prefix;
        if(suffix>0) (void)cycle_pool_eligible(n[i].weight*suffix/pivot);
    }
    tree_elimination{}.sample_clique(n, pivot, seed, out);
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
inline void sample_trace_cycle(std::span<weighted_neighbor> n, double pivot,
                               std::uint64_t seed, edge_emitter out,
                               std::size_t exact_core_max_h = 0,
                               std::size_t double_cycle_min_h = 0) {
    static thread_local trace_cycle_plan workspace;
    static thread_local std::vector<std::size_t> cycle;
    static thread_local std::vector<deferred_edge> staged;
    staged.clear(); staged.reserve(2 * n.size() + exact_core_max_h * exact_core_max_h);
    // Only plan construction may choose an input-only numerical fallback.
    // Once random sampling starts, errors remain errors; never redraw.
    try { workspace.prepare(n); }
    catch (const std::domain_error&) {
        sample_cycle_gks_fallback(n, pivot, seed, out);
        return;
    }
    const auto cut=workspace.cut,h=n.size()-cut,d=n.size();
    // Input-only all-outcome checks before any RNG or publication. Positive
    // operations make cycle weights and each source's HT weight monotone in
    // the receiver weight; endpoint checks cover every possible edge.
    bool eligible = cycle_pool_eligible(cycle_weight(n,pivot,h,cut,cut+1));
    eligible &= cycle_pool_eligible(cycle_weight(n,pivot,h,d-2,d-1));
    for(std::size_t i=0;i<cut;++i) {
        const double probability=workspace.uniform_probability(i);
        if(!(probability>=0 && probability<=.5) || !std::isfinite(probability))
            throw std::domain_error("relative trace: invalid mixture");
        eligible &= probability > 0;
        eligible &= cycle_pool_eligible(workspace.parent_weight(n,pivot,i,i+1));
        eligible &= cycle_pool_eligible(workspace.parent_weight(n,pivot,i,d-1));
    }
    if (!eligible) {
        sample_cycle_gks_fallback(n, pivot, seed, out);
        return;
    }
    random_stream rng{seed};
    if (exact_core_max_h > 0 && h <= exact_core_max_h) {
        // Exact core: every core pair with its clique weight. Zero core
        // variance; light attachments below are unchanged.
        for (std::size_t x = cut; x < d; ++x)
            for (std::size_t y = x + 1; y < d; ++y) {
                const double weight = n[x].weight * n[y].weight / pivot;
                require_pool_edge(weight);
                staged.push_back({n[x].vertex, n[y].vertex, weight});
            }
    } else {
        // One uniform Hamiltonian cycle, or two independent ones with halved
        // weights on a large core (random 4-regular core). Both unbiased.
        const int cycles = (double_cycle_min_h > 0 && h >= double_cycle_min_h) ? 2 : 1;
        for (int c = 0; c < cycles; ++c) {
            cycle.resize(h);std::iota(cycle.begin(),cycle.end(),cut);
            for(std::size_t k=h;k>1;--k)
                std::swap(cycle[k-1],cycle[uniform_index(rng,k)]);
            for(std::size_t k=0;k<h;++k) {
                const auto i=cycle[k],j=cycle[(k+1)%h];
                const double weight=cycle_weight(n,pivot,h,i,j)/cycles;
                require_pool_edge(weight);
                staged.push_back({n[i].vertex,n[j].vertex,weight});
            }
        }
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
inline void sample_cycle_clique(std::span<weighted_neighbor> n, double pivot,
                               std::uint64_t seed, edge_emitter out, clique_sampler sampler,
                               std::size_t exact_core_max_h,
                               std::size_t double_cycle_min_h) {
    if (sampler != clique_sampler::trace_cycle)
        throw std::invalid_argument("unknown clique sampler");
    if (!cycle_inputs_eligible(n, pivot)) {
        sample_cycle_gks_fallback(n, pivot, seed, out);
        return;
    }
    sample_trace_cycle(n, pivot, seed, out, exact_core_max_h, double_cycle_min_h);
}
} // namespace apxchol::detail
