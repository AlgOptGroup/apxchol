#pragma once
#include "k2_reference.h"
#include <numeric>
#include <string_view>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace apxchol::heavy_research {
enum class mode { gks, k2, k2_cycle, heavy_gks, heavy_k2, heavy_kcore, heavy_cycle_order };
inline mode parse_mode(std::string_view name) {
    if(name=="gks") return mode::gks;
    if(name=="k2") return mode::k2;
    if(name=="k2_cycle") return mode::k2_cycle;
    if(name=="heavy_gks") return mode::heavy_gks;
    if(name=="heavy_k2") return mode::heavy_k2;
    if(name=="heavy_kcore") return mode::heavy_kcore;
    if(name=="heavy_cycle_order") return mode::heavy_cycle_order;
    throw std::invalid_argument("unknown sampler arm");
}
struct alignas(64) counters {
    std::uint64_t calls=0, vertices=0, cycles=0, core_vertices=0, receiver_passes=0,
                  fractional_items=0, numerical_fallbacks=0;
};
using moments=std::array<double,8>;
inline double checked_nonnegative(double value,double scale) {
    const double tolerance=512*std::numeric_limits<double>::epsilon()*std::max(1.,scale);
    if(!std::isfinite(value) || value < -tolerance)throw std::domain_error("invalid variance moment");
    return std::max(0.,value);
}

// Scores the independent-parent law. The same cutoff is deliberately frozen
// for all coordination arms, so their comparison isolates dependence, not a
// retrospective change in topology selection. O(d) work/storage after sorting.
inline std::size_t choose_cut(std::span<const weighted_neighbor> neighbors) {
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
inline std::size_t uniform_index(random_stream& rng,std::size_t bound) {
    const auto b=static_cast<std::uint64_t>(bound),threshold=-b%b;
    std::uint64_t value;do {value=rng.next();} while(value<threshold);
    return value%b;
}
struct sampler {
    mode arm;
    std::vector<counters>* statistics=nullptr;
    std::size_t max_clique_edges(node_index d) const {
        if(d<2)return 0;
        if(d==2 || arm==mode::gks || arm==mode::k2 || (arm==mode::k2_cycle && d>6))return d-1;
        return d;
    }
    void sample_clique(std::span<weighted_neighbor> n,double D,std::uint64_t seed,edge_emitter out) const {
        counters unused;auto* st=&unused;
        if(statistics) {
            std::size_t thread=0;
#ifdef _OPENMP
            thread=omp_get_thread_num();
#endif
            st=&statistics->at(thread);
        }
        ++st->calls;st->vertices+=n.size();
        if(n.size()<2 || !(D>0))return;
        if(arm==mode::gks) {tree_elimination{}.sample_clique(n,D,seed,out);return;}
        if(arm==mode::k2_cycle || arm==mode::k2) {
            if(arm==mode::k2_cycle && research_k2::sample_variance_gated_cycle(n,D,seed,out)) {++st->cycles;st->core_vertices+=n.size();return;}
            if(research_k2::sample_coordinated_gks_clique(n,D,seed,out))return;
            tree_elimination{}.sample_clique(n,D,seed,out);return;
        }
        if(n.size()==2) {tree_elimination{}.sample_clique(n,D,seed,out);return;}
        research_k2::canonical_sort_neighbors(n);
        std::size_t cut;
        try {cut=choose_cut(n);} catch(const std::domain_error&) {
            ++st->numerical_fallbacks;tree_elimination{}.sample_clique(n,D,seed,out);return;
        }
        const std::size_t d=n.size(),h=d-cut;
        ++st->cycles;st->core_vertices+=h;
        static thread_local std::vector<double> prefix,mass,remaining;
        static thread_local std::vector<std::size_t> cycle,partner,receivers;
        prefix.resize(d+1);prefix[0]=0;
        for(std::size_t i=0;i<d;++i)prefix[i+1]=prefix[i]+n[i].weight;
        mass.resize(cut);remaining.resize(cut);partner.assign(cut,d);
        for(std::size_t i=0;i<cut;++i) {
            remaining[i]=prefix[d]-prefix[i+1];mass[i]=n[i].weight*remaining[i]/D;
        }
        random_stream rng{seed};cycle.resize(h);std::iota(cycle.begin(),cycle.end(),cut);
        for(std::size_t k=h;k>1;--k)std::swap(cycle[k-1],cycle[uniform_index(rng,k)]);
        for(std::size_t k=0;k<h;++k) {
            const auto i=cycle[k],j=cycle[(k+1)%h];
            out(n[i].vertex,n[j].vertex,n[i].weight*n[j].weight/D*(h-1)*.5);
        }
        receivers.clear();
        if(arm==mode::heavy_cycle_order) {
            // Anchor at the heaviest vertex, then traverse the sampled cycle.
            // Conditional probabilities below stay exact for every cycle.
            auto start=std::find(cycle.begin(),cycle.end(),d-1)-cycle.begin();
            for(std::size_t k=0;k<h;++k)receivers.push_back(cycle[(start+k)%h]);
        } else if(arm!=mode::heavy_gks) {
            const auto count=arm==mode::heavy_k2?std::min<std::size_t>(2,h):h;
            for(std::size_t k=0;k<count;++k)receivers.push_back(d-1-k);
        }
        for(auto receiver:receivers) {
            bool open=false;double cumulative=0;const double phase=rng.next_unit();
            ++st->receiver_passes;
            for(std::size_t i=0;i<cut;++i) {
                if(partner[i]!=d)continue;
                open=true;
                const double p=std::clamp(n[receiver].weight/remaining[i],0.,1.);
                if(p>=1 || phase+std::ceil(cumulative-phase)<cumulative+p)partner[i]=receiver;
                else remaining[i]-=n[receiver].weight;
                cumulative+=p;++st->fractional_items;
            }
            if(!open)break;
        }
        for(std::size_t i=0;i<cut;++i) {
            if(partner[i]==d) {
                // Every arm processes either a heaviest suffix or the entire
                // core, so the remaining eligible range is contiguous. Keep
                // ordinary GKS's logarithmic inverse-CDF lookup.
                const auto last=d-1-receivers.size();
                if(last<=i) {
                    // Only possible after floating-point endpoint rounding;
                    // preserve a structural edge to a genuine heavier vertex.
                    partner[i]=receivers.back();++st->numerical_fallbacks;
                } else {
                    const double mass_left=prefix[last+1]-prefix[i+1];
                    const double target=prefix[i+1]+rng.next_unit()*mass_left;
                    const auto it=std::upper_bound(prefix.begin()+i+2,prefix.begin()+last+2,target);
                    partner[i]=std::min(last,static_cast<std::size_t>(it-prefix.begin()-1));
                }
            }
            out(n[i].vertex,n[partner[i]].vertex,mass[i]);
        }
    }
};
static_assert(eliminator<sampler>);
} // namespace apxchol::heavy_research
