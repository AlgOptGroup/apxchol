#include "heavy_core.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <set>

using namespace apxchol;
using namespace apxchol::heavy_research;
void require(bool pass,const char* what) {if(!pass)throw std::runtime_error(what);}
std::vector<deferred_edge> sample(const std::vector<double>& w,mode arm,std::uint64_t seed,bool reverse=false) {
    std::vector<weighted_neighbor> n;for(std::size_t i=0;i<w.size();++i)n.push_back({static_cast<node_index>(i),w[i]});
    if(reverse)std::reverse(n.begin(),n.end());
    std::vector<deferred_edge> edges;sampler{arm}.sample_clique(n,std::accumulate(w.begin(),w.end(),0.),seed,edge_emitter(edges));return edges;
}
int main(int argc,char** argv) try {
    const std::vector<mode> modes={mode::gks,mode::k2,mode::k2_cycle,mode::heavy_gks,mode::heavy_k2,mode::heavy_kcore,mode::heavy_cycle_order};
    const std::vector<std::vector<double>> fixtures={{1,1,1},{1,1,1,1},{1,3,5,10,20},{1,1,1,1,8,8},{1,2,4,8,16,32,64,128},{1,1.01,1.02,1.03,1.04,1.05,1.06,1.07}};
    std::size_t outcomes=0,means=0;
    constexpr std::size_t N=40000;
    double max_z=0;
    for(const auto& w:fixtures)for(auto arm:modes) {
        const auto d=w.size();std::vector<double> sum(d*d),sq(d*d);
        const auto first=sample(w,arm,42),reordered=sample(w,arm,42,true);
        require(first.size()==reordered.size(),"deterministic size");
        for(std::size_t k=0;k<first.size();++k)require(first[k].u==reordered[k].u&&first[k].v==reordered[k].v&&first[k].w==reordered[k].w,"canonical deterministic output");
        for(std::size_t seed=0;seed<N;++seed) {
            const auto edges=sample(w,arm,seed);++outcomes;
            require(edges.size()>=d-1&&edges.size()<=d,"edge budget");
            std::vector<std::size_t> parent(d);std::iota(parent.begin(),parent.end(),0);
            auto find=[&](std::size_t v){while(parent[v]!=v)v=parent[v];return v;};
            std::set<std::pair<std::size_t,std::size_t>> seen;
            for(const auto& e:edges) {
                require(e.u!=e.v&&e.u<d&&e.v<d&&e.w>0&&std::isfinite(e.w),"valid edge");
                const auto lo=std::min(e.u,e.v),hi=std::max(e.u,e.v);
                require(seen.insert({lo,hi}).second,"distinct edges");
                parent[find(lo)]=find(hi);sum[lo*d+hi]+=e.w;sq[lo*d+hi]+=e.w*e.w;
            }
            for(std::size_t i=1;i<d;++i)require(find(0)==find(i),"connected sample");
        }
        const double D=std::accumulate(w.begin(),w.end(),0.);
        for(std::size_t i=0;i<d;++i)for(std::size_t j=i+1;j<d;++j) {
            const auto k=i*d+j;const double mean=sum[k]/N,want=w[i]*w[j]/D;
            const double variance=std::max(0.,sq[k]/N-mean*mean);
            const double error=std::abs(mean-want),allow=7*std::sqrt(variance/N)+1e-9*std::max(1.,want);
            require(error<=allow,"unbiasedness seven-sigma check");
            if(variance>1e-12)max_z=std::max(max_z,error/std::sqrt(variance/N));
            ++means;
        }
    }
    std::size_t cuts=0;
    if(argc==2) {
        std::ifstream in(argv[1]);std::string line;
        while(std::getline(in,line)) {
            if(line.empty())continue;std::istringstream row(line);std::size_t d,want;row>>d>>want;
            std::vector<weighted_neighbor> n;for(std::size_t i=0;i<d;++i){double w;row>>w;n.push_back({static_cast<node_index>(i),w});}
            require(choose_cut(n)==want,"frozen cutoff reference");++cuts;
        }
        require(cuts==180,"complete frozen cutoff population");
    }
    std::cout<<"{\"outcomes\":"<<outcomes<<",\"edge_means\":"<<means<<",\"max_mean_z\":"<<max_z<<",\"frozen_cutoffs\":"<<cuts<<",\"status\":\"pass\"}\n";
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
