#include "heavy_core.h"
#include "apxchol.h"
#include "mtx_input.h"
#include <fast_matrix_market/app/Eigen.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/resource.h>

int main(int argc,char** argv) try {
    if(argc!=6) throw std::invalid_argument("usage: heavy_core_bench matrix arm threads seed rhs_count");
    const int threads=std::stoi(argv[3]),count=std::stoi(argv[5]);
    const auto seed=std::stoull(argv[4]);
    if(threads<1 || count<1) throw std::invalid_argument("positive threads/RHS count required");
#ifdef _OPENMP
    omp_set_dynamic(0);omp_set_num_threads(threads);
#endif
    Eigen::SparseMatrix<double> A;fast_matrix_market::matrix_market_header hdr;
    {std::ifstream f(argv[1]);if(!f)throw std::runtime_error("input unavailable");fast_matrix_market::read_matrix_market_eigen(f,hdr,A);}
    const auto scan=apxchol::scan_input(A);std::string reason;
    const auto kind=apxchol::resolve_input_kind(apxchol::input_kind::automatic,scan,hdr.field==fast_matrix_market::pattern,reason);
    if(kind==apxchol::input_kind::adjacency)apxchol::adjacency_to_laplacian(A);
    apxchol::solve_options options;options.factor_opts.seed=seed;options.tol=1e-8;options.max_iter=5000;
    std::vector<Eigen::VectorXd> rhs;
    for(int r=0;r<count;++r) {
        std::srand(static_cast<unsigned>(20260905+r));
        auto b=apxchol::generate_test_rhs(A.rows());
        if(kind==apxchol::input_kind::adjacency || (scan.excess_rows==0 && scan.deficient_rows==0))
            apxchol::project_laplacian_rhs_components(A,b);
        rhs.push_back(std::move(b));
    }
    std::vector<apxchol::heavy_research::counters> counters(threads);
    apxchol::heavy_research::sampler sampler{apxchol::heavy_research::parse_mode(argv[2]),&counters};
    apxchol::checkpoint cp;
    const auto start=std::chrono::steady_clock::now();
    using Graph=apxchol::graph<apxchol::directed_vec_pool_incidence>;
    auto F=apxchol::factorize<apxchol::block_greedy_partitioner,decltype(sampler),Graph>(A,sampler,options.factor_opts,&cp);
    const auto raw_nnz=F.L.nonZeros();
    apxchol::cpu_solver solver(A,std::move(F),options,&cp);
    const double setup=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout<<std::setprecision(17);
    bool all=true;double solve_total=0;
    for(int r=0;r<count;++r) {
        const auto t=std::chrono::steady_clock::now();
        auto result=solver.solve(rhs[r]);
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
        const double residual=(A*result.x-rhs[r]).norm()/rhs[r].norm();
        const bool pass=std::isfinite(residual)&&residual<=options.tol;
        all=all&&pass;solve_total+=seconds;
        std::cout<<"{\"kind\":\"solve\",\"rhs\":"<<r<<",\"seconds\":"<<seconds
            <<",\"iterations\":"<<result.iterations<<",\"reported_residual\":"<<result.residual
            <<",\"true_residual\":"<<residual<<",\"pass\":"<<(pass?"true":"false")<<"}\n";
    }
    apxchol::heavy_research::counters total;
    for(const auto& c:counters) {
        total.calls+=c.calls;total.vertices+=c.vertices;total.cycles+=c.cycles;
        total.core_vertices+=c.core_vertices;total.receiver_passes+=c.receiver_passes;
        total.fractional_items+=c.fractional_items;total.numerical_fallbacks+=c.numerical_fallbacks;
    }
    rusage usage{};getrusage(RUSAGE_SELF,&usage);
    std::cout<<"{\"kind\":\"summary\",\"arm\":\""<<argv[2]<<"\",\"threads\":"<<threads<<",\"seed\":"<<seed
        <<",\"n\":"<<A.rows()<<",\"operator_nnz\":"<<A.nonZeros()<<",\"raw_factor_nnz\":"<<raw_nnz
        <<",\"stored_factor_nnz\":"<<solver.preconditioner().trsv().stored_nnz()<<",\"setup_s\":"<<setup
        <<",\"solve_total_s\":"<<solve_total<<",\"rhs_count\":"<<count<<",\"all_pass\":"<<(all?"true":"false")
        <<",\"max_rss_kib\":"<<usage.ru_maxrss<<",\"stars\":"<<total.calls<<",\"neighbor_visits\":"<<total.vertices
        <<",\"cycles\":"<<total.cycles<<",\"core_vertices\":"<<total.core_vertices<<",\"receiver_passes\":"<<total.receiver_passes
        <<",\"fractional_items\":"<<total.fractional_items<<",\"numerical_fallbacks\":"<<total.numerical_fallbacks<<"}\n";
    cp.report(std::cerr);
    return all?0:1;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 2;}
