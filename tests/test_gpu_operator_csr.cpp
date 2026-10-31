#include <gtest/gtest.h>
#include "scoped_omp_threads.h"
#include "apxchol/solver/solve.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#if defined(APXCHOL_USE_CUDA)
#include "apxchol/solver/pcg_cuda_kernels.h"
#include "apxchol/solver/pcg_cuda.h"
#include "apxchol/solver/detail/solver_route.h"
#include <cuda_runtime.h>
namespace {
using apxchol::detail::cuda_pcg_kernels::operator_csr;
using apxchol::detail::cuda_pcg_kernels::try_build_permuted_operator_csr;
struct HostCsc {
    int n = 0;
    std::vector<int> outer{0}, inner;
    std::vector<double> values;
};
struct Entry { int row, col; double value; };
HostCsc from_entries(int n, const std::vector<Entry>& entries) {
    std::vector<std::vector<std::pair<int,double>>> columns(n);
    for (const auto& e : entries) columns.at(e.col).emplace_back(e.row,e.value);
    HostCsc out; out.n=n;
    for (auto& col : columns) {
        std::sort(col.begin(),col.end(),[](const auto& a,const auto& b){return a.first<b.first;});
        for (auto [row,value] : col) { out.inner.push_back(row); out.values.push_back(value); }
        out.outer.push_back(static_cast<int>(out.inner.size()));
    }
    return out;
}
struct Reference {
    std::vector<int> ptr{0}, index;
    std::vector<double> values;
    bool exact_fp32=true;
};
// Independent scalar lower-triangle scatter. This does not call either host
// constructor or follow the device's column-owner/inverse-permutation route.
Reference reference(const HostCsc& input,const std::vector<std::uint32_t>& perm) {
    Reference ref;
    std::vector<std::vector<std::pair<int,double>>> rows(input.n);
    for (int col=0;col<input.n;++col) for (int k=input.outer[col];k<input.outer[col+1];++k) {
        const int row=input.inner[k]; if (row<col) continue;
        const double value=input.values[k];
        ref.exact_fp32 &= static_cast<double>(static_cast<float>(value))==value;
        rows[perm[row]].emplace_back(static_cast<int>(perm[col]),value);
        if (row!=col) rows[perm[col]].emplace_back(static_cast<int>(perm[row]),value);
    }
    for (auto& row : rows) {
        std::sort(row.begin(),row.end(),[](const auto& a,const auto& b){return a.first<b.first;});
        for (auto [col,value] : row) { ref.index.push_back(col);ref.values.push_back(value); }
        ref.ptr.push_back(static_cast<int>(ref.index.size()));
    }
    return ref;
}
struct DeviceOwner {
    operator_csr output;
    DeviceOwner()=default;
    DeviceOwner(const DeviceOwner&)=delete;
    DeviceOwner& operator=(const DeviceOwner&)=delete;
    ~DeviceOwner() {
        if (output.row_ptr) cudaFree(output.row_ptr);
        if (output.col_idx) cudaFree(output.col_idx);
        if (output.values_f64) cudaFree(output.values_f64);
        if (output.values_f32) cudaFree(output.values_f32);
        if (output.row_map) cudaFree(output.row_map);
    }
};
bool runtime_available() { int count=0;return cudaGetDeviceCount(&count)==cudaSuccess && count>0; }
template<class T> std::vector<T> download(const T* device,std::size_t count) {
    std::vector<T> host(count);
    if (count) {
        if (!device) throw std::runtime_error("missing owned operator buffer");
        const auto status=cudaMemcpy(host.data(),device,count*sizeof(T),cudaMemcpyDeviceToHost);
        if (status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    }
    return host;
}
// Stored rows keep input order; reassemble them into factor-order rows with
// row_map before comparing against the sorted permuted reference.
template<class T>
std::vector<T> by_vector_row(const operator_csr& output,const std::vector<int>& ptr,
                             const std::vector<int>& map,const std::vector<T>& stored,
                             const Reference& ref,std::vector<int>* index=nullptr) {
    const int n=static_cast<int>(ptr.size())-1;
    std::vector<std::vector<std::pair<int,T>>> rows(n);
    const auto col=download(output.col_idx,ref.index.size());
    for (int r=0;r<n;++r) for (int k=ptr[r];k<ptr[r+1];++k) rows.at(map.at(r)).emplace_back(col[k],stored[k]);
    std::vector<T> values;
    for (auto& row : rows) {
        std::sort(row.begin(),row.end(),[](const auto& a,const auto& b){return a.first<b.first;});
        for (auto [c,v] : row) { if (index) index->push_back(c); values.push_back(v); }
    }
    return values;
}
void expect_output(const operator_csr& output,const Reference& ref,int precision) {
    ASSERT_EQ(output.nnz,static_cast<int>(ref.values.size()));
    const bool fp32=precision==1 || (precision==-1 && ref.exact_fp32);
    ASSERT_EQ(output.fp32,fp32);
    const std::size_t n=ref.ptr.size()-1;
    const auto ptr=download(output.row_ptr,ref.ptr.size());
    const auto map=download(output.row_map,n);
    std::vector<int> sorted_map=map;std::sort(sorted_map.begin(),sorted_map.end());
    std::vector<int> identity(n);std::iota(identity.begin(),identity.end(),0);
    ASSERT_EQ(sorted_map,identity);
    std::vector<int> counts(n);
    for (std::size_t r=0;r<n;++r) counts.at(map[r])=ptr[r+1]-ptr[r];
    for (std::size_t r=0;r<n;++r) EXPECT_EQ(counts[r],ref.ptr[r+1]-ref.ptr[r]) << r;
    if (ref.index.empty()) {
        EXPECT_EQ(output.col_idx,nullptr);
        EXPECT_EQ(output.values_f64,nullptr);
        EXPECT_EQ(output.values_f32,nullptr);
    }
    std::vector<int> index;
    if (fp32) {
        EXPECT_EQ(output.values_f64,nullptr);
        const auto values=by_vector_row(output,ptr,map,download(output.values_f32,ref.values.size()),ref,&index);
        for (std::size_t i=0;i<values.size();++i) {
            SCOPED_TRACE(i);
            EXPECT_EQ(std::bit_cast<std::uint32_t>(values[i]),
                      std::bit_cast<std::uint32_t>(static_cast<float>(ref.values[i])));
        }
    } else {
        EXPECT_EQ(output.values_f32,nullptr);
        const auto values=by_vector_row(output,ptr,map,download(output.values_f64,ref.values.size()),ref,&index);
        for (std::size_t i=0;i<values.size();++i) {
            SCOPED_TRACE(i);
            EXPECT_EQ(std::bit_cast<std::uint64_t>(values[i]),std::bit_cast<std::uint64_t>(ref.values[i]));
        }
    }
    EXPECT_EQ(index,ref.index);
}
void compare(const HostCsc& input,const std::vector<std::uint32_t>& perm,int precision) {
    const auto wanted=reference(input,perm);DeviceOwner result;
    if (precision == 1 && !wanted.exact_fp32) {
        EXPECT_THROW(try_build_permuted_operator_csr(input.n,static_cast<int>(input.values.size()),
            input.outer.data(),input.inner.data(),input.values.data(),perm.data(),precision,result.output),
            std::invalid_argument);
        EXPECT_EQ(result.output.row_ptr, nullptr);
        EXPECT_EQ(result.output.values_f32, nullptr);
        return;
    }
    ASSERT_TRUE(try_build_permuted_operator_csr(input.n,static_cast<int>(input.values.size()),
        input.outer.data(),input.inner.data(),input.values.data(),perm.data(),precision,result.output));
    expect_output(result.output,wanted,precision);
}
std::array<std::vector<std::uint32_t>,3> permutations(int n) {
    std::vector<std::uint32_t> identity(n);std::iota(identity.begin(),identity.end(),0u);
    auto reverse=identity;std::reverse(reverse.begin(),reverse.end());
    auto random=identity;std::mt19937 rng(9317+n);std::shuffle(random.begin(),random.end(),rng);
    return {identity,reverse,random};
}
HostCsc geometry(int n) {
    std::vector<Entry> entries;const int active=n==257 ? n-2:n;
    for (int col=0;col<n;++col) if (col%3!=1) entries.push_back({col,col,col%7==6 ? -0.0:4.0});
    auto edge=[&](int row,int col,double value) {
        entries.push_back({row,col,value});
        // The upper values deliberately differ; only LOWER controls output.
        entries.push_back({col,row,value+0.1});
    };
    for (int row=1;row<active;++row) edge(row,0,row%17 ? -0.25*(1+row%5):-0.0);
    for (int col=1;col+1<active;++col) if (col/11==(col+1)/11) edge(col+1,col,-0.5);
    return from_entries(n,entries);
}
HostCsc precision_input(bool inexact) {
    std::vector<double> values={0.0,-0.0,0.25,-0.5,
        static_cast<double>(std::numeric_limits<float>::denorm_min()),
        -static_cast<double>(std::numeric_limits<float>::denorm_min()),
        static_cast<double>(std::numeric_limits<float>::min())};
    if (inexact) {
        const double half=std::ldexp(1.0,-150),tie=1.0+std::ldexp(1.0,-24);
        for (double x : {half,-half,std::numeric_limits<double>::denorm_min(),
                -std::numeric_limits<double>::denorm_min(),tie,std::nextafter(tie,0.0),
                std::nextafter(tie,2.0),-tie,0.1,-0.1}) values.push_back(x);
    }
    std::vector<Entry> entries{{0,0,8.0}};
    for (std::size_t i=0;i<values.size();++i) {
        entries.push_back({static_cast<int>(i+1),0,values[i]});
        entries.push_back({0,static_cast<int>(i+1),values[i]+0.123456789});
        if (i%2) entries.push_back({static_cast<int>(i+1),static_cast<int>(i+1),-0.0});
    }
    return from_entries(static_cast<int>(values.size()+1),entries);
}

// Scalar PCG through the panel entry point.
template<class Precond>
void pcg_solve(apxchol::detail::cuda_pcg& pcg, const Precond& precond, const Eigen::VectorXd& b,
               Eigen::VectorXd& x, double tol, int max_iter, int& iterations, double& residual,
               bool laplacian, const Eigen::VectorXd* x0 = nullptr) {
    std::array<int, 1> count{};
    std::array<double, 1> rr{};
    x.resize(b.size());
    pcg.solve_panel<1>(precond, b, x, tol, max_iter, count, rr, laplacian, x0);
    iterations = count[0];
    residual = rr[0];
}

const apxchol::detail::gpu_solve_session& session(const apxchol::solver& owner) {
    return std::get<apxchol::detail::gpu_solve_session>(apxchol::detail::solver_access::route(owner));
}

struct TestPreconditioner {
    int n;
    bool breakdown = false;
    unsigned breakdown_after = std::numeric_limits<unsigned>::max();
    mutable unsigned calls = 0;
    const auto& trsv() const { return *this; }
    template<unsigned Columns>
    void solve_LLt_dev(const double* input, double* output, const unsigned*, unsigned) const {
        const auto bytes = n * Columns * sizeof(double);
        const bool stopped = breakdown || calls >= breakdown_after;
        ++calls;
        const auto status = stopped ? cudaMemset(output, 0, bytes)
            : cudaMemcpy(output, input, bytes, cudaMemcpyDeviceToDevice);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    }
};

double independent_residual(const Eigen::SparseMatrix<double>& a,
                            const Eigen::VectorXd& b, const Eigen::VectorXd& x) {
    std::vector<long double> residual(b.size());
    for (int i = 0; i < b.size(); ++i) residual[i] = b[i];
    for (int col = 0; col < a.outerSize(); ++col)
        for (Eigen::SparseMatrix<double>::InnerIterator it(a, col); it; ++it)
            residual[it.row()] -= static_cast<long double>(it.value()) * x[col];
    long double rnorm = 0, bnorm = 0;
    for (int i = 0; i < b.size(); ++i) {
        rnorm = std::hypot(rnorm, residual[i]);
        bnorm = std::hypot(bnorm, static_cast<long double>(b[i]));
    }
    return static_cast<double>(bnorm == 0 ? rnorm : rnorm / bnorm);
}
}
#endif

TEST(GpuOperatorCsr, BoundariesAndPermutationsMatchIndependentLowerReference) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP()<<"CUDA operator CSR required";
#else
    if (!runtime_available()) GTEST_SKIP()<<"CUDA unavailable";
    for (int n : {0,1,2,31,32,33,63,64,65,255,256,257,1025}) {
        SCOPED_TRACE(n);
        const auto input=geometry(n);const auto orders=permutations(n);
        for (std::size_t order=0;order<orders.size();++order) {
            SCOPED_TRACE(order);
            for (int precision : {-1,0,1}) {
                SCOPED_TRACE(precision);
                compare(input,orders[order],precision);
            }
        }
    }
#endif
}

TEST(GpuOperatorCsr, RandomPairedCscMatchesIndependentLowerReference) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA operator CSR required";
#else
    if (!runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    std::mt19937 rng(418932);
    for (int trial = 0; trial < 80; ++trial) {
        SCOPED_TRACE(trial);
        const int n = trial * 13 % 131;
        std::vector<Entry> entries;
        for (int col = 0; col < n; ++col) {
            if (trial % 3) entries.push_back({col, col, 4.0});
            for (int row = col + 1; row < n; ++row) {
                if ((trial % 2 && row / 8 != col / 8) || rng() % 13) continue;
                const double value = trial % 5 ? -0.25 * (rng() % 8) : -0.1;
                entries.push_back({row, col, value});
                // Canonical LOWER values determine precision and both entries.
                entries.push_back({col, row, value + 1e-10});
            }
        }
        const auto input = from_entries(n, entries);
        std::vector<std::uint32_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0u);
        std::shuffle(perm.begin(), perm.end(), rng);
        compare(input, perm, -1);
    }
#endif
}

TEST(GpuOperatorCsr, CanonicalLowerControlsPrecisionAndPreservesEveryValueBit) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP()<<"CUDA operator CSR required";
#else
    if (!runtime_available()) GTEST_SKIP()<<"CUDA unavailable";
    for (bool inexact : {false,true}) {
        SCOPED_TRACE(inexact);
        const auto input=precision_input(inexact);
        for (const auto& perm : permutations(input.n)) {
            ASSERT_EQ(reference(input,perm).exact_fp32,!inexact);
            for (int precision : {-1,0,1}) {
                SCOPED_TRACE(precision);
                compare(input,perm,precision);
            }
        }
    }
#endif
}

TEST(GpuOperatorCsr, UnsupportedLayoutsLeaveOutputUnpublished) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP()<<"CUDA operator CSR required";
#else
    if (!runtime_available()) GTEST_SKIP()<<"CUDA unavailable";
    const auto base=from_entries(3,{{0,0,2.0},{1,0,-1.0},{0,1,-1.0},{1,1,2.0},{2,1,-1.0},{1,2,-1.0},{2,2,2.0}});
    std::vector<HostCsc> bad;
    auto add=[&](auto mutate){auto x=base;mutate(x);bad.push_back(std::move(x));};
    add([](auto& x){x.outer[0]=1;});
    add([](auto& x){--x.outer.back();});
    add([](auto& x){x.outer[1]=x.outer[2]+1;});
    add([](auto& x){x.inner[0]=-1;});
    add([](auto& x){x.inner[0]=x.n;});
    add([](auto& x){std::swap(x.inner[0],x.inner[1]);std::swap(x.values[0],x.values[1]);});
    add([](auto& x){x.inner[1]=x.inner[0];});
    bad.push_back(from_entries(3,{{0,0,2.0},{1,0,-1.0},{1,1,2.0},{2,2,2.0}}));
    // Equal upper/lower counts do not establish coordinate pairing.
    bad.push_back(from_entries(3,{{0,0,2.0},{1,0,-1.0},{1,1,2.0},{0,2,-1.0},{2,2,2.0}}));
    ASSERT_EQ(bad.size(),9u);
    const std::vector<std::uint32_t> perm{2,0,1};
    for (std::size_t i=0;i<bad.size();++i) {
        SCOPED_TRACE(i);
        for (int precision : {-1,0,1}) {
            SCOPED_TRACE(precision);
            operator_csr output;output.nnz=73;output.fp32=true;
            const auto& x=bad[i];
            EXPECT_FALSE(try_build_permuted_operator_csr(x.n,static_cast<int>(x.values.size()),
                x.outer.data(),x.inner.data(),x.values.data(),perm.data(),precision,output));
            EXPECT_EQ(output.row_ptr,nullptr);EXPECT_EQ(output.col_idx,nullptr);
            EXPECT_EQ(output.values_f64,nullptr);EXPECT_EQ(output.values_f32,nullptr);
            EXPECT_EQ(output.row_map,nullptr);
            EXPECT_EQ(output.nnz,73);EXPECT_TRUE(output.fp32);
            // If a broken implementation publishes buffers, still reap them.
            if(output.row_ptr)cudaFree(output.row_ptr);
            if(output.col_idx)cudaFree(output.col_idx);
            if(output.values_f64)cudaFree(output.values_f64);
            if(output.values_f32)cudaFree(output.values_f32);
            if(output.row_map)cudaFree(output.row_map);
        }
    }
#endif
}

TEST(GpuOperatorCsr, NonemptyOutputOwnerIsRejectedWithoutModification) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA operator CSR required";
#else
    if (!runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    const auto valid = geometry(5);
    const auto perm = permutations(valid.n)[2];
    DeviceOwner owner;
    ASSERT_TRUE(try_build_permuted_operator_csr(valid.n,
        static_cast<int>(valid.values.size()), valid.outer.data(), valid.inner.data(),
        valid.values.data(), perm.data(), -1, owner.output));
    const auto before = owner.output;
    // Balanced triangle counts still lack coordinate pairing.
    const auto bad = from_entries(5, {{1, 0, -1.0}, {0, 2, -1.0}});
    // The device builder's output is an ownership transfer, not a replace
    // operation: it rejects a live owner before validating incoming storage.
    EXPECT_THROW(try_build_permuted_operator_csr(bad.n,
        static_cast<int>(bad.values.size()), bad.outer.data(), bad.inner.data(),
        bad.values.data(), perm.data(), -1, owner.output), std::invalid_argument);
    EXPECT_EQ(owner.output.row_ptr, before.row_ptr);
    EXPECT_EQ(owner.output.col_idx, before.col_idx);
    EXPECT_EQ(owner.output.values_f64, before.values_f64);
    EXPECT_EQ(owner.output.values_f32, before.values_f32);
    EXPECT_EQ(owner.output.row_map, before.row_map);
    EXPECT_EQ(owner.output.nnz, before.nnz);
    EXPECT_EQ(owner.output.fp32, before.fp32);
    expect_output(owner.output, reference(valid, perm), -1);
#endif
}

TEST(GpuOperatorCsr, HostInputStorageMayEndAtSuccessfulReturn) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP()<<"CUDA operator CSR required";
#else
    if (!runtime_available()) GTEST_SKIP()<<"CUDA unavailable";
    for (int precision : {-1,0,1}) {
        SCOPED_TRACE(precision);
        DeviceOwner result;Reference wanted;
        {
            auto input=geometry(65);auto perm=permutations(input.n)[2];wanted=reference(input,perm);
            ASSERT_TRUE(try_build_permuted_operator_csr(input.n,static_cast<int>(input.values.size()),
                input.outer.data(),input.inner.data(),input.values.data(),perm.data(),precision,result.output));
            std::fill(input.outer.begin(),input.outer.end(),0);std::fill(input.inner.begin(),input.inner.end(),0);
            std::fill(input.values.begin(),input.values.end(),99.0);std::fill(perm.begin(),perm.end(),0);
        }
        expect_output(result.output,wanted,precision);
    }
#endif
}

TEST(GpuOperatorCsr, ConsumingSolveKeepsDefaultStorageAndOriginalResidual) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP()<<"CUDA operator CSR required";
#else
    if (!runtime_available()) GTEST_SKIP()<<"CUDA unavailable";
    scoped_omp_threads serial(1);
    constexpr int n=65;
    for (double shift : {0.0,1.0}) {
        SCOPED_TRACE(shift);
        std::vector<Eigen::Triplet<double>> entries;
        std::vector<double> diag(n,shift);
        for(int u=0;u<n;++u)for(int offset : {1,7}) {
            const int v=(u+offset)%n;const double w=offset==1?1.0:0.5;
            diag[u]+=w;diag[v]+=w;entries.emplace_back(u,v,-w);entries.emplace_back(v,u,-w);
        }
        for(int u=0;u<n;++u)entries.emplace_back(u,u,diag[u]);
        Eigen::SparseMatrix<double> A(n,n);A.setFromTriplets(entries.begin(),entries.end());
        Eigen::VectorXd exact(n);for(int u=0;u<n;++u)exact[u]=std::sin(u+0.375);
        const Eigen::VectorXd b=A*exact;
        const apxchol::solver owner(A, {.backend = apxchol::solve_backend::gpu});
        const auto& preconditioner = session(owner);
        ASSERT_TRUE(preconditioner.trsv().adopted_device_factor());
        EXPECT_TRUE(preconditioner.trsv().fp16());
        EXPECT_GT(preconditioner.trsv().drop_stats().rel,0.0);
        apxchol::detail::cuda_pcg solver;
        solver.setup(A,preconditioner.factor().perm);
        EXPECT_TRUE(solver.fp32_operator());
        Eigen::VectorXd result;int iterations=0;double reported=0;
        pcg_solve(solver, preconditioner,b,result,1e-8,1000,iterations,reported,shift==0.0);
        ASSERT_EQ(result.size(),n);
        EXPECT_GT(iterations,0);
        EXPECT_LE((A*result-b).norm()/b.norm(),1e-8);
        EXPECT_TRUE(std::isfinite(reported));
        EXPECT_NEAR(reported, independent_residual(A, b, result), 2e-14);
    }
#endif
}

TEST(GpuSolveHonesty, ScalarAndBatchContinueAfterSlowProgress) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "32-bit-node CUDA solve required";
#else
    if (!runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    constexpr int n = 256;
    std::vector<Eigen::Triplet<double>> entries;
    for (int i = 0; i < n; ++i) {
        entries.emplace_back(i, i, 2.0);
        if (i) {
            entries.emplace_back(i, i - 1, -1.0);
            entries.emplace_back(i - 1, i, -1.0);
        }
    }
    Eigen::SparseMatrix<double> matrix(n, n);
    matrix.setFromTriplets(entries.begin(), entries.end());
    std::vector<apxchol::node_index> permutation(n);
    std::iota(permutation.begin(), permutation.end(), 0);
    apxchol::detail::cuda_pcg solver;
    solver.setup(matrix, permutation);
    const TestPreconditioner identity{n};
    const Eigen::VectorXd rhs = Eigen::VectorXd::Ones(n);
    Eigen::MatrixXd B(n, 4), X(n, 4);
    B.col(0) = rhs;
    B.col(1).setZero();
    B.col(2) = -2 * rhs;
    B.col(3) = 0.5 * rhs;
    // The residual still exceeds 0.5 at the former 50-iteration stop, but
    // CG converges when allowed to continue on this positive-definite system.
    for (const int budget : {50, 51, 512}) {
        SCOPED_TRACE(budget);
        Eigen::VectorXd x;
        int iterations = 0;
        double residual = 0;
        pcg_solve(solver, identity, rhs, x, 1e-8, budget, iterations, residual, false);
        const auto check = [&](const Eigen::VectorXd& b, const Eigen::VectorXd& output,
                               int count, double reported) {
            EXPECT_NEAR(reported, independent_residual(matrix, b, output), 2e-13);
            if (budget < 512) {
                EXPECT_EQ(count, budget);
                EXPECT_GT(reported, 0.5);
            } else {
                EXPECT_GT(count, 50);
                EXPECT_LE(count, budget);
                EXPECT_LT(reported, 1e-8);
            }
        };
        check(rhs, x, iterations, residual);
        std::array<int, 4> counts{};
        std::array<double, 4> residuals{};
        solver.solve_panel<4>(identity, B, X, 1e-8, budget, counts, residuals, false);
        EXPECT_EQ(counts[1], 0);
        EXPECT_EQ(residuals[1], 0.0);
        EXPECT_TRUE(X.col(1).isZero(0.0));
        for (const int c : {0, 2, 3}) check(B.col(c), X.col(c), counts[c], residuals[c]);
    }
#endif
}

TEST(GpuSolveHonesty, FinalResidualChecksReturnedScalarAndPairOnReuse) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "CUDA solve required";
#else
    if (!runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    constexpr int n = 33;
    for (bool laplacian : {false, true}) for (bool fp32 : {false, true}) {
        SCOPED_TRACE(::testing::Message() << laplacian << ' ' << fp32);
        const double weight = fp32 ? 1.0 : std::nextafter(1.0, 2.0);
        std::vector<Eigen::Triplet<double>> entries;
        for (int i = 0; i < n; ++i) {
            entries.emplace_back(i, i, (laplacian ? 2.0 : 4.0) * weight);
            entries.emplace_back(i, (i + 1) % n, -weight);
            entries.emplace_back((i + 1) % n, i, -weight);
        }
        Eigen::SparseMatrix<double> a(n, n);
        a.setFromTriplets(entries.begin(), entries.end());
        std::vector<apxchol::node_index> perm(n);
        for (int i = 0; i < n; ++i) perm[i] = n - 1 - i;
        apxchol::detail::cuda_pcg solver;
        solver.setup(a, perm);
        EXPECT_EQ(solver.fp32_operator(), fp32);
        TestPreconditioner preconditioner{n};
        Eigen::VectorXd exact(n);
        for (int i = 0; i < n; ++i) exact[i] = std::sin(i + 0.25);
        const Eigen::VectorXd base = a * exact;
        for (double magnitude : {1.0, 1e-300, 1e300}) {
            SCOPED_TRACE(magnitude);
            const Eigen::VectorXd b = base * magnitude;
            Eigen::VectorXd x;
            int iterations = 0;
            double reported = 0;
            pcg_solve(solver, preconditioner, b, x, 1e-12, 100, iterations, reported, laplacian);
            EXPECT_GT(iterations, 0);
            EXPECT_NEAR(reported, independent_residual(a, b, x), 2e-14);
            EXPECT_LT(reported, 1e-10);

            Eigen::MatrixXd rhs(n, 2), output(n, 2), warm(n, 2);
            rhs.col(0) = b;
            rhs.col(1).setZero();
            warm.col(0) = 0.25 * x;
            warm.col(1).setZero();
            std::array<int, 2> counts{};
            std::array<double, 2> residuals{};
            for (int limit : {0, 2, 100}) {
                SCOPED_TRACE(limit);
                solver.solve_panel<2>(preconditioner, rhs, output, 1e-12, limit,
                                  counts, residuals, laplacian, &warm);
                for (int q = 0; q < 2; ++q)
                    EXPECT_NEAR(residuals[q], independent_residual(a, rhs.col(q), output.col(q)), 2e-14);
                EXPECT_LE(counts[0], limit);
                EXPECT_EQ(counts[1], 0);
                EXPECT_EQ(residuals[1], 0);
            }
            // Shrinking the logical width must not inherit the pair's layout.
            pcg_solve(solver, preconditioner, b, x, 1e-12, 0, iterations, reported, laplacian);
            EXPECT_EQ(iterations, 0);
            EXPECT_NEAR(reported, 1.0, 2e-15);
        }
        preconditioner.breakdown = true;
        const Eigen::VectorXd warm = 0.25 * exact;
        Eigen::VectorXd output;
        int iterations;
        double reported;
        pcg_solve(solver, preconditioner, base, output, 1e-12, 10, iterations, reported, laplacian, &warm);
        EXPECT_EQ(iterations, 0);
        EXPECT_NEAR(reported, independent_residual(a, base, output), 2e-14);

        // A breakdown after progress must not start another solve attempt.
        preconditioner.breakdown = false;
        preconditioner.breakdown_after = 1;
        preconditioner.calls = 0;
        pcg_solve(solver, preconditioner, base, output, 1e-12, 100, iterations, reported, laplacian, &warm);
        EXPECT_EQ(iterations, 1);
        EXPECT_EQ(preconditioner.calls, 2u);
        EXPECT_GT(reported, 1e-12);
        EXPECT_NEAR(reported, independent_residual(a, base, output), 2e-14);
        Eigen::MatrixXd rhs(n, 4), panel(n, 4);
        rhs.col(0) = base; rhs.col(1).setZero(); rhs.col(2) = -2 * base; rhs.col(3) = 0.5 * base;
        std::array<int, 4> counts{};
        std::array<double, 4> residuals{};
        preconditioner.calls = 0;
        solver.solve_panel<4>(preconditioner, rhs, panel, 1e-12, 100, counts, residuals, laplacian);
        EXPECT_EQ(preconditioner.calls, 2u);
        EXPECT_EQ(counts, (std::array<int, 4>{1, 0, 1, 1}));
        for (unsigned q = 0; q < 4; ++q)
            EXPECT_NEAR(residuals[q], independent_residual(a, rhs.col(q), panel.col(q)), 2e-14);
    }
#endif
}

TEST(GpuSolveHonesty, ResidualIncludesRoundingWhenRestoringSubnormalSolutions) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "CUDA solve required";
#else
    if (!runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    constexpr int n = 3;
    Eigen::SparseMatrix<double> a(n, n);
    for (int i = 0; i < n; ++i) a.insert(i, i) = 3.25;
    a.makeCompressed();
    const std::vector<apxchol::node_index> perm{2, 0, 1};
    apxchol::detail::cuda_pcg solver;
    solver.setup(a, perm);
    TestPreconditioner preconditioner{n};
    const double tiny = std::numeric_limits<double>::denorm_min();
    Eigen::MatrixXd b(n, 2), output(n, 2);
    b.col(0).setConstant(100 * tiny);
    b.col(1).setOnes();
    std::array<int, 2> iterations{};
    std::array<double, 2> residuals{};
    solver.solve_panel<2>(preconditioner, b, output, 1e-12, 10, iterations, residuals, false);
    EXPECT_GT(residuals[0], 1e-3);
    for (int q = 0; q < 2; ++q)
        EXPECT_NEAR(residuals[q], independent_residual(a, b.col(q), output.col(q)), 2e-14);

    Eigen::VectorXd x;
    int count;
    double residual;
    const Eigen::VectorXd underflow = Eigen::VectorXd::Constant(n, tiny);
    EXPECT_THROW(pcg_solve(solver, preconditioner, underflow, x, 1e-12, 10, count, residual, false),
                 std::underflow_error);
    const Eigen::VectorXd ordinary = Eigen::VectorXd::Ones(n);
    pcg_solve(solver, preconditioner, ordinary, x, 1e-12, 10, count, residual, false);
    EXPECT_NEAR(residual, independent_residual(a, ordinary, x), 2e-14);

    a *= 1.0 / 13.0; // 0.25 I: the restored solution exceeds the finite range.
    solver.setup(a, perm);
    const Eigen::VectorXd overflow = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::max() / 2);
    EXPECT_THROW(pcg_solve(solver, preconditioner, overflow, x, 1e-12, 10, count, residual, false),
                 std::overflow_error);
#endif
}

TEST(GpuSolveHonesty, FinalResidualDoesNotSquareAwayTinyErrors) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "CUDA solve required";
#else
    if (!runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    struct Case { int n; double leading, tail, residual; };
    const double tiny = std::numeric_limits<double>::denorm_min();
    for (const auto test : {Case{2, 1, 1e-200, 1e-200}, Case{5, 2, tiny, tiny},
                           Case{101, 1e100, 1e-224, 1e-323}}) {
        const int n = test.n;
        SCOPED_TRACE(n);
        Eigen::SparseMatrix<double> a(n, n);
        a.setIdentity();
        std::vector<apxchol::node_index> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        apxchol::detail::cuda_pcg solver;
        solver.setup(a, perm);
        TestPreconditioner preconditioner{n};
        const double expected = test.residual;
        Eigen::VectorXd b = Eigen::VectorXd::Constant(n, test.tail);
        b[0] = test.leading;
        Eigen::VectorXd warm = Eigen::VectorXd::Zero(n);
        warm[0] = b[0];
        Eigen::VectorXd x;
        int iterations;
        double residual;
        pcg_solve(solver, preconditioner, b, x, 1e-250, 0, iterations, residual, false, &warm);
        EXPECT_GT(residual, 0);
        EXPECT_DOUBLE_EQ(residual / expected, 1.0);
        EXPECT_EQ(iterations, 0);
        Eigen::MatrixXd rhs(n, 2), initial(n, 2), output(n, 2);
        rhs.col(0) = b;
        rhs.col(1).setZero();
        initial.col(0) = warm;
        initial.col(1).setZero();
        std::array<int, 2> counts{};
        std::array<double, 2> residuals{};
        solver.solve_panel<2>(preconditioner, rhs, output, 1e-250, 0, counts, residuals, false, &initial);
        EXPECT_GT(residuals[0], 0);
        EXPECT_DOUBLE_EQ(residuals[0] / expected, 1.0);
        EXPECT_DOUBLE_EQ(residuals[1], 0.0);
        EXPECT_EQ(counts, (std::array<int, 2>{0, 0}));
    }
#endif
}

TEST(GpuSolveHonesty, FinalResidualDoesNotOverflowBeforeNormalization) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "CUDA solve required";
#else
    if (!runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    constexpr int n = 5;
    const double coefficient = std::numeric_limits<double>::max() / 2;
    Eigen::SparseMatrix<double> a(n, n);
    for (int i = 0; i < n; ++i) a.insert(i, i) = coefficient;
    a.makeCompressed();
    apxchol::detail::cuda_pcg solver;
    solver.setup(a, {0, 1, 2, 3, 4});
    TestPreconditioner preconditioner{n};
    const Eigen::VectorXd b = Eigen::VectorXd::Constant(n, 2);
    const Eigen::VectorXd warm = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd x;
    int iterations;
    double residual;
    pcg_solve(solver, preconditioner, b, x, 1e-8, 0, iterations, residual, false, &warm);
    EXPECT_DOUBLE_EQ(residual / (coefficient / 2), 1.0);
    EXPECT_EQ(iterations, 0);
    Eigen::MatrixXd rhs(n, 2), initial(n, 2), output(n, 2);
    rhs.col(0) = b;
    rhs.col(1).setZero();
    initial.col(0) = warm;
    initial.col(1).setZero();
    std::array<int, 2> counts{};
    std::array<double, 2> residuals{};
    solver.solve_panel<2>(preconditioner, rhs, output, 1e-8, 0, counts, residuals, false, &initial);
    EXPECT_DOUBLE_EQ(residuals[0] / (coefficient / 2), 1.0);
    EXPECT_DOUBLE_EQ(residuals[1], 0.0);
    EXPECT_EQ(counts, (std::array<int, 2>{0, 0}));
#endif
}

TEST(GpuOperatorCsr, DeviceSetupRejectsUnsupportedStorage) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "32-bit-node CUDA operator CSR required";
#else
    if (!runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    scoped_omp_threads serial(1);
    constexpr int n = 9;
    std::vector<Eigen::Triplet<double>> entries;
    for (int i = 0; i < n; ++i) {
        entries.emplace_back(i, i, 4.0);
        if (i + 1 < n) {
            entries.emplace_back(i, i + 1, -1.0);
            entries.emplace_back(i + 1, i, -1.0);
        }
    }
    Eigen::SparseMatrix<double> A(n, n);
    A.setFromTriplets(entries.begin(), entries.end());
    const apxchol::solver owner(A, {.backend = apxchol::solve_backend::gpu});
    const auto& preconditioner = session(owner);
    ASSERT_TRUE(preconditioner.trsv().adopted_device_factor());
    apxchol::detail::cuda_pcg solver;
    testing::internal::CaptureStderr();
    solver.setup(A, preconditioner.factor().perm, true);
    const auto diagnostic = testing::internal::GetCapturedStderr();
    EXPECT_NE(diagnostic.find("[gpu-operator-builder] route=device completion_wait=1"),
              std::string::npos);
    EXPECT_EQ(cudaStreamQuery(nullptr), cudaSuccess);
    Eigen::VectorXd exact(n);
    for (int i = 0; i < n; ++i) exact[i] = std::sin(i + 0.25);
    const Eigen::VectorXd b = A * exact;
    Eigen::VectorXd result;
    int iterations = 0;
    double reported = 0;
    pcg_solve(solver, preconditioner, b, result, 1e-8, 100, iterations, reported, false);
    ASSERT_EQ(result.size(), n);
    EXPECT_LE((A * result - b).norm() / b.norm(), 1e-8);
    EXPECT_TRUE(std::isfinite(reported));

    // Paired CSC, compression, square shape and a complete permutation are
    // mandatory. None of these failures may invoke the retired host builder.
    Eigen::SparseMatrix<double> lower = A.triangularView<Eigen::Lower>();
    Eigen::SparseMatrix<double> upper = A.triangularView<Eigen::Upper>();
    auto uncompressed = A;
    uncompressed.uncompress();
    for (const auto* bad : {&lower, &upper, &uncompressed}) {
        apxchol::detail::cuda_pcg rejected;
        EXPECT_THROW(rejected.setup(*bad, preconditioner.factor().perm),
                     std::invalid_argument);
    }
    Eigen::SparseMatrix<double> nonsquare(n, n + 1);
    apxchol::detail::cuda_pcg rejected;
    EXPECT_THROW(rejected.setup(nonsquare, preconditioner.factor().perm),
                 std::invalid_argument);
    auto short_perm = preconditioner.factor().perm;
    short_perm.pop_back();
    EXPECT_THROW(rejected.setup(A, short_perm), std::invalid_argument);
#endif
}

TEST(GpuSolveHonesty, ZeroIterationPublicSolveReportsRelativeResidual) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "CUDA solve required";
#else
    if (!runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    scoped_omp_threads serial(1);
    constexpr int n = 5;
    std::vector<Eigen::Triplet<double>> entries;
    for (int i = 0; i < n; ++i) {
        entries.emplace_back(i, i, 4.0);
        if (i + 1 < n) {
            entries.emplace_back(i, i + 1, -1.0);
            entries.emplace_back(i + 1, i, -1.0);
        }
    }
    Eigen::SparseMatrix<double> A(n, n);
    A.setFromTriplets(entries.begin(), entries.end());
    for (double scale : {1e-12, 1.0, 1e6}) {
        SCOPED_TRACE(scale);
        const Eigen::VectorXd b = Eigen::VectorXd::Constant(n, scale);
        const auto result = apxchol::solve(A, b, {.tol = 1e-8, .max_iter = 0,
                                                 .backend = apxchol::solve_backend::gpu});
        ASSERT_EQ(result.x.size(), n);
        EXPECT_TRUE(result.x.isZero(0.0));
        EXPECT_EQ(result.iterations, 0);
        EXPECT_NEAR(result.residual, 1.0, 2e-15);
        EXPECT_DOUBLE_EQ((A * result.x - b).norm() / b.norm(), result.residual);
    }
#endif
}

TEST(GpuSolveHonesty, ZeroIterationSessionReportsRelativeResidualOnReuse) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "CUDA solve required";
#else
    if (!runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    scoped_omp_threads serial(1);
    constexpr int n = 5;
    std::vector<Eigen::Triplet<double>> entries;
    for (int i = 0; i < n; ++i) {
        entries.emplace_back(i, i, 4.0);
        if (i + 1 < n) {
            entries.emplace_back(i, i + 1, -1.0);
            entries.emplace_back(i + 1, i, -1.0);
        }
    }
    Eigen::SparseMatrix<double> A(n, n);
    A.setFromTriplets(entries.begin(), entries.end());
    apxchol::solver solver(A, {.backend = apxchol::solve_backend::gpu});
    ASSERT_TRUE(session(solver).trsv().adopted_device_factor());
    EXPECT_THROW(solver.solve(Eigen::VectorXd::Zero(n - 1), 1e-8, 0),
                 std::invalid_argument);
    for (double scale : {1e-12, 0.0, 1e6}) {
        SCOPED_TRACE(scale);
        const Eigen::VectorXd b = Eigen::VectorXd::Constant(n, scale);
        const auto result = solver.solve(b, 1e-8, 0);
        EXPECT_EQ(result.backend, apxchol::solve_backend::gpu);
        ASSERT_EQ(result.x.size(), n);
        EXPECT_TRUE(result.x.isZero(0.0));
        EXPECT_EQ(result.iterations, 0);
        EXPECT_NEAR(result.residual, scale == 0.0 ? 0.0 : 1.0, 2e-15);
        const double denominator = b.norm() > 0.0 ? b.norm() : 1.0;
        EXPECT_DOUBLE_EQ((A * result.x - b).norm() / denominator, result.residual);
    }
#endif
}

TEST(GpuOperatorCsr, ReusableOwnersFollowTheirSetupDeviceAcrossThreads) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "32-bit-node CUDA solve required";
#else
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        GTEST_SKIP() << "CUDA unavailable";
    // On multi-GPU hosts the new thread starts on a different device. The
    // single-GPU case still checks cross-thread reuse and destruction.
    const apxchol::detail::cuda_device_scope setup_device(
        count - 1, apxchol::detail::cuda_device_scope::checked{});
    scoped_omp_threads serial(1);
    constexpr int n = 65;
    Eigen::SparseMatrix<double> A(n, n);
    for (int j = 0; j < n; ++j) {
        if (j) A.insert(j - 1, j) = -1;
        A.insert(j, j) = 3;
        if (j + 1 < n) A.insert(j + 1, j) = -1;
    }
    A.makeCompressed();
    const Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(n, 1, 2);
    apxchol::solve_options options;
    options.backend = apxchol::solve_backend::gpu;
    options.num_threads = 1;
    options.factor_opts.factor_storage = apxchol::factor_storage_type::fp32;
    options.keep_factor = true;
    const auto host_factor = apxchol::factorize(A);
    for (const bool adopt : {false, true}) {
        SCOPED_TRACE(adopt);
        auto owner = adopt ? std::make_unique<apxchol::solver>(A, host_factor, options)
                           : std::make_unique<apxchol::solver>(A, options);
        const Eigen::VectorXd applied = owner->apply(rhs);
        std::exception_ptr failure;
        std::thread worker([&, owner = std::move(owner)]() mutable {
            try {
                apxchol::detail::check_cuda(cudaSetDevice(0), "cudaSetDevice");
                const auto expect_caller_device = [] {
                    int current = -1;
                    apxchol::detail::check_cuda(cudaGetDevice(&current), "cudaGetDevice");
                    EXPECT_EQ(current, 0);
                };
                const auto scalar = owner->solve(rhs, 1e-8, 200);
                EXPECT_TRUE(scalar.converged);
                EXPECT_LT(independent_residual(A, rhs, scalar.x), 1e-8);
                expect_caller_device();
                Eigen::MatrixXd B(n, 4), X(n, 4);
                for (int c = 0; c < 4; ++c) B.col(c) = (c + 1.0) * rhs;
                const auto batch = owner->solve(B, X, 1e-8, 200);
                for (int c = 0; c < 4; ++c) {
                    EXPECT_TRUE(batch.columns[c].converged);
                    EXPECT_LT(independent_residual(A, B.col(c), X.col(c)), 1e-8);
                }
                expect_caller_device();
                EXPECT_EQ((owner->apply(rhs) - applied).norm(), 0);
                expect_caller_device();
                owner.reset();
                expect_caller_device();
            } catch (...) { failure = std::current_exception(); }
        });
        worker.join();
        if (failure) std::rethrow_exception(failure);
    }
#endif
}
