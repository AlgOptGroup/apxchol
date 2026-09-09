#include <gtest/gtest.h>
#include "apxchol/solver/solve.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(APXCHOL_USE_CUDA)
#include "apxchol/solver/pcg_cuda_kernels.h"
#include "apxchol/solver/pcg_cuda.h"
#include <cuda_runtime.h>
namespace {
using apxchol::pcg_cuda::operator_csr;
using apxchol::pcg_cuda::try_build_permuted_operator_csr;
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
void expect_output(const operator_csr& output,const Reference& ref,int precision) {
    ASSERT_EQ(output.nnz,static_cast<int>(ref.values.size()));
    const bool fp32=precision==1 || (precision==-1 && ref.exact_fp32);
    ASSERT_EQ(output.fp32,fp32);
    EXPECT_EQ(download(output.row_ptr,ref.ptr.size()),ref.ptr);
    EXPECT_EQ(download(output.col_idx,ref.index.size()),ref.index);
    if (ref.index.empty()) {
        EXPECT_EQ(output.col_idx,nullptr);
        EXPECT_EQ(output.values_f64,nullptr);
        EXPECT_EQ(output.values_f32,nullptr);
    }
    if (fp32) {
        EXPECT_EQ(output.values_f64,nullptr);
        const auto values=download(output.values_f32,ref.values.size());
        for (std::size_t i=0;i<values.size();++i) {
            SCOPED_TRACE(i);
            EXPECT_EQ(std::bit_cast<std::uint32_t>(values[i]),
                      std::bit_cast<std::uint32_t>(static_cast<float>(ref.values[i])));
        }
    } else {
        EXPECT_EQ(output.values_f32,nullptr);
        const auto values=download(output.values_f64,ref.values.size());
        for (std::size_t i=0;i<values.size();++i) {
            SCOPED_TRACE(i);
            EXPECT_EQ(std::bit_cast<std::uint64_t>(values[i]),std::bit_cast<std::uint64_t>(ref.values[i]));
        }
    }
}
void compare(const HostCsc& input,const std::vector<std::uint32_t>& perm,int precision) {
    const auto wanted=reference(input,perm);DeviceOwner result;
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
class scoped_environment {
public:
    scoped_environment(const char* key,const char* value):key_(key) {
        if (const char* old=std::getenv(key)) { present_=true;old_=old; }
        if (value) setenv(key,value,1);else unsetenv(key);
    }
    ~scoped_environment(){if(present_)setenv(key_.c_str(),old_.c_str(),1);else unsetenv(key_.c_str());}
private:std::string key_,old_;bool present_=false;
};
class scoped_threads {
public:
    scoped_threads() {
#ifdef _OPENMP
        previous_=omp_get_max_threads();omp_set_num_threads(1);
#endif
    }
    ~scoped_threads() {
#ifdef _OPENMP
        omp_set_num_threads(previous_);
#endif
    }
private:int previous_=1;
};
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
            EXPECT_EQ(output.nnz,73);EXPECT_TRUE(output.fp32);
            // If a broken implementation publishes buffers, still reap them.
            if(output.row_ptr)cudaFree(output.row_ptr);
            if(output.col_idx)cudaFree(output.col_idx);
            if(output.values_f64)cudaFree(output.values_f64);
            if(output.values_f32)cudaFree(output.values_f32);
        }
    }
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
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP()<<"CUDA operator CSR required";
#else
    if (!runtime_available()) GTEST_SKIP()<<"CUDA unavailable";
    scoped_threads serial;
    scoped_environment frontend("APXCHOL_GPU_BLOCK_FRONTEND","force");
    scoped_environment shadow("APXCHOL_GPU_ROUND_SHADOW","force");
    scoped_environment finalize("APXCHOL_GPU_FACTOR_FINALIZE","force");
    scoped_environment fp16("APXCHOL_SPTRSV_FP16",nullptr);
    scoped_environment drop("APXCHOL_FACTOR_DROP",nullptr);
    scoped_environment backend("APXCHOL_GPU_SPTRSV",nullptr);
    scoped_environment tail("APXCHOL_RESIDUAL_SPARSIFY","0");
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
        apxchol::apx_cholesky preconditioner;
        preconditioner.compute(A);
        ASSERT_TRUE(preconditioner.trsv().adopted_device_factor());
        EXPECT_TRUE(preconditioner.trsv().fp16());
        EXPECT_GT(preconditioner.trsv().drop_stats().rel,0.0);
        for(const char* precision : {static_cast<const char*>(nullptr),"0"}) {
            scoped_environment op_precision("APXCHOL_GPU_FP32_OPERATOR",precision);
            apxchol::cuda_pcg solver;
            solver.setup(A,preconditioner.factor().perm);
            EXPECT_EQ(solver.fp32_operator(),precision==nullptr);
            Eigen::VectorXd result;int iterations=0;double reported=0;
            solver.solve(preconditioner,b,result,1e-8,1000,iterations,reported,shift==0.0);
            ASSERT_EQ(result.size(),n);
            EXPECT_GT(iterations,0);
            EXPECT_LE((A*result-b).norm()/b.norm(),1e-8);
            EXPECT_TRUE(std::isfinite(reported));
        }
    }
#endif
}

TEST(GpuOperatorCsr, OptInAndHostFallbackKeepOriginalSystemSolve) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP()<<"CUDA operator CSR required";
#else
    if (!runtime_available()) GTEST_SKIP()<<"CUDA unavailable";
    scoped_threads serial;
    scoped_environment frontend("APXCHOL_GPU_BLOCK_FRONTEND",nullptr);
    scoped_environment shadow("APXCHOL_GPU_ROUND_SHADOW",nullptr);
    scoped_environment finalize("APXCHOL_GPU_FACTOR_FINALIZE",nullptr);
    constexpr int n=9;
    std::vector<Eigen::Triplet<double>> entries;
    for (int i=0;i<n;++i) {
        entries.emplace_back(i,i,4.0);
        if(i+1<n) { entries.emplace_back(i,i+1,-1.0);entries.emplace_back(i+1,i,-1.0); }
    }
    Eigen::SparseMatrix<double> A(n,n);A.setFromTriplets(entries.begin(),entries.end());
    Eigen::SparseMatrix<double> lower=A.triangularView<Eigen::Lower>();
    Eigen::VectorXd exact(n);for(int i=0;i<n;++i)exact[i]=std::sin(i+0.25);
    const Eigen::VectorXd b=A*exact;
    apxchol::apx_cholesky preconditioner;
    preconditioner.compute(A);
    ASSERT_FALSE(preconditioner.trsv().adopted_device_factor());
    // Full CSC exercises default-off, incomplete opt-in, and explicit opt-in.
    // Lower-only storage fails the paired-format proof and must fall back even
    // with all three flags enabled. Each uses the same original-system factor.
    for (int route=0;route<4;++route) {
        SCOPED_TRACE(route);
        scoped_environment block("APXCHOL_GPU_BLOCK_FRONTEND",route>=2?"force":nullptr);
        scoped_environment round("APXCHOL_GPU_ROUND_SHADOW",route>=1?"force":nullptr);
        scoped_environment factor("APXCHOL_GPU_FACTOR_FINALIZE",route>=1?"force":nullptr);
        apxchol::cuda_pcg solver;
        scoped_environment trace("APXCHOL_VERBOSE","1");
        testing::internal::CaptureStderr();
        solver.setup(route==3?lower:A,preconditioner.factor().perm);
        const auto diagnostic=testing::internal::GetCapturedStderr();
        const bool device=route==2 && sizeof(apxchol::node_index)==sizeof(std::uint32_t);
        EXPECT_NE(diagnostic.find(device ? "[gpu-operator-builder] route=device completion_wait=1"
                                         : "[gpu-operator-builder] route=host completion_wait=0"),
                  std::string::npos);
        if(device) EXPECT_EQ(cudaStreamQuery(nullptr),cudaSuccess);
        Eigen::VectorXd result;int iterations=0;double reported=0;
        solver.solve(preconditioner,b,result,1e-8,100,iterations,reported,false);
        ASSERT_EQ(result.size(),n);
        EXPECT_LE((A*result-b).norm()/b.norm(),1e-8);
        EXPECT_TRUE(std::isfinite(reported));
    }
#endif
}
