// apxchol_mex.cpp — Octave/MATLAB MEX gateway for the apxchol CPU solver.
//
// One gateway, op-dispatched (keeps the binary surface to a single MEX file):
//   [h, setup_seconds]         = apxchol_mex('factorize', A)        % A sparse double, square
//   [x, iterations, resid, conv, solve_seconds] = apxchol_mex('solve', h, b, rtol, maxiter, x0)
//   z                          = apxchol_mex('apply', h, r)         % z = M^{-1} r
//   apxchol_mex('save', h, path)
//   [h, setup_seconds]         = apxchol_mex('load', path, options)
//   apxchol_mex('free', h)
//
// M-file wrappers (apxchol_solver.m / apxchol_solve.m) give the user-facing API;
// nobody should call this gateway directly.

#include "mex.h"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include "apxchol.h"
#include "apxchol/detail/csc_import.h"
#include "apxchol/operator_class.h"

namespace {

// ── error reporting ─────────────────────────────────────────────────────────
// mexErrMsgIdAndTxt must NOT be called from inside mexFunction's `try`: under
// Octave it raises an exception of its own, which the generic
// `catch (const std::exception&)` then swallows -- rewriting every documented
// identifier to `apxchol:error` and prefixing the message twice. Throw this
// instead and report it from the catch chain, where the identifier survives.
// Under MATLAB, where mexErrMsgIdAndTxt longjmps, the behaviour is the same.
struct mex_error : std::runtime_error {
    mex_error(const char* id, const std::string& msg)
        : std::runtime_error(msg), id(id) {}
    std::string id;
};

[[noreturn]] void fail(const char* id, const std::string& msg) {
    throw mex_error(id, msg);
}

double scalar_option(const mxArray* value, const char* name) {
    // Keep numeric classes, logical scalars, and sparse scalars accepted by
    // mxGetScalar; reject shapes/types before asking it to convert anything.
    const std::string message = std::string(name) +
        " must be a finite real numeric or logical scalar";
    if ((!mxIsNumeric(value) && !mxIsLogical(value)) || mxIsComplex(value) ||
        mxGetNumberOfElements(value) != 1)
        fail("apxchol:badInput", message);
    const double result = mxGetScalar(value);
    if (!std::isfinite(result))
        fail("apxchol:badInput", message);
    return result;
}

template<class Integer>
Integer integer_option(const mxArray* value, const char* name) {
    const double result = scalar_option(value, name);
    if (std::trunc(result) != result ||
        static_cast<long double>(result) < std::numeric_limits<Integer>::lowest() ||
        static_cast<long double>(result) > std::numeric_limits<Integer>::max())
        fail("apxchol:badInput", std::string(name) + " must be an integer in range");
    return static_cast<Integer>(result);
}

std::string string_option(const mxArray* value, const char* name) {
    if (!mxIsChar(value) || mxGetM(value) != 1)
        fail("apxchol:badInput", std::string(name) + " must be a character row vector");
    const mxChar* characters = mxGetChars(value);
    for (mwSize i = 0; i < mxGetNumberOfElements(value); ++i)
        if (characters[i] == 0)
            fail("apxchol:badInput", std::string(name) + " contains a null character");
    // Octave stores character arrays as UTF-8 bytes; MATLAB stores UTF-16.
    // An allocated conversion avoids sizing a byte buffer by character count.
#ifdef HAVE_OCTAVE
    char* utf8 = mxArrayToString(value);
#else
    char* utf8 = mxArrayToUTF8String(value);
#endif
    const std::unique_ptr<char, decltype(&mxFree)> converted(utf8, &mxFree);
    if (!converted)
        fail("apxchol:badInput", std::string(name) + " could not be converted to UTF-8");
    return converted.get();
}

apxchol::solve_options setup_options(const mxArray* options) {
    apxchol::solve_options result{.backend = apxchol::solve_backend::cpu};
    if (options != nullptr) {
        if (!mxIsStruct(options) || mxGetNumberOfElements(options) != 1)
            fail("apxchol:badInput", "options must be a scalar struct");
        for (int field = 0; field < mxGetNumberOfFields(options); ++field) {
            const std::string name = mxGetFieldNameByNumber(options, field);
            const mxArray* value = mxGetFieldByNumber(options, 0, field);
            if (value == nullptr) fail("apxchol:badInput", "option '" + name + "' is empty");
            if (name == "num_threads") result.num_threads = integer_option<int>(value, name.c_str());
            else if (name == "seed") result.factor_opts.seed = integer_option<unsigned>(value, name.c_str());
            else if (name == "partitioner")
                result.factor_opts.partitioner = apxchol::detail::parse_option_name(
                    apxchol::detail::partitioner_names, string_option(value, name.c_str()), name);
            else if (name == "sampler")
                result.factor_opts.sampler = apxchol::detail::parse_option_name(
                    apxchol::detail::clique_sampler_names, string_option(value, name.c_str()), name);
            else if (name == "degree_quantile") result.factor_opts.partition.degree_quantile = scalar_option(value, name.c_str());
            else if (name == "degree_multiplier") result.factor_opts.partition.degree_multiplier = scalar_option(value, name.c_str());
            else if (name == "factor_storage")
                result.factor_opts.factor_storage = apxchol::detail::parse_factor_storage(string_option(value, name.c_str()));
            else if (name == "drop_tolerance") result.factor_opts.drop_tolerance = scalar_option(value, name.c_str());
            else if (name == "regularization") result.factor_opts.regularization = scalar_option(value, name.c_str());
            else if (name == "lump_max_mass") result.factor_opts.lump_max_mass = scalar_option(value, name.c_str());
            else if (name == "verbose") {
                const double flag = scalar_option(value, name.c_str());
                if (flag != 0 && flag != 1) fail("apxchol:badInput", name + " must be logical or 0/1");
                result.factor_opts.verbose = flag != 0;
            } else fail("apxchol:badInput", "unknown option '" + name + "'");
        }
    }
    return result;
}

// ── handle store ────────────────────────────────────────────────────────────────
std::map<std::uint64_t, std::unique_ptr<apxchol::solver>>& store() {
    static std::map<std::uint64_t, std::unique_ptr<apxchol::solver>> s;
    return s;
}
std::uint64_t next_handle = 1;

void at_exit() { store().clear(); }

// Stores the solver: plhs[0] receives its handle and plhs[1], if requested, its setup time.
void store_solver(std::unique_ptr<apxchol::solver> solver, int nlhs, mxArray* plhs[]) {
    if (nlhs > 1) plhs[1] = mxCreateDoubleScalar(solver->setup_seconds());
    const std::uint64_t h = next_handle++;
    store()[h] = std::move(solver);
    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *static_cast<std::uint64_t*>(mxGetData(plhs[0])) = h;
}

apxchol::solver& get_solver(const mxArray* mh) {
    if (!mxIsUint64(mh) || mxGetNumberOfElements(mh) != 1)
        fail("apxchol:badInput", "handle must be a uint64 scalar");
    const std::uint64_t h = *static_cast<std::uint64_t*>(mxGetData(mh));
    auto it = store().find(h);
    if (it == store().end())
        fail("apxchol:badInput", "invalid or freed apxchol handle");
    return *it->second;
}

// MATLAB/Octave sparse (CSC: jc/ir/pr) -> Eigen. Same int32 scope guard as the
// Python binding (this build uses 32-bit Eigen storage indices).
Eigen::SparseMatrix<double> mx_to_eigen(const mxArray* mA) {
    if (!mxIsSparse(mA) || !mxIsDouble(mA) || mxIsComplex(mA))
        fail("apxchol:badInput", "A must be a real sparse double matrix");
    const mwSize nr = mxGetM(mA), nc = mxGetN(mA);
    if (nr != nc)
        fail("apxchol:badInput", "A must be square (got " + std::to_string(nr) +
                                  "x" + std::to_string(nc) + ")");
    const mwIndex* jc = mxGetJc(mA);
    const mwIndex* ir = mxGetIr(mA);
    const double*  pr = mxGetPr(mA);
    const std::uint64_t nnz = jc[nc];
    if (std::cmp_greater(nr, std::numeric_limits<int>::max()) ||
        std::cmp_greater(nnz, std::numeric_limits<int>::max()))
        fail("apxchol:badInput",
             "matrix too large for 32-bit indices (n or nnz exceeds 2^31)");

    return apxchol::detail::import_csc(nr, jc, ir, pr);
}

Eigen::Map<const Eigen::MatrixXd> mx_to_rhs(const mxArray* value, Eigen::Index rows,
                                           const char* name) {
    if (!mxIsDouble(value) || mxIsComplex(value) || mxIsSparse(value) ||
        mxGetNumberOfDimensions(value) != 2)
        fail("apxchol:badInput", std::string(name) + " must be a dense real double vector or matrix");
    const mwSize nr = mxGetM(value), nc = mxGetN(value);
    // Preserve row-vector input; for a one-row system, columns are separate RHSs.
    if (rows > 1 && nr == 1 && nc == static_cast<mwSize>(rows))
        return {mxGetPr(value), rows, 1};
    if (nr != static_cast<mwSize>(rows))
        fail("apxchol:badInput", std::string(name) + " must have one row per matrix row");
    return {mxGetPr(value), rows, static_cast<Eigen::Index>(nc)};
}

void solve_metadata(int nlhs, mxArray* plhs[], std::span<const apxchol::solve_result> columns,
                    double solve_seconds) {
    if (nlhs > 4) plhs[4] = mxCreateDoubleScalar(solve_seconds);
    if (nlhs > 1) plhs[1] = mxCreateDoubleMatrix(1, columns.size(), mxREAL);
    if (nlhs > 2) plhs[2] = mxCreateDoubleMatrix(1, columns.size(), mxREAL);
    if (nlhs > 3) plhs[3] = mxCreateLogicalMatrix(1, columns.size());
    for (std::size_t j = 0; j < columns.size(); ++j) {
        if (nlhs > 1) mxGetPr(plhs[1])[j] = columns[j].iterations;
        if (nlhs > 2) mxGetPr(plhs[2])[j] = columns[j].residual;
        if (nlhs > 3) mxGetLogicals(plhs[3])[j] = columns[j].converged;
    }
}

mxArray* sparse_to_mx(Eigen::SparseMatrix<double> A) {
    A.makeCompressed();
    mxArray* m = mxCreateSparse(A.rows(), A.cols(), std::max<Eigen::Index>(A.nonZeros(), 1), mxREAL);
    std::copy_n(A.outerIndexPtr(), A.cols() + 1, mxGetJc(m));
    std::copy_n(A.innerIndexPtr(), A.nonZeros(), mxGetIr(m));
    std::copy_n(A.valuePtr(), A.nonZeros(), mxGetPr(m));
    return m;
}

mxArray* vec_to_mx(const Eigen::VectorXd& v) {
    mxArray* m = mxCreateDoubleMatrix(v.size(), 1, mxREAL);
    Eigen::Map<Eigen::VectorXd>(mxGetPr(m), v.size()) = v;
    return m;
}

}  // namespace

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    static bool registered = false;
    if (!registered) { mexAtExit(at_exit); registered = true; }

    if (nrhs < 1 || !mxIsChar(prhs[0]))
        mexErrMsgIdAndTxt("apxchol:usage", "first argument must be an op string");
    char opbuf[16] = {0};
    mxGetString(prhs[0], opbuf, sizeof(opbuf));
    const std::string op(opbuf);

    try {
        if (op == "factorize") {
            if (nrhs != 2 && nrhs != 3)
                fail("apxchol:usage", "usage: h = apxchol_mex('factorize', A, options)");
            auto solver = std::make_unique<apxchol::solver>(
                mx_to_eigen(prhs[1]), setup_options(nrhs == 3 ? prhs[2] : nullptr));
            store_solver(std::move(solver), nlhs, plhs);
        } else if (op == "as_operator") {
            if (nrhs != 2)
                fail("apxchol:usage", "usage: A = apxchol_mex('as_operator', M)");
            const auto M = mx_to_eigen(prhs[1]);
            plhs[0] = apxchol::detail::is_graph_adjacency(M)
                ? sparse_to_mx(apxchol::detail::adjacency_laplacian(M))
                : mxDuplicateArray(prhs[1]);
        } else if (op == "load") {
            if (nrhs != 2 && nrhs != 3)
                fail("apxchol:usage", "usage: h = apxchol_mex('load', path, options)");
            const auto path = string_option(prhs[1], "path");
            auto options = setup_options(nrhs == 3 ? prhs[2] : nullptr);
            store_solver(std::make_unique<apxchol::solver>(
                apxchol::solver::load(path, options)), nlhs, plhs);
        } else if (op == "save") {
            if (nrhs != 3)
                fail("apxchol:usage", "usage: apxchol_mex('save', h, path)");
            get_solver(prhs[1]).save(string_option(prhs[2], "path"));
        } else if (op == "solve") {
            if (nrhs != 5 && nrhs != 6)
                fail("apxchol:usage",
                     "usage: [x,iterations,resid,conv,seconds] = apxchol_mex('solve', h, b, rtol, maxiter, x0)");
            auto& s = get_solver(prhs[1]);
            const auto B = mx_to_rhs(prhs[2], s.rows(), "b");
            std::optional<double> tol;
            std::optional<int> maxiter;
            if (!(mxIsDouble(prhs[3]) && mxIsEmpty(prhs[3]))) tol = scalar_option(prhs[3], "rtol");
            if (!(mxIsDouble(prhs[4]) && mxIsEmpty(prhs[4])))
                maxiter = integer_option<int>(prhs[4], "maxiter");
            std::optional<apxchol::const_matrix_view> X0;
            if (nrhs == 6 && !(mxIsDouble(prhs[5]) && mxIsEmpty(prhs[5]))) {
                const auto guess = mx_to_rhs(prhs[5], s.rows(), "x0");
                if (guess.cols() != B.cols())
                    fail("apxchol:badInput", "x0 must have the same shape as b");
                X0.emplace(guess);
            }
            if (B.cols() == 1) {
                Eigen::VectorXd x0;
                if (X0) x0 = X0->col(0);
                const auto result = s.solve(B.col(0), tol, maxiter, X0 ? &x0 : nullptr);
                plhs[0] = vec_to_mx(result.x);
                solve_metadata(nlhs, plhs, {&result, 1}, result.solve_seconds);
            } else {
                plhs[0] = mxCreateDoubleMatrix(B.rows(), B.cols(), mxREAL);
                Eigen::Map<Eigen::MatrixXd> X(mxGetPr(plhs[0]), B.rows(), B.cols());
                const auto result = s.solve(B, X, tol, maxiter, X0 ? &*X0 : nullptr);
                solve_metadata(nlhs, plhs, result.columns, result.solve_seconds);
            }
        } else if (op == "apply") {
            if (nrhs != 3)
                fail("apxchol:usage", "usage: z = apxchol_mex('apply', h, r)");
            auto& s = get_solver(prhs[1]);
            const auto R = mx_to_rhs(prhs[2], s.rows(), "r");
            if (R.cols() == 1) {
                plhs[0] = vec_to_mx(s.apply(R.col(0)));
            } else {
                plhs[0] = mxCreateDoubleMatrix(R.rows(), R.cols(), mxREAL);
                Eigen::Map<Eigen::MatrixXd> Z(mxGetPr(plhs[0]), R.rows(), R.cols());
                s.apply(R, Z);
            }
        } else if (op == "free") {
            if (nrhs != 2)
                fail("apxchol:usage", "usage: apxchol_mex('free', h)");
            if (mxIsUint64(prhs[1]) && mxGetNumberOfElements(prhs[1]) == 1)
                store().erase(*static_cast<std::uint64_t*>(mxGetData(prhs[1])));
        } else {
            fail("apxchol:usage", "unknown op '" + op + "'");
        }
    // mex_error FIRST: it derives from std::runtime_error, and it is the only
    // handler that keeps the identifier the thrower chose.
    } catch (const mex_error& e) {
        mexErrMsgIdAndTxt(e.id.c_str(), "%s", e.what());
    } catch (const std::invalid_argument& e) {
        mexErrMsgIdAndTxt("apxchol:badInput", "%s", e.what());
    } catch (const std::exception& e) {
        mexErrMsgIdAndTxt("apxchol:error", "%s", e.what());
    }
}
