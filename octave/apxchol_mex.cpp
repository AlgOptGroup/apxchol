// apxchol_mex.cpp — Octave/MATLAB MEX gateway for the apxchol CPU solver.
//
// One gateway, op-dispatched (keeps the binary surface to a single MEX file):
//   h                          = apxchol_mex('factorize', A)        % A sparse double, square
//   [x, iters, resid, conv]    = apxchol_mex('solve', h, b, tol, maxiter)
//   z                          = apxchol_mex('apply', h, r)         % z = M^{-1} r
//   apxchol_mex('free', h)
//
// The Solver class is a thin shell over apxchol::cpu_solver and MIRRORS
// python/src/apxchol_ext.cpp — keep the two in sync when touching either.
// M-file wrappers (apxchol_solver.m / apxchol_solve.m) give the user-facing API;
// nobody should call this gateway directly.

#include "mex.h"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "apxchol.h"
#include "mtx_input.h"

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

// Reject the one input class that is UNAMBIGUOUSLY not an operator: a graph
// adjacency matrix. Handed to the core it would negate every edge weight,
// `sample_clique` would early-return on the negative weighted degree, and the
// caller would get a zero-fill factor and a PCG that breaks down before its
// first update -- surfacing only as `converged == false`, with no diagnosis.
//
// This is an ERROR, not a silent conversion. The CLI can auto-convert because
// it prints how it read the file on every run; a library caller does not
// reliably see anything we print, so converting would risk handing back a
// confident answer to a DIFFERENT system than the one they believe they asked
// about. `apxchol_laplacian(A)` makes the conversion explicit instead.
//
// Narrow by construction (see `apxchol::adjacency_signature`, shared with the
// CLI and the Python binding): it fires only when NOT ONE row carries a
// positive diagonal entry while positive off-diagonals exist. Mixed-sign
// FEM/structural operators keep working.
void reject_adjacency_input(const Eigen::SparseMatrix<double>& A) {
    const apxchol::adjacency_signature sig = apxchol::detect_adjacency_signature(A);
    if (!sig.detected()) return;
    fail("apxchol:adjacencyInput",
         "A looks like a graph adjacency matrix, not an assembled operator: "
         "none of the " + std::to_string(sig.n) +
         " rows carries a positive diagonal entry, while " +
         std::to_string(sig.positive_offdiag) +
         " off-diagonal entries are positive. apxchol solves the assembled "
         "Laplacian/SDDM operator, whose off-diagonals are non-positive by "
         "definition; an adjacency matrix would be read with every edge weight "
         "negated, giving a zero-fill factor and a PCG that cannot take a step. "
         "Pass the assembled operator, or convert an adjacency matrix with "
         "apxchol_laplacian(A) (equivalently "
         "L = spdiags(sum(abs(A), 2), 0, n, n) - abs(A)).");
}

// ── reusable solver: thin shell over apxchol::cpu_solver (mirrors the Python
// binding) — the library's factor-once/solve-many class with the SAME
// parallel-SpMV PCG the benchmark's one-shot solve() runs.
class Solver {
public:
    explicit Solver(Eigen::SparseMatrix<double> A) : slv_(A) {}
        // default solve_options: vec_pool storage + block_greedy (the library defaults)

    Eigen::Index rows() const { return slv_.rows(); }

    Eigen::VectorXd apply(const Eigen::VectorXd& r) const {
        if (r.size() != slv_.rows()) throw std::invalid_argument("apply: r length mismatch");
        return slv_.apply(r);
    }

    void solve(const Eigen::VectorXd& b, double tol, int maxiter,
               Eigen::VectorXd& x, int& iters, double& rnorm, bool& converged) const {
        if (b.size() != slv_.rows()) throw std::invalid_argument("solve: b length mismatch");
        const apxchol::solve_result res = slv_.solve(b, tol, maxiter);
        x = res.x; iters = static_cast<int>(res.iterations); rnorm = res.residual;
        // residual < tol iff the PCG converged (b == 0 leaves residual 0 with
        // the exact x = 0); all other exits leave it >= tol.
        converged = res.residual < tol;
    }

private:
    apxchol::cpu_solver slv_;
};

// ── handle store ────────────────────────────────────────────────────────────────
std::map<std::uint64_t, std::unique_ptr<Solver>>& store() {
    static std::map<std::uint64_t, std::unique_ptr<Solver>> s;
    return s;
}
std::uint64_t next_handle = 1;

void at_exit() { store().clear(); }

Solver& get_solver(const mxArray* mh) {
    if (!mxIsUint64(mh) || mxGetNumberOfElements(mh) != 1)
        fail("apxchol:badHandle", "handle must be a uint64 scalar");
    const std::uint64_t h = *static_cast<std::uint64_t*>(mxGetData(mh));
    auto it = store().find(h);
    if (it == store().end())
        fail("apxchol:badHandle", "invalid or freed apxchol handle");
    return *it->second;
}

// MATLAB/Octave sparse (CSC: jc/ir/pr) -> Eigen. Same int32 scope guard as the
// Python binding (this build uses 32-bit Eigen storage indices).
Eigen::SparseMatrix<double> mx_to_eigen(const mxArray* mA) {
    if (!mxIsSparse(mA) || !mxIsDouble(mA) || mxIsComplex(mA))
        fail("apxchol:badMatrix", "A must be a real sparse double matrix");
    const mwSize nr = mxGetM(mA), nc = mxGetN(mA);
    if (nr != nc)
        fail("apxchol:badMatrix", "A must be square (got " + std::to_string(nr) +
                                  "x" + std::to_string(nc) + ")");
    const mwIndex* jc = mxGetJc(mA);
    const mwIndex* ir = mxGetIr(mA);
    const double*  pr = mxGetPr(mA);
    const std::uint64_t nnz = jc[nc];
    if (nr > (std::uint64_t)std::numeric_limits<int>::max() ||
        nnz > (std::uint64_t)std::numeric_limits<int>::max())
        fail("apxchol:tooLarge",
             "matrix too large for 32-bit indices (n or nnz exceeds 2^31)");

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(nnz);
    for (mwSize j = 0; j < nc; ++j)
        for (mwIndex p = jc[j]; p < jc[j + 1]; ++p)
            trips.emplace_back((int)ir[p], (int)j, pr[p]);

    Eigen::SparseMatrix<double> A(nr, nc);
    A.setFromTriplets(trips.begin(), trips.end());
    A.makeCompressed();
    // After assembly, so duplicate (i, j) entries have been summed and the
    // signs tested are the ones the solver will actually see.
    reject_adjacency_input(A);
    return A;
}

Eigen::VectorXd mx_to_vec(const mxArray* mb, const char* what) {
    if (!mxIsDouble(mb) || mxIsComplex(mb) || mxIsSparse(mb))
        fail("apxchol:badVector",
             std::string(what) + " must be a dense real double vector");
    const mwSize n = mxGetNumberOfElements(mb);
    return Eigen::Map<const Eigen::VectorXd>(mxGetPr(mb), n);
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
            if (nrhs != 2)
                fail("apxchol:usage", "usage: h = apxchol_mex('factorize', A)");
            auto solver = std::make_unique<Solver>(mx_to_eigen(prhs[1]));
            const std::uint64_t h = next_handle++;
            store()[h] = std::move(solver);
            plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
            *static_cast<std::uint64_t*>(mxGetData(plhs[0])) = h;
        } else if (op == "solve") {
            if (nrhs != 5)
                fail("apxchol:usage",
                     "usage: [x,iters,resid,conv] = apxchol_mex('solve', h, b, tol, maxiter)");
            Solver& s = get_solver(prhs[1]);
            const Eigen::VectorXd b = mx_to_vec(prhs[2], "b");
            const double tol = mxGetScalar(prhs[3]);
            const int maxiter = (int)mxGetScalar(prhs[4]);
            Eigen::VectorXd x; int iters; double rnorm; bool conv;
            s.solve(b, tol, maxiter, x, iters, rnorm, conv);
            plhs[0] = vec_to_mx(x);
            if (nlhs > 1) plhs[1] = mxCreateDoubleScalar(iters);
            if (nlhs > 2) plhs[2] = mxCreateDoubleScalar(rnorm);
            if (nlhs > 3) plhs[3] = mxCreateLogicalScalar(conv);
        } else if (op == "apply") {
            if (nrhs != 3)
                fail("apxchol:usage", "usage: z = apxchol_mex('apply', h, r)");
            Solver& s = get_solver(prhs[1]);
            plhs[0] = vec_to_mx(s.apply(mx_to_vec(prhs[2], "r")));
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
