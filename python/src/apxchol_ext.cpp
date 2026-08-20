#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "apxchol.h"

namespace py = pybind11;

namespace {

// scipy CSC (indptr, indices, data) -> column-major Eigen::SparseMatrix<double>.
// forcecast accepts int32/int64 index arrays and any float data dtype.
Eigen::SparseMatrix<double> csc_to_eigen(py::array indptr, py::array indices,
                                         py::array data, Eigen::Index n) {
    auto ip = indptr.cast<py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>>();
    auto ii = indices.cast<py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>>();
    auto dd = data.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    const std::int64_t* P = ip.data();
    const std::int64_t* I = ii.data();
    const double*        V = dd.data();
    const Eigen::Index ncol = static_cast<Eigen::Index>(ip.size()) - 1;
    if (ncol != n)
        throw std::invalid_argument("indptr length must be n+1");

    // This build uses 32-bit Eigen storage indices; matrices with n or nnz beyond
    // 2^31 would silently truncate below. Reject them with a clear message.
    const std::int64_t int_max = std::numeric_limits<int>::max();
    if (n > int_max || dd.size() > int_max)
        throw std::invalid_argument(
            "matrix too large for 32-bit indices (n or nnz exceeds 2^31); "
            "this CPU package build does not support 64-bit indices");

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<std::size_t>(dd.size()));
    for (Eigen::Index j = 0; j < ncol; ++j)
        for (std::int64_t p = P[j]; p < P[j + 1]; ++p)
            trips.emplace_back(static_cast<int>(I[p]), static_cast<int>(j), V[p]);

    Eigen::SparseMatrix<double> A(n, n);
    A.setFromTriplets(trips.begin(), trips.end());
    A.makeCompressed();
    // No input-class check here: the library asserts the operator contract
    // itself, inside the factorization, from ONE implementation shared with
    // the CLI and the Octave binding (apxchol/operator_class.h). A violation
    // arrives as std::invalid_argument -> ValueError, naming which condition
    // failed; an adjacency matrix is caught by the positive-diagonal condition
    // and the message names apxchol.laplacian(A).
    return A;
}

// ── options dict -> solve_options ────────────────────────────────────────────
// Every key the binding understands, in the order reported by the error message.
const char* const kValidKeys[] = {
    "seed", "partitioner", "storage", "keep_factor",
    "degree_quantile", "degree_multiplier", "degree_tiebreak",
    "exact_clique_max_degree",
    "residual_peel", "stagnation_window",
};

std::string join(const char* const* items, std::size_t count) {
    std::string out;
    for (std::size_t i = 0; i < count; ++i) {
        if (i) out += ", ";
        out += items[i];
    }
    return out;
}

apxchol::graph_storage parse_storage(const std::string& name) {
    if (name == "vec_pool")     return apxchol::graph_storage::vec_pool;
    if (name == "forward_star") return apxchol::graph_storage::forward_star;
    if (name == "vec")          return apxchol::graph_storage::vec;
    if (name == "bstr")         return apxchol::graph_storage::bstr;
    throw std::invalid_argument(
        "unknown storage '" + name +
        "'; valid: vec_pool, forward_star, vec, bstr");
}

apxchol::residual_peel_strategy parse_peel(const std::string& name) {
    if (name == "natural")    return apxchol::residual_peel_strategy::natural;
    if (name == "min_degree") return apxchol::residual_peel_strategy::min_degree;
    if (name == "bk_serial")  return apxchol::residual_peel_strategy::bk_serial;
    throw std::invalid_argument(
        "unknown residual_peel '" + name +
        "'; valid: natural, min_degree, bk_serial");
}

apxchol::solve_options parse_options(const py::dict& d) {
    apxchol::solve_options so;
    for (auto item : d) {
        const std::string key = py::str(item.first).cast<std::string>();
        const py::handle v = item.second;
        try {
        // solve_options
        if      (key == "keep_factor")       so.keep_factor_values = v.cast<bool>();
        else if (key == "storage")           so.storage = parse_storage(v.cast<std::string>());
        else if (key == "stagnation_window") so.stagnation_window = v.cast<int>();
        // factor_options
        else if (key == "seed")               so.factor_opts.seed = v.cast<unsigned>();
        else if (key == "partitioner")        so.factor_opts.is_select = v.cast<std::string>();
        else if (key == "degree_quantile")    so.factor_opts.partition.degree_quantile = v.cast<double>();
        else if (key == "degree_multiplier")  so.factor_opts.partition.degree_multiplier = v.cast<double>();
        else if (key == "degree_tiebreak")    so.factor_opts.partition.degree_tiebreak = v.cast<bool>();
        else if (key == "exact_clique_max_degree")
            so.factor_opts.exact_clique_max_degree = v.cast<std::size_t>();
        else if (key == "residual_peel")
            so.factor_opts.residual_peel = parse_peel(v.cast<std::string>());
        else
            throw std::invalid_argument(
                "unknown option '" + key + "'; valid keys: " +
                join(kValidKeys, sizeof(kValidKeys) / sizeof(*kValidKeys)));
        } catch (const py::cast_error&) {
            throw std::invalid_argument(
                "invalid value for option '" + key + "': cannot convert " +
                std::string(py::str(py::type::of(item.second.cast<py::object>())
                                        .attr("__name__"))) +
                " to the expected type");
        } catch (const py::error_already_set& e) {
            throw std::invalid_argument(
                "invalid value for option '" + key + "': " + std::string(e.what()));
        }
    }
    return so;
}

// Reusable CPU solver: thin marshalling shell over apxchol::cpu_solver (the
// library's factor-once / solve-many class — the SAME parallel-SpMV PCG the
// benchmark's one-shot solve() runs, so package and bench numbers match).
class Solver {
public:
    Solver(py::array indptr, py::array indices, py::array data, Eigen::Index n,
           const py::dict& options)
        : slv_(csc_to_eigen(indptr, indices, data, n), parse_options(options)) {}

    Eigen::Index rows() const { return slv_.rows(); }
    bool sddm() const { return factor().sddm; }
    // Positive off-diagonal entries M-matrix lumping moved onto the diagonal
    // while building the preconditioner; 0 for a Laplacian/SDDM operator.
    std::int64_t lumped() const {
        return static_cast<std::int64_t>(factor().lumped_offdiag);
    }

    // M^{-1} r (one forward/back SpTRSV; reused tuned factor).
    Eigen::VectorXd apply(const Eigen::VectorXd& r) const {
        if (r.size() != slv_.rows())
            throw std::invalid_argument("apply: r length mismatch");
        return slv_.apply(r);
    }

    py::dict solve(const Eigen::VectorXd& b, double tol, int maxiter,
                   py::object x0, py::object out_arr) const {
        if (b.size() != slv_.rows())
            throw std::invalid_argument("solve: b length mismatch");
        Eigen::VectorXd x0v;
        const Eigen::VectorXd* x0p = nullptr;
        if (!x0.is_none()) {
            x0v = x0.cast<Eigen::VectorXd>();
            if (x0v.size() != slv_.rows())
                throw std::invalid_argument("solve: x0 length mismatch");
            x0p = &x0v;
        }

        // Caller-provided output memory: solve straight into the numpy array
        // (no per-solve solution allocation; the array must alias no copy).
        if (!out_arr.is_none()) {
            auto a = py::cast<py::array>(out_arr);
            const bool ok = py::isinstance<py::array_t<double>>(a) &&
                            a.ndim() == 1 && a.shape(0) == slv_.rows() &&
                            (a.flags() & py::array::c_style) && a.writeable();
            if (!ok)
                throw std::invalid_argument(
                    "solve: out must be a writable C-contiguous float64 array "
                    "of length n");
            Eigen::Map<Eigen::VectorXd> xmap(
                static_cast<double*>(a.mutable_data()), slv_.rows());
            const apxchol::solve_result res = slv_.solve(b, xmap, tol, maxiter, x0p);
            py::dict out;
            out["x"]         = a;
            out["iters"]     = static_cast<int>(res.iterations);
            out["residual"]  = res.residual;
            out["converged"] = res.residual < tol;
            return out;
        }

        const apxchol::solve_result res = slv_.solve(b, tol, maxiter, x0p);
        py::dict out;
        out["x"]         = Eigen::VectorXd(res.x);
        out["iters"]     = static_cast<int>(res.iterations);
        out["residual"]  = res.residual;
        // The PCG breaks with residual < tol on convergence; every other exit
        // (maxiter, stagnation, indefinite pAp, zero-b shortcut) leaves it >= tol
        // except b == 0, where residual 0 < tol is correct (x = 0 is exact).
        out["converged"] = res.residual < tol;
        return out;
    }

    // nnz of the lower-triangular factor G (diagonal included). Survives
    // release_values() — the column pointers are always retained.
    std::int64_t factor_nnz() const {
        return static_cast<std::int64_t>(factor().L.nonZeros());
    }

    // (indptr, indices, data) copy of the factor in PERMUTED space. Each column
    // holds its diagonal first, then the off-diagonals in ascending row order.
    py::tuple factor_csc() const {
        const apxchol::factorization& F = factor();
        const std::int64_t nnz = static_cast<std::int64_t>(F.L.nonZeros());
        if (nnz > 0 && F.L.vals_.empty())
            throw std::runtime_error(
                "factor values not retained; construct with keep_factor=True");

        const std::int64_t ncol = static_cast<std::int64_t>(F.L.rows());
        py::array_t<std::int64_t> indptr(ncol + 1);
        py::array_t<std::int32_t> indices(nnz);
        py::array_t<double>       data(nnz);
        auto* ip = indptr.mutable_data();
        auto* ii = indices.mutable_data();
        auto* dd = data.mutable_data();
        for (std::int64_t c = 0; c <= ncol; ++c)
            ip[c] = static_cast<std::int64_t>(F.L.outer_[static_cast<std::size_t>(c)]);
        for (std::int64_t k = 0; k < nnz; ++k) {
            ii[k] = static_cast<std::int32_t>(F.L.inner_[static_cast<std::size_t>(k)]);
            dd[k] = static_cast<double>(F.L.vals_[static_cast<std::size_t>(k)]);
        }
        return py::make_tuple(indptr, indices, data);
    }

    // perm[original_vertex] = position in the elimination order.
    py::array_t<std::int64_t> perm() const {
        const std::vector<apxchol::node_index>& p = factor().perm;
        py::array_t<std::int64_t> out(static_cast<py::ssize_t>(p.size()));
        auto* o = out.mutable_data();
        for (std::size_t i = 0; i < p.size(); ++i)
            o[i] = static_cast<std::int64_t>(p[i]);
        return out;
    }

private:
    const apxchol::factorization& factor() const {
        return slv_.preconditioner().factor();
    }

    apxchol::cpu_solver slv_;
};

}  // namespace

PYBIND11_MODULE(_apxchol, m) {
    m.doc() = "apxchol CPU approximate-Cholesky preconditioner (pybind11 binding)";
    py::class_<Solver>(m, "Solver")
        .def(py::init<py::array, py::array, py::array, Eigen::Index, const py::dict&>(),
             py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("n"),
             py::arg("options") = py::dict())
        .def("solve", &Solver::solve, py::arg("b"), py::arg("tol"), py::arg("maxiter"),
             py::arg("x0") = py::none(), py::arg("out") = py::none())
        .def("apply", &Solver::apply, py::arg("r"))
        .def("rows", &Solver::rows)
        .def("sddm", &Solver::sddm)
        .def("lumped", &Solver::lumped)
        .def("factor_nnz", &Solver::factor_nnz)
        .def("factor_csc", &Solver::factor_csc)
        .def("perm", &Solver::perm);
}
