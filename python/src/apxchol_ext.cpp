#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "apxchol.h"
#include "apxchol/detail/csc_import.h"

namespace py = pybind11;

namespace {

template<class Import>
auto with_csc_indices(const py::array& array, Import import) {
    if (py::isinstance<py::array_t<std::int32_t>>(array))
        return import(array.cast<py::array_t<std::int32_t, py::array::c_style>>());
    return import(array.cast<py::array_t<std::int64_t,
                                       py::array::c_style | py::array::forcecast>>());
}

// CSC arrays -> column-major Eigen::SparseMatrix<double>.
Eigen::SparseMatrix<double> csc_to_eigen(py::array indptr, py::array indices,
                                         py::array data) {
    if (indptr.ndim() != 1 || indices.ndim() != 1 || data.ndim() != 1)
        throw std::invalid_argument("CSC data, indices and indptr must be one-dimensional");
    if (indptr.size() == 0)
        throw std::invalid_argument("CSC indptr must contain at least its initial zero");

    auto dd = data.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    if (indices.size() != dd.size())
        throw std::invalid_argument("CSC data and indices must have the same length");
    return with_csc_indices(indptr, [&](const auto& ip) {
        const auto n = ip.size() - 1;
        if (ip.at(n) != dd.size())
            throw std::invalid_argument("CSC indptr must end at the length of data and indices");
        return with_csc_indices(indices, [&](const auto& ii) {
            return apxchol::detail::import_csc(n, ip.data(), ii.data(), dd.data());
        });
    });
}

// ── options dict -> solve_options ────────────────────────────────────────────
// Every key the binding understands, in the order reported by the error message.
const char* const kValidKeys[] = {
    "backend", "seed", "partitioner", "keep_factor", "factor_storage_dtype",
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

apxchol::solve_backend parse_backend(const std::string& name) {
    if (name == "auto") return apxchol::solve_backend::automatic;
    if (name == "cpu") return apxchol::solve_backend::cpu;
    if (name == "gpu") return apxchol::solve_backend::gpu;
    throw std::invalid_argument("backend must be auto, cpu or gpu");
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
        if      (key == "backend")           so.backend = parse_backend(v.cast<std::string>());
        else if (key == "keep_factor")       so.keep_factor_values = v.cast<bool>();
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
        else if (key == "factor_storage_dtype")
            so.factor_opts.factor_storage = apxchol::detail::parse_factor_storage(v.cast<std::string>());
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

// Python marshalling around the common solver.
class Solver {
public:
    Solver(py::array indptr, py::array indices, py::array data,
           const py::dict& options)
        : slv_(csc_to_eigen(indptr, indices, data), parse_options(options)) {}

    Eigen::Index rows() const { return slv_.rows(); }
    const char* backend() const {
        return slv_.backend() == apxchol::solve_backend::cpu ? "cpu" : "gpu";
    }
    bool sddm() const { return factor().sddm; }
    const char* factor_storage_dtype() const {
        return slv_.factor_storage() == apxchol::factor_storage_type::fp16
            ? "float16" : "float32";
    }
    // Positive off-diagonal entries M-matrix lumping moved onto the diagonal
    // while building the preconditioner; 0 for a Laplacian/SDDM operator.
    std::int64_t lumped() const {
        return static_cast<std::int64_t>(factor().lumped_offdiag);
    }

    // M^{-1} r (one forward/back SpTRSV; reused tuned factor).
    Eigen::VectorXd apply(const Eigen::VectorXd& r) const {
        return slv_.apply(r);
    }

    py::dict solve(const Eigen::VectorXd& b, double tol, int maxiter,
                   py::object x0, py::object out_arr) const {
        if (tol < 0.0) tol = apxchol::default_tol;
        Eigen::VectorXd x0v;
        const Eigen::VectorXd* x0p = nullptr;
        if (!x0.is_none()) {
            x0v = x0.cast<Eigen::VectorXd>();
            x0p = &x0v;
        }

        apxchol::solve_result res;
        py::dict out;
        // Preserve the caller's output array; GPU solves may use temporary storage.
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
            res = slv_.solve(b, xmap, tol, maxiter, x0p);
            out["x"] = a;
        } else {
            res = slv_.solve(b, tol, maxiter, x0p);
            out["x"] = std::move(res.x);
        }

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
        py::array_t<apxchol::factor_value_t> data(nnz);
        auto* ip = indptr.mutable_data();
        auto* ii = indices.mutable_data();
        auto* dd = data.mutable_data();
        for (std::int64_t c = 0; c <= ncol; ++c)
            ip[c] = static_cast<std::int64_t>(F.L.outer_[static_cast<std::size_t>(c)]);
        for (std::int64_t k = 0; k < nnz; ++k) {
            ii[k] = static_cast<std::int32_t>(F.L.inner_[static_cast<std::size_t>(k)]);
            dd[k] = F.L.vals_[static_cast<std::size_t>(k)];
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
        return slv_.factor();
    }

    apxchol::solver slv_;
};

}  // namespace

PYBIND11_MODULE(_apxchol, m) {
#ifdef APXCHOL_REQUIRE_X86_64_V2
    if (!__builtin_cpu_supports("x86-64-v2")) {
        throw py::import_error("apxchol requires an x86-64-v2 CPU.");
    }
#endif
    m.doc() = "apxchol approximate-Cholesky preconditioner (pybind11 binding)";
    py::class_<Solver>(m, "Solver")
        .def(py::init<py::array, py::array, py::array, const py::dict&>(),
             py::arg("indptr"), py::arg("indices"), py::arg("data"),
             py::arg("options") = py::dict())
        .def("solve", &Solver::solve, py::arg("b"), py::arg("tol"), py::arg("maxiter"),
             py::arg("x0") = py::none(), py::arg("out") = py::none())
        .def("apply", &Solver::apply, py::arg("r"))
        .def("rows", &Solver::rows)
        .def("backend", &Solver::backend)
        .def("sddm", &Solver::sddm)
        .def("factor_storage_dtype", &Solver::factor_storage_dtype)
        .def("lumped", &Solver::lumped)
        .def("factor_nnz", &Solver::factor_nnz)
        .def("factor_csc", &Solver::factor_csc)
        .def("perm", &Solver::perm);
}
