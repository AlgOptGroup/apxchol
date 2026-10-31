#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "apxchol.h"
#include "apxchol/detail/csc_import.h"
#include "apxchol/operator_class.h"
#include "apxchol/version.h"

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
    "backend", "num_threads", "seed", "partitioner", "sampler", "keep_factor",
    "factor_storage_dtype", "degree_quantile", "degree_multiplier", "drop_tolerance",
    "regularization", "lump_max_mass", "verbose",
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

apxchol::solve_options parse_options(const py::dict& d) {
    apxchol::solve_options so{.collect_timings = false};
    for (auto item : d) {
        const std::string key = py::str(item.first).cast<std::string>();
        const py::handle v = item.second;
        try {
        // solve_options
        if      (key == "backend")           so.backend = parse_backend(v.cast<std::string>());
        else if (key == "num_threads")       so.num_threads = v.cast<int>();
        else if (key == "keep_factor")       so.keep_factor_values = v.cast<bool>();
        // factor_options
        else if (key == "seed")               so.factor_opts.seed = v.cast<unsigned>();
        else if (key == "partitioner")
            so.factor_opts.partitioner = apxchol::detail::parse_option_name(
                apxchol::detail::partitioner_names, v.cast<std::string>(), key);
        else if (key == "sampler")
            so.factor_opts.sampler = apxchol::detail::parse_option_name(
                apxchol::detail::clique_sampler_names, v.cast<std::string>(), key);
        else if (key == "degree_quantile") {
            if (!v.is_none()) so.factor_opts.partition.degree_quantile = v.cast<double>();
        }
        else if (key == "degree_multiplier")  so.factor_opts.partition.degree_multiplier = v.cast<double>();
        else if (key == "factor_storage_dtype")
            so.factor_opts.factor_storage = apxchol::detail::parse_factor_storage(v.cast<std::string>());
        else if (key == "drop_tolerance")     so.factor_opts.drop_tolerance = v.cast<double>();
        else if (key == "regularization")     so.factor_opts.regularization = v.cast<double>();
        else if (key == "lump_max_mass")       so.factor_opts.lump_max_mass = v.cast<double>();
        else if (key == "verbose")             so.factor_opts.verbose = v.cast<bool>();
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

// Python marshalling around the common solver. Factorization, solves and
// applications release the GIL once inputs are converted; a call made while
// another thread uses the same solver raises RuntimeError.
class Solver {
public:
    Solver(py::array indptr, py::array indices, py::array data,
           const py::dict& options)
        : slv_(setup(csc_to_eigen(indptr, indices, data), parse_options(options))) {}

    void save(const std::filesystem::path& path) const {
        py::gil_scoped_release release;
        slv_.save(path);
    }

    static Solver load(const std::filesystem::path& path, const py::dict& options) {
        const auto parsed = parse_options(options);
        py::gil_scoped_release release;
        return Solver(apxchol::solver::load(path, parsed));
    }

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

    using Array = py::array_t<double, py::array::forcecast>;
    using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

    py::object apply(Array r) {
        r = checked_rhs(r);
        if (r.ndim() == 1) {
            const auto rhs = r.cast<Eigen::VectorXd>();
            Eigen::VectorXd z;
            {
                py::gil_scoped_release release;
                z = slv_.apply(rhs);
            }
            return py::cast(std::move(z));
        }
        auto out = output_array(r, py::none());
        auto z = mutable_matrix(out);
        {
            py::gil_scoped_release release;
            slv_.apply(matrix(r), z);
        }
        return out;
    }

    py::dict solve(Array b, std::optional<double> tol, std::optional<int> maxiter,
                   py::object x0, py::object out_arr) {
        b = checked_rhs(b);
        auto output = output_array(b, out_arr);
        auto copy_alias = [&](Array input) {
            if (out_arr.is_none()) return input;
            if (py::cast<bool>(py::module_::import("numpy").attr("may_share_memory")(
                    input, output)))
                return input.attr("copy")().cast<Array>();
            return input;
        };
        b = copy_alias(b);
        Array guess;
        if (!x0.is_none()) {
            guess = x0.cast<Array>();
            guess = checked_rhs(guess);
            if (!same_shape(b, guess))
                throw std::invalid_argument("x0 must have the same shape as b");
            guess = copy_alias(guess);
        }

        py::dict out;
        out["x"] = output;
        if (b.ndim() == 1) {
            Eigen::VectorXd x0v;
            if (!x0.is_none()) x0v = guess.cast<Eigen::VectorXd>();
            const auto rhs = b.cast<Eigen::VectorXd>();
            Eigen::Map<Eigen::VectorXd> x(output.mutable_data(), rows());
            apxchol::solve_result res;
            {
                py::gil_scoped_release release;
                res = slv_.solve(rhs, x, tol, maxiter, x0.is_none() ? nullptr : &x0v);
            }
            out["iters"] = static_cast<int>(res.iterations);
            out["residual"] = res.residual;
            out["converged"] = res.converged;
        } else {
            apxchol::batch_solve_result res;
            auto x = mutable_matrix(output);
            if (x0.is_none()) {
                py::gil_scoped_release release;
                res = slv_.solve(matrix(b), x, tol, maxiter);
            } else {
                apxchol::const_matrix_view x0m = matrix(guess);
                py::gil_scoped_release release;
                res = slv_.solve(matrix(b), x, tol, maxiter, &x0m);
            }
            const auto count = b.shape(1);
            py::array_t<int> iters(count);
            py::array_t<double> residual(count);
            py::array_t<bool> converged(count);
            for (py::ssize_t j = 0; j < count; ++j) {
                const auto& column = res.columns[static_cast<std::size_t>(j)];
                iters.mutable_data()[j] = static_cast<int>(column.iterations);
                residual.mutable_data()[j] = column.residual;
                converged.mutable_data()[j] = column.converged;
            }
            out["iters"] = iters;
            out["residual"] = residual;
            out["converged"] = converged;
        }
        return out;
    }

    // nnz of the lower-triangular factor G (diagonal included). Survives
    // released values — the column pointers are always retained.
    std::int64_t factor_nnz() const {
        return static_cast<std::int64_t>(factor().L.nonZeros());
    }

    // (indptr, indices, data) copy of the factor in PERMUTED space. Each column
    // holds its diagonal first, then the off-diagonals in ascending row order.
    py::tuple factor_csc() const {
        const apxchol::factorization& F = factor();
        const std::int64_t nnz = static_cast<std::int64_t>(F.L.nonZeros());
        if (nnz > 0 && !F.L.has_values())
            throw std::runtime_error(
                "factor values not retained; construct with keep_factor=True");

        const std::int64_t ncol = static_cast<std::int64_t>(F.L.rows());
        py::array_t<std::int64_t> indptr(ncol + 1);
        py::array_t<std::int32_t> indices(nnz);
        py::array_t<apxchol::factor_value_t> data(nnz);
        auto* ip = indptr.mutable_data();
        auto* ii = indices.mutable_data();
        auto* dd = data.mutable_data();
        const auto outer = F.L.col_ptrs();
        const auto inner = F.L.row_indices();
        const auto vals = F.L.values();
        for (std::int64_t c = 0; c <= ncol; ++c)
            ip[c] = static_cast<std::int64_t>(outer[static_cast<std::size_t>(c)]);
        for (std::int64_t k = 0; k < nnz; ++k) {
            ii[k] = static_cast<std::int32_t>(inner[static_cast<std::size_t>(k)]);
            dd[k] = vals[static_cast<std::size_t>(k)];
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
    explicit Solver(apxchol::solver owner) : slv_(std::move(owner)) {}

    static apxchol::solver setup(const Eigen::SparseMatrix<double>& A,
                                 const apxchol::solve_options& options) {
        py::gil_scoped_release release;
        return apxchol::solver(A, options);
    }

    static bool same_shape(const py::array& a, const py::array& b) {
        return a.ndim() == b.ndim() && std::equal(a.shape(), a.shape() + a.ndim(), b.shape());
    }

    Array checked_rhs(Array a) const {
        if ((a.ndim() != 1 && a.ndim() != 2) || a.shape(0) != rows())
            throw std::invalid_argument("RHS must have shape (n,) or (n, nrhs)");
        for (py::ssize_t d = 0; d < a.ndim(); ++d) {
            if (a.strides(d) % py::ssize_t(sizeof(double)) != 0)
                throw std::invalid_argument("RHS strides must be multiples of float64 item size");
            // Eigen::Ref interprets a zero stride as contiguous, not broadcast.
            if (a.size() && a.shape(d) > 1 && a.strides(d) == 0)
                return a.attr("copy")().cast<Array>();
        }
        return a;
    }

    static Array output_array(const Array& rhs, const py::object& out) {
        if (out.is_none())
            return Array(std::vector<py::ssize_t>(rhs.shape(), rhs.shape() + rhs.ndim()));
        if (!py::isinstance<py::array_t<double>>(out))
            throw std::invalid_argument("out must be a float64 NumPy array");
        auto a = out.cast<Array>();
        if (!same_shape(a, rhs) || !a.writeable() ||
            !(a.flags() & (py::array::c_style | py::array::f_style)))
            throw std::invalid_argument(
                "out must be writable, C- or F-contiguous, and have the same shape as b");
        return a;
    }

    static Eigen::Map<const Eigen::MatrixXd, Eigen::Unaligned, Stride>
    matrix(const Array& a) {
        return Eigen::Map<const Eigen::MatrixXd, Eigen::Unaligned, Stride>(
            a.data(), a.shape(0), a.shape(1),
            Stride(a.strides(1) / py::ssize_t(sizeof(double)), a.strides(0) / py::ssize_t(sizeof(double))));
    }

    static Eigen::Map<Eigen::MatrixXd, Eigen::Unaligned, Stride>
    mutable_matrix(Array& a) {
        return Eigen::Map<Eigen::MatrixXd, Eigen::Unaligned, Stride>(
            a.mutable_data(), a.shape(0), a.shape(1),
            Stride(a.strides(1) / py::ssize_t(sizeof(double)), a.strides(0) / py::ssize_t(sizeof(double))));
    }

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
    m.attr("__version__") = APXCHOL_VERSION;
    m.attr("DEFAULT_TOL") = apxchol::default_tol;
    m.attr("DEFAULT_MAXITER") = apxchol::default_max_iter;
    // Returns None when the input is already a valid operator, else the CSC
    // arrays of its graph Laplacian; throws for anything else.
    m.def("as_operator",
          [](py::array indptr, py::array indices, py::array data) -> py::object {
              const auto M = csc_to_eigen(std::move(indptr), std::move(indices), std::move(data));
              if (!apxchol::detail::is_graph_adjacency(M))
                  return py::none();
              Eigen::SparseMatrix<double> L = apxchol::detail::adjacency_laplacian(M);
              L.makeCompressed();
              using Index = Eigen::SparseMatrix<double>::StorageIndex;
              return py::make_tuple(
                  py::array_t<Index>(L.outerSize() + 1, L.outerIndexPtr()),
                  py::array_t<Index>(L.nonZeros(), L.innerIndexPtr()),
                  py::array_t<double>(L.nonZeros(), L.valuePtr()));
          },
          py::arg("indptr"), py::arg("indices"), py::arg("data"));
    py::class_<Solver>(m, "Solver")
        .def(py::init<py::array, py::array, py::array, const py::dict&>(),
             py::arg("indptr"), py::arg("indices"), py::arg("data"),
             py::arg("options") = py::dict())
        .def("__copy__", [](const Solver& self) { return Solver(self); })
        .def("save", &Solver::save, py::arg("path"))
        .def_static("load", &Solver::load, py::arg("path"),
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
