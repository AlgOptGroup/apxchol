#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "apxchol.h"
#include "apxchol/detail/csc_import.h"
#include "apxchol/detail/csc_input.h"
#include "apxchol/solver/detail/solver_route.h"
#include "apxchol/operator_class.h"
#include "apxchol/version.h"

namespace py = pybind11;

namespace {

template<class Index>
using index_array = py::array_t<Index, py::array::c_style | py::array::forcecast>;

// NumPy CSC arrays as a solver input. Contiguous float64 values and int32 or
// int64 index arrays of one width are borrowed when the columns are sorted;
// anything else is converted once (apxchol::detail::imported_csc).
class csc_arrays {
public:
    csc_arrays(const py::array& indptr, const py::array& indices, const py::array& data) {
        if (indptr.ndim() != 1 || indices.ndim() != 1 || data.ndim() != 1)
            throw std::invalid_argument("CSC data, indices and indptr must be one-dimensional");
        if (indptr.size() == 0)
            throw std::invalid_argument("CSC indptr must contain at least its initial zero");
        values_ = data.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        if (indices.size() != values_.size())
            throw std::invalid_argument("CSC data and indices must have the same length");
        if (py::isinstance<py::array_t<std::int32_t>>(indptr) &&
            py::isinstance<py::array_t<std::int32_t>>(indices))
            import<std::int32_t>(indptr, indices);
        else
            import<std::int64_t>(indptr, indices);
    }

    template<class F>
    decltype(auto) visit(F f) const {
        return std::visit([&](const auto& imported) { return f(imported->view()); }, imported_);
    }

private:
    template<class Index>
    void import(const py::array& indptr, const py::array& indices) {
        auto ip = indptr.cast<index_array<Index>>();
        auto ii = indices.cast<index_array<Index>>();
        const auto n = ip.size() - 1;
        if (ip.at(n) != values_.size())
            throw std::invalid_argument("CSC indptr must end at the length of data and indices");
        imported_ = std::make_unique<apxchol::detail::imported_csc<Index>>(
            n, ip.data(), ii.data(), values_.data());
        indptr_ = std::move(ip);
        indices_ = std::move(ii);
    }

    py::array_t<double> values_;
    py::array indptr_, indices_;
    std::variant<std::unique_ptr<apxchol::detail::imported_csc<std::int32_t>>,
                 std::unique_ptr<apxchol::detail::imported_csc<std::int64_t>>> imported_;
};

// An owned CSC matrix as NumPy arrays, with int32 indices when they fit.
py::tuple csc_to_numpy(const apxchol::detail::csc_matrix<double>& A) {
    const auto n = static_cast<py::ssize_t>(A.rows());
    const auto nnz = static_cast<py::ssize_t>(A.ptr()[A.rows()]);
    const auto as = [&](auto index) {
        using Index = decltype(index);
        py::array_t<Index> indptr(n + 1), indices(nnz);
        std::copy_n(A.ptr(), n + 1, indptr.mutable_data());
        std::copy_n(A.idx(), nnz, indices.mutable_data());
        return py::make_tuple(indptr, indices, py::array_t<double>(nnz, A.val()));
    };
    if (std::cmp_less_equal(nnz, std::numeric_limits<std::int32_t>::max()) &&
        std::cmp_less_equal(n, std::numeric_limits<std::int32_t>::max()))
        return as(std::int32_t{});
    return as(std::int64_t{});
}

// ── options dict -> solve_options ────────────────────────────────────────────
// Every key the binding understands, in the order reported by the error message.
const char* const kValidKeys[] = {
    "backend", "num_threads", "seed", "partitioner", "sampler",
    "factor_storage", "degree_quantile", "degree_multiplier", "drop_tolerance",
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
    apxchol::solve_options so;
    for (auto item : d) {
        const std::string key = py::str(item.first).cast<std::string>();
        const py::handle v = item.second;
        try {
        // solve_options
        if      (key == "backend")           so.backend = parse_backend(v.cast<std::string>());
        else if (key == "num_threads")       so.num_threads = v.cast<int>();
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
        else if (key == "factor_storage")
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
        : slv_(setup(csc_arrays(indptr, indices, data), parse_options(options))) {}

    void save(const std::filesystem::path& path) const {
        py::gil_scoped_release release;
        slv_.save(path);
    }

    static Solver load(const std::filesystem::path& path, const py::dict& options) {
        const auto parsed = parse_options(options);
        py::gil_scoped_release release;
        return Solver(apxchol::solver::load(path, parsed));
    }

    std::int64_t rows() const { return slv_.rows(); }
    double setup_seconds() const { return slv_.setup_seconds(); }
    const char* backend() const {
        return slv_.backend() == apxchol::solve_backend::cpu ? "cpu" : "gpu";
    }
    bool sddm() const { return factor().sddm; }
    const char* factor_storage() const {
        return slv_.factor_storage() == apxchol::factor_storage_type::fp16
            ? "float16" : "float32";
    }
    // Positive off-diagonal entries M-matrix lumping moved onto the diagonal
    // while building the preconditioner; 0 for a Laplacian/SDDM operator.
    std::int64_t lumped_offdiag() const {
        return static_cast<std::int64_t>(factor().lumped_offdiag);
    }

    using Array = py::array_t<double, py::array::forcecast>;
    // Column-major storage the native views accept; contiguous vectors qualify.
    using Fortran = py::array_t<double, py::array::f_style | py::array::forcecast>;

    py::object apply(Array r) {
        r = checked_rhs(r);
        const Fortran input = Fortran::ensure(r);
        if (r.ndim() == 1) {
            std::vector<double> z;
            {
                py::gil_scoped_release release;
                z = slv_.apply(vector(input));
            }
            return py::array_t<double>(static_cast<py::ssize_t>(z.size()), z.data());
        }
        Fortran out(std::vector<py::ssize_t>(r.shape(), r.shape() + r.ndim()));
        auto z = mutable_matrix(out);
        {
            py::gil_scoped_release release;
            slv_.apply(matrix(input), z);
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
        const Fortran rhs = Fortran::ensure(b);
        const Fortran initial = x0.is_none() ? Fortran() : Fortran::ensure(guess);
        // A C-ordered matrix output is solved into column-major scratch and copied back.
        Fortran solution = Fortran::ensure(output);

        py::dict out;
        out["x"] = output;
        if (b.ndim() == 1) {
            apxchol::solve_result res;
            {
                py::gil_scoped_release release;
                res = slv_.solve(vector(rhs), mutable_vector(solution), tol, maxiter,
                                 x0.is_none() ? std::span<const double>() : vector(initial));
            }
            out["iterations"] = static_cast<int>(res.iterations);
            out["residual"] = res.residual;
            out["converged"] = res.converged;
            out["solve_seconds"] = res.solve_seconds;
        } else {
            apxchol::batch_solve_result res;
            {
                py::gil_scoped_release release;
                res = slv_.solve(matrix(rhs), mutable_matrix(solution), tol, maxiter,
                                 x0.is_none() ? apxchol::const_matrix_view{} : matrix(initial));
            }
            const auto count = b.shape(1);
            py::array_t<int> iterations(count);
            py::array_t<double> residual(count);
            py::array_t<bool> converged(count);
            for (py::ssize_t j = 0; j < count; ++j) {
                const auto& column = res.columns[static_cast<std::size_t>(j)];
                iterations.mutable_data()[j] = static_cast<int>(column.iterations);
                residual.mutable_data()[j] = column.residual;
                converged.mutable_data()[j] = column.converged;
            }
            out["iterations"] = iterations;
            out["residual"] = residual;
            out["converged"] = converged;
            out["solve_seconds"] = res.solve_seconds;
        }
        if (!solution.is(output)) output[py::ellipsis()] = solution;
        return out;
    }

    // nnz of the installed factor G (diagonal included), without exporting it.
    std::int64_t factor_nnz() const {
        return std::visit([](const auto& route) {
            return static_cast<std::int64_t>(route.factor_nnz());
        }, apxchol::detail::solver_access::route(slv_));
    }

    // (indptr, indices, data) copy of the installed factor G = L S in PERMUTED
    // space (factorization.h), exact in float64. Each column holds its
    // diagonal first, then the off-diagonals in ascending row order.
    py::tuple factor_csc() const {
        apxchol::factorization F;
        {
            py::gil_scoped_release release;
            F = slv_.factor();
        }
        const std::int64_t nnz = static_cast<std::int64_t>(F.L.nonZeros());
        const std::int64_t ncol = static_cast<std::int64_t>(F.L.rows());
        py::array_t<std::int64_t> indptr(ncol + 1);
        py::array_t<std::int32_t> indices(nnz);
        py::array_t<double> data(nnz);
        auto* ip = indptr.mutable_data();
        auto* ii = indices.mutable_data();
        auto* dd = data.mutable_data();
        const auto outer = F.L.col_ptrs();
        const auto inner = F.L.row_indices();
        const auto vals = F.L.values();
        for (std::int64_t c = 0; c <= ncol; ++c)
            ip[c] = static_cast<std::int64_t>(outer[static_cast<std::size_t>(c)]);
        for (std::int64_t c = 0; c < ncol; ++c) {
            const double s = F.scale.empty() ? 1.0 : F.scale[static_cast<std::size_t>(c)];
            for (std::int64_t k = ip[c]; k < ip[c + 1]; ++k) {
                ii[k] = static_cast<std::int32_t>(inner[static_cast<std::size_t>(k)]);
                dd[k] = static_cast<double>(vals[static_cast<std::size_t>(k)]) * s;
            }
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

    static apxchol::solver setup(const csc_arrays& A, const apxchol::solve_options& options) {
        return A.visit([&](const auto& view) {
            py::gil_scoped_release release;
            return apxchol::solver(view, options);
        });
    }

    static bool same_shape(const py::array& a, const py::array& b) {
        return a.ndim() == b.ndim() && std::equal(a.shape(), a.shape() + a.ndim(), b.shape());
    }

    Array checked_rhs(Array a) const {
        if ((a.ndim() != 1 && a.ndim() != 2) || a.shape(0) != rows())
            throw std::invalid_argument("RHS must have shape (n,) or (n, nrhs)");
        return a;
    }

    static Array output_array(const Array& rhs, const py::object& out) {
        if (out.is_none())
            return Fortran(std::vector<py::ssize_t>(rhs.shape(), rhs.shape() + rhs.ndim()));
        if (!py::isinstance<py::array_t<double>>(out))
            throw std::invalid_argument("out must be a float64 NumPy array");
        auto a = out.cast<Array>();
        if (!same_shape(a, rhs) || !a.writeable() ||
            !(a.flags() & (py::array::c_style | py::array::f_style)))
            throw std::invalid_argument(
                "out must be writable, C- or F-contiguous, and have the same shape as b");
        return a;
    }

    static std::span<const double> vector(const Fortran& a) {
        return {a.data(), static_cast<std::size_t>(a.size())};
    }
    static std::span<double> mutable_vector(Fortran& a) {
        return {a.mutable_data(), static_cast<std::size_t>(a.size())};
    }
    static apxchol::const_matrix_view matrix(const Fortran& a) {
        return {a.data(), a.shape(0), a.shape(1), a.shape(0)};
    }
    static apxchol::matrix_view mutable_matrix(Fortran& a) {
        return {a.mutable_data(), a.shape(0), a.shape(1), a.shape(0)};
    }

    // Permutation and flags of the route; does not export the factor.
    const apxchol::factorization& factor() const {
        return std::visit([](const auto& route) -> const apxchol::factorization& {
            return route.factor();
        }, apxchol::detail::solver_access::route(slv_));
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
    // The native contract has two error classes: std::invalid_argument for bad
    // input, options or unsupported requests, and std::runtime_error (with its
    // standard subclasses) for execution failures. Map them to ValueError and
    // RuntimeError; pybind11's own errors keep their Python types.
    py::register_local_exception_translator([](std::exception_ptr error) {
        try {
            std::rethrow_exception(error);
        } catch (const py::builtin_exception&) {
            throw;
        } catch (const std::invalid_argument& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (const std::runtime_error& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });
    // Returns None when the input is already a valid operator, else the CSC
    // arrays of its graph Laplacian; throws for anything else.
    m.def("as_operator",
          [](py::array indptr, py::array indices, py::array data) -> py::object {
              const csc_arrays arrays(indptr, indices, data);
              return arrays.visit([](const auto& view) -> py::object {
                  const apxchol::detail::csc_input M(view);
                  if (!apxchol::detail::is_graph_adjacency(M))
                      return py::none();
                  return csc_to_numpy(apxchol::detail::adjacency_laplacian(M));
              });
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
        .def("setup_seconds", &Solver::setup_seconds)
        .def("backend", &Solver::backend)
        .def("sddm", &Solver::sddm)
        .def("factor_storage", &Solver::factor_storage)
        .def("lumped_offdiag", &Solver::lumped_offdiag)
        .def("factor_nnz", &Solver::factor_nnz)
        .def("factor_csc", &Solver::factor_csc)
        .def("perm", &Solver::perm);
}
