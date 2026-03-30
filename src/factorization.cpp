#include "apxchol/solver/factorization.h"
#include "apxchol/graph/graph.h"
#include "apxchol/checkpoint.h"
#include <stdexcept>

namespace apxchol {

// Pre-compiled instantiations for the three built-in backends.
// Other backends (custom containers, different SSO sizes, etc.) are
// instantiated on demand via the template definitions in factorization_impl.h.
template factorization factorize<vec_incidence>(
    const graph<vec_incidence>&, const factor_options&, checkpoint*);
template factorization factorize<forward_star_incidence>(
    const graph<forward_star_incidence>&, const factor_options&, checkpoint*);
template factorization factorize<small_vec_incidence>(
    const graph<small_vec_incidence>&, const factor_options&, checkpoint*);

factorization factorize(const Eigen::SparseMatrix<double>& L,
                        graph_storage storage,
                        const factor_options& opts,
                        checkpoint* cp) {
    if (L.rows() != L.cols())
        throw std::invalid_argument("factorize: matrix must be square");

    switch (storage) {
    case graph_storage::forward_star:
        return factorize<graph<forward_star_incidence>>(L, opts, cp);
    case graph_storage::small_vec:
        return factorize<graph<small_vec_incidence>>(L, opts, cp);
    default:
        return factorize<graph<vec_incidence>>(L, opts, cp);
    }
}

} // namespace apxchol
