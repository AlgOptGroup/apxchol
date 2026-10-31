APXCHOL_GPU_CODE(
namespace residual_sparsify {
using bits_type = APXCHOL_GPU_UINT64;
template<class Wide> struct Scale {
    Wide value, target;
    unsigned passes, advance;
};
// Heaviest-first weight bucket from the FP32 encoding of a nonnegative weight:
// the inverted exponent and top three mantissa bits, eight buckets per octave.
APXCHOL_GPU_FUNCTION unsigned bucket(unsigned float_bits) {
    return ~float_bits >> 20;
}
APXCHOL_GPU_FUNCTION bits_type priority(unsigned bucket, unsigned ordinal) {
    return (bits_type(bucket) << 32) | ordinal;
}
APXCHOL_GPU_FUNCTION bits_type draw_bits(unsigned u, unsigned v, bits_type seed) {
    auto z = seed ^ (bits_type(u) * bits_type(0x9E3779B97F4A7C15UL)) ^
        (bits_type(v) * bits_type(0xBF58476D1CE4E5B9UL));
    z = (z ^ (z >> 30)) * bits_type(0xBF58476D1CE4E5B9UL);
    z = (z ^ (z >> 27)) * bits_type(0x94D049BB133111EBUL);
    return (z ^ (z >> 31)) >> 11;
}
template<class Wide> struct Coalesced { Wide value; unsigned end; };
template<class Math, class Input>
APXCHOL_GPU_FUNCTION Coalesced<typename Math::wide> coalesce(Math math, Input in, unsigned begin, unsigned end) {
    auto sum = math.from_float(0);
    unsigned at = begin;
    do { sum = math.add(sum, in.weight(at++)); } while (at < end && in.same(begin, at));
    return {sum, at};
}
template<class Memory>
APXCHOL_GPU_FUNCTION void choose(Memory m, unsigned id) {
    const auto edge = m.edge(id);
    const unsigned u = m.label(edge.a), v = m.label(edge.b);
    if (u == v) return;
    const unsigned bucket = m.bucket(edge.value);
    m.offer(u, bucket, id); m.offer(v, bucket, id);
}
template<class Memory>
APXCHOL_GPU_FUNCTION unsigned root(Memory m, unsigned v) {
    for (;;) { const unsigned p = m.parent(v); if (p == v) return v; v = p; }
}
template<class Memory>
APXCHOL_GPU_FUNCTION void contract(Memory m, unsigned i) {
    if (m.label(i) != i) return;
    const unsigned id = m.choice(i);
    if (id == ~0u) return;
    const auto edge = m.edge(id);
    m.mark(id);
    // Strict priorities make these Boruvka choices a forest. Linking larger
    // root IDs to smaller ones keeps concurrent parent updates acyclic.
    for (;;) {
        const unsigned u = root(m, edge.a), v = root(m, edge.b);
        if (u == v) return;
        const unsigned high = u > v ? u : v, low = u > v ? v : u;
        if (m.link(high, low)) { m.joined(); return; }
    }
}
template<class Math>
APXCHOL_GPU_FUNCTION bool worthwhile(Math math, typename Math::wide edges,
        typename Math::wide forest, typename Math::wide rounds, typename Math::wide keep) {
    const auto zero = math.from_float(0), one = math.from_float(1);
    const auto difference = math.subtract(edges, forest);
    const auto off_tree = math.less(zero, difference) ? difference : zero;
    const auto dropped = math.multiply(math.subtract(one, keep), off_tree);
    return !math.less(math.multiply(dropped, rounds), math.multiply(math.from_float(8), edges));
}
template<class Math>
APXCHOL_GPU_FUNCTION typename Math::wide importance(Math math, typename Math::wide weight) {
    return math.root(weight);
}
template<class Math>
APXCHOL_GPU_FUNCTION typename Math::wide probability(Math math, typename Math::wide scale,
                                                     typename Math::wide importance) {
    const auto value = math.multiply(scale, importance), one = math.from_float(1);
    return math.less(value, one) ? value : one;
}
template<class Math, class ScalePointer>
APXCHOL_GPU_FUNCTION void normalize(Math math, ScalePointer scale,
                                    typename Math::wide sum, bool initial) {
    const auto zero = math.from_float(0);
    if (!initial && !scale->advance) return;
    if (initial) {
        scale->value = math.less(zero, scale->target) && math.less(zero, sum)
            ? math.divide(scale->target, sum) : zero;
    } else {
        ++scale->passes;
        if (!math.less(zero, sum)) { scale->advance = 0; return; }
        scale->value = math.multiply(scale->value, math.divide(scale->target, sum));
    }
    scale->advance = math.less(zero, scale->value);
}
template<class Wide> struct Sample { Wide probability, value; bool kept, valid; };
template<class Math>
APXCHOL_GPU_FUNCTION Sample<typename Math::wide> sample(Math math, typename Math::wide weight,
        typename Math::wide importance, typename Math::wide scale, bool backbone,
        unsigned u, unsigned v, bits_type seed) {
    const auto zero = math.from_float(0), one = math.from_float(1);
    const auto p = backbone ? one : probability(math, scale, importance);
    const bool kept = !math.less(p, one) || math.less(math.uniform(draw_bits(u, v, seed)), p);
    const auto value = kept && !backbone ? math.reweight(weight, p) : weight;
    return {p, value, kept, math.finite(scale) && math.finite(p) && !math.less(p, zero) &&
                              (!kept || math.finite(value))};
}
}
)
