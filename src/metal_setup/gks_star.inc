APXCHOL_GPU_CODE(
namespace sparse_round {
using seed_type = APXCHOL_GPU_UINT64;
enum status_code : unsigned { success, oversized, invalid_input, arithmetic_range, insufficient_output };
template<class Wide> struct neighbor { unsigned node, slot; Wide value; };
template<class Wide> struct star_input {
    unsigned node, begin, end, nodes;
    Wide excess;
    seed_type seed;
};
struct row_result {
    unsigned status, entries, fills, excess_updates;
    float diagonal;
};
template<class Record>
APXCHOL_GPU_FUNCTION bool neighbor_before(Record a, Record b) {
    return a.node != b.node ? a.node < b.node : a.slot < b.slot;
}
template<class Math, class Record>
APXCHOL_GPU_FUNCTION bool weight_before(Math math, Record a, Record b) {
    if (math.less(a.value, b.value)) return true;
    if (math.less(b.value, a.value)) return false;
    return a.node < b.node;
}
APXCHOL_GPU_FUNCTION seed_type random_bits(seed_type state) {
    state = (state ^ (state >> 30)) * seed_type(0xBF58476D1CE4E5B9UL);
    state = (state ^ (state >> 27)) * seed_type(0x94D049BB133111EBUL);
    return state ^ (state >> 31);
}

template<class Wide> struct canonical_star { row_result result; Wide degree; };

// Compact in encounter order; fold strided partials before ordered deduplication.
template<class Math,class Input,class Records,class Group>
APXCHOL_GPU_FUNCTION canonical_star<typename Math::wide> gather_star(Math math,Input input,
        star_input<typename Math::wide> p,Records rows,Group group) {
    const auto zero = math.from_float(0.0f);
    row_result result{};
    if (p.end < p.begin) {
        result.status = oversized; return {result,zero};
    }
    if (p.node >= p.nodes || !input.active(p.node) || !math.finite(p.excess) ||
        math.less(p.excess, zero)) {
        result.status = invalid_input; return {result,zero};
    }
    auto degree = zero;
    unsigned count = 0, invalid = 0;
    for (seed_type chunk = p.begin; chunk < p.end; chunk += group.width) {
        const auto at = chunk + group.lane;
        unsigned other = 0, keep = 0;
        auto value = zero;
        if (at < p.end) {
            other = input.node(unsigned(at));
            if (other >= p.nodes || other == p.node) invalid = 1;
            else if (input.active(other)) {
                value = math.from_float(input.weight(unsigned(at)));
                if (!math.finite(value) || math.less(value, zero)) invalid = 1;
                else { keep = 1; degree = math.add(degree, value); }
            }
        }
        const unsigned rank = group.prefix(keep);
        if (keep) rows[count + rank] = {other, unsigned(at - p.begin), value};
        count += group.broadcast(rank + keep, group.width - 1);
    }
    if (group.sum(invalid)) { result.status = invalid_input; return {result,zero}; }
    for (unsigned offset = group.width / 2; offset; offset /= 2) {
        const auto other = group.broadcast(degree, group.lane ^ offset);
        if (group.lane < offset) degree = math.add(degree, other);
    }
    result.entries=count;
    return {result,degree};
}

template<class Math,class Records>
APXCHOL_GPU_FUNCTION canonical_star<typename Math::wide> combine_star(Math math,
        star_input<typename Math::wide> p,Records rows,canonical_star<typename Math::wide> gathered) {
    const auto zero=math.from_float(0.0f);
    auto result=gathered.result;
    auto degree=gathered.degree;
    const unsigned count=result.entries;
    unsigned unique = 0;
    for (unsigned at = 0; at < count;) {
        auto row = rows[at++];
        while (at < count && rows[at].node == row.node)
            row.value = math.add(row.value, rows[at++].value);
        rows[unique++] = row;
    }
    degree = unique ? math.add(degree, p.excess) : p.excess;
    if (!math.finite(degree)) { result.status = arithmetic_range; return {result,zero}; }
    if (!math.less(zero, degree)) degree = math.from_float(1.0f);
    result.entries = unique;
    result.fills = unique ? unique - 1 : 0;
    result.excess_updates = math.less(zero, p.excess) ? unique : 0;
    return {result,degree};
}


template<class Math,class Output,class Records>
APXCHOL_GPU_FUNCTION row_result factor_star(Math math,Output out,Records rows,
        canonical_star<typename Math::wide> prepared,unsigned lane=0,unsigned width=1) {
    const auto zero=math.from_float(0.0f),degree=prepared.degree;
    auto result=prepared.result;
    const unsigned unique=result.entries;

    if (out.entries < result.entries || out.fills < result.fills ||
        out.excess_updates < result.excess_updates) {
        result.status = insufficient_output; return result;
    }
    const auto root = math.root(degree);
    result.diagonal = math.to_float(root);
    if (!math.finite(root) || !math.less(zero, math.from_float(result.diagonal))) {
        result.status = arithmetic_range; return result;
    }
    for (unsigned i = lane; i < unique; i += width) {
        const auto value = math.divide(rows[i].value, root);
        const float stored = math.to_float(value);
        if (!math.finite(value) || !math.finite(math.from_float(stored))) {
            result.status = arithmetic_range; return result;
        }
        out.factor(i, rows[i].node, stored);
    }
    return result;
}

template<class Math,class Records,class Prefix>
APXCHOL_GPU_FUNCTION row_result prepare_sample(Math math,Records rows,Prefix prefix,
        row_result result) {
    const auto zero=math.from_float(0.0f);
    const unsigned unique=result.entries;
    for (unsigned i=0;i<unique;++i) {
        prefix[i]=i ? math.add(prefix[i-1],rows[i].value) : rows[i].value;
        if (!math.finite(prefix[i]) || (i && math.less(prefix[i],prefix[i-1]))) {
            result.status=arithmetic_range; return result;
        }
    }
    // Nonnegative prefixes make skipped nonpositive suffixes trailing.
    result.fills=unique ? unique-1 : 0;
    while (result.fills && !math.less(zero,
            math.subtract(prefix[unique-1],prefix[result.fills-1]))) --result.fills;
    return result;
}

template<class Math,class Output,class Records,class Prefix>
APXCHOL_GPU_FUNCTION unsigned sample_item(Math math,Output out,
        star_input<typename Math::wide> p,Records rows,Prefix prefix,
        canonical_star<typename Math::wide> prepared,unsigned i) {
    const auto zero=math.from_float(0.0f),degree=prepared.degree;
    const auto result=prepared.result;
    const unsigned unique=result.entries;
    if (result.excess_updates) {
        const auto update=math.multiply_divide(rows[i].value,p.excess,degree);
        if (!math.finite(update)) return arithmetic_range;
        out.excess(i,rows[i].node,update);
    }
    if (i>=result.fills) return success;
    const auto suffix=math.subtract(prefix[unique-1],prefix[i]);
    const auto bits=random_bits(p.seed+seed_type(0x9E3779B97F4A7C15UL)*seed_type(i+1));
    const auto target=math.add(prefix[i],math.multiply(math.uniform(bits>>11),suffix));
    unsigned low=i+1,high=unique;
    while (low<high) {
        const unsigned mid=low+(high-low)/2;
        if (!math.less(target,prefix[mid])) low=mid+1;
        else high=mid;
    }
    const unsigned partner=low<unique ? low : unique-1;
    const auto weight=math.multiply_divide(rows[i].value,suffix,degree);
    const float stored=math.to_float(weight);
    if (!math.finite(weight) || !math.finite(math.from_float(stored)) || math.less(weight,zero))
        return arithmetic_range;
    const unsigned a=rows[i].node,b=rows[partner].node;
    out.fill(i,a<b ? a : b,a<b ? b : a,stored);
    return success;
}



}
)
