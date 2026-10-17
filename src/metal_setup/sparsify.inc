APXCHOL_GPU_CODE(
namespace resident_setup {
template<class Wide> struct SparseState {
    residual_sparsify::Scale<Wide> scale;
    unsigned count, tree, joins, threshold, selected, enabled, overflow;
};
enum SparseStage : unsigned {
    sparse_initialize, sparse_coalesce, sparse_collect, sparse_gate, sparse_reset,
    sparse_bucket, sparse_ordinal, sparse_contract, sparse_labels,
    sparse_importance, sparse_partials, sparse_fold, sparse_clear,
    sparse_sample, sparse_offsets, sparse_emit, sparse_finish, sparse_stage_count
};
struct EdgeOrder {
    APXCHOL_GPU_FUNCTION bool operator()(Edge a, Edge b) const {
        return a.node != b.node ? a.node < b.node : a.value < b.value;
    }
};
template<class Math, class Edges> struct SparseInput {
    Math math; Edges edges;
    APXCHOL_GPU_FUNCTION typename Math::wide weight(unsigned i) const { return math.from_float(edges[i].value); }
    APXCHOL_GPU_FUNCTION bool same(unsigned a, unsigned b) const { return edges[a].node == edges[b].node; }
};
template<class Memory, class Math, class Group>
APXCHOL_GPU_FUNCTION void sparse_coalesce_row(Memory m, Math math, unsigned row, Group group) {
    const auto s = m.data;
    if (group.broadcast(group.lane == 0 ? unsigned(m.failed() || row >= s.state->active) : 0)) return;
    const unsigned v = s.entry_offsets[row], begin = s.begin[v], count = s.cursor[v] - begin;
    auto edges = s.graph[s.state->slot] + begin;
    group_sort(edges, count, EdgeOrder{}, group);
    if (group.lane) return;
    unsigned out = 0, canonical = 0;
    for (unsigned at = 0; at < count;) {
        const unsigned first = at, node = edges[at].node;
        const auto folded = residual_sparsify::coalesce(math,
            SparseInput<Math, decltype(edges)>{math, edges}, at, count);
        at = folded.end;
        if (!s.active[node]) continue;
        const auto sum = folded.value;
        const float value = math.to_float(sum);
        if (math.finite(sum) && math.finite(math.from_float(value))) {
            edges[out++] = {node, value};
            canonical += v < node;
        } else {
            m.retain_parallel();
            for (unsigned i = first; i < at; ++i) edges[out++] = edges[i];
            canonical += v < node ? at - first : 0;
        }
    }
    s.cursor[v] = begin + out;
    s.degree_bound[v] = out;
    s.counts[v].edges = canonical;
}
template<class Memory, class Math>
APXCHOL_GPU_FUNCTION void sparse_step(Memory m, Math math, unsigned stage, unsigned i, bool initial) {
    const auto s = m.data;
    if (m.failed()) return;
    if (stage == sparse_initialize) {
        if (i == 0) { s.state->prune_gap=1; s.state->prune_streak=s.state->prune_hidden=0; }
        if (i <= s.state->nodes) s.counts[i] = {};
        if (i < s.state->nodes) s.candidates[i] = s.winners[i] = i;
        return;
    }
    if (stage == sparse_collect) {
        if (i == 0) m.sparse->count = s.counts[s.state->nodes].edges;
        if (i >= s.state->active) return;
        const unsigned v = s.entry_offsets[i];
        unsigned out = s.counts[v].edges;
        for (unsigned at = s.begin[v]; at < s.cursor[v]; ++at) {
            const auto e = s.graph[s.state->slot][at];
            if (v < e.node) s.fills[out++] = {v, e.node, e.value};
        }
        return;
    }
    if (stage == sparse_gate && i == 0) {
        const auto number = [&](unsigned value) {
            return math.add(math.from_float(float(value >> 16) * 65536.0f), math.from_float(float(value & 65535u)));
        };
        const unsigned active = s.state->active, edges = m.sparse->count;
        if (!m.sparse->overflow && active > m.sparse->threshold && m.sparse->selected && edges > active - 1) {
            const auto rounds = math.divide(number(active - m.sparse->threshold), number(m.sparse->selected));
            m.sparse->enabled = residual_sparsify::worthwhile(math, number(edges), number(active - 1),
                                                             rounds, math.from_float(0.25f));
        }
        return;
    }
    if (stage >= sparse_reset && !m.sparse->enabled) return;
    if (stage == sparse_reset && i < s.state->active) {
        const auto v = s.entry_offsets[i];
        s.update_ptr[v] = s.update_cursor[v] = ~0u;
    }
    if (stage == sparse_bucket || stage == sparse_ordinal) {
        if (i < m.sparse->count) residual_sparsify::choose(m, i);
    }
    if (stage == sparse_contract && i < s.state->active)
        residual_sparsify::contract(m, s.entry_offsets[i]);
    if (stage == sparse_labels && i < s.state->active) {
        const auto v = s.entry_offsets[i];
        s.winners[v] = residual_sparsify::root(m, v);
    }
    if (stage == sparse_importance && i < m.sparse->count) {
        s.update_ids[i] = 0;
        s.prefix[i] = residual_sparsify::importance(math, math.from_float(s.fills[i].value));
    }
    if (stage == sparse_fold && i == 0 && (initial || m.sparse->scale.advance)) {
        if (initial) {
            // Split the count before conversion to preserve all integer bits.
            const unsigned off_tree = m.sparse->count - m.sparse->tree;
            const auto count = math.add(math.from_float(float(off_tree >> 16) * 65536.0f),
                                        math.from_float(float(off_tree & 65535u)));
            m.sparse->scale.target = math.multiply(count, math.from_float(0.25f));
        }
        auto sum = math.from_float(0);
        for (unsigned block = 0; block < (m.sparse->count + 16383) / 16384; ++block)
            sum = math.add(sum, s.updates[block].value);
        residual_sparsify::normalize(math, &m.sparse->scale, sum, initial);
    }
    if (stage == sparse_clear && i <= s.state->nodes) s.counts[i] = {};
    if (stage == sparse_sample && i < m.sparse->count) {
        const auto e = s.fills[i];
        if (s.winners[e.a] != s.winners[e.b]) { m.fail(bad_graph); return; }
        const auto sampled = residual_sparsify::sample(math, math.from_float(e.value), s.prefix[i],
            m.sparse->scale.value, s.update_ids[i], e.a, e.b, s.state->seed ^ s.state->round);
        const float value = math.to_float(sampled.value);
        if (!sampled.valid || (sampled.kept && !math.finite(math.from_float(value)))) {
            m.fail(bad_graph); return;
        }
        s.update_ids[i] = sampled.kept;
        if (sampled.kept) {
            s.fills[i].value = value;
            m.count(e.a); m.count(e.b);
        }
    }
    if (stage == sparse_offsets && i <= s.state->nodes) {
        s.begin[i] = s.counts[i].edges;
        if (i < s.state->nodes) {
            if (s.counts[i+1].edges<s.counts[i].edges) { m.fail(bad_graph); return; }
            s.cursor[i] = s.counts[i].edges;
            s.degree_bound[i] = s.counts[i+1].edges-s.counts[i].edges;
        }
    }
    if (stage == sparse_emit && i < m.sparse->count && s.update_ids[i]) {
        const auto e = s.fills[i];
        s.graph[s.state->slot ^ 1][m.reserve_edge(e.a)] = {e.b, e.value};
        s.graph[s.state->slot ^ 1][m.reserve_edge(e.b)] = {e.a, e.value};
    }
    if (stage == sparse_finish && i == 0) {
        if (m.sparse->tree != m.sparse->joins || s.counts[s.state->nodes].edges > s.state->edge_capacity)
            m.fail(bad_graph);
        else { s.state->edges = s.counts[s.state->nodes].edges; s.state->slot ^= 1; }
    }
}
}
)
