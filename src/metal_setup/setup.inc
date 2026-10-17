APXCHOL_GPU_CODE(
namespace resident_setup {
using sparse_round::seed_type;
enum Stage : unsigned {
    stage_initialize = 0,
    stage_reset = 1,
    stage_histogram = 2,
    stage_threshold = 3,
    stage_candidates = 4,
    stage_decide = 5,
    stage_commit = 6,
    stage_select = 7,
    stage_count_stars = 8,
    stage_offsets = 9,
    stage_emit_stars = 10,
    stage_deactivate = 11,
    stage_count_retained = 12,
    stage_count_incoming = 13,
    stage_copy_retained = 14,
    stage_scatter_incoming = 15,
    stage_update_excess = 16,
    stage_finish = 17,
    stage_validate_compaction = 18,
};

APXCHOL_GPU_FUNCTION bool stage_is_grouped(unsigned stage) {
    return stage == stage_decide || stage == stage_commit ||
           stage == stage_count_stars || stage == stage_emit_stars ||
           stage == stage_count_incoming || stage == stage_copy_retained ||
           stage == stage_scatter_incoming;
}

struct Edge { unsigned node; float value; };
struct Fill { unsigned a, b; float value; };
struct Column { unsigned node, begin, count; float diagonal; };
template<class Wide> struct Update { unsigned node; Wide value; };
struct Counts { unsigned edges, updates; };
struct State {
    unsigned nodes, edges, active, selected, columns, entries, round, slot, status;
    unsigned pending_entries, pending_fills, threshold, edge_capacity, has_excess, eligible;
    unsigned prune_gap, prune_streak, prune_hidden, prune_enabled, degree_divisor;
    seed_type seed;
};
enum class Field : unsigned {
    graph0, graph1, begin, active_ids, active, candidates, winners, selected,
    cursor, entry_offsets, fill_offsets, excess, columns, fills, updates,
    rows, counts, update_ptr, update_cursor, update_ids, records, prefix, histogram, round_ptr, degree_bound, count
};
struct Layout { seed_type bytes, offsets[unsigned(Field::count)]; };

template<class Wide> struct View {
    APXCHOL_GPU_BUFFER(State) state;
    APXCHOL_GPU_BUFFER(Edge) graph[2];
    APXCHOL_GPU_BUFFER(unsigned) begin;
    APXCHOL_GPU_BUFFER(unsigned) active_ids;
    APXCHOL_GPU_BUFFER(unsigned) active;
    APXCHOL_GPU_BUFFER(unsigned) degree_bound;
    APXCHOL_GPU_BUFFER(unsigned) candidates;
    APXCHOL_GPU_BUFFER(unsigned) winners;
    APXCHOL_GPU_BUFFER(unsigned) selected;
    APXCHOL_GPU_BUFFER(unsigned) cursor;
    APXCHOL_GPU_BUFFER(unsigned) entry_offsets;
    APXCHOL_GPU_BUFFER(unsigned) fill_offsets;
    APXCHOL_GPU_BUFFER(Wide) excess;
    APXCHOL_GPU_BUFFER(Column) columns;
    APXCHOL_GPU_BUFFER(Edge) factors;
    APXCHOL_GPU_BUFFER(Fill) fills;
    APXCHOL_GPU_BUFFER(Update<Wide>) updates;
    APXCHOL_GPU_BUFFER(sparse_round::row_result) rows;
    APXCHOL_GPU_BUFFER(Counts) counts;
    APXCHOL_GPU_BUFFER(unsigned) update_ptr;
    APXCHOL_GPU_BUFFER(unsigned) update_cursor;
    APXCHOL_GPU_BUFFER(unsigned) update_ids;
    APXCHOL_GPU_BUFFER(sparse_round::neighbor<Wide>) records;
    APXCHOL_GPU_BUFFER(Wide) prefix;
    APXCHOL_GPU_BUFFER(unsigned) histogram;
    APXCHOL_GPU_BUFFER(unsigned) round_ptr;
};
enum failure : unsigned { bad_graph = 256, no_progress = 1024, capacity_exceeded = 2048 };

// A pointer parameter lets the same ordering work in device or local scratch.
struct Sort {
    template<class Rows, class Before>
    APXCHOL_GPU_FUNCTION void sort(Rows rows, unsigned n, Before before) const {
        for (unsigned start = n / 2; start; --start) sift(rows, start - 1, n, before);
        for (unsigned end = n; end > 1;) {
            --end;
            auto x = rows[end]; rows[end] = rows[0]; rows[0] = x;
            sift(rows, 0, end, before);
        }
    }
    template<class Rows, class Before>
    APXCHOL_GPU_FUNCTION void sift(Rows rows, unsigned root, unsigned n, Before before) const {
        const auto value = rows[root];
        for (unsigned child = root * 2 + 1; child < n; child = root * 2 + 1) {
            if (child + 1 < n && before(rows[child], rows[child + 1])) ++child;
            if (!before(value, rows[child])) break;
            rows[root] = rows[child]; root = child;
        }
        rows[root] = value;
    }
    struct ByNode {
        template<class T> APXCHOL_GPU_FUNCTION bool operator()(T a, T b) const {
            return sparse_round::neighbor_before(a,b);
        }
    };
    template<class Math> struct ByWeight {
        Math math;
        template<class T> APXCHOL_GPU_FUNCTION bool operator()(T a,T b) const {
            return sparse_round::weight_before(math,a,b);
        }
    };
};

template<class Memory> struct Input {
    Memory m;
    APXCHOL_GPU_FUNCTION unsigned node(unsigned at) const { return m.data.graph[m.data.state->slot][at].node; }
    APXCHOL_GPU_FUNCTION float weight(unsigned at) const { return m.data.graph[m.data.state->slot][at].value; }
    APXCHOL_GPU_FUNCTION bool active(unsigned node) const { return m.data.active[node] != 0; }
};
template<class Memory> struct Output {
    Memory m; unsigned row, entries, fills, excess_updates;
    APXCHOL_GPU_FUNCTION void factor(unsigned i,unsigned node,float value) const {
        m.data.factors[m.data.state->entries+m.data.entry_offsets[row]+i] = {node,value};
    }
    APXCHOL_GPU_FUNCTION void fill(unsigned i,unsigned a,unsigned b,float value) const {
        m.data.fills[m.data.fill_offsets[row]+i] = {a,b,value};
    }
    APXCHOL_GPU_FUNCTION void excess(unsigned i,unsigned node,typename Memory::wide value) const {
        m.data.updates[m.data.entry_offsets[row]+i] = {node,value};
    }
};

template<class Memory,class Math>
APXCHOL_GPU_FUNCTION void initialize(Memory m,Math math,unsigned v) {
    auto s=m.data; const unsigned n=s.state->nodes,e=s.state->edges;
    if (v>=n) return;
    const auto begin=s.begin[v],end=s.cursor[v];
    if ((v==0 && begin!=0) || (v+1==n && end!=e) || end<begin || end>e || e>s.state->edge_capacity ||
        !math.finite(s.excess[v]) || math.less(s.excess[v],math.from_float(0))) { m.fail(bad_graph); return; }
    for (unsigned at=begin;at<end;++at) {
        const auto edge=s.graph[0][at];
        if (edge.node>=n || edge.node==v || (at!=begin && s.graph[0][at-1].node>=edge.node) ||
            !math.finite(math.from_float(edge.value)) || edge.value<0) { m.fail(bad_graph); return; }
    }
    s.active[v]=1; s.entry_offsets[v]=v; s.degree_bound[v]=end-begin;
}

template<class Memory>
APXCHOL_GPU_FUNCTION bool precedes(Memory m,unsigned a,unsigned b) {
    const auto s=m.data;
    const auto da=s.degree_bound[a],db=s.degree_bound[b];
    if (da!=db) return da<db;
    const auto key=s.state->seed^(seed_type(s.state->round)*seed_type(0xD6E8FEB86659FD93UL));
    const auto ha=sparse_round::random_bits((key^a)+seed_type(0x9E3779B97F4A7C15UL));
    const auto hb=sparse_round::random_bits((key^b)+seed_type(0x9E3779B97F4A7C15UL));
    return ha!=hb ? ha<hb : a<b;
}
APXCHOL_GPU_FUNCTION unsigned degree_bucket(unsigned degree) {
    unsigned b=0; while (degree) { ++b; degree>>=1; } return b;
}
template<class Memory> APXCHOL_GPU_FUNCTION void histogram(Memory m,unsigned v) {
    const auto s=m.data;
    if (m.failed() || !s.state->active || v>=s.state->nodes || !s.active[v]) return;
    const unsigned actual=s.cursor[v]-s.begin[v],old=s.degree_bound[v],previous=s.state->threshold;
    const unsigned cutoff=previous>~0u/4 ? ~0u : 4*previous;
    const bool enabled=s.state->prune_enabled && previous;
    const bool skipping=enabled && s.state->prune_streak<s.state->prune_gap;
    if (enabled && !skipping && old>cutoff && actual<=previous) m.hidden_degree();
    if (!skipping || old<=cutoff) s.degree_bound[v]=actual;
    m.add_histogram(degree_bucket(s.degree_bound[v]));
}
template<class Memory> APXCHOL_GPU_FUNCTION void threshold(Memory m) {
    auto s=m.data;
    if (m.failed() || !s.state->active) return;
    // Advance the audit cadence using the previous threshold, after all rows ran.
    if (s.state->prune_enabled && s.state->threshold) {
        if (s.state->prune_streak<s.state->prune_gap) ++s.state->prune_streak;
        else {
            s.state->prune_gap=s.state->prune_hidden ? 1 : (s.state->prune_gap<8 ? 2*s.state->prune_gap : 8);
            s.state->prune_streak=0;
        }
    }
    const unsigned rank=(s.state->active-1)/s.state->degree_divisor;
    unsigned cumulative=0,b=0;
    while (b<32 && cumulative+s.histogram[b]<=rank) cumulative+=s.histogram[b++];
    s.state->threshold=b==32 ? ~0u : (1u<<b)-1;
    s.state->eligible=cumulative+s.histogram[b];
}
template<class Memory> APXCHOL_GPU_FUNCTION void candidates(Memory m,unsigned v) {
    auto s=m.data;
    if (v<s.state->nodes) s.candidates[v]=s.active[v] &&
        s.degree_bound[v]<=s.state->threshold ? 0 : 2;
}
template<class Memory,class Group> APXCHOL_GPU_FUNCTION void decide(Memory m,unsigned v,Group group) {
    auto s=m.data; if (m.failed() || v>=s.state->nodes) return;
    unsigned blocked=s.candidates[v]!=0;
    for (seed_type at=seed_type(s.begin[v])+group.lane;!blocked && at<s.cursor[v];at+=group.width) {
        const auto u=s.graph[s.state->slot][at].node;
        blocked=s.candidates[u]==0 && precedes(m,u,v);
    }
    const bool win=!group.sum(blocked);
    if (group.lane==0) s.winners[v]=win;
}
template<class Memory,class Group> APXCHOL_GPU_FUNCTION void commit(Memory m,unsigned v,Group group) {
    auto s=m.data; if (m.failed() || v>=s.state->nodes || s.candidates[v]!=0) return;
    if (s.winners[v]) { if (group.lane==0) s.candidates[v]=1; return; }
    unsigned blocked=0;
    for (seed_type at=seed_type(s.begin[v])+group.lane;!blocked && at<s.cursor[v];at+=group.width)
        blocked=s.winners[s.graph[s.state->slot][at].node];
    if (group.sum(blocked) && group.lane==0) s.candidates[v]=2;
}
template<class Memory> APXCHOL_GPU_FUNCTION void select(Memory m,unsigned row) {
    auto s=m.data; if (m.failed() || row>=s.state->active) return;
    const unsigned v=s.entry_offsets[row];
    // Preserve the frontier before factor offsets reuse its buffer.
    s.active_ids[row]=v;
    if (s.candidates[v]==1) s.selected[m.reserve_selected()]=v;
}

// Ascending compare-exchange layers need no padded storage for partial rows.
template<class Rows,class Before,class Group>
APXCHOL_GPU_FUNCTION void group_sort(Rows rows,unsigned n,Before before,Group group) {
    for(seed_type p=1;p<n;p*=2) for(seed_type k=p;k;k/=2) {
        const seed_type start=k==p?0:k,span=2*k;
        const seed_type pairs=((n-start+span-1)&~(span-1))/2;
        for(seed_type q=group.lane;q<pairs;q+=group.width) {
            const seed_type a=start+q+(q&~(k-1)),b=a+k;
            if(b<n && (a&~(2*p-1))==(b&~(2*p-1)) && before(rows[b],rows[a])) {
                const auto value=rows[a];rows[a]=rows[b];rows[b]=value;
            }
        }
        group.barrier();
    }
}

template <bool Emit, class Memory, class Math, class Group>
APXCHOL_GPU_FUNCTION void group_star(Memory m, Math math, unsigned row, Group group) {
    auto s = m.data;
    if (group.broadcast(group.lane == 0
                            ? unsigned(m.failed() || !s.state->active || row >= s.state->nodes)
                            : 0))
        return;
    if (row >= s.state->selected) {
        if (!Emit && group.lane == 0)
            s.counts[row] = {};
        return;
    }
    const unsigned v = s.selected[row], begin = s.begin[v];
    const auto seed = s.state->seed ^ ((seed_type(v) + 1) * seed_type(0x9E3779B97F4A7C15UL));
    const sparse_round::star_input<typename Math::wide> input{
        v, begin, s.cursor[v], s.state->nodes, s.excess[v], seed};
    auto rows = s.records + begin;
    if (!Emit) {
        const auto gathered = sparse_round::gather_star(math, Input<Memory>{m}, input, rows, group);
        const unsigned status = group.broadcast(gathered.result.status);
        if (status) {
            if (group.lane == 0)
                m.fail(1u << status);
            return;
        }
        const unsigned count = group.broadcast(gathered.result.entries);
        group.barrier();
        group_sort(rows, count, Sort::ByNode{}, group);
        if (group.lane == 0) {
            const auto prepared = sparse_round::combine_star(math, input, rows, gathered);
            s.rows[row] = prepared.result;
            s.counts[row] = {prepared.result.entries, prepared.result.fills};
            if (prepared.result.status)
                m.fail(1u << prepared.result.status);
            if (prepared.result.entries)
                s.prefix[begin] = prepared.degree;
        }
        return;
    }
    auto reserved = s.rows[row];
    auto degree = reserved.entries ? s.prefix[begin] : s.excess[v];
    if (!math.less(math.from_float(0), degree))
        degree = math.from_float(1);
    const Output<Memory> out{m, row, reserved.entries, reserved.fills, reserved.excess_updates};
    reserved = sparse_round::factor_star(
        math, out, rows, sparse_round::canonical_star<typename Math::wide>{reserved, degree},
        group.lane, group.width);
    if (reserved.status)
        m.fail(1u << reserved.status);
    group.barrier();
    if (group.broadcast(group.lane == 0 ? unsigned(m.failed()) : 0))
        return;
    group_sort(rows, reserved.entries, typename Sort::ByWeight<Math>{math}, group);
    if (group.lane == 0) {
        s.rows[row] = sparse_round::prepare_sample(math, rows, s.prefix + begin, reserved);
        if (s.rows[row].status)
            m.fail(1u << s.rows[row].status);
    }
    group.barrier();
    if (group.broadcast(group.lane == 0 ? unsigned(m.failed()) : 0))
        return;
    reserved = s.rows[row];
    for (unsigned i = group.lane; i < reserved.entries; i += group.width) {
        const auto status = sparse_round::sample_item(
            math, out, input, rows, s.prefix + begin,
            sparse_round::canonical_star<typename Math::wide>{reserved, degree}, i);
        if (status)
            m.fail(1u << status);
    }
    group.barrier();
    if (group.lane == 0 && !m.failed())
        s.columns[s.state->columns + row] = {v, s.state->entries + s.entry_offsets[row],
                                             reserved.entries, reserved.diagonal};
}

template<class Memory> APXCHOL_GPU_FUNCTION void offsets(Memory m,unsigned row,unsigned total) {
    auto s=m.data; if (row>total || m.failed()) return;
    s.entry_offsets[row]=s.counts[row].edges; s.fill_offsets[row]=s.counts[row].updates;
    if (row==total) {
        s.state->pending_entries=s.counts[row].edges; s.state->pending_fills=s.counts[row].updates;
        if (!s.state->selected) m.fail(no_progress);
        if (s.state->columns+s.state->selected>s.state->nodes || s.state->pending_entries>s.state->edge_capacity ||
            s.state->pending_entries>~0u-s.state->entries ||
            s.state->pending_fills>s.state->edge_capacity) m.fail(capacity_exceeded);
    }
}


template<class Memory> APXCHOL_GPU_FUNCTION void deactivate(Memory m,unsigned row) {
    if (!m.failed() && row<m.data.state->selected) m.data.active[m.data.selected[row]]=0;
}
template<class Memory> APXCHOL_GPU_FUNCTION void count_retained(Memory m, unsigned row) {
    auto s = m.data;
    if (m.failed() || !s.state->active || row > s.state->nodes) return;
    if (row >= s.state->active) {
        s.counts[row] = {};
        return;
    }
    const unsigned v = s.active_ids[row];
    const unsigned count = s.active[v] ? s.cursor[v] - s.begin[v] : 0;
    s.counts[row] = {count, 0};
    s.candidates[v] = row;
}
template<bool Scatter,class Memory> APXCHOL_GPU_FUNCTION void incoming(Memory m,unsigned row,unsigned lane=0,unsigned width=1) {
    auto s=m.data; if (m.failed() || row>=s.state->selected) return;
    const unsigned next=s.state->slot^1,n=s.state->nodes;
    if (!Scatter) {
        const unsigned v = s.selected[row];
        // Paired physical incidences include duplicate and zero-weight edges.
        for (APXCHOL_GPU_UINT64 at = APXCHOL_GPU_UINT64(s.begin[v]) + lane;
             at < s.cursor[v]; at += width) {
            const unsigned other = s.graph[s.state->slot][at].node;
            if (other >= n || !s.active[other]) {
                m.fail(bad_graph);
                return;
            }
            m.remove_edge(other);
        }
    }
    for (unsigned i=lane;i<s.rows[row].fills;i+=width) {
        const auto e=s.fills[s.fill_offsets[row]+i];
        if (e.a>=n || e.b>=n || e.a==e.b || !s.active[e.a] || !s.active[e.b]) {m.fail(bad_graph);return;}
        if (Scatter) {
            s.graph[next][m.reserve_edge(e.a)]={e.b,e.value};
            s.graph[next][m.reserve_edge(e.b)]={e.a,e.value};
        } else { m.count_edge(e.a);m.count_edge(e.b); }
    }
    for (unsigned i=lane;i<s.rows[row].excess_updates;i+=width) {
        const unsigned at=s.entry_offsets[row]+i,node=s.updates[at].node;
        if (node>=n || !s.active[node]) {m.fail(bad_graph);return;}
        if (Scatter) s.update_ids[m.reserve_update(node)]=at;
        else m.count_update(node);
    }
}
template<class Memory> APXCHOL_GPU_FUNCTION void validate_compact(Memory m) {
    const auto s=m.data;const auto total=s.counts[s.state->active];
    if(total.edges>s.state->edge_capacity || total.edges>s.state->edges || total.updates>s.state->edge_capacity)
        m.fail(capacity_exceeded);
}
template<class Memory,class Group> APXCHOL_GPU_FUNCTION void retained(Memory m,unsigned row,Group group) {
    auto s=m.data;
    if (group.broadcast(group.lane==0 ? unsigned(m.failed() || row>=s.state->active) : 0)) return;
    const unsigned v=s.active_ids[row];
    if (!s.active[v]) return;
    const unsigned slot=s.state->slot,next=slot^1;
    // All lanes receive the old bounds before lane zero publishes the new row.
    const unsigned old_begin=group.broadcast(group.lane==0 ? s.begin[v] : 0);
    const unsigned old_end=group.broadcast(group.lane==0 ? s.cursor[v] : 0);
    const auto begin=s.counts[row];
    unsigned at=begin.edges;
    for (APXCHOL_GPU_UINT64 chunk=old_begin;chunk<old_end;chunk+=group.width) {
        const auto old=chunk+group.lane;
        Edge edge{}; unsigned keep=0;
        if (old<old_end) {edge=s.graph[slot][old];keep=s.active[edge.node]!=0;}
        const unsigned rank=group.prefix(keep);
        if (keep) s.graph[next][at+rank]=edge;
        at+=group.broadcast(rank+keep,group.width-1);
    }
    if (group.lane==0) {
        const unsigned end=s.counts[row+1].edges;
        if (end<at) { m.fail(bad_graph); return; }
        const unsigned added=end-at;
        s.degree_bound[v]=added>~0u-s.degree_bound[v] ? ~0u : s.degree_bound[v]+added;
        s.begin[v]=begin.edges; s.cursor[v]=at;
        s.update_ptr[v]=begin.updates; s.update_cursor[v]=begin.updates;
    }
}
template<class Memory,class Math> APXCHOL_GPU_FUNCTION void excess(Memory m,Math math,unsigned row) {
    auto s=m.data; if (m.failed() || row>=s.state->active) return;
    const unsigned v=s.active_ids[row];
    if (!s.active[v]) return;
    auto delta=math.from_float(0);
    for (unsigned at=s.update_ptr[v];at<s.update_cursor[v];++at) delta=math.add(delta,s.updates[s.update_ids[at]].value);
    s.excess[v]=math.add(s.excess[v],delta);
    if (!math.finite(s.excess[v])) m.fail(1u<<sparse_round::arithmetic_range);
    // Scatter has consumed factor offsets; publish the next frontier here.
    s.entry_offsets[m.reserve_active()]=v;
}
template<class Memory> APXCHOL_GPU_FUNCTION void finish(Memory m) {
    auto s=m.data; if (m.failed()) return;
    s.round_ptr[s.state->round] = s.state->columns;
    s.state->edges=s.counts[s.state->active].edges;
    s.state->active-=s.state->selected; s.state->columns+=s.state->selected;
    s.round_ptr[s.state->round + 1] = s.state->columns;
    s.state->entries+=s.state->pending_entries;
    s.state->slot^=1; ++s.state->round;
}
// Both the queued encoder and the one-workgroup tail use this sequence.
template <class Dispatch>
APXCHOL_GPU_FUNCTION void factor_round(Dispatch dispatch, unsigned bound) {
    dispatch.stage(stage_reset, 33);
    dispatch.stage(stage_histogram, bound);
    dispatch.stage(stage_threshold, 1);
    dispatch.stage(stage_candidates, bound);
    for (unsigned pass = 0; pass < 4; ++pass) {
        dispatch.stage(stage_decide, bound);
        dispatch.stage(stage_commit, bound);
    }
    dispatch.stage(stage_select, bound);
    dispatch.stage(stage_count_stars, bound + 1);
    dispatch.scan(bound + 1);
    dispatch.stage(stage_offsets, bound + 1, bound);
    dispatch.stage(stage_emit_stars, bound);
    dispatch.stage(stage_deactivate, bound);
    dispatch.stage(stage_count_retained, bound + 1);
    dispatch.stage(stage_count_incoming, bound);
    dispatch.scan(bound + 1);
    dispatch.stage(stage_validate_compaction, 1);
    dispatch.stage(stage_copy_retained, bound);
    dispatch.stage(stage_scatter_incoming, bound);
    dispatch.stage(stage_update_excess, bound);
    dispatch.stage(stage_finish, 1);
}

}
)
