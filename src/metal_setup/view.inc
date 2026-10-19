APXCHOL_GPU_CODE(
namespace resident_setup {
template<class Wide>
APXCHOL_GPU_FUNCTION View<Wide> view(APXCHOL_GPU_BUFFER(unsigned char) base,
                                    APXCHOL_GPU_BUFFER(Edge) factors, Layout l) {
    View<Wide> v{};
    v.state=reinterpret_cast<APXCHOL_GPU_BUFFER(State)>(base);
    v.graph[0]=reinterpret_cast<APXCHOL_GPU_BUFFER(Edge)>(base+l.offsets[unsigned(Field::graph0)]);
    v.graph[1]=reinterpret_cast<APXCHOL_GPU_BUFFER(Edge)>(base+l.offsets[unsigned(Field::graph1)]);
    v.begin=reinterpret_cast<APXCHOL_GPU_BUFFER(unsigned)>(base+l.offsets[unsigned(Field::begin)]);
    v.active_ids=reinterpret_cast<APXCHOL_GPU_BUFFER(unsigned)>(base+l.offsets[unsigned(Field::active_ids)]);
    v.active=reinterpret_cast<APXCHOL_GPU_BUFFER(unsigned)>(base+l.offsets[unsigned(Field::active)]);
    v.candidates=reinterpret_cast<APXCHOL_GPU_BUFFER(unsigned)>(base+l.offsets[unsigned(Field::candidates)]);
    v.winners=reinterpret_cast<APXCHOL_GPU_BUFFER(unsigned)>(base+l.offsets[unsigned(Field::winners)]);
    v.selected=reinterpret_cast<APXCHOL_GPU_BUFFER(unsigned)>(base+l.offsets[unsigned(Field::selected)]);
    v.cursor=reinterpret_cast<APXCHOL_GPU_BUFFER(unsigned)>(base+l.offsets[unsigned(Field::cursor)]);
    v.entry_offsets=reinterpret_cast<APXCHOL_GPU_BUFFER(unsigned)>(base+l.offsets[unsigned(Field::entry_offsets)]);
    v.fill_offsets=reinterpret_cast<APXCHOL_GPU_BUFFER(unsigned)>(base+l.offsets[unsigned(Field::fill_offsets)]);
    v.excess=reinterpret_cast<APXCHOL_GPU_BUFFER(Wide)>(base+l.offsets[unsigned(Field::excess)]);
    v.columns=reinterpret_cast<APXCHOL_GPU_BUFFER(Column)>(base+l.offsets[unsigned(Field::columns)]);
    v.factors=factors;
    v.fills=reinterpret_cast<APXCHOL_GPU_BUFFER(Fill)>(base+l.offsets[unsigned(Field::fills)]);
    v.updates=reinterpret_cast<APXCHOL_GPU_BUFFER(Update<Wide>)>(base+l.offsets[unsigned(Field::updates)]);
    v.rows=reinterpret_cast<APXCHOL_GPU_BUFFER(sparse_round::row_result)>(base+l.offsets[unsigned(Field::rows)]);
    v.counts=reinterpret_cast<APXCHOL_GPU_BUFFER(Counts)>(base+l.offsets[unsigned(Field::counts)]);
    v.update_ptr=reinterpret_cast<APXCHOL_GPU_BUFFER(unsigned)>(base+l.offsets[unsigned(Field::update_ptr)]);
    v.update_cursor=reinterpret_cast<APXCHOL_GPU_BUFFER(unsigned)>(base+l.offsets[unsigned(Field::update_cursor)]);
    v.update_ids=reinterpret_cast<APXCHOL_GPU_BUFFER(unsigned)>(base+l.offsets[unsigned(Field::update_ids)]);
    v.records=reinterpret_cast<APXCHOL_GPU_BUFFER(sparse_round::neighbor<Wide>)>(base+l.offsets[unsigned(Field::records)]);
    v.histogram=reinterpret_cast<APXCHOL_GPU_BUFFER(unsigned)>(base+l.offsets[unsigned(Field::histogram)]);
    v.round_ptr=reinterpret_cast<APXCHOL_GPU_BUFFER(unsigned)>(base+l.offsets[unsigned(Field::round_ptr)]);
    v.degree_bound=reinterpret_cast<APXCHOL_GPU_BUFFER(unsigned)>(base+l.offsets[unsigned(Field::degree_bound)]);
    return v;
}
}
)
