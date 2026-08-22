#!/usr/bin/env bash
# download_graphs.sh – fetch SuiteSparse Matrix Collection test graphs
#
# These are standard benchmark Laplacians / graph matrices used in
# the Laplacian solver literature.
set -euo pipefail

DEST="${1:-$(dirname "$0")/../data/matrices}"
mkdir -p "$DEST"

# Canonical SuiteSparse collection host (the old herokuapp mirror is a legacy
# alias that 301s here; curl -fsSL follows the hop to sparse-files.engr.tamu.edu).
BASE="https://sparse.tamu.edu/MM"

# Format: GROUP/NAME  (will download GROUP/NAME.tar.gz, extract .mtx)
MATRICES=(
    # Small structural (coordinate format)
    "HB/bcsstk01"               # 48x48 structural
    "HB/bcsstk13"               # 2003x2003 structural
    # Road networks / Delaunay (coordinate, symmetric, pattern - graph Laplacians)
    "DIMACS10/chesapeake"       # 39x39 small
    "DIMACS10/delaunay_n10"     # 1024 Delaunay
    "DIMACS10/delaunay_n13"     # 8192 Delaunay
    "DIMACS10/delaunay_n15"     # 32768 Delaunay
    "DIMACS10/delaunay_n17"     # 131072 Delaunay
    # 2D/3D FEM (coordinate, symmetric)
    "Wissgott/parabolic_fem"    # 525825 FEM
    "MaxPlanck/shallow_water1"  # 81920 shallow water
    # GPU RCHOL paper (Liang et al.) benchmark matrices
    "McRae/ecology1"            # 1000000 ecology model
    "McRae/ecology2"            # 999999 ecology model
    "GHS_psdef/apache2"         # 715176 structural
    "AMD/G3_circuit"            # 1585478 circuit simulation
    "MaxPlanck/shallow_water2"  # 81920 shallow water (variant)
    # Fair-sweep registry (benchmarks/runner_common.py)
    "Schmid/thermal2"           # 1228045 FEM thermal
    "SNAP/com-Amazon"           # 334863 social
    "DIMACS10/coAuthorsDBLP"    # 299067 coauthorship
    "DIMACS10/kron_g500-logn16" # 65536 synthetic kronecker
    "SNAP/com-Youtube"          # 1134890 social
    "DIMACS10/coPapersDBLP"     # 540486 citation
    "SNAP/as-Skitter"           # 1696415 internet topology
    "SNAP/com-LiveJournal"      # 3997962 social
    "SNAP/com-Orkut"            # 3072441 social (fits default unsigned 32-bit offsets)
)

for entry in "${MATRICES[@]}"; do
    group=$(dirname "$entry")
    name=$(basename "$entry")
    outfile="$DEST/${name}.mtx"

    if [[ -f "$outfile" ]]; then
        echo "[skip] $outfile already exists"
        continue
    fi

    url="$BASE/$entry.tar.gz"
    echo "[download] $url"
    tmpdir=$(mktemp -d)

    if curl -fsSL "$url" -o "$tmpdir/archive.tar.gz"; then
        tar xzf "$tmpdir/archive.tar.gz" -C "$tmpdir"
        # The matrix file is usually at name/name.mtx inside the archive
        # Prefer exact match over other .mtx files (e.g. name_b.mtx is the RHS)
        mtx=$(find "$tmpdir" -name "${name}.mtx" | head -1)
        if [[ -z "$mtx" ]]; then
            mtx=$(find "$tmpdir" -name "*.mtx" | head -1)
        fi
        if [[ -n "$mtx" ]]; then
            cp "$mtx" "$outfile"
            echo "[ok] $outfile ($(wc -l < "$outfile") lines)"
        else
            echo "[warn] no .mtx found in archive for $entry"
        fi
    else
        echo "[fail] could not download $entry"
    fi
    rm -rf "$tmpdir"
done

echo ""
echo "Done. Matrices in: $DEST"
ls -lh "$DEST"/*.mtx 2>/dev/null || echo "(no .mtx files)"
