#!/usr/bin/env python3
"""Generate four deterministic reweightings of an undirected matrix support.

Input diagonal and original weights are discarded. Output files are graph
adjacency matrices, not assembled operators. Random weights are log-uniform.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy import io, sparse


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def support(matrix):
    a = sparse.csr_matrix(matrix)
    if a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ValueError("input must be a nonempty square matrix")
    if np.iscomplexobj(a.data) or not np.isfinite(a.data).all():
        raise ValueError("input must have finite real values")
    a.sum_duplicates()
    a.setdiag(0)
    a.eliminate_zeros()
    a.data = np.ones(a.nnz, dtype=np.int8)
    a = a.maximum(a.T).tocsr()
    a.sort_indices()
    return a


def spanning_forest(a):
    """BFS from the smallest vertex in each component, including isolated ones."""
    parent = np.full(a.shape[0], -1, dtype=np.int64)
    seen = np.zeros(a.shape[0], dtype=bool)
    components = 0
    for root in range(a.shape[0]):
        if seen[root]:
            continue
        components += 1
        seen[root] = True
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for v in a.indices[a.indptr[u]:a.indptr[u + 1]]:
                if not seen[v]:
                    seen[v] = True
                    parent[v] = u
                    queue.append(int(v))
    return parent, components


def generate(source, out, name, seed):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("name must be a simple matrix identifier")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    source, out = Path(source).resolve(), Path(out)
    initial_hash = file_sha256(source)
    a = support(io.mmread(source))
    upper = sparse.triu(a, k=1, format="coo")
    parent, components = spanning_forest(a)
    tree = ((parent[upper.row] == upper.col) |
            (parent[upper.col] == upper.row))
    assert int(tree.sum()) == a.shape[0] - components
    # Coupled variates make narrow/wide comparisons differ only in their range.
    u = np.random.Generator(np.random.PCG64(seed)).random(upper.nnz)
    narrow, wide = 10.0 ** (2.0 * u - 1.0), 10.0 ** (6.0 * u - 3.0)
    models = [("unit", np.ones(upper.nnz)), ("log2", narrow),
              ("log6", wide), ("backbone", np.where(tree, narrow, wide))]
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for model, weights in models:
        mid = f"{name}_{model}_s{seed}"
        path = out / (mid + ".mtx")
        lower = sparse.coo_matrix((weights, (upper.col, upper.row)), shape=a.shape)
        io.mmwrite(path, lower, symmetry="symmetric", precision=17)
        rows.append({"id": mid, "family": "weighted", "path": path.name,
                     "kind": "graph", "n": a.shape[0], "sha256": file_sha256(path),
                     "weight_model": {"name": model, "seed": seed,
                         "distribution": "unit" if model == "unit" else "log-uniform",
                         "rng": "NumPy PCG64", "coupled_ranges": True,
                         "components": components, "forest_edges": int(tree.sum()),
                         "undirected_edges": upper.nnz, "source_sha256": initial_hash,
                         "support": "union of off-diagonal nonzero support; diagonal discarded"}})
    if file_sha256(source) != initial_hash:
        raise RuntimeError("source changed during generation")
    manifest = {"schema_version": 1, "matrices": rows}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path, help="new output directory")
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = generate(args.source, args.out, args.name, args.seed)
    print(json.dumps({"planned": 4, "generated": len(result["matrices"]),
                      "manifest": str(args.out / "manifest.json")}))


if __name__ == "__main__":
    main()
