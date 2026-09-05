#!/usr/bin/env python3
"""Use the same current-cell renderers and presentation for either machine."""
import argparse
import collections
import csv
import json
from pathlib import Path
import subprocess
import sys

import chart_cells
import runner_common as rc

HERE = Path(__file__).resolve().parent


def render(cells, output, threads, platform, scaling_store=None, scaling_matrices=None,
           scaling_threads=None):
    records, report = chart_cells.load_current_records(
        cells, include=lambda r: r.get('cell', {}).get('threads') == threads,
        stale_policy='reject', source=f'{platform} snapshot')
    if not records:
        raise ValueError('no current cells at the declared thread count')
    keys = [(r['cell']['matrix_id'], r['cell']['solver'], r['cell'].get('config', ''),
             r['cell'].get('device', 'cpu')) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate logical cells in snapshot')
    output.mkdir(parents=True, exist_ok=True)
    for script, flags in (
        ('fair_charts.py', ['--root', str(cells), '--out', str(output)]),
        ('gpu_charts.py', ['--root', str(cells), '--out', str(output/'figures')]),
        ('combined_charts.py', ['--cells', str(cells), '--gpu-root', str(cells),
                                '--out', str(output/'figures')]),
    ):
        subprocess.run([sys.executable, str(HERE/script), *flags,
                        '--threads', str(threads)], check=True)
    fields = ['family', 'matrix', 'device', 'solver', 'config', 'threads', 'status',
              'n', 'nnz', 'setup_s', 'solve_s', 'total_s', 'iters', 'rel_res',
              'max_repeat_rel_res', 'cuda_init_s', 'warmup_repeats',
              'retained_repeats', 'representative_repeat', 'max_rss_mb',
              'max_vram_mb', 'timeout_cap_s', 'git_sha', 'binary_sha256']
    rows = []
    for r in records:
        c, m, p = r['cell'], r.get('metrics', {}), r.get('provenance', {})
        row = {k: m.get(k, '') for k in fields}
        row.update({k: c.get(k, '') for k in ('family','device','solver','config','threads')})
        row.update(matrix=c['matrix_id'], status=r['status'], timeout_cap_s=rc.timeout_cap(r),
                   git_sha=p.get('git_sha', ''), binary_sha256=p.get('binary_sha256', p.get('driver_sha256', '')))
        rows.append(row)
    with (output/'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r['family'], r['matrix'], r['device'], r['solver'], r['config'])))
    if scaling_store:
        if not scaling_matrices or not scaling_threads:
            raise ValueError('scaling requires declared matrices and thread counts')
        for device in ('cpu', 'gpu'):
            subprocess.run([sys.executable, str(HERE/'thread_scaling.py'),
                '--store', str(scaling_store), '--device', device, '--render-only',
                '--series', 'apxchol bg+tree', '--matrices', scaling_matrices,
                '--thread-counts', scaling_threads, '--out', str(output)], check=True)
    counts = collections.Counter(r['status'] for r in records)
    matrices = {r['cell']['matrix_id'] for r in records}
    series = sorted({(r['cell'].get('device','cpu'),r['cell']['solver'],r['cell'].get('config','')) for r in records})
    missing = [dict(matrix=mid, device=device, solver=solver, config=config)
               for mid in rc.MATRICES for device,solver,config in series
               if (mid,solver,config,device) not in set(keys)]
    (output/'coverage.json').write_text(json.dumps(dict(
        platform=platform,threads=threads,registered_matrices=len(rc.MATRICES),
        measured_matrices=len(matrices),present=len(records),
        expected_in_present_series=len(rc.MATRICES)*len(series),status_counts=counts,
        missing=missing,series=series),indent=2)+'\n')
    def links(pattern):
        return ', '.join(f'[{p.stem.removeprefix("combined_")}]({p.relative_to(output).as_posix()})'
                         for p in sorted((output/'figures').glob(pattern))) or 'Pending measurements'
    lines = [f'# {platform} benchmark snapshot', '',
        f'This snapshot selects T={threads} before comparing outcomes. '
        f'It contains {len(records)} cells over {len(matrices)}/{len(rc.MATRICES)} registered matrices.', '',
        'Every completed retained solve must meet the original-operator true-relative-residual '
        'target of `1e-8`. CUDA initialization is reported separately. Setup includes mandatory '
        'solver preparation; the solve column includes the remaining complete solver call. '
        'Explicit warmup and retained-repetition fields are preserved in the CSV.', '',
        '[Values and outcomes](results.csv) · [Coverage and missing cells](coverage.json) · '
        '[Common protocol](../README.md)', '',
        '| View | Figures |', '|---|---|',
        '| Total time | '+links('combined_overview_grids*.png')+', '+links('combined_overview_ipm*.png')+', '+links('combined_overview_suitesparse*.png')+' |',
        '| CPU totals | '+links('combined_overview_cpu_*.png')+' |',
        '| GPU totals | '+links('combined_overview_gpu_*.png')+' |',
        '| Setup and solve | '+links('combined_breakdown_*.png')+' |',
        '| Setup scaling | '+links('threads*setup_speedup.png')+' |',
        '| Converged-solve scaling | '+links('threads*solve_speedup.png')+' |', '',
        'Heatmap colours normalize within each matrix column; they do not compare absolute '
        'speed between machines. Timeout, numerical non-convergence, execution failure, '
        'unsupported input, and missing measurement remain distinct.', '',
        'The 2D grids have a coefficient jump from 1 to 0.01. The 3D grids have unit weights. '
        'Native CMG is labelled as a serial packed implementation; canonical MATLAB CMG '
        'and serial Julia reference solvers retain their own labels and timing boundaries.', '',
        'Status counts: '+', '.join(f'{key}: {value}' for key,value in sorted(counts.items()))+'.', '']
    (output/'README.md').write_text('\n'.join(lines))
    print(f'{platform}: checked {len(records)}/{len(records)} cells; {len(missing)} absent within present series')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cells',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--threads',type=int,required=True)
    p.add_argument('--platform',required=True)
    p.add_argument('--scaling-store',type=Path)
    p.add_argument('--scaling-matrices')
    p.add_argument('--scaling-threads')
    a=p.parse_args()
    render(a.cells.resolve(),a.out.resolve(),a.threads,a.platform,a.scaling_store,
           a.scaling_matrices,a.scaling_threads)
