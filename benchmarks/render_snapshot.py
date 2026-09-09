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
import fair_charts
import gpu_charts
import runner_common as rc

HERE = Path(__file__).resolve().parent


def render(cells, output, threads, platform, scaling_store=None, scaling_matrices=None,
           scaling_threads=None, sampler_comparison=False, fill_cells=None):
    records, report = chart_cells.load_current_records(
        cells, include=lambda r: r.get('cell', {}).get('threads') == threads,
        stale_policy='reject', source=f'{platform} snapshot')
    fair_charts.select_sampler_comparison(sampler_comparison)
    gpu_charts.select_sampler_comparison(sampler_comparison)
    # Declare the complete profile before selecting any observed outcomes.
    series = sorted(
        [('cpu', solver, config) for (solver, config), label in fair_charts.LABELS.items()
         if label in fair_charts.ORDER]
        + [('gpu', solver, config) for (solver, config), label in gpu_charts.LABELS.items()
           if label in gpu_charts.ORDER])
    if sampler_comparison:
        records = [r for r in records if (r['cell'].get('device', 'cpu'),
                   r['cell']['solver'], r['cell'].get('config', '')) in series]
        for record in records:
            gpu_charts.validate_owned_route(record)
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
                        '--threads', str(threads),
                        *(['--sampler-comparison'] if sampler_comparison else [])], check=True)
    if fill_cells is not None:
        subprocess.run([sys.executable, str(HERE/'fill_chart.py'),
                        '--cells', str(fill_cells), '--out', str(output/'figures')], check=True)
    fields = ['family', 'matrix', 'device', 'solver', 'config', 'threads', 'effective_threads', 'status',
              'n', 'nnz', 'setup_s', 'solve_s', 'total_s', 'iters', 'rel_res',
              'max_repeat_rel_res', 'cuda_init_s', 'warmup_repeats',
              'retained_repeats', 'representative_repeat', 'max_rss_mb',
              'max_vram_mb', 'timeout_cap_s', 'timeout_scope',
              'per_solve_timeout_lower_bound_s', 'git_sha', 'binary_sha256',
              'setup_route', 'sampler', 'degree_quantile', 'actual_device_factor_adopted',
              'fp16', 'factor_drop_rel', 'stored_factor_nnz', 'fillin', 'compiler',
              'openmp_wait_policy', 'source_manifest_sha256', 'timing_protocol',
              'timing_stability_warning', 'phase_observations']
    rows = []
    for r in records:
        c, m, p = r['cell'], r.get('metrics', {}), r.get('provenance', {})
        meta = r.get('matrix_meta', {})
        row = {k: m.get(k, '') for k in fields}
        row.update({k: c.get(k, '') for k in ('family','device','solver','config','threads')})
        row.update(matrix=c['matrix_id'], status=r['status'],
                   effective_threads=m.get('effective_threads', p.get('effective_threads', '')),
                   timeout_cap_s=r.get('timeout_cap_s', ''),
                   timeout_scope=meta.get('timeout_scope', ''),
                   per_solve_timeout_lower_bound_s=rc.timeout_cap(r),
                   warmup_repeats=m.get('warmup_repeats', meta.get('warmup_runs', p.get('warmup', ''))),
                   retained_repeats=m.get('retained_repeats', meta.get('retained_count', '')),
                   git_sha=p.get('git_sha', ''), binary_sha256=p.get('binary_sha256', p.get('driver_sha256', '')))
        row.update({key: p.get(key, meta.get(key, '')) for key in (
            'setup_route', 'sampler', 'degree_quantile', 'compiler',
            'openmp_wait_policy', 'source_manifest_sha256', 'timing_protocol')})
        if isinstance(row['phase_observations'], (list, dict)):
            row['phase_observations'] = json.dumps(row['phase_observations'], sort_keys=True)
        rows.append(row)
    with (output/'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
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
    missing = [dict(matrix=mid, device=device, solver=solver, config=config)
               for mid in rc.MATRICES for device,solver,config in series
               if (mid,solver,config,device) not in set(keys)]
    (output/'coverage.json').write_text(json.dumps(dict(
        platform=platform,threads=threads,registered_matrices=len(rc.MATRICES),
        measured_matrices=len(matrices),present=len(records),
        expected_headline_cells=len(rc.MATRICES)*len(series),status_counts=counts,
        series_profile="sampler-comparison" if sampler_comparison else "historical-default",
        missing=missing,series=series),indent=2)+'\n')
    def previews(pattern):
        return ' '.join(f'![{p.stem.removeprefix("combined_")}]({p.relative_to(output).as_posix()})'
                        for p in sorted((output/'figures').glob(pattern))) or 'Pending measurements'
    def heatmap(metric, device, family):
        return previews(f'combined_{metric}{device}_{family}.png')
    lines = [f'# {platform} benchmark snapshot', '',
        f'This snapshot selects T={threads} before comparing outcomes. '
        f'It contains {len(records)} cells over {len(matrices)}/{len(rc.MATRICES)} registered matrices. '
        f'{len(missing)} of {len(rc.MATRICES)*len(series)} declared headline cells are missing. '
        'T is the requested headline thread budget; the CSV records effective thread '
        'counts separately for serial and thread-limited solvers.', '',
        'Every completed retained solve must meet the original-operator true-relative-residual '
        'target of `1e-8`. CUDA initialization is reported separately. Setup includes mandatory '
        'solver preparation; the solve column includes the remaining complete solver call. '
        'The CSV preserves configured warmup counts and observed retained counts. '
        'Timeout scope and the raw deadline are separate from any valid per-solve lower bound.', '',
        '[Values and outcomes](results.csv) · [Coverage and missing cells](coverage.json) · '
        '[Common protocol](../README.md)', '',
        '| Time | Devices | Grids | IPM | SuiteSparse |', '|---|---|---|---|---|',
        *[f'| {label} | {device_label} | '+ ' | '.join(heatmap(metric, device, family)
                                                  for family in ('grids', 'ipm', 'suitesparse'))+' |'
          for metric, label in [('setup', 'Setup'), ('solve', 'Solve'), ('overview', 'Total')]
          for device, device_label in [('_cpu', 'CPU'), ('_gpu', 'GPU'), ('', 'CPU + GPU')]], '',
        '| Detail | Grids | IPM | SuiteSparse |', '|---|---|---|---|',
        *[f'| {label} | '+ ' | '.join(heatmap(metric, '', family)
                                    for family in ('grids', 'ipm', 'suitesparse'))+' |'
          for metric, label in [('iters', 'Iterations'), ('rss_peak', 'Peak host memory')]],
        '| Factor fill | '+ ' | '.join(previews(f'fill_heatmap_{family}.png')
                                      for family in ('grids', 'ipm', 'suitesparse'))+' |', '',
        '| Other views | Figures |', '|---|---|',
        '| CPU setup and solve | '+previews('combined_breakdown_cpu_*.png')+' |',
        '| GPU setup and solve | '+previews('combined_breakdown_gpu_*.png')+' |',
        '| Setup scaling | '+previews('threads*setup_speedup.png')+' |',
        '| Converged-solve scaling | '+previews('threads*solve_speedup.png')+' |', '',
        'Heatmap colours normalize within each matrix column; they do not compare absolute '
        'speed between machines. Timeout, numerical non-convergence, execution failure, '
        'unsupported input, and missing measurement remain distinct.', '',
        'The 2D grids have a coefficient jump from 1 to 0.01. The 3D grids have unit weights. '
        'Native CMG is labelled as a serial packed implementation; serial Julia references '
        'retain their own labels and timing boundaries. '
        + ('Canonical MATLAB CMG is outside the current comparison.' if sampler_comparison
           else 'Canonical MATLAB CMG retains its historical label.'), '',
        'Status counts: '+', '.join(f'{key}: {value}' for key,value in sorted(counts.items()))+'.', '']
    if sampler_comparison:
        lines += ['CPU rows compare GKS and trace-cycle at degree quantile 0.2; GPU-owned '
                  'rows compare GKS at quantiles 0.8 and 0.2. The CSV records the '
                  'requested route, actual device-factor adoption, storage and timing '
                  'provenance. Missing cells are not filled from older CPU-setup/GPU-solve '
                  'measurements or another sampler.', '']
    if (output/'PLATFORM.md').is_file():
        lines += ['[Platform-specific availability and exceptions](PLATFORM.md)', '']
    (output/'README.md').write_text('\n'.join(lines))
    print(f'{platform}: inspected {len(records)} records; '
          f'{len(missing)} missing of {len(rc.MATRICES)*len(series)} declared headline cells')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cells',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--threads',type=int,required=True)
    p.add_argument('--platform',required=True)
    p.add_argument('--scaling-store',type=Path)
    p.add_argument('--scaling-matrices')
    p.add_argument('--scaling-threads')
    p.add_argument('--sampler-comparison', action='store_true')
    p.add_argument('--fill-cells', type=Path, help='Explicit derived factor-fill cell store')
    a=p.parse_args()
    render(a.cells.resolve(),a.out.resolve(),a.threads,a.platform,a.scaling_store,
           a.scaling_matrices,a.scaling_threads,a.sampler_comparison,a.fill_cells)
