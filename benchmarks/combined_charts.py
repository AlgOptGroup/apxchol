#!/usr/bin/env python3
"""Combined comparison figures from the unified per-cell store.

Per family, per matrix, EVERY (solver, device) is its own bar -- there is no
CPU/GPU pairing; all bars in a matrix are sorted fastest->slowest by total time.
Colour identifies the solver, a black outline marks GPU (CPU is plain), and the
setup/solve split is solid base + /// hatch. Includes the CPU-only solvers
(RCHOL/pRCHOL/AC/AC2) alongside the CPU+GPU solvers. (Geometric
multigrid is structured-grid-only, not a general-matrix solver.)

Per family it emits:
  combined_breakdown_{fam}.png  setup (solid) + solve (///), stacked = total height
  combined_setup_{fam}.png      setup only (log)
  combined_solve_{fam}.png      solve only (log)
  combined_iters_{fam}.png      PCG iterations

Reuses the two loaders (no data duplication): fair_charts.load / _pick / headline_mats
for CPU (device=cpu cells), gpu_charts.load for GPU.

  PYTHONPATH=benchmarks python3 benchmarks/combined_charts.py --out benchmarks/latest/figures
"""
import argparse, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
import numpy as np

from runner_common import mat_labels   # registry-backed matrix axis labels
import fair_charts as cpu
import gpu_charts as gpu

FAMS = ["grids", "ipm", "suitesparse"]
TOL = 1e-8
ENC = ("colour = solver  ·  GPU = black-outlined bar, CPU = plain"
       "  ·  solid = setup, /// = solve  ·  bars sorted fastest→slowest"
       "  ·  red frame + ≥ (last) = timed out at its persisted cap")

# (name, colour, cpu_label, gpu_label). cpu_label/gpu_label = None when the solver
# has no bar on that device (RCHOL/pRCHOL/AC/AC2 are CPU-only).
#
# THE SERIES RULE (fair_charts.LABELS): one row per (solver, configuration).
# Headline figures show apxchol's declared bg default only; selector comparisons
# live in the compact dedicated ablation heatmaps.
SOLVERS = [
    ("apxchol/bg",    "#0b5394", "apxchol/bg",    "apxchol/bg (GPU)"),
    ("BoomerAMG",     "#2ca02c", "BoomerAMG",     "BoomerAMG (GPU)"),
    ("BoomerAMG/cut", "#74c476", "BoomerAMG/cut", "BoomerAMG/cut (GPU)"),
    ("AMGCL",         "#8c564b", "AMGCL",         "AMGCL (GPU)"),
    ("CMG (MATLAB)†", "#e377c2", "CMG (MATLAB)†", None),
    ("ParAC Graph",   "#ff8c00", "ParAC Graph",   "ParAC Graph (GPU)"),
    ("ParAC Physics", "#e6550d", "ParAC Physics", "ParAC Physics (GPU)"),
    ("RCHOL",         "#d62728", "RCHOL",         None),
    ("pRCHOL",        "#ff9896", "pRCHOL",        None),
    ("AC",            "#7f7f7f", "AC (Jl ref)†",  None),
    ("AC2",           "#bcbd22", "AC2 (Jl ref)†", None),
]


def _cpu(recs, fam, mat, lab):
    if lab is None:
        return None
    b = cpu._pick(recs, fam, mat, lab)
    if b is None or b["status"] != "complete":
        return None
    m = b.get("metrics", {})
    if m.get("setup_s") is None or m.get("solve_s") is None:
        return None
    return dict(setup=m["setup_s"], solve=m["solve_s"],
                total=m.get("total_s", m["setup_s"] + m["solve_s"]),
                iters=m.get("iters"),
                rss_peak=m.get("max_rss_mb"), rss_solve=m.get("solve_rss_mb"))


def _gpu(grows, fam, mat, glab):
    if glab is None:
        return None
    d = grows.get((fam, mat), {}).get(glab)
    # THE GRADING RULE (benchmarks/README.md): true relative residual <= exactly TOL,
    # same for every solver. gpu_charts.load has already filtered to complete cells;
    # this re-check must use the same mark, not a looser one.
    if not d or d.get("rel_res", 1.0) > TOL:
        return None
    return dict(setup=d.get("setup", 0.0), solve=d.get("solve", 0.0),
                total=d.get("total", 0.0), iters=d.get("iters"),
                rss_peak=d.get("rss_peak"), rss_solve=d.get("rss_solve"),
                vram_peak=d.get("vram_peak"))


def _cpu_status(recs, fam, mat, lab):
    b = cpu._pick(recs, fam, mat, lab) if lab else None
    return b.get("status") if b else None

def _cpu_outcome(recs, fam, mat, lab):
    return cpu._pick(recs, fam, mat, lab) if lab else None

def _collect(recs, grows, fam, goutcomes=None, device="both"):
    """-> mats (present), data[mat] = list of (name, colour, device, vals). Completed
    solvers sort by total; timed-out solvers get a sentinel bar only when the cell
    records the exact cap used. Such bars always sort last.
    device: 'both' | 'cpu' | 'gpu' -> filter which device's bars to include."""
    goutcomes = goutcomes or {}
    mats = cpu.headline_mats(recs, fam)
    present, data = [], {}
    for mat in mats:
        bars = []
        for name, col, cl, gl in SOLVERS:
            if device != "gpu":
                c = _cpu(recs, fam, mat, cl)
                if c:
                    bars.append((name, col, "CPU", c))
                else:
                    outcome = _cpu_outcome(recs, fam, mat, cl)
                    cap = cpu.timeout_cap(outcome or {})
                    if outcome and outcome.get("status") == "timeout" and cap:
                        bars.append((name, col, "CPU", {"total": cap, "setup": cap,
                                    "solve": 0.0, "timeout": True}))
            if device != "cpu":
                g = _gpu(grows, fam, mat, gl)
                if g:
                    bars.append((name, col, "GPU", g))
                elif gl:
                    outcome = goutcomes.get((fam, mat, gl), {})
                    cap = outcome.get("timeout_cap_s")
                    if outcome.get("status") == "timeout" and cap:
                        bars.append((name, col, "GPU", {"total": cap, "setup": cap,
                                    "solve": 0.0, "timeout": True}))
        if bars:
            present.append(mat); data[mat] = bars
    return present, data


def _find_break(heights):
    """Decide a BROKEN linear y-axis (top range + bottom range, the empty band between
    removed) for a set of bar heights, or None for plain linear. Only triggers when the
    heights are genuinely bimodal -- a few bars towering over a low cluster (e.g.
    BoomerAMG 212s vs the <45s field on the social giants) with a wide EMPTY band in
    between. When heights are smoothly spread (e.g. giants_xl: 5..1163s, no single gap)
    there is no honest break point -> plain linear. NEVER a log/symlog transform (a
    setup+solve stacked bar must stay linearly readable). Returns (lo_top, hi_bot, hi_top)."""
    h = sorted(v for v in heights if v and np.isfinite(v) and v > 0)
    if len(h) < 4:
        return None
    hi = h[-1]
    gap, lo, hi_b = 0.0, None, None
    for a, b in zip(h, h[1:]):
        if b - a > gap:
            gap, lo, hi_b = b - a, a, b
    if lo is None:
        return None
    # break only if the empty band is a big slice of the range AND the low cluster really
    # is "low" (its top < half the global max) -- otherwise the spread is continuous.
    if gap < 0.45 * hi or lo > 0.5 * hi:
        return None
    lo_top = lo * 1.12                       # headroom above the low cluster
    hi_bot = hi_b - 0.12 * (hi_b - lo)       # a little below the lowest towering bar
    return lo_top, hi_bot, hi * 1.08


def _bars(recs, grows, fam, out, *, ylabel, title, logy=False, val=None, stacked=False,
          goutcomes=None, device="both", ycompress=False):
    """Every (solver,device) bar individually, sorted by total per matrix; TIMED-OUT
    solvers (no breakdown) sort LAST as a red-framed '≥cap' bar. device filters CPU/GPU.
    val(vals) -> scalar; for the stacked breakdown val(vals) -> (setup, solve).
    ycompress: use a BROKEN linear y-axis (not log) when the heights are bimodal, so one
    towering bar (e.g. BoomerAMG 212s) stops flattening the <45s field while the bulk
    keeps true linear proportion; plain linear when the spread is continuous."""
    mats, data = _collect(recs, grows, fam, goutcomes, device)
    if not mats:
        return
    maxslots = max(len(data[m]) for m in mats)
    w = 0.86 / maxslots
    x = np.arange(len(mats))

    # Pre-sort the bars per matrix (shared by the height scan + the draw closure).
    sorted_bars = {}
    for mat in mats:
        if stacked:
            sorted_bars[mat] = sorted(data[mat], key=lambda b: (b[3].get("timeout", False), b[3]["total"]))
        else:
            sorted_bars[mat] = sorted(data[mat], key=lambda b: (b[3].get("timeout", False),
                                                                val(b[3]) is None, val(b[3]) or 0))

    # Broken-axis decision (linear top + linear bottom). Off for log charts.
    brk = None
    if ycompress and not logy:
        heights = []
        for mat in mats:
            for (_, _, _, vals) in sorted_bars[mat]:
                hgt = vals["total"] if (stacked or vals.get("timeout")) else val(vals)
                if hgt:
                    heights.append(hgt)
        brk = _find_break(heights)

    def draw(ax):
        for mi, mat in enumerate(mats):
            for slot, (name, col, dev, vals) in enumerate(sorted_bars[mat]):
                xx = x[mi] + slot * w
                if vals.get("timeout"):
                    # red frame + '≥' at the cap (ran at least this long), no inner hatch
                    v = vals["total"]
                    ax.bar(xx, v, w, color=col, alpha=0.55, edgecolor="#c00000", linewidth=1.9)
                    ax.text(xx, v, "≥", ha="center", va="bottom", fontsize=9,
                            color="#c00000", fontweight="bold", clip_on=True)
                    continue
                edge = dict(edgecolor="black", linewidth=1.1) if dev == "GPU" \
                    else dict(edgecolor="none")
                if stacked:
                    s, v = val(vals)
                    ax.bar(xx, s, w, color=col, **edge)
                    ax.bar(xx, v, w, bottom=s, color=col, alpha=0.5, hatch="///", **edge)
                else:
                    v = val(vals)
                    if v is not None:
                        ax.bar(xx, v, w, color=col, **edge)

    if brk:
        lo_top, hi_bot, hi_top = brk
        # Two stacked panels sharing x: top = the towering bars, bottom = the low cluster,
        # with the empty band between them removed and diagonal break marks drawn.
        fig, (axhi, axlo) = plt.subplots(
            2, 1, sharex=True, figsize=(max(9, 2.6 * len(mats)), 6.5),
            gridspec_kw=dict(height_ratios=[1, 2.2], hspace=0.07))
        draw(axhi); draw(axlo)
        axlo.set_ylim(0, lo_top)
        axhi.set_ylim(hi_bot, hi_top)
        axhi.spines["bottom"].set_visible(False)
        axlo.spines["top"].set_visible(False)
        axhi.tick_params(axis="x", which="both", bottom=False)
        for ax_, edge in ((axhi, 0.0), (axlo, 1.0)):   # diagonal break marks (axes coords)
            dk = dict(transform=ax_.transAxes, color="k", clip_on=False, lw=1)
            ax_.plot((-0.006, 0.006), (edge - 0.012, edge + 0.012), **dk)
            ax_.plot((1 - 0.006, 1 + 0.006), (edge - 0.012, edge + 0.012), **dk)
        main_ax, bottom_ax, grid_axes = axhi, axlo, (axhi, axlo)
        axhi.set_title(ENC, fontsize=7.5, color="0.4")
        fig.supylabel(ylabel, fontsize=10)
    else:
        fig, ax = plt.subplots(figsize=(max(9, 2.6 * len(mats)), 6))
        if logy:
            ax.set_yscale("log")
        draw(ax)
        main_ax, bottom_ax, grid_axes = ax, ax, (ax,)
        ax.set_title(ENC, fontsize=7.5, color="0.4")
        ax.set_ylabel(ylabel)

    bottom_ax.set_xticks(x + 0.43 - w / 2)
    bottom_ax.set_xticklabels(mat_labels(mats), rotation=30, ha="right")
    fig.suptitle(title, fontsize=11)
    # ONE legend entry per solver: the CPU swatch (plain) and GPU swatch (black-outlined)
    # sit side-by-side under "Solver (CPU / GPU)"; a device that only ever timed out gets a
    # red-framed swatch. Halves the legend vs separate (CPU)/(GPU) rows.
    order = [s[0] for s in SOLVERS]
    info = {}
    for m in mats:
        for (name, col, dev, vals) in data[m]:
            e = info.setdefault(name, dict(color=col, conv=set(), tmo=set()))
            (e["tmo"] if vals.get("timeout") else e["conv"]).add(dev)
    handles, labels = [], []
    for name in sorted(info, key=lambda n: order.index(n) if n in order else 99):
        e = info[name]; parts, devlabs = [], []
        for d in ("CPU", "GPU"):
            if d in e["conv"]:
                parts.append(Patch(facecolor=e["color"], edgecolor="black" if d == "GPU" else "none"))
            elif d in e["tmo"]:
                parts.append(Patch(facecolor=e["color"], alpha=0.55, edgecolor="#c00000"))
            else:
                continue
            devlabs.append(d)
        if parts:
            handles.append(tuple(parts) if len(parts) > 1 else parts[0])
            labels.append(f"{name} ({' / '.join(devlabs)})")
    main_ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)},
                   ncol=4, fontsize=6.3, loc="upper left", handlelength=1.7)
    for gax in grid_axes:
        gax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)



_METRIC_TITLE = {"total": "total time", "setup": "setup time",
                 "solve": "solve time", "iters": "PCG iterations",
                 "rss_peak": "peak host RSS (setup-dominated), CPU solvers",
                 "rss_solve": "solve-held host RSS, CPU solvers",
                 "vram_peak": "peak device VRAM (whole run, per-process), GPU solvers"}

# overview metric name -> raw per-cell metrics key (the store uses the _s suffix).
_RAW_KEY = {"total": "total_s", "setup": "setup_s", "solve": "solve_s", "iters": "iters",
            "rss_peak": "max_rss_mb", "rss_solve": "solve_rss_mb",
            "vram_peak": "max_vram_mb"}

def _cpu_time_cell(recs, fam, mat, cpu_lab, metric):
    """(value_or_nan, is_timeout, exact_cap_or_none) for one CPU series."""
    b = cpu._pick(recs, fam, mat, cpu_lab)
    if b is None:
        return np.nan, False, None
    if b["status"] == "timeout":
        return np.nan, True, cpu.timeout_cap(b)
    if b["status"] != "complete":
        return np.nan, False, None
    m = b.get("metrics", {})
    v = m.get(_RAW_KEY.get(metric, metric))
    if v is None and metric == "total" and m.get("setup_s") is not None and m.get("solve_s") is not None:
        v = m["setup_s"] + m["solve_s"]
    return (v if v is not None else np.nan), False, None

def overview_heatmap(recs, grows, fam, out, mode="combined", metric="total", mats=None,
                     goutcomes=None, thread_label="t16"):
    """solver x matrix 'who wins where' heatmap for ONE metric (total / setup / solve /
    iters). rows = (solver, device), cols = matrix; colour = ratio to the best (lowest)
    in that matrix column (green = winner, log scale); each cell annotated with the
    absolute value over the ratio. This is the house heatmap that REPLACES the per-metric
    grouped-bar charts (setup-only / solve-only / iterations) -- a heatmap stays legible
    across the full matrix ladder (all grids), where grouped bars get cramped. mode:
      'cpu'      -> CPU rows only,
      'gpu'      -> GPU rows only,
      'combined' -> both, each (solver, device) its own row, so a GPU variant beating
                    its CPU twin (or vice-versa) is visible and the per-column winner
                    is taken across BOTH devices.
    mats: matrix columns (default headline_mats; the heatmap callers pass the FULL
    matrix_order so every measured grid shows)."""
    if mats is None:
        mats = cpu.headline_mats(recs, fam)
    is_iters = metric == "iters"
    is_rss = metric in ("rss_peak", "rss_solve")
    is_vram = metric == "vram_peak"
    is_mem = is_rss or is_vram                       # GB-formatted, blank-on-zero
    # A timeout cap is a TOTAL wall-clock cap, so it is only meaningful on the
    # TOTAL heatmap. A timed-out solver produced NO setup/solve/iters breakdown, so on
    # those metrics it's marked 'T/O' (timed out) rather than a fabricated ≥cap value.
    clamp = metric == "total"
    goutcomes = goutcomes or {}
    rows_data = []   # (rowlabel, vals, timeouts, timeout_caps, oom, failed, device)
    for name, col, cl, gl in SOLVERS:
        if mode in ("cpu", "combined") and cl:
            vals, tmo, tcaps, oomf, failf = [], [], [], [], []
            for m in mats:
                v, t, cap = _cpu_time_cell(recs, fam, m, cl, metric)
                if is_mem and not v:        # 0 / None memory = not recorded -> blank, not "0.0GB / inf×"
                    v = np.nan
                st = _cpu_status(recs, fam, m, cl)
                vals.append(v); tmo.append(t); tcaps.append(cap or np.nan)
                oomf.append(st == "oom"); failf.append(st == "failed")
            # Memory heatmaps: a Timeout marker is informative NEXT TO real measurements
            # ("this config timed out, no memory recorded") but a row with ZERO memory
            # info (no value, no OOM -- e.g. ParAC's never-recorded solve-held RSS, all
            # cells Timeout) is noise -> dropped. OOM counts as info (it IS a memory
            # outcome). Time/iters heatmaps keep the old any-marker condition.
            keep = (any(np.isfinite(x) for x in vals) or any(oomf)) if is_mem else \
                   (any(np.isfinite(x) for x in vals) or any(tmo) or any(oomf) or any(failf))
            if keep:
                rows_data.append((f"{name} (CPU)" if mode == "combined" else name,
                                  vals, tmo, tcaps, oomf, failf, "cpu"))
        if mode in ("gpu", "combined") and gl:
            vals, tmo, tcaps, oomf, failf = [], [], [], [], []
            for m in mats:
                vals.append((_gpu(grows, fam, m, gl) or {}).get(metric) or np.nan)
                outcome = goutcomes.get((fam, m, gl), {})
                st = outcome.get("status")
                tmo.append(st == "timeout")
                tcaps.append(outcome.get("timeout_cap_s") or np.nan)
                oomf.append(st == "oom"); failf.append(st == "failed")
            # same row gate as the CPU branch: memory rows need real memory info.
            keep = (any(np.isfinite(x) for x in vals) or any(oomf)) if is_mem else \
                   (any(np.isfinite(x) for x in vals) or any(tmo) or any(oomf) or any(failf))
            if keep:
                rows_data.append((f"{name} (GPU)" if mode == "combined" else name,
                                  vals, tmo, tcaps, oomf, failf, "gpu"))
    if not rows_data or not mats:
        return
    names = [r[0] for r in rows_data]
    M = np.array([r[1] for r in rows_data], dtype=float)
    TMO = np.array([r[2] for r in rows_data], dtype=bool)
    TCAP = np.array([r[3] for r in rows_data], dtype=float)
    OOM = np.array([r[4] for r in rows_data], dtype=bool)
    FAIL = np.array([r[5] for r in rows_data], dtype=bool)
    DEV = [r[6] for r in rows_data]
    # Clamp ONLY cells whose runtime we genuinely DON'T KNOW: a timed-out solver (no
    # real measurement -- e.g. RCHOL/pRCHOL hitting the cap) is rendered AT the cap with
    # '≥' + a black border. A solver that actually COMPLETED is rendered at its real
    # measured time, however slow (e.g. uncapped BoomerAMG 212s) -- we measured it, so we
    # show it honestly rather than flattening it to the cap.
    capped = np.zeros_like(M, dtype=bool)
    if clamp:
        known = TMO & np.isfinite(TCAP)
        M[known] = TCAP[known]
        capped[known] = True
    ratio = np.full_like(M, np.nan)
    for j in range(M.shape[1]):
        colv = M[:, j]
        # Lower bounds never compete for "best" against completed measurements.
        pos = colv[np.isfinite(colv) & (colv > 0) & ~capped[:, j]]
        if pos.size:
            ratio[:, j] = colv / pos.min()
    cmap = plt.cm.RdYlGn_r.copy(); cmap.set_bad("white")   # never-run cells -> white
    # LOG colour scale on the ratio (not a capped linear one) -- otherwise every
    # solver beyond ~4x best saturates to the same red and the slow tier is one
    # indistinguishable wall. Log spreads 1x..max across the full ramp.
    finite = ratio[np.isfinite(ratio)]
    vmax = min(float(finite.max()), 16.0) if finite.size else 4.0
    norm = mcolors.LogNorm(vmin=1.0, vmax=max(vmax, 1.6))
    fig, ax = plt.subplots(figsize=(max(8, 1.25 * len(mats) + 3.5), 0.55 * len(names) + 2))
    im = ax.imshow(np.ma.masked_invalid(ratio), cmap=cmap, aspect="auto", norm=norm)
    ax.set_xticks(range(len(mats))); ax.set_xticklabels(mat_labels(mats), rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8.5)
    for i in range(len(names)):
        for j in range(len(mats)):
            if np.isfinite(M[i, j]):
                pre = "≥" if capped[i, j] else ""
                val = (f"{int(round(M[i, j]))}" if is_iters
                       else f"{M[i, j] / 1024:.1f}GB" if is_mem
                       else f"{M[i, j]:.2f}s")
                ax.text(j, i, f"{pre}{val}\n{pre}{ratio[i, j]:.1f}×", ha="center",
                        va="center", fontsize=6.3)
                if capped[i, j]:   # black border marks an exact persisted lower bound
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor="black", lw=1.4))
            elif TMO[i, j]:        # timed out (TOTAL cap) -> red flag, no setup/solve/iters breakdown
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f2a0a0",
                                           edgecolor="#a00000", lw=0.9))
                ax.text(j, i, "Timeout", ha="center", va="center", fontsize=5.6,
                        color="black", fontweight="bold")
            elif OOM[i, j]:        # ran but didn't fit -> red flag + RAM/GPU, distinct from white 'not run'
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f2a0a0",
                                           edgecolor="#a00000", lw=0.9))
                ax.text(j, i, f"OOM\n{'GPU' if DEV[i] == 'gpu' else 'RAM'}", ha="center",
                        va="center", fontsize=6.0, color="black", fontweight="bold")
            elif FAIL[i, j]:       # ran but errored (no result) -> distinct from white 'not run'
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f2a0a0",
                                           edgecolor="#a00000", lw=0.9))
                ax.text(j, i, "FAIL", ha="center", va="center", fontsize=6.0,
                        color="black", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, ticks=[1, 1.5, 2, 3, 4, 6, 8, 12, 16])
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}×"))
    cb.set_label("× best solver in column (log scale)")
    dev = {"cpu": "CPU", "gpu": "GPU", "combined": "CPU + GPU"}[mode]
    unit = "fewest iters" if is_iters else "least memory" if is_mem else "fastest"
    capnote = "; ≥/border = exact persisted timeout cap (real time unknown)" if clamp else ""
    # Two-line title (was one over-long line that clipped off the figure edges); the
    # 2nd line carries the legend incl. the OOM (RAM/GPU) and white-'not run' keys.
    # Explicit 3 lines (title + two legend lines) so the marker key never wraps
    # mid-phrase onto a ragged extra line. Memory heatmaps don't mark timeouts (a
    # timeout is a time outcome, not a memory measurement -> blank), so drop that key.
    legend = ("OOM = RAM/GPU · FAIL = errored · blank = not run/recorded" if is_mem
              else "≥/Timeout = timed out (real time unknown) · OOM = RAM/GPU · "
                   "FAIL = errored · blank = not run")
    ax.set_title(f"{fam}: solver × matrix — {dev} {_METRIC_TITLE[metric]}, {thread_label}\n"
                 f"green = {unit} per matrix; cell = value / ×best{capnote}\n"
                 f"{legend}",
                 fontsize=8.0, wrap=True)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="results/cells")
    ap.add_argument("--gpu-root", default="results/cells",
                    help="unified per-cell store containing device=gpu cells")
    ap.add_argument("--out", default="benchmarks/latest/figures")
    a = ap.parse_args()
    recs = cpu.load(a.cells)          # device=cpu only (fair_charts filters)
    grows = gpu.load(a.gpu_root)
    goutcomes = gpu.load_outcomes(a.gpu_root)
    os.makedirs(a.out, exist_ok=True)
    n = 0
    sd = lambda d: (d["setup"], d["solve"])
    for fam in FAMS:
        if not any(r["cell"]["family"] == fam for r in recs):
            continue
        base = f"{a.out}/combined"
        # SuiteSparse splits into _small/_giants (the social giants flatten the FEM
        # matrices otherwise); grids/ipm stay unsuffixed. Two matrix sets per group:
        #   - bars (combined_breakdown): headline_mats (grids reduced to largest 2D+3D;
        #     grouped bars get cramped past a handful of matrices)
        #   - heatmaps (overview / setup / solve / iters): the FULL matrix_order, so
        #     every measured grid shows -- a heatmap stays legible at that width.
        # Both split suitesparse the same way (family_groups), so the _small/_giants
        # suffixes line up across bars and heatmaps.
        def rsub_of(gmats):
            gset = set(gmats)
            return [r for r in recs
                    if r["cell"]["family"] != fam or r["cell"]["matrix_id"] in gset]
        # Each section (combined / CPU-only / GPU-only) is the SAME chart, just device-
        # filtered: bars via _bars(device=), heatmaps via overview_heatmap(mode=).
        DEV = (("both", "", "CPU+GPU"), ("cpu", "_cpu", "CPU"), ("gpu", "_gpu", "GPU"))
        for suffix, gmats in cpu.family_groups(recs, fam, cpu.headline_mats(recs, fam),
                                               split=True):
            rs = rsub_of(gmats)
            for dev, tag, lab in DEV:
                _bars(rs, grows, fam, f"{base}_breakdown{tag}_{fam}{suffix}.png",
                      stacked=True, val=sd, goutcomes=goutcomes, device=dev,
                      # broken axis on the giants (all devices) + every GPU breakdown:
                      # the GPU has the extreme outliers (e.g. ParAC on kron_g500 ~7s vs
                      # <1s for the rest). _find_break auto-gates, so non-bimodal GPU
                      # charts render normally; CPU non-giant charts stay un-broken.
                      ycompress=suffix.startswith("_giants") or dev == "gpu",
                      ylabel="time (s) — t16  [solid = setup, /// = solve]",
                      title=f"{fam}{suffix}: all solvers {lab}, setup+solve (sorted fastest→slowest)")
                n += 1
        for suffix, hmats in cpu.family_groups(recs, fam, cpu.matrix_order(recs, fam)):
            rsub = rsub_of(hmats)
            for metric in ("total", "setup", "solve", "iters", "rss_peak", "rss_solve",
                           "vram_peak"):
                stem = "overview" if metric == "total" else metric
                # RSS is HOST memory -> meaningless for GPU (footprint is VRAM): CPU-only.
                # VRAM is DEVICE memory -> GPU-only (CPU solvers never touch the device).
                # Each is a single chart, no _cpu/_gpu split.
                if metric in ("rss_peak", "rss_solve"):
                    modes = [("cpu", "")]
                elif metric in ("vram_peak",):
                    modes = [("gpu", "")]
                else:
                    modes = [("combined", ""), ("cpu", "_cpu"), ("gpu", "_gpu")]
                for mode, tag in modes:
                    overview_heatmap(rsub, grows, fam, f"{base}_{stem}{tag}_{fam}{suffix}.png",
                                     mode=mode, metric=metric, mats=hmats,
                                     goutcomes=goutcomes)
                    n += 1
    print(f"combined_charts: {len(recs)} cpu + {len(grows)} gpu cells -> {n} figures in {a.out}")


if __name__ == "__main__":
    main()
