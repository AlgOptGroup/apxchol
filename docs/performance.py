"""Render public benchmark summaries; raw measurements remain downloadable JSON."""
import argparse
import collections
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ("#167c80", "#b4552d", "#725aa6")
plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "apxchol-performance", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def hosted(reports, output):
    fig, axes = plt.subplots(max(1, len(reports)), 2, figsize=(11, 3.6 * max(1, len(reports))),
                             squeeze=False, layout="constrained")
    if not reports:
        for ax in axes[0]:
            ax.axis("off")
        axes[0, 0].text(0, .5, "Hosted measurements appear after CI runs this revision.\n"
                               "No timings are inferred from other machines.", wrap=True)
    for row, report in enumerate(reports):
        cases = report["cases"]
        labels = [f"{case['fixture']}\n{case['configuration']}" for case in cases]
        for col, (metric, title) in enumerate((("setup_plus_one", "Setup + one scalar solve"),
                                              ("batch4", "Four RHSs: batched versus sequential"))):
            ax = axes[row, col]
            for i, case in enumerate(cases):
                summary = case["summary"]
                if summary["status"] == "failed":
                    ax.text(i, 0, "FAILED", rotation=90, va="bottom", ha="center")
                    continue
                noisy = summary["status"] != "accepted"
                value = summary["median_seconds"][metric] * 1000
                color = COLORS[i % (3 if len(cases) == 6 else 2)]
                ax.bar(i, value, width=.6, color=color, alpha=.7 if noisy else 1,
                       hatch="///" if noisy else None)
                points = [record["seconds"][metric] * 1000 for record in case["records"]]
                ax.plot(np.full(len(points), i), points, ".", color="#222222", markersize=3)
                if metric == "setup_plus_one" and summary["reference_status"] == "passed":
                    ax.plot(i, summary["median_seconds"]["scipy_cg"] * 1000, "_", color="#111111",
                            markersize=17, label="SciPy CG, no preconditioner" if not ax.get_legend_handles_labels()[0] else None)
                if metric == "batch4":
                    sequential = summary["median_seconds"]["sequential4"] * 1000
                    ax.plot(i, sequential, "_", color="#111111", markersize=17,
                            label="4 sequential calls" if not ax.get_legend_handles_labels()[0] else None)
            ax.set_xticks(range(len(cases)), labels, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("milliseconds (lower is better)")
            ax.set_title(title)
            ax.grid(axis="y", alpha=.15)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
        hw = report["hardware"]
        axes[row, 0].text(0, 1.16, f"{hw['os']} · {hw['cpu']} · {report['source'][:10]}",
                          transform=axes[row, 0].transAxes, fontsize=10)
    title = "Current hosted-runner measurements" if all(r.get("kind") == "hosted" for r in reports) else "Local validation preview (not hosted results)"
    fig.suptitle(title + "\nDots: all six observations; hatching: repeated runs differ by more than 10%", fontsize=13)
    save(fig, output / "hosted.svg")


def historical(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), layout="constrained")
    # These are within-factor ratios. CPU and GPU each use their own factor.
    maximum = max(b["routes"][route]["primary"]["batching_gain"][str(width)]
                  for b in data["blocks"] for route in ("cpu", "metal") for width in (4, 16))
    for index, route in enumerate(("cpu", "metal")):
        for width, ax in zip((4, 16), axes):
            values = [b["routes"][route]["primary"]["batching_gain"][str(width)] for b in data["blocks"]]
            ax.bar(index, np.median(values), color=COLORS[index], width=.55)
            ax.plot(np.full(len(values), index), values, ".", color="#222222", markersize=5)
            ax.set_xticks((0, 1), ("CPU (up to 8 workers)", "Metal"))
            ax.set_title(f"{width} right-hand sides")
            ax.axhline(1, color="#666666", linewidth=.8)
            ax.set_ylim(0, max(1, maximum) * 1.2)
            ax.set_ylabel("speedup over sequential solves")
            ax.grid(axis="y", alpha=.15)
    fig.suptitle("YouTube graph · 1.13 million rows · Apple M1 Max\nBatched solves reuse work; three independent seeds", fontsize=12)
    fig.supxlabel(f"{data['date']} · revision {data['source_commit'][:7]} · FP32 factors · tolerance 1e−8", fontsize=9)
    save(fig, output / "youtube.svg")


FAMILIES = {"amgcl": "AMGCL", "amgcl_cuda": "AMGCL", "amgx": "AMGX",
            "hypre_boomeramg": "BoomerAMG", "hypre_boomeramg_gpu": "BoomerAMG",
            "cmg_packed": "CMG", "cmg_matlab": "CMG", "ac": "Laplacians.jl", "ac2": "Laplacians.jl",
            "rchol": "RCHOL", "rchol_par": "RCHOL", "accelerate_cholesky": "Accelerate", "accelerate_cg": "Accelerate",
            "parac": "ParAC", "parac_graph": "ParAC", "parac_physics": "ParAC"}
FAMILY_ORDER = ["AMGCL", "AMGX", "BoomerAMG", "CMG", "Laplacians.jl", "RCHOL", "ParAC", "Accelerate"]
VARIANTS = {"ac": "AC", "ac2": "AC2", "rchol": "serial", "rchol_par": "pRCHOL", "cmg_packed": "packed port",
            "cmg_matlab": "MATLAB", "parac": "Graph", "parac_graph": "Graph", "parac_physics": "Physics",
            "accelerate_cholesky": "Cholesky", "accelerate_cg": "CG"}
CLASSES = {"grids": "Grids", "meshes": "Meshes and circuits", "ipm": "Linear programs (IPM)",
           "social": "Social and web graphs"}
DEVICES = {"cpu": "CPU", "gpu": "CUDA", "metal": "Metal"}
MARKS = {"failed": "fail", "not_converged": "nc", "timeout": "t/o", "exceeded_cap": "≥20×", "oom": "oom",
         "n/a": "n/a", "missing": "—"}
UNFINISHED = np.inf  # a run that did not finish counts as an apxchol win
INK, MUTED, GRID = "#1f1f1f", "#5f5e5a", "#e4e3df"
DEVICE_COLORS = {"cpu": "#2a78d6", "gpu": "#eb6834", "metal": "#1baf7a"}
TICKS = [0.5, 1, 2, 5, 10, 20, 50, 100]
CLIP = 100  # draw larger speedups and unfinished runs at the right edge


NUMBERS = {"n": int, "nnz": int, "threads": int, "iters": int, "setup_s": float, "solve_s": float,
           "rel_res": float, "cap_s": float, "max_rss_mb": float}


def read_cells(path, tolerance):
    """Rows of a compact campaign CSV, with numbers parsed and the tolerance verdict added."""
    cells = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            cell = {k: (NUMBERS[k](v) if v != "" else None) if k in NUMBERS else v for k, v in row.items()}
            cell["meets_tolerance"] = (cell["status"] == "complete" and cell["rel_res"] is not None
                                       and cell["rel_res"] <= tolerance)
            cells.append(cell)
    return cells


def load(header_path):
    """One campaign: the shared header plus its cells (and thread scaling, when published)."""
    header_path = Path(header_path)
    data = json.loads(header_path.read_text())
    campaign = data["campaign"]
    data["cells"] = read_cells(header_path.with_name(f"{campaign}.csv"), data["tolerance"])
    scaling = header_path.with_name(f"{campaign}-scaling.csv")
    if scaling.is_file():
        cells = read_cells(scaling, data["tolerance"])
        data["scaling"] = {"matrices": list(dict.fromkeys(c["matrix"] for c in cells)),
                           "thread_counts": sorted({c["threads"] for c in cells}), "cells": cells}
    return data


def apx_label(config):
    config = config.split("[")[0].replace("bg+", "").replace("trace_cycle", "trace-cycle")
    config = config.replace("/gpu-owned-q0", "/q0.").replace("/q", " q")  # Daint and Euler spellings
    return "apxchol " + config.replace("tree", "GKS")


def series_label(cell):
    if cell["solver"] == "apxchol_v1":
        return apx_label(cell["config"])
    name = FAMILIES.get(cell["solver"], cell["solver"])
    extra = [VARIANTS.get(cell["solver"])] if cell["solver"] in VARIANTS else []
    extra += [v for v in (cell.get("variant"), "" if cell["config"] in ("", "original-operator") else cell["config"]) if v]
    return name + (f" ({', '.join(extra)})" if extra else "")


def good(cell):
    return cell is not None and cell["status"] == "complete" and cell.get("meets_tolerance")


def reference(data, device):
    """The apxchol series a user gets by default: GKS at the default degree quantile.

    Daint predates the GPU default of 0.5 and has no such series; its 0.8 run stands in."""
    configs = {c["config"] for c in data["cells"] if c["device"] == device and c["solver"] == "apxchol_v1"}
    for config in ("bg+tree[vec_pool_aos]", "bg+tree/gpu-owned-q08[vec_pool_aos]"):
        if config in configs:
            return config
    raise ValueError(f"no default apxchol series on {device}")


class Campaign:
    def __init__(self, data):
        self.data = data
        self.cells = {}
        for c in data["cells"]:
            self.cells[c["device"], series_label(c), c["matrix"]] = c
        self.devices = [d for d in DEVICES if any(c["device"] == d for c in data["cells"])]
        self.cls = {c["matrix"]: CLASSES[c["class"]] for c in data["cells"]}
        rows = {c["matrix"]: c["n"] for c in data["cells"]}
        order = list(CLASSES.values())
        self.matrices = sorted(self.cls, key=lambda m: (order.index(self.cls[m]), rows[m], m))

    def series(self, device):
        return sorted({s for d, s, _ in self.cells if d == device},
                      key=lambda s: (not s.startswith("apxchol"), s))

    def competitors(self, device):
        return [s for s in self.series(device) if not s.startswith("apxchol")]

    def families(self, device):
        found = {s.split(" (")[0] for s in self.competitors(device)}
        return [f for f in FAMILY_ORDER if f in found]

    def apx(self, device, matrix):
        c = self.cells.get((device, apx_label(reference(self.data, device)), matrix))
        return c["setup_s"] + c["solve_s"] if good(c) else None

    def family_cell(self, device, family, matrix):
        """Speedup of apxchol over the fastest variant of a family, the winning cell and a status mark.

        Returns (speedup, cell, mark); speedup is UNFINISHED when no variant finished and None when
        every variant marks the input unsupported or apxchol itself has no result."""
        ref = self.apx(device, matrix)
        cells = [self.cells[device, s, matrix] for s in self.competitors(device)
                 if s.split(" (")[0] == family and (device, s, matrix) in self.cells]
        done = [c for c in cells if good(c)]
        if ref is None:
            return None, None, "—"
        if done:
            best = min(done, key=lambda c: c["setup_s"] + c["solve_s"])
            return (best["setup_s"] + best["solve_s"]) / ref, best, None
        statuses = [c["status"] for c in cells]
        if not statuses or all(s == "n/a" for s in statuses):
            return None, None, "n/a"
        for status in ("exceeded_cap", "timeout", "oom", "not_converged", "failed", "missing"):
            if status in statuses:
                return UNFINISHED, None, MARKS[status]
        return UNFINISHED, None, MARKS.get(statuses[0], statuses[0])


def log_axis(ax, label):
    ax.set_xscale("log")
    ax.set_xlim(0.25, CLIP * 1.25)
    ax.set_xticks(TICKS, [f"{t:g}" for t in TICKS])
    ax.minorticks_off()
    ax.axvline(1, color=MUTED, linewidth=1)
    ax.grid(axis="x", color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    ax.set_xlabel(label, color=MUTED, fontsize=9)


def family_chart(run, platform, output):
    """Headline: speedup of apxchol over each solver family, one dot per device."""
    devices = [d for d in run.devices if run.families(d)]
    families = [f for f in FAMILY_ORDER if any(f in run.families(d) for d in devices)]
    fig, ax = plt.subplots(figsize=(8.6, .62 * len(families) + 1.3), layout="constrained")
    offsets = np.linspace(-.17, .17, len(devices)) if len(devices) > 1 else [0]
    for device, offset in zip(devices, offsets):
        color = DEVICE_COLORS[device]
        for row, family in enumerate(families):
            if family not in run.families(device):
                continue
            y = row + offset
            values = [run.family_cell(device, family, m)[0] for m in run.matrices]
            values = sorted(v for v in values if v is not None)
            finished = [v for v in values if v < UNFINISHED]
            median = np.median(values)
            lo, hi = np.percentile(np.minimum(values, 1e9), [25, 75])
            ax.plot([min(lo, CLIP), min(hi, CLIP)], [y, y], color=color, linewidth=5, alpha=.3, solid_capstyle="round")
            ax.plot(np.minimum(finished, CLIP), np.full(len(finished), y), "o", color=color, alpha=.5, markersize=3,
                    markeredgewidth=0)
            ax.plot(min(median, CLIP), y, "o", color=color, markersize=8, markeredgecolor="white", markeredgewidth=1.2)
            text = (f"{median:.1f}×" if median < 10 else f"{median:.0f}×") if median < UNFINISHED else "—"
            unfinished = len(values) - len(finished)
            note = f", {unfinished} of {len(values)} unfinished" if unfinished else f", {len(values)} matrices"
            ax.annotate(f"{DEVICES[device]} {text}{note}", (1.01, y), xycoords=("axes fraction", "data"),
                        va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(families)), families)
    ax.set_ylim(len(families) - .5, -.5)
    log_axis(ax, "speedup of apxchol in setup + solve time  (right of 1: apxchol faster)")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=DEVICE_COLORS[d], label=DEVICES[d]) for d in devices]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1), ncol=len(devices), frameon=False, fontsize=9)
    save(fig, output / f"{platform}-families.svg")


def best_other_chart(run, platform, output):
    """Each matrix: apxchol against whichever other solver was fastest on it."""
    devices = [d for d in run.devices if run.families(d)]
    fig, axes = plt.subplots(1, len(devices), figsize=(4.9 * len(devices), 3.3), sharey=True, layout="constrained",
                             squeeze=False)
    winners = collections.Counter()
    for ax, device in zip(axes[0], devices):
        for matrix in run.matrices:
            options = [(v, f) for f in run.families(device)
                       for v in [run.family_cell(device, f, matrix)[0]] if v is not None]
            if not options:
                continue
            value, family = min(options)
            jitter = (sum(map(ord, matrix)) % 7 - 3) * .07
            y = list(CLASSES.values()).index(run.cls[matrix]) + jitter
            if value == UNFINISHED:
                winners["none"] += 1
                ax.plot(CLIP, y, "o", markersize=7, markerfacecolor="white", markeredgecolor=MUTED)
                continue
            winners[family] += 1
            ax.plot(value, y, "o", markersize=7, color=FAMILY_COLORS.get(family, "#9a9993"),
                    markeredgecolor="white", markeredgewidth=1)
        ax.set_yticks(range(len(CLASSES)), list(CLASSES.values()))
        ax.set_ylim(len(CLASSES) - .5, -.5)
        ax.set_title(DEVICES[device], loc="left", fontsize=10)
        log_axis(ax, "speedup over the fastest other solver")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=f) for f, c in FAMILY_COLORS.items()
               if f in winners]
    if set(winners) - set(FAMILY_COLORS) - {"none"}:
        handles.append(plt.Line2D([], [], marker="o", linestyle="", color="#9a9993", label="another solver"))
    if winners["none"]:
        handles.append(plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=MUTED,
                                  label="no other solver finished"))
    fig.legend(handles=handles, loc="outside upper left", ncol=len(handles), frameon=False, fontsize=8.5,
               title="Fastest other solver:", title_fontsize=8.5, alignment="left")
    save(fig, output / f"{platform}-best-other.svg")


FAMILY_COLORS = {"AMGCL": "#2a78d6", "BoomerAMG": "#eb6834", "AMGX": "#6250d6"}


def heatmap(ax, rows, columns, value_of, classes=None):
    grid = np.full((len(rows), len(columns)), np.nan)
    for i, row in enumerate(rows):
        for j, column in enumerate(columns):
            value, mark = value_of(row, column)
            if mark is not None:
                ax.text(j, i, mark, ha="center", va="center", fontsize=6.5, color=MUTED)
            else:
                grid[i, j] = np.log10(value)
                ax.text(j, i, f"{value:.0f}" if value >= 10 else f"{value:.2g}", ha="center", va="center",
                        fontsize=6.5, color="white" if abs(grid[i, j]) > .8 else INK)
    ax.imshow(grid, cmap="PuOr", vmin=-1.3, vmax=1.3, aspect="auto")
    ax.set_xticks(range(len(columns)), columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if classes:
        previous = None
        for j, column in enumerate(columns):
            if classes[column] != previous:
                if previous is not None:
                    ax.axvline(j - .5, color="white", linewidth=3)
                ax.text(j - .4, -.75, classes[column], fontsize=8, color=MUTED)
                previous = classes[column]


def family_grid(run, platform, device, output):
    families = run.families(device)

    def value_of(family, matrix):
        value, _, mark = run.family_cell(device, family, matrix)
        return (value, mark) if mark is None else (None, mark)
    fig, ax = plt.subplots(figsize=(.44 * len(run.matrices) + 2.4, .42 * len(families) + 1.9), layout="constrained")
    heatmap(ax, families, run.matrices, value_of, run.cls)
    save(fig, output / f"{platform}-{device}-families.svg")


def variant_grid(run, platform, device, output):
    rows = run.series(device)
    reference_label = apx_label(reference(run.data, device))

    def value_of(series, matrix):
        c = run.cells.get((device, series, matrix))
        ref = run.apx(device, matrix)
        if c is None:
            return None, "—"
        if not good(c):
            return None, MARKS.get(c["status"], c["status"])
        if ref is None:
            return None, "—"
        return (c["setup_s"] + c["solve_s"]) / ref, None
    labels = [r + (" (reference)" if r == reference_label else "") for r in rows]
    fig, ax = plt.subplots(figsize=(.44 * len(run.matrices) + 3.2, .3 * len(rows) + 1.9), layout="constrained")
    heatmap(ax, rows, run.matrices, value_of, run.cls)
    ax.set_yticks(range(len(rows)), labels, fontsize=7.5)
    save(fig, output / f"{platform}-{device}-variants.svg")


def agreement_table(run, device):
    """Markdown: how far apart the variants of each family are, on matrices where both finished."""
    lines = ["| Family | Variants compared | Both finished | Median ratio | Within 15% | Largest difference | "
             "Finished in only one |", "|---|---|---:|---:|---:|---|---:|"]
    by_family = collections.defaultdict(list)
    for s in run.series(device):
        by_family["apxchol" if s.startswith("apxchol") else s.split(" (")[0]].append(s)
    for family, members in by_family.items():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                pairs, only = [], 0
                for m in run.matrices:
                    ca, cb = run.cells.get((device, a, m)), run.cells.get((device, b, m))
                    if good(ca) and good(cb):
                        pairs.append(((cb["setup_s"] + cb["solve_s"]) / (ca["setup_s"] + ca["solve_s"]), m))
                    elif good(ca) != good(cb):
                        only += 1
                if not pairs:
                    continue
                ratios_ = [r for r, _ in pairs]
                worst, matrix = max(pairs, key=lambda p: abs(np.log(p[0])))
                within = sum(abs(np.log(r)) <= np.log(1.15) for r in ratios_)
                lines.append(f"| {family} | {b} vs {a} | {len(pairs)} | {np.median(ratios_):.2f} | "
                             f"{within}/{len(pairs)} | {worst:.2f} ({matrix}) | {only} |")
    return "\n".join(lines) + "\n"


def times_table(run, device):
    """Markdown: setup and solve seconds for apxchol and each family's fastest variant."""
    families = run.families(device)
    ref = apx_label(reference(run.data, device))
    head = ["Matrix", "apxchol setup", "apxchol solve"] + [f"{f} {p}" for f in families for p in ("setup", "solve")]
    lines = ["| " + " | ".join(head) + " |", "|---" + "|---:" * (len(head) - 1) + "|"]
    for m in run.matrices:
        c = run.cells.get((device, ref, m))
        row = [m] + ([f"{c['setup_s']:.3g}", f"{c['solve_s']:.3g}"] if good(c) else [MARKS.get(c["status"], "—")] * 2)
        for f in families:
            _, best, mark = run.family_cell(device, f, m)
            row += [f"{best['setup_s']:.3g}", f"{best['solve_s']:.3g}"] if best else [mark or "—", ""]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def scaling_chart(data, platform, output):
    """Setup + solve time against thread count; a cross on the top edge marks a capped run."""
    scaling = data["scaling"]
    series = list(dict.fromkeys(series_label(c) for c in scaling["cells"]))
    colors = dict(zip(series, ("#2a78d6", "#1baf7a", "#eb6834", "#6250d6", "#e34948", "#eda100")))
    fig, axes = plt.subplots(1, len(scaling["matrices"]), figsize=(3.6 * len(scaling["matrices"]), 3.4),
                             layout="constrained", squeeze=False)
    for ax, matrix in zip(axes[0], scaling["matrices"]):
        capped = []
        for s in series:
            points = sorted(((c["threads"], c) for c in scaling["cells"]
                             if c["matrix"] == matrix and series_label(c) == s), key=lambda p: p[0])
            done = [(t, c["setup_s"] + c["solve_s"]) for t, c in points if good(c)]
            capped += [(t, s) for t, c in points if not good(c)]
            if done:
                ax.plot(*zip(*done), "o-", color=colors[s], markersize=3.5, linewidth=1.4)
        ax.set_yscale("log")
        top = ax.get_ylim()[1]
        for t, s in capped:
            ax.plot(t, top, "x", color=colors[s], clip_on=False, markersize=6)
        ax.set_xscale("log", base=2)
        ax.set_xticks(scaling["thread_counts"], [str(t) for t in scaling["thread_counts"]])
        ax.minorticks_off()
        ax.grid(color=GRID, linewidth=.8)
        ax.set_title(matrix, loc="left", fontsize=10)
        ax.set_xlabel("threads", color=MUTED, fontsize=9)
    axes[0][0].set_ylabel("setup + solve (s)", color=MUTED, fontsize=9)
    handles = [plt.Line2D([], [], marker="o", color=colors[s], label=s, markersize=4) for s in series]
    handles.append(plt.Line2D([], [], marker="x", linestyle="", color=MUTED, label="over the cap"))
    fig.legend(handles=handles, loc="outside upper left", ncol=len(handles), frameon=False, fontsize=8.5)
    save(fig, output / f"{platform}-scaling.svg")


def absolute(run, platform, device, output):
    """apxchol alone on a device: setup and solve as one stacked bar per matrix, plus a table."""
    cells = [run.cells.get((device, apx_label(reference(run.data, device)), m)) for m in run.matrices]
    cells = [c for c in cells if c is not None]
    fig, ax = plt.subplots(figsize=(7.5, .26 * len(cells) + 1.3), layout="constrained")
    y = np.arange(len(cells))
    setup = np.array([c["setup_s"] for c in cells])
    solve = np.array([c["solve_s"] for c in cells])
    ax.barh(y, setup, color="#2a78d6", height=.7, label="setup")
    ax.barh(y, solve, left=setup, color="#eb6834", height=.7, label="solve")
    for i, (a, b) in enumerate(zip(setup, solve)):
        ax.text(a + b, i, f"  {a + b:.2g} s", va="center", fontsize=7, color=INK)
    ax.set_yticks(y, [c["matrix"] for c in cells], fontsize=8)
    ax.set_ylim(len(cells) - .5, -.5)
    ax.set_xlim(0, (setup + solve).max() * 1.15)
    ax.set_xlabel("seconds: setup + solve", color=MUTED, fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    save(fig, output / f"{platform}-{device}.svg")
    lines = ["| Matrix | Rows | Nonzeros | Setup (s) | Solve (s) | Setup + solve (s) | Iterations | Peak memory (MB) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in cells:
        if not good(c):
            lines.append(f"| {c['matrix']} | {c['n']:,} | {c['nnz']:,} | {MARKS.get(c['status'], c['status'])} | | | | |")
            continue
        lines.append(f"| {c['matrix']} | {c['n']:,} | {c['nnz']:,} | {c['setup_s']:.3g} | {c['solve_s']:.3g} | "
                     f"{c['setup_s'] + c['solve_s']:.3g} | {c['iters']} | {c.get('max_rss_mb', 0):,.0f} |")
    (output / f"{platform}-{device}-table.md").write_text("\n".join(lines) + "\n")


def campaign(data, platform, output):
    """Every chart and table for one platform extract."""
    output.mkdir(parents=True, exist_ok=True)
    run = Campaign(data)
    if any(run.families(d) for d in run.devices):
        family_chart(run, platform, output)
        best_other_chart(run, platform, output)
    for device in run.devices:
        if not run.families(device):
            absolute(run, platform, device, output)
            continue
        family_grid(run, platform, device, output)
        variant_grid(run, platform, device, output)
        (output / f"{platform}-{device}-agreement.md").write_text(agreement_table(run, device))
        (output / f"{platform}-{device}-times.md").write_text(times_table(run, device))
    if "scaling" in data:
        scaling_chart(data, platform, output)


def campaigns(directory, output):
    """Render every docs/performance/<platform>-<date>.json campaign header and its CSV files."""
    for path in sorted(directory.glob("*-2*.json")):
        platform = path.name.split("-")[0]
        if platform in ("euler", "daint", "mac"):
            campaign(load(path), platform, output)


def render(report_paths, output, revision=None):
    output.mkdir(parents=True, exist_ok=True)
    reports = [json.loads(path.read_text()) for path in report_paths]
    for report in reports:
        if report.get("dirty") is not False or (revision is not None and report.get("source") != revision):
            raise ValueError("benchmark reports must come from the clean documentation revision")
    for path in report_paths:
        shutil.copyfile(path, output / path.name)
    hosted(reports, output)
    lines = ["Each report records its source revision and hardware. "
             "Hosted runners vary; compare configurations within a run, not absolute times between commits.", "",
             "![Hosted CPU and Metal timings](performance/hosted.svg)", ""]
    for path, report in zip(report_paths, reports):
        hw = report["hardware"]
        lines += [f"# {hw['os']}", "", f"{hw['cpu']}; {hw['logical_cpus']} logical CPUs. "
                  + (f"GPU: {', '.join(hw['gpus'])}. " if hw.get('gpus') else "") +
                  f"Source `{report['source'][:12]}`. "
                  f"Numerical checks: {report['passed_owners']}/{report['planned_owners']} runs, "
                  f"{report['rhs_checks']}/{report['planned_rhs_checks']} apxchol RHSs. "
                  f"SciPy CG: {report['reference_passes']}/{report['planned_owners']} residual checks passed.", "",
                  f"[Measurements and environment](performance/{path.name})" +
                  (f" · [CI run]({report['run_url']})" if "run_url" in report else ""), "",
                  "| Case | Configuration | Repeat controls within 10% | Timing status | SciPy CG |", "|---|---|---:|---|---|"]
        for case in report["cases"]:
            result = case["summary"]
            lines.append(f"| {case['fixture']} | {case['configuration']} | {result['accepted_pairs']}/3 | {result['status']} | {result.get('reference_status', 'unavailable')} |")
        lines.append("")
    if not reports:
        lines += ["No hosted measurements were supplied for this local preview.", ""]
    (output / "latest.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("reports", nargs="*", type=Path)
    parser.add_argument("--historical", type=Path)
    parser.add_argument("--campaigns", type=Path, metavar="DIRECTORY",
                        help="render every platform extract in DIRECTORY")
    args = parser.parse_args()
    if args.campaigns:
        campaigns(args.campaigns, args.output)
        raise SystemExit
    render(args.reports, args.output)
    if args.historical:
        historical(json.loads(args.historical.read_text()), args.output)
