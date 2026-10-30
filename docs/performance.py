"""Render public benchmark summaries; raw measurements remain downloadable JSON."""
import argparse
import collections
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


SOLVERS = {"ac": "Julia AC", "ac2": "Julia AC2", "amgcl": "AMGCL", "amgcl_cuda": "AMGCL",
           "hypre_boomeramg": "BoomerAMG", "hypre_boomeramg_gpu": "BoomerAMG", "rchol": "RCHOL",
           "rchol_par": "pRCHOL", "cmg_packed": "CMG (packed)", "cmg_matlab": "CMG (MATLAB)",
           "parac": "ParAC Graph", "parac_graph": "ParAC Graph", "parac_physics": "ParAC Physics"}
MARKS = {"failed": "fail", "not_converged": "nc", "timeout": "t/o", "exceeded_cap": "cap", "oom": "oom",
         "n/a": "n/a", "missing": "—"}


def series_label(cell):
    if cell["solver"] == "apxchol_v1":
        config = cell["config"].split("[")[0].replace("bg+", "").replace("trace_cycle", "trace-cycle")
        return "apxchol " + config.replace("/gpu-owned-q0", " q0.").replace("tree", "GKS")
    name = SOLVERS.get(cell["solver"], cell["solver"])
    extra = [v for v in (cell.get("variant"), "" if cell["config"] in ("", "original-operator") else cell["config"]) if v]
    return name + (f" ({', '.join(extra)})" if extra else "")


def ratios(data, device):
    """Total time relative to the best apxchol series on the same matrix and device."""
    cells = [c for c in data["cells"] if c["device"] == device]
    cells += [dict(m, status="missing") for m in data["missing"] if m.get("device") == device]
    best = {}
    for c in cells:
        if c["solver"] == "apxchol_v1" and c["meets_tolerance"]:
            best[c["matrix"]] = min(best.get(c["matrix"], np.inf), c["total_s"])
    table = {}
    for c in cells:
        ok = c["status"] == "complete" and c.get("meets_tolerance") and c["matrix"] in best
        table[series_label(c), c["matrix"]] = c["total_s"] / best[c["matrix"]] if ok else MARKS.get(c["status"], c["status"])
    return table


def campaign(data, platform, output):
    """One headline strip and one full grid per device; every planned cell appears in the grid."""
    for device in sorted({c["device"] for c in data["cells"]}):
        table = ratios(data, device)
        rows = sorted({s for s, _ in table}, key=lambda s: (not s.startswith("apxchol"), s))
        matrices = list(dict.fromkeys(c["matrix"] for c in data["cells"] if c["device"] == device))
        competitors = [s for s in rows if not s.startswith("apxchol")]
        fig, ax = plt.subplots(figsize=(7.5, .45 * len(competitors) + 1.2), layout="constrained")
        for y, s in enumerate(competitors):
            values = [v for (r, _), v in table.items() if r == s and not isinstance(v, str)]
            marks = collections.Counter(v for (r, _), v in table.items() if r == s and isinstance(v, str))
            ax.plot(values, np.full(len(values), y), "o", color=COLORS[0], alpha=.6, markersize=4)
            if values:
                ax.plot(np.median(values), y, "|", color="#111111", markersize=14, markeredgewidth=2)
            if marks:
                ax.annotate(", ".join(f"{n} {m}" for m, n in sorted(marks.items())), (1.01, y), xycoords=("axes fraction", "data"),
                            va="center", fontsize=8, color=COLORS[1])
        ax.axvline(1, color="#666666", linewidth=.8)
        ax.set_xscale("log")
        ax.set_yticks(range(len(competitors)), competitors)
        ax.invert_yaxis()
        ax.set_xlabel("one-RHS time (setup + solve) relative to apxchol; right of 1 = apxchol faster")
        ax.grid(axis="x", alpha=.15)
        ax.set_title(f"{platform} {device.upper()}: dots are matrices, bars are medians", fontsize=10)
        save(fig, output / f"{platform.lower()}-{device}.svg")

        grid = np.full((len(rows), len(matrices)), np.nan)
        fig, ax = plt.subplots(figsize=(.42 * len(matrices) + 3, .3 * len(rows) + 1.6), layout="constrained")
        for i, s in enumerate(rows):
            for j, m in enumerate(matrices):
                v = table.get((s, m), "—")
                if isinstance(v, str):
                    ax.text(j, i, v, ha="center", va="center", fontsize=6, color="#444444")
                else:
                    grid[i, j] = np.log10(v)
                    ax.text(j, i, f"{v:.0f}" if v >= 10 else f"{v:.2g}", ha="center", va="center", fontsize=6,
                            color="white" if abs(grid[i, j]) > .8 else "black")
        ax.imshow(grid, cmap="PuOr", vmin=-1.5, vmax=1.5, aspect="auto")
        ax.set_xticks(range(len(matrices)), matrices, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(rows)), rows, fontsize=7)
        ax.set_title(f"{platform} {device.upper()}: one-RHS time relative to the best apxchol series", fontsize=10)
        save(fig, output / f"{platform.lower()}-{device}-cells.svg")


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
    parser.add_argument("--campaign", nargs=2, metavar=("PLATFORM", "EXTRACT"))
    args = parser.parse_args()
    if args.campaign:
        campaign(json.loads(Path(args.campaign[1]).read_text()), args.campaign[0], args.output)
        raise SystemExit
    render(args.reports, args.output)
    if args.historical:
        historical(json.loads(args.historical.read_text()), args.output)
