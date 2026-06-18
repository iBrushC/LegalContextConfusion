"""plot_results.py — matplotlib degradation charts from run results.

Optional visualisation layer on top of analyze.py. Reads the same runs.jsonl,
aggregates with score_outputs.aggregate, and renders PNGs:

  * degradation_length.png — headline metric vs context length, one line per
    model, one panel per modality, with a +/-stdev band.
  * degradation_depth.png  — headline metric vs target depth (start/mid/end) at
    the longest context, one line per model, one panel per modality.
  * heatmap.png            — per (model x modality), a length x depth heatmap.

The headline metric follows each modality's focus (span_f1 for rot/confusion,
abstention_rate for missing_answer); override with --metric to plot any metric.

matplotlib is the only non-stdlib dependency in the project and is needed only
for this script (pip install matplotlib). Everything else stays stdlib-only.

Usage:
    python src/plot_results.py
    python src/plot_results.py --metric wrong_doc_rate
    python src/plot_results.py --charts length depth
    python src/plot_results.py --results data/results/runs.jsonl --out data/results/plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_outputs import aggregate  # noqa: E402
from analyze import (build_curves, load_runs, HEADLINE,  # noqa: E402
                     POSITION_ORDER, SHORT_POS, MODALITY_ORDER)

try:
    import matplotlib
    matplotlib.use("Agg")  # headless: render straight to files
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    raise SystemExit("error: plot_results.py needs matplotlib + numpy "
                     "(pip install matplotlib).")

DEFAULT_RESULTS = Path("data/results/runs.jsonl")
DEFAULT_OUT = Path("data/results/plots")

METRIC_LABEL = {
    "span_f1": "span token-F1", "exact_match": "exact match",
    "abstention_rate": "abstention rate", "hallucination_rate": "hallucination rate",
    "wrong_doc_rate": "wrong-document rate", "answered_positive_rate": "answered rate",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _metric_for(cells: dict, override: str | None) -> str:
    focus = next(iter(cells.values()))["focus"]
    return override or HEADLINE.get(focus, "span_f1")


def _budgets(cells: dict) -> list[int]:
    return sorted({b for (b, _) in cells})


def _positions(cells: dict, budgets: list[int]) -> list[str]:
    return [p for p in POSITION_ORDER if any((b, p) in cells for b in budgets)]


def _human_budget(b: int) -> str:
    return f"{b//1_000_000}M" if b >= 1_000_000 else f"{b//1000}k" if b >= 1000 else str(b)


def _shared_legend(fig, axes) -> None:
    handles, labels = {}, []
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in handles:
                handles[lab] = h
                labels.append(lab)
    if labels:
        fig.legend([handles[l] for l in labels], labels, loc="upper center",
                   ncol=min(len(labels), 6), frameon=False, bbox_to_anchor=(0.5, 1.0))


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def plot_length(curves, models, modalities, override, colors, out: Path, dpi: int):
    fig, axes = plt.subplots(1, len(modalities), figsize=(5.2 * len(modalities), 4.4),
                             squeeze=False)
    for ax, modality in zip(axes[0], modalities):
        metric_name = None
        for model in models:
            cells = curves.get((model, modality))
            if not cells:
                continue
            metric = _metric_for(cells, override)
            metric_name = metric
            xs, ys, es = [], [], []
            for b in _budgets(cells):
                vals = [cells[(b, p)].get(f"{metric}_mean")
                        for p in POSITION_ORDER if (b, p) in cells
                        and cells[(b, p)].get(f"{metric}_mean") is not None]
                stds = [cells[(b, p)].get(f"{metric}_std") or 0
                        for p in POSITION_ORDER if (b, p) in cells
                        and cells[(b, p)].get(f"{metric}_mean") is not None]
                if vals:
                    xs.append(b)
                    ys.append(float(np.mean(vals)))
                    es.append(float(np.mean(stds)) if stds else 0.0)
            if not xs:
                continue
            xs, ys, es = np.array(xs), np.array(ys), np.array(es)
            ax.plot(xs, ys, marker="o", color=colors[model], label=model, linewidth=2)
            ax.fill_between(xs, np.clip(ys - es, 0, 1), np.clip(ys + es, 0, 1),
                            color=colors[model], alpha=0.15)
        ax.set_xscale("log")
        ax.set_title(modality)
        ax.set_xlabel("context length (tokens, log)")
        ax.set_ylabel(METRIC_LABEL.get(metric_name, metric_name or "score"))
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3, which="both")
    _shared_legend(fig, axes[0])
    fig.suptitle("Degradation vs context length", y=1.04, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_depth(curves, models, modalities, override, colors, out: Path, dpi: int):
    fig, axes = plt.subplots(1, len(modalities), figsize=(5.2 * len(modalities), 4.4),
                             squeeze=False)
    for ax, modality in zip(axes[0], modalities):
        metric_name, bmax = None, None
        for model in models:
            cells = curves.get((model, modality))
            if not cells:
                continue
            metric = _metric_for(cells, override)
            metric_name = metric
            budgets = _budgets(cells)
            bmax = budgets[-1]
            positions = [p for p in POSITION_ORDER if (bmax, p) in cells]
            # keep the x-axis aligned to all positions, but only plot points
            # that actually have a value (a parse-failed cell has None)
            pts = [(i, p) for i, p in enumerate(positions)
                   if cells[(bmax, p)].get(f"{metric}_mean") is not None]
            if pts:
                xs = [i for i, _ in pts]
                ys = [cells[(bmax, p)][f"{metric}_mean"] for _, p in pts]
                es = [cells[(bmax, p)].get(f"{metric}_std") or 0 for _, p in pts]
                ax.errorbar(xs, ys, yerr=es, marker="s", capsize=3, linewidth=2,
                            color=colors[model], label=model)
            ax.set_xticks(range(len(positions)))
            ax.set_xticklabels([SHORT_POS.get(p, p) for p in positions])
        ax.set_title(f"{modality}" + (f"  @ {_human_budget(bmax)}" if bmax else ""))
        ax.set_xlabel("target depth")
        ax.set_ylabel(METRIC_LABEL.get(metric_name, metric_name or "score"))
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
    _shared_legend(fig, axes[0])
    fig.suptitle("Depth effect at longest context", y=1.04, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_heatmaps(curves, models, modalities, override, out: Path, dpi: int):
    fig, axes = plt.subplots(len(models), len(modalities),
                             figsize=(4.0 * len(modalities), 3.0 * len(models)),
                             squeeze=False)
    im = None
    for i, model in enumerate(models):
        for j, modality in enumerate(modalities):
            ax = axes[i][j]
            cells = curves.get((model, modality))
            if not cells:
                ax.axis("off")
                continue
            metric = _metric_for(cells, override)
            budgets = _budgets(cells)
            positions = _positions(cells, budgets)
            grid = np.full((len(budgets), len(positions)), np.nan)
            for bi, b in enumerate(budgets):
                for pi, p in enumerate(positions):
                    r = cells.get((b, p))
                    if r and r.get(f"{metric}_mean") is not None:
                        grid[bi, pi] = r[f"{metric}_mean"]
            im = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(positions)))
            ax.set_xticklabels([SHORT_POS.get(p, p) for p in positions])
            ax.set_yticks(range(len(budgets)))
            ax.set_yticklabels([_human_budget(b) for b in budgets])
            for bi in range(len(budgets)):
                for pi in range(len(positions)):
                    if not np.isnan(grid[bi, pi]):
                        ax.text(pi, bi, f"{grid[bi, pi]:.2f}", ha="center",
                                va="center", fontsize=8,
                                color="white" if grid[bi, pi] < 0.6 else "black")
            if j == 0:
                ax.set_ylabel(f"{model}\ncontext length")
            if i == 0:
                ax.set_title(modality)
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02, label="headline metric")
    fig.suptitle("Headline metric by length × depth", y=1.0, fontsize=13)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                        help=f"runs.jsonl (or its dir) (default: {DEFAULT_RESULTS})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"output dir for PNGs (default: {DEFAULT_OUT})")
    parser.add_argument("--metric", default=None,
                        help="force a metric to plot (default: headline by focus)")
    parser.add_argument("--charts", nargs="+", default=["length", "depth", "heatmap"],
                        choices=["length", "depth", "heatmap"],
                        help="which charts to render (default: all)")
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    runs = load_runs(args.results)
    summary = aggregate(runs)
    if not summary:
        raise SystemExit("error: no scored runs to plot.")
    curves = build_curves(summary)

    models = sorted({m for (m, _) in curves})
    modalities = [m for m in MODALITY_ORDER if any((mo, m) in curves for mo in models)]
    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(models)}

    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    if "length" in args.charts:
        written.append(plot_length(curves, models, modalities, args.metric,
                                   colors, args.out / "degradation_length.png", args.dpi))
    if "depth" in args.charts:
        written.append(plot_depth(curves, models, modalities, args.metric,
                                  colors, args.out / "degradation_depth.png", args.dpi))
    if "heatmap" in args.charts:
        written.append(plot_heatmaps(curves, models, modalities, args.metric,
                                     args.out / "heatmap.png", args.dpi))

    print(f"Plotted {len(models)} model(s) × {len(modalities)} modality(ies) "
          f"from {len(runs)} runs:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
