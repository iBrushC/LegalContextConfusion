"""plot_results.py — matplotlib degradation charts from run results.

Optional visualisation layer on top of analyze.py. Reads the same runs.jsonl,
pools every document's runs into one curve per (model, modality, budget) via
score_outputs.aggregate_curves, and renders PNGs:

  * degradation_length.png — headline metric vs context length, one line per
    model, one panel per modality, with a +/-stdev band (the band is now the
    cross-document + multirun spread). The no-filler baseline, where present,
    is drawn as a dashed reference line.
  * overview.png (optional) — a compact (model x budget) heatmap of the
    headline metric, one panel per modality. Replaces the old length x depth
    heatmap now that the target always sits at the end of the window.

The headline metric follows each modality's focus (span_f1 for rot/confusion,
abstention_rate for missing_answer, mc_accuracy for MAUD); override with
--metric to plot any metric.

matplotlib is the only non-stdlib dependency in the project and is needed only
for this script (pip install matplotlib). Everything else stays stdlib-only.

Each dataset is read from its own data/results_<dataset>/ stream and rendered
into data/results_<dataset>/plots/; --dataset selects which (default both).

Usage:
    python src/plot_results.py
    python src/plot_results.py --dataset cuad
    python src/plot_results.py --metric wrong_doc_rate
    python src/plot_results.py --charts length overview
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_outputs import aggregate_curves  # noqa: E402
from analyze import (  # noqa: E402
    DATASETS, build_curves, load_runs, results_dir, HEADLINE, MODALITY_ORDER,
)

try:
    import matplotlib
    matplotlib.use("Agg")  # headless: render straight to files
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    raise SystemExit("error: plot_results.py needs matplotlib + numpy "
                     "(pip install matplotlib).")

# Per-dataset results live under data/results_<dataset>/ (CUAD and MAUD never
# share a stream); --dataset selects which to plot, into that dir's plots/.

METRIC_LABEL = {
    "span_f1": "span token-F1", "exact_match": "exact match",
    "mc_accuracy": "MC accuracy",
    "abstention_rate": "abstention rate", "hallucination_rate": "hallucination rate",
    "wrong_doc_rate": "wrong-document rate", "answered_positive_rate": "answered rate",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _metric_for(cells: dict, override: str | None) -> str:
    focus = next(iter(cells.values()))["focus"]
    return override or HEADLINE.get(focus, "span_f1")


def _interference_budgets(cells: dict) -> list[int]:
    """Sorted non-baseline budgets (the actual filler-length points)."""
    return sorted(b for b, r in cells.items() if not r.get("is_baseline"))


def _baseline_row(cells: dict):
    """The no-filler baseline row for this (model, modality), if any."""
    for r in cells.values():
        if r.get("is_baseline"):
            return r
    return None


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
    scaled_any = False
    for ax, modality in zip(axes[0], modalities):
        metric_name = None
        ymax = 1.0
        ticks: set[int] = set()
        for model in models:
            cells = curves.get((model, modality))
            if not cells:
                continue
            metric = _metric_for(cells, override)
            metric_name = metric
            # Scale each model's curve by its own no-filler baseline, so the
            # y-axis reads as "fraction of baseline performance retained" and
            # every modality shares the same 1.0 = no-degradation anchor.
            base = _baseline_row(cells)
            base_mean = base.get(f"{metric}_mean") if base is not None else None
            scale = base_mean if (base_mean is not None and base_mean > 0) else None
            budgets = _interference_budgets(cells)
            xs, ys, es = [], [], []
            for b in budgets:
                r = cells[b]
                mean = r.get(f"{metric}_mean")
                if mean is None:
                    continue
                xs.append(b)
                val = float(mean)
                err = float(r.get(f"{metric}_std") or 0.0)
                if scale:
                    val /= scale
                    err /= scale
                ys.append(val)
                es.append(err)
            if xs:
                ticks.update(xs)
                xs, ys, es = np.array(xs), np.array(ys), np.array(es)
                lo, hi = ys - es, ys + es
                if not scale:  # absolute metrics live in [0, 1]; clip the band
                    lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
                ymax = max(ymax, float(hi.max()))
                ax.plot(xs, ys, marker="o", color=colors[model], label=model,
                        linewidth=2)
                ax.fill_between(xs, np.clip(lo, 0, None), hi,
                                color=colors[model], alpha=0.15)
            # No-filler baseline reference: 1.0 when scaled, else the raw level.
            if scale:
                scaled_any = True
                ax.axhline(1.0, color=colors[model], linestyle="--",
                           linewidth=1, alpha=0.6)
            elif base is not None and base.get(f"{metric}_mean") is not None:
                ax.axhline(base[f"{metric}_mean"], color=colors[model],
                           linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xscale("log", base=2)
        if ticks:  # exact filler-length labels (64k/128k/256k/512k), not log2
            xt = sorted(ticks)
            ax.set_xticks(xt)
            ax.set_xticklabels([_human_budget(b) for b in xt])
            ax.minorticks_off()
        ax.set_title(modality)
        ax.set_xlabel("context length (filler tokens)")
        ylab = METRIC_LABEL.get(metric_name, metric_name or "score")
        ax.set_ylabel(ylab + " (rel. to baseline)" if scaled_any else ylab)
        ax.set_ylim(-0.02, max(1.05, ymax * 1.05))
        ax.grid(True, alpha=0.3, which="both")
    _shared_legend(fig, axes[0])
    fig.suptitle("Degradation vs context length  (pooled across documents; "
                 "dashed = no-filler baseline"
                 + (" = 1.0)" if scaled_any else ")"), y=1.04, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_overview(curves, models, modalities, override, out: Path, dpi: int):
    """Compact (model x budget) heatmap of the headline metric, per modality."""
    fig, axes = plt.subplots(1, len(modalities),
                             figsize=(0.9 * len(modalities) + 3.0 * len(modalities),
                                      1.6 + 0.6 * len(models)),
                             squeeze=False, constrained_layout=True)
    im = None
    for ax, modality in zip(axes[0], modalities):
        # Union of interference budgets seen by any model for this modality.
        budgets = sorted({b for model in models
                          for b in _interference_budgets(curves.get((model, modality), {}))})
        if not budgets:
            ax.axis("off")
            ax.set_title(modality)
            continue
        metric = None
        grid = np.full((len(models), len(budgets)), np.nan)
        for mi, model in enumerate(models):
            cells = curves.get((model, modality))
            if not cells:
                continue
            metric = _metric_for(cells, override)
            for bi, b in enumerate(budgets):
                r = cells.get(b)
                if r and r.get(f"{metric}_mean") is not None:
                    grid[mi, bi] = r[f"{metric}_mean"]
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(budgets)))
        ax.set_xticklabels([_human_budget(b) for b in budgets])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_title(f"{modality}" + (f"\n({METRIC_LABEL.get(metric, metric)})"
                                      if metric else ""))
        ax.set_xlabel("filler tokens")
        for mi in range(len(models)):
            for bi in range(len(budgets)):
                if not np.isnan(grid[mi, bi]):
                    ax.text(bi, mi, f"{grid[mi, bi]:.2f}", ha="center", va="center",
                            fontsize=8,
                            color="white" if grid[mi, bi] < 0.6 else "black")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02, label="headline metric")
    # constrained_layout reserves room for the suptitle above the (two-line)
    # per-axis titles, so it no longer collides with the per-modality metric.
    fig.suptitle("Headline metric by model × context length", fontsize=13)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def plot_dataset(dataset: str, args) -> None:
    """Render one dataset's degradation charts into its results_<dataset>/plots/."""
    results = args.results or (results_dir(dataset) / "runs.jsonl")
    out = args.out or (results_dir(dataset) / "plots")
    runs = load_runs(results)
    summary = aggregate_curves(runs)
    if not summary:
        print(f"warning: no scored runs to plot for {dataset} ({results}).")
        return
    curves = build_curves(summary)

    models = sorted({m for (m, _) in curves})
    modalities = [m for m in MODALITY_ORDER if any((mo, m) in curves for mo in models)]
    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(models)}

    out.mkdir(parents=True, exist_ok=True)
    written = []
    if "length" in args.charts:
        written.append(plot_length(curves, models, modalities, args.metric,
                                   colors, out / "degradation_length.png", args.dpi))
    if "overview" in args.charts:
        written.append(plot_overview(curves, models, modalities, args.metric,
                                     out / "overview.png", args.dpi))

    print(f"{dataset.upper()}: plotted {len(models)} model(s) × "
          f"{len(modalities)} modality(ies) from {len(runs)} runs:")
    for p in written:
        print(f"  {p}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("cuad", "maud", "both"), default="both",
                        help="which dataset(s) to plot (default: both — each from "
                             "its own data/results_<dataset>/ stream)")
    parser.add_argument("--results", type=Path, default=None,
                        help="override runs.jsonl (or its dir); default "
                             "data/results_<dataset>/runs.jsonl (single --dataset)")
    parser.add_argument("--out", type=Path, default=None,
                        help="override the PNG output dir; default "
                             "data/results_<dataset>/plots (single --dataset)")
    parser.add_argument("--metric", default=None,
                        help="force a metric to plot (default: headline by focus)")
    parser.add_argument("--charts", nargs="+", default=["length", "overview"],
                        choices=["length", "overview"],
                        help="which charts to render (default: length overview)")
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "both" else (args.dataset,)
    if (args.results is not None or args.out is not None) and len(datasets) > 1:
        raise SystemExit("error: --results/--out target a single dataset; "
                         "pass --dataset cuad or --dataset maud.")

    for ds in datasets:
        plot_dataset(ds, args)


if __name__ == "__main__":
    main()
