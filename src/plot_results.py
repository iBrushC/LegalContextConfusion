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

Usage:
    python src/plot_results.py
    python src/plot_results.py --metric wrong_doc_rate
    python src/plot_results.py --charts length overview
    python src/plot_results.py --results data/results/runs.jsonl --out data/results/plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_outputs import aggregate_curves  # noqa: E402
from analyze import build_curves, load_runs, HEADLINE, MODALITY_ORDER  # noqa: E402

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
    for ax, modality in zip(axes[0], modalities):
        metric_name = None
        for model in models:
            cells = curves.get((model, modality))
            if not cells:
                continue
            metric = _metric_for(cells, override)
            metric_name = metric
            budgets = _interference_budgets(cells)
            xs, ys, es = [], [], []
            for b in budgets:
                r = cells[b]
                mean = r.get(f"{metric}_mean")
                if mean is None:
                    continue
                xs.append(b)
                ys.append(float(mean))
                es.append(float(r.get(f"{metric}_std") or 0.0))
            if xs:
                xs, ys, es = np.array(xs), np.array(ys), np.array(es)
                ax.plot(xs, ys, marker="o", color=colors[model], label=model,
                        linewidth=2)
                ax.fill_between(xs, np.clip(ys - es, 0, 1), np.clip(ys + es, 0, 1),
                                color=colors[model], alpha=0.15)
            # No-filler baseline as a dashed reference line (CUAD rot/confusion).
            base = _baseline_row(cells)
            if base is not None and base.get(f"{metric}_mean") is not None:
                ax.axhline(base[f"{metric}_mean"], color=colors[model],
                           linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xscale("log")
        ax.set_title(modality)
        ax.set_xlabel("context length (filler tokens, log)")
        ax.set_ylabel(METRIC_LABEL.get(metric_name, metric_name or "score"))
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3, which="both")
    _shared_legend(fig, axes[0])
    fig.suptitle("Degradation vs context length  (pooled across documents; "
                 "dashed = no-filler baseline)", y=1.04, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_overview(curves, models, modalities, override, out: Path, dpi: int):
    """Compact (model x budget) heatmap of the headline metric, per modality."""
    fig, axes = plt.subplots(1, len(modalities),
                             figsize=(0.9 * len(modalities) + 3.0 * len(modalities),
                                      1.0 + 0.6 * len(models)),
                             squeeze=False)
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
    fig.suptitle("Headline metric by model × context length", y=1.02, fontsize=13)
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
    parser.add_argument("--charts", nargs="+", default=["length", "overview"],
                        choices=["length", "overview"],
                        help="which charts to render (default: length overview)")
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    runs = load_runs(args.results)
    summary = aggregate_curves(runs)
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
    if "overview" in args.charts:
        written.append(plot_overview(curves, models, modalities, args.metric,
                                     args.out / "overview.png", args.dpi))

    print(f"Plotted {len(models)} model(s) × {len(modalities)} modality(ies) "
          f"from {len(runs)} runs:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
