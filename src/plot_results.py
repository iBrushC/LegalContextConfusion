"""plot_results.py — matplotlib degradation charts from run results.

Optional visualisation layer on top of analyze.py. Reads the same runs.jsonl,
pools every document's runs into one curve per (model, modality, budget) via
score_outputs.aggregate_curves, and renders PNGs:

  * degradation_length.png — headline metric vs context length, relative to
    each model's no-filler baseline (1.0 = no degradation), one line per model,
    one panel per modality, with a +/-stdev band (the band is now the
    cross-document + multirun spread). The no-filler baseline is the leftmost
    point on each line and is also drawn as a dashed reference at 1.0.
  * degradation_actual_vs_relative.png — the same curves shown two ways:
    actual metric value (top row) alongside relative-to-baseline (bottom row),
    one column per modality. Both rows include the no-filler baseline point.
  * overview.png (optional) — a compact (model x budget) heatmap of the
    headline metric, one panel per modality. Replaces the old length x depth
    heatmap now that the target always sits at the end of the window.
  * position_accuracy.png (MAUD only) — per-question accuracy vs WHERE the
    question's evidence fragment landed in the window (0 = front, 1 = end). MAUD
    fragment-mode cells disperse the target's clauses among the filler and record
    each question's evidence position (`question_locations`), so this joins those
    positions to per-question correctness (re-derived from the logged raw output)
    and plots a binned-mean accuracy curve per model, one panel per modality.
    CUAD (and MAUD --full-documents) carry no per-question location, so the chart
    is skipped for them.

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
from score_outputs import (  # noqa: E402
    aggregate_curves, load_gold, mc_is_correct, ms_score, predictions_from_run,
    prepared_dir,
)
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
    "ms_exact_match": "select-all exact", "ms_f1": "select-all set-F1",
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
def _draw_modality(ax, curves, models, modality, override, colors, *,
                   relative: bool, title_suffix: str = ""):
    """Draw one (modality) panel of degradation lines.

    relative=True scales each model's curve by its own no-filler baseline so the
    y-axis reads as "fraction of baseline retained" (1.0 = no degradation);
    relative=False plots the actual metric value. In both cases the no-filler
    baseline is now the leftmost point ON each model's line (y=1.0 when scaled,
    the raw baseline level otherwise) — drawn one octave left of the smallest
    filler budget, since a log2 axis can't place budget 0 directly.

    Returns (metric_name, any_scaled).
    """
    metric_name = None
    ymax = 1.0
    any_scaled = False
    # Modality-wide smallest interference budget; the baseline marker sits one
    # octave (factor of 2) to its left so every model's baseline point aligns.
    all_budgets = sorted({b for m in models
                          for b in _interference_budgets(curves.get((m, modality), {}))})
    base_x = all_budgets[0] / 2 if all_budgets else None
    ticks: set[int] = set()
    for model in models:
        cells = curves.get((model, modality))
        if not cells:
            continue
        metric = _metric_for(cells, override)
        metric_name = metric
        base = _baseline_row(cells)
        base_mean = base.get(f"{metric}_mean") if base is not None else None
        scale = base_mean if (relative and base_mean is not None and base_mean > 0) else None
        xs, ys, es = [], [], []
        # Baseline anchor first, so the line starts at the no-filler point.
        if base is not None and base_mean is not None and base_x is not None:
            xs.append(base_x)
            ys.append(1.0 if scale else float(base_mean))
            es.append(0.0 if scale else float(base.get(f"{metric}_std") or 0.0))
        for b in all_budgets:
            r = cells.get(b)
            mean = r.get(f"{metric}_mean") if r else None
            if mean is None:
                continue
            ticks.add(b)
            val = float(mean)
            err = float(r.get(f"{metric}_std") or 0.0)
            if scale:
                val /= scale
                err /= scale
            xs.append(b)
            ys.append(val)
            es.append(err)
        if not xs:
            continue
        if scale:
            any_scaled = True
        xs, ys, es = np.array(xs, dtype=float), np.array(ys), np.array(es)
        lo, hi = ys - es, ys + es
        if not scale:  # absolute metrics live in [0, 1]; clip the band
            lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
        ymax = max(ymax, float(hi.max()))
        ax.plot(xs, ys, marker="o", color=colors[model], label=model, linewidth=2)
        ax.fill_between(xs, np.clip(lo, 0, None), hi, color=colors[model], alpha=0.15)
        # No-filler baseline reference: 1.0 when scaled, else the raw level.
        if scale:
            ax.axhline(1.0, color=colors[model], linestyle="--", linewidth=1, alpha=0.6)
        elif base is not None and base_mean is not None:
            ax.axhline(base_mean, color=colors[model], linestyle="--",
                       linewidth=1, alpha=0.6)
    ax.set_xscale("log", base=2)
    if ticks:  # exact filler-length labels (64k/128k/256k/512k), not log2
        xt = sorted(ticks)
        labels = [_human_budget(b) for b in xt]
        if base_x is not None:  # prepend the no-filler anchor tick
            xt = [base_x] + xt
            labels = ["base"] + labels
        ax.set_xticks(xt)
        ax.set_xticklabels(labels)
        ax.minorticks_off()
    ax.set_title(modality + title_suffix)
    ax.set_xlabel("context length (filler tokens)")
    ylab = METRIC_LABEL.get(metric_name, metric_name or "score")
    if relative and any_scaled:
        ylab += " (rel. to baseline)"
    ax.set_ylabel(ylab)
    ax.set_ylim(-0.02, max(1.05, ymax * 1.05))
    ax.grid(True, alpha=0.3, which="both")
    return metric_name, any_scaled


def plot_length(curves, models, modalities, override, colors, out: Path, dpi: int):
    fig, axes = plt.subplots(1, len(modalities), figsize=(5.2 * len(modalities), 4.4),
                             squeeze=False)
    scaled_any = False
    for ax, modality in zip(axes[0], modalities):
        _, any_scaled = _draw_modality(ax, curves, models, modality, override,
                                       colors, relative=True)
        scaled_any = scaled_any or any_scaled
    _shared_legend(fig, axes[0])
    fig.suptitle("Degradation vs context length  (pooled across documents; "
                 "dashed = no-filler baseline"
                 + (" = 1.0)" if scaled_any else ")"), y=1.04, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_actual_vs_relative(curves, models, modalities, override, colors,
                            out: Path, dpi: int):
    """Actual accuracy (top row) alongside relative-to-baseline (bottom row),
    one column per modality. Both rows include the no-filler baseline point."""
    fig, axes = plt.subplots(2, len(modalities),
                             figsize=(5.2 * len(modalities), 8.4), squeeze=False)
    for ax, modality in zip(axes[0], modalities):
        _draw_modality(ax, curves, models, modality, override, colors,
                       relative=False, title_suffix=" — actual")
    scaled_any = False
    for ax, modality in zip(axes[1], modalities):
        _, any_scaled = _draw_modality(ax, curves, models, modality, override,
                                       colors, relative=True,
                                       title_suffix=" — rel. to baseline")
        scaled_any = scaled_any or any_scaled
    _shared_legend(fig, axes[0])
    fig.suptitle("Actual vs relative accuracy by context length  "
                 "(pooled across documents; dashed = no-filler baseline"
                 + (" = 1.0 below)" if scaled_any else ")"), y=1.02, fontsize=12)
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
# Position -> accuracy (MAUD fragment mode only)
# --------------------------------------------------------------------------- #
def location_points(runs, gold) -> dict:
    """(model, modality) -> [(min_frac, score)] for located MAUD questions.

    Joins each run's per-question predictions (re-derived from the logged raw
    output, exactly as score_outputs does) to its cell's `question_locations`,
    which only MAUD fragment-mode cells carry. `min_frac` is where the question's
    nearest evidence fragment landed in the window (0 = front, 1 = end). `score`
    is 1/0 for single-pick multiple choice (mc_accuracy) and the set-F1 for
    select-all questions (ms_f1) — the same correctness the headline metrics use.
    Questions left unscored by a parse/response failure are dropped, and in the
    missing_answer modality only the safe-negative questions count, mirroring
    score_outputs.tally_cell so the points line up with the curve metrics.
    """
    points: dict[tuple, list] = {}
    for run in runs:
        cell = gold.get(run.get("cell_id"))
        if not cell:
            continue
        locs = cell.get("question_locations")
        if not locs:  # whole-document cells (CUAD, MAUD --full-documents) have none
            continue
        modality = cell.get("modality")
        model = run.get("model")
        pred_by_qid, unscored, *_ = predictions_from_run(run)
        for q in cell["questions"]:
            qid = q["qa_id"]
            if qid in unscored:
                continue
            loc = locs.get(qid)
            if not loc or loc.get("min_frac") is None:
                continue
            atype = q.get("answer_type")
            if atype not in ("multiple_choice", "multi_select"):
                continue
            if modality == "missing_answer" and not q.get("is_negative"):
                continue
            pred = pred_by_qid.get(qid)
            answer = str(pred.get("answer", "")).strip() if pred else ""
            options = q.get("answer_options") or q.get("gold_answers") or []
            golds = q.get("gold_answers") or [a["text"] for a in q.get("answers", [])]
            if atype == "multi_select":
                _, score, _ = ms_score(answer, options, golds)
            else:
                score = 1.0 if mc_is_correct(answer, options, golds) else 0.0
            points.setdefault((model, modality), []).append(
                (float(loc["min_frac"]), float(score)))
    return points


def _bin_means(pts, nbins: int):
    """(centers, means, counts) of `pts` [(frac, score)] over `nbins` bins in [0,1]."""
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    edges = np.linspace(0.0, 1.0, nbins + 1)
    idx = np.clip(np.digitize(xs, edges) - 1, 0, nbins - 1)
    centers, means, counts = [], [], []
    for b in range(nbins):
        m = idx == b
        if m.any():
            centers.append((edges[b] + edges[b + 1]) / 2)
            means.append(float(ys[m].mean()))
            counts.append(int(m.sum()))
    return np.array(centers), np.array(means), np.array(counts)


def plot_position_accuracy(points_by, models, modalities, colors, out: Path,
                           dpi: int, nbins: int):
    """Per-question accuracy vs evidence position in the window, per modality.

    One panel per modality; one binned-mean line per model (markers sized by the
    bin's question count), with the individual per-question scores drawn as faint
    dots behind the line so the binning and sample density stay visible.
    """
    fig, axes = plt.subplots(1, len(modalities),
                             figsize=(5.2 * len(modalities), 4.4), squeeze=False)
    for ax, modality in zip(axes[0], modalities):
        for model in models:
            pts = points_by.get((model, modality))
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.scatter(xs, ys, s=10, color=colors[model], alpha=0.10, linewidths=0)
            cx, cy, ns = _bin_means(pts, nbins)
            if len(cx):
                ax.plot(cx, cy, marker="o", color=colors[model], linewidth=2,
                        markersize=4, label=f"{model} (n={len(pts)})")
        ax.set_title(modality)
        ax.set_xlabel("evidence position in window (0 = front, 1 = end)")
        ax.set_ylabel("per-question accuracy")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
    _shared_legend(fig, axes[0])
    fig.suptitle("MAUD accuracy vs. evidence position in context window  "
                 f"(binned mean per model, {nbins} bins; faint dots = individual "
                 "questions)", y=1.04, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
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
    if "actual" in args.charts:
        written.append(plot_actual_vs_relative(
            curves, models, modalities, args.metric, colors,
            out / "degradation_actual_vs_relative.png", args.dpi))
    if "overview" in args.charts:
        written.append(plot_overview(curves, models, modalities, args.metric,
                                     out / "overview.png", args.dpi))
    if "position" in args.charts:
        # Needs the gold cells' per-question evidence locations, which only MAUD
        # fragment-mode cells carry; CUAD (and MAUD --full-documents) have none.
        prepared = args.prepared or prepared_dir(dataset)
        try:
            gold = load_gold(prepared)
        except SystemExit as e:
            print(f"note: {dataset}: skipping position chart ({e}).")
            gold = None
        points_by = location_points(runs, gold) if gold else {}
        if points_by:
            loc_models = [m for m in models if any((m, mo) in points_by
                                                   for mo in MODALITY_ORDER)]
            loc_mods = [m for m in MODALITY_ORDER
                        if any((mo, m) in points_by for mo in loc_models)]
            written.append(plot_position_accuracy(
                points_by, loc_models, loc_mods, colors,
                out / "position_accuracy.png", args.dpi, args.position_bins))
        else:
            print(f"note: {dataset}: no per-question evidence locations found; "
                  f"skipping position chart (MAUD fragment mode only).")

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
    parser.add_argument("--prepared", type=Path, default=None,
                        help="override gold cells dir (for the position chart); "
                             "default data/prepared_<dataset> (single --dataset)")
    parser.add_argument("--charts", nargs="+",
                        default=["length", "actual", "overview", "position"],
                        choices=["length", "actual", "overview", "position"],
                        help="which charts to render (default: length actual "
                             "overview position; position is MAUD-only)")
    parser.add_argument("--position-bins", type=int, default=10,
                        help="number of position bins for the position chart "
                             "(default: 10)")
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "both" else (args.dataset,)
    if (args.results is not None or args.out is not None
            or args.prepared is not None) and len(datasets) > 1:
        raise SystemExit("error: --results/--out/--prepared target a single "
                         "dataset; pass --dataset cuad or --dataset maud.")

    for ds in datasets:
        plot_dataset(ds, args)


if __name__ == "__main__":
    main()
