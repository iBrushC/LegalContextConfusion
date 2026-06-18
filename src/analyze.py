"""analyze.py — turn run results into degradation curves.

Reads the run records written by run_models.py and reshapes them into the
deliverable from the board's `degradation-curves` card: for each
(model x modality), the headline eval score plotted against context LENGTH
(budget), as mean +/- stdev, alongside the base-procedure signatures (latency,
tokens, cost).

Runs are POOLED across the several target documents (aggregate_curves): every
document's runs for a given (model, modality, budget) are treated as repeated
trials, so the band reflects cross-document + multirun spread. (The target is
always at the end of the window now, so there is no depth/position axis.)

The headline metric is chosen by the cell's `focus`:
  * focus=spans      (CUAD rot, confusion) -> span_f1   [+ exact_match, wrong_doc_rate]
  * focus=abstention (CUAD missing_answer) -> abstention_rate [+ hallucination_rate]
  * focus=labels     (MAUD, multiple choice) -> mc_accuracy [+ wrong_doc_rate]
    (for the missing_answer modality, mc_accuracy is over only the safe-negative
    questions — the abstention analogue)
Override with --metric to curve any metric.

Outputs: human-readable tables + ASCII sparklines to the console, and a tidy
long-format curves.csv (model, modality, budget, metric, mean, std)
ready for matplotlib / seaborn / a spreadsheet.

Develop/test it against mock outputs:
    python src/run_models.py --mock --models claude gemini   # writes runs.jsonl
    python src/analyze.py                                     # analyze them

Usage:
    python src/analyze.py
    python src/analyze.py --results data/results/runs.jsonl
    python src/analyze.py --metric wrong_doc_rate
    python src/analyze.py --csv data/results/curves.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_outputs import _AGG_METRICS, aggregate_curves  # noqa: E402

DEFAULT_RESULTS = Path("data/results/runs.jsonl")
DEFAULT_CSV = Path("data/results/curves.csv")

MODALITY_ORDER = ["rot", "confusion", "missing_answer", "missing_document"]
# The single shared zero-filler baseline cell (modality="baseline") anchors these
# interference modalities; build_curves fans it out as their budget-0 point.
BASELINE_ANCHORS = ("rot", "confusion")

# Headline + secondary metrics per focus.
HEADLINE = {"spans": "span_f1", "abstention": "abstention_rate",
            "labels": "mc_accuracy"}
SECONDARY = {"spans": ["exact_match", "wrong_doc_rate"],
             "abstention": ["hallucination_rate"],
             "labels": ["wrong_doc_rate"]}

_BLOCKS = "▁▂▃▄▅▆▇█"  # sparkline ramp for values in [0, 1]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_runs(path: Path) -> list[dict]:
    if path.is_dir():
        path = path / "runs.jsonl"
    if not path.exists():
        raise SystemExit(
            f"error: {path} not found. Run run_models.py (e.g. --mock) first."
        )
    runs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            runs.append(json.loads(line))
    if not runs:
        raise SystemExit(f"error: {path} is empty.")
    return runs


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _spark(values: list) -> str:
    out = []
    for v in values:
        if v is None:
            out.append(" ")
        else:
            out.append(_BLOCKS[max(0, min(len(_BLOCKS) - 1, round(v * (len(_BLOCKS) - 1))))])
    return "".join(out)


def _cell(mean, std) -> str:
    if mean is None:
        return f"{'-':^11}"
    body = f"{mean:.2f}±{std:.2f}" if std is not None else f"{mean:.2f}"
    return f"{body:^11}"


def _human_tokens(n) -> str:
    if n is None:
        return "?"
    return f"{n/1000:.0f}k" if n >= 1000 else f"{n:.0f}"


def _human_budget(n: int) -> str:
    return f"{n//1000}k" if n >= 1000 else str(n)


# --------------------------------------------------------------------------- #
# Reshape into (model, modality) curve tables
# --------------------------------------------------------------------------- #
def build_curves(summary: list[dict]) -> dict:
    """Index pooled curve rows by (model, modality) -> {budget_tokens: row}.

    Budget 0 is the zero-filler baseline anchor; other budgets are the
    interference points. Each row is already pooled across documents.

    The baseline is now emitted once per document as its own modality="baseline"
    cell (rather than duplicated per interference modality), so here it is fanned
    back out: the shared baseline row becomes the budget-0 anchor of every
    modality it anchors (rot/confusion), and the standalone "baseline" group is
    dropped. Downstream table/plot code then sees the same shape as before.
    """
    curves: dict[tuple, dict] = {}
    for row in summary:
        curves.setdefault((row["model"], row["modality"]), {})[
            row["budget_tokens"]] = row

    for model in {m for (m, _) in curves}:
        base_group = curves.pop((model, "baseline"), None)
        if not base_group:
            continue
        base_row = base_group.get(0) or next(iter(base_group.values()))
        for modality in BASELINE_ANCHORS:
            group = curves.get((model, modality))
            if group is not None:
                group.setdefault(0, base_row)
    return curves


# --------------------------------------------------------------------------- #
# Console report
# --------------------------------------------------------------------------- #
def report(summary: list[dict], metric_override: str | None) -> None:
    curves = build_curves(summary)
    models = sorted({m for (m, _) in curves})
    print(f"Degradation curves — {len(summary)} (model x modality x budget) rows, "
          f"{len(models)} model(s)\n")

    for model in models:
        print(f"{'='*72}\nMODEL: {model}\n{'='*72}")
        modalities = [m for m in MODALITY_ORDER if (model, m) in curves]
        for modality in modalities:
            cells = curves[(model, modality)]
            sample = next(iter(cells.values()))
            focus = sample["focus"]
            metric = metric_override or HEADLINE.get(focus, "span_f1")
            budgets = sorted(cells)
            ndocs = sample.get("num_documents")
            ndesc = f" across {ndocs} doc(s)" if ndocs else ""

            print(f"\n  modality={modality}   focus={focus}   "
                  f"headline={metric}  (mean±stdev{ndesc})")
            print(f"  {'length':>8} | {metric:^13}")
            print("  " + "-" * 24)
            for b in budgets:
                r = cells.get(b)
                cell_str = (_cell(r.get(f"{metric}_mean"), r.get(f"{metric}_std"))
                            if r else _cell(None, None))
                print(f"  {_human_budget(b):>8} | {cell_str}")

            # Sparkline of the headline metric vs length (single pooled series).
            series = [cells[b].get(f"{metric}_mean") for b in budgets]
            print(f"  curve ({metric} over length {_human_budget(budgets[0])}"
                  f"->{_human_budget(budgets[-1])}): {_spark(series)}   "
                  + " ".join(f"{v:.2f}" if v is not None else " - " for v in series))

            _print_extras(cells, focus, budgets, metric)
        print()


def _print_extras(cells, focus, budgets, metric) -> None:
    """Secondary metrics + length delta + cost signatures (length-only now)."""
    # Secondary metrics (averaged over all budgets of this modality).
    for sec in SECONDARY.get(focus, []):
        if sec == metric:
            continue
        vals = [r.get(f"{sec}_mean") for r in cells.values()
                if r.get(f"{sec}_mean") is not None]
        if vals:
            print(f"  {sec}: {sum(vals)/len(vals):.3f} (avg over budgets)")

    # Length effect: headline at the longest vs shortest budget.
    def at(budget):
        r = cells.get(budget)
        return r.get(f"{metric}_mean") if r else None

    lo, hi = at(budgets[0]), at(budgets[-1])
    if lo is not None and hi is not None and len(budgets) > 1:
        print(f"  length effect: {metric} {lo:.2f} @ {_human_budget(budgets[0])} "
              f"-> {hi:.2f} @ {_human_budget(budgets[-1])}  (Δ {hi-lo:+.2f})")

    # Signatures (averaged).
    lat = [r.get("latency_mean") for r in cells.values() if r.get("latency_mean") is not None]
    tok = [r.get("prompt_tokens_mean") for r in cells.values() if r.get("prompt_tokens_mean") is not None]
    cost = [r.get("cost_mean") for r in cells.values() if r.get("cost_mean") is not None]
    sig = []
    if lat:
        sig.append(f"latency~{sum(lat)/len(lat):.1f}s")
    if tok:
        sig.append(f"prompt~{_human_tokens(sum(tok)/len(tok))} tok")
    if cost:
        sig.append(f"cost~${sum(cost)/len(cost):.4f}")
    if sig:
        print("  signatures: " + "  ".join(sig))


# --------------------------------------------------------------------------- #
# Tidy long CSV (for plotting)
# --------------------------------------------------------------------------- #
def write_curves_csv(summary: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in summary:
        for metric in _AGG_METRICS:
            mean = r.get(f"{metric}_mean")
            if mean is None:
                continue
            rows.append({
                "model": r["model"], "modality": r["modality"], "focus": r["focus"],
                "budget_tokens": r["budget_tokens"],
                "is_baseline": r.get("is_baseline", False),
                "metric": metric, "mean": mean, "std": r.get(f"{metric}_std"),
                "runs": r["runs"], "ok_runs": r["ok_runs"],
                "num_documents": r.get("num_documents"),
                "latency_mean": r.get("latency_mean"),
                "prompt_tokens_mean": r.get("prompt_tokens_mean"),
                "cost_mean": r.get("cost_mean"),
            })
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                        help=f"runs.jsonl (or its dir) (default: {DEFAULT_RESULTS})")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help=f"tidy long-format curves CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--metric", default=None,
                        help="force a headline metric (default: by focus)")
    parser.add_argument("--no-csv", action="store_true",
                        help="skip writing the curves CSV")
    args = parser.parse_args()

    # Sparklines use Unicode block glyphs; force UTF-8 so a legacy Windows
    # console codepage (cp1252) doesn't crash the whole report mid-print.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    runs = load_runs(args.results)
    summary = aggregate_curves(runs)
    if not summary:
        raise SystemExit("error: no scored runs to analyze.")

    report(summary, args.metric)

    if not args.no_csv:
        n = write_curves_csv(summary, args.csv)
        print(f"Wrote {n} tidy rows -> {args.csv}")


if __name__ == "__main__":
    main()
