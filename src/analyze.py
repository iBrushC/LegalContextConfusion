"""analyze.py — turn run results into degradation curves.

Reads the run records written by run_models.py and reshapes them into the
deliverable from the board's `degradation-curves` card: for each
(model x modality), the headline eval score plotted against context LENGTH
(budget) and target DEPTH (start / middle / end), as mean +/- stdev, alongside
the base-procedure signatures (latency, tokens, cost).

The headline metric is chosen by the cell's `focus`:
  * focus=spans      (rot, confusion)      -> span_f1   [+ exact_match, wrong_doc_rate]
  * focus=abstention (missing_answer)      -> abstention_rate [+ hallucination_rate]
Override with --metric to curve any metric.

Outputs: human-readable tables + ASCII sparklines to the console, and a tidy
long-format curves.csv (model, modality, budget, position, metric, mean, std)
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
from score_outputs import _AGG_METRICS, aggregate  # noqa: E402

DEFAULT_RESULTS = Path("data/results/runs.jsonl")
DEFAULT_CSV = Path("data/results/curves.csv")

POSITION_ORDER = ["target_at_start", "target_at_middle", "target_at_end"]
SHORT_POS = {"target_at_start": "start", "target_at_middle": "mid",
             "target_at_end": "end"}
MODALITY_ORDER = ["rot", "confusion", "missing_answer", "missing_document"]

# Headline + secondary metrics per focus.
HEADLINE = {"spans": "span_f1", "abstention": "abstention_rate"}
SECONDARY = {"spans": ["exact_match", "wrong_doc_rate"],
             "abstention": ["hallucination_rate"]}

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
    """Index summary rows by (model, modality) -> {(budget, position): row}."""
    curves: dict[tuple, dict] = {}
    for row in summary:
        curves.setdefault((row["model"], row["modality"]), {})[
            (row["budget_tokens"], row["position"])] = row
    return curves


def _axes(cells: dict) -> tuple[list[int], list[str]]:
    budgets = sorted({b for (b, _) in cells})
    positions = [p for p in POSITION_ORDER if any(p == pp for (_, pp) in cells)]
    return budgets, positions


# --------------------------------------------------------------------------- #
# Console report
# --------------------------------------------------------------------------- #
def report(summary: list[dict], metric_override: str | None) -> None:
    curves = build_curves(summary)
    models = sorted({m for (m, _) in curves})
    print(f"Degradation curves — {len(summary)} (model x cell) rows, "
          f"{len(models)} model(s)\n")

    for model in models:
        print(f"{'='*72}\nMODEL: {model}\n{'='*72}")
        modalities = [m for m in MODALITY_ORDER if (model, m) in curves]
        for modality in modalities:
            cells = curves[(model, modality)]
            focus = next(iter(cells.values()))["focus"]
            metric = metric_override or HEADLINE.get(focus, "span_f1")
            budgets, positions = _axes(cells)

            print(f"\n  modality={modality}   focus={focus}   "
                  f"headline={metric}  (mean±stdev)")
            header = "  " + f"{'length':>8} | " + " | ".join(
                f"{SHORT_POS.get(p, p):^11}" for p in positions)
            print(header)
            print("  " + "-" * (len(header) - 2))
            for b in budgets:
                row_cells = []
                for p in positions:
                    r = cells.get((b, p))
                    if r:
                        row_cells.append(_cell(r.get(f"{metric}_mean"),
                                               r.get(f"{metric}_std")))
                    else:
                        row_cells.append(_cell(None, None))
                print("  " + f"{_human_budget(b):>8} | " + " | ".join(row_cells))

            # Sparkline of the headline metric vs length, one line per position.
            print(f"  curve ({metric} over length {_human_budget(budgets[0])}"
                  f"->{_human_budget(budgets[-1])}):")
            for p in positions:
                series = [cells.get((b, p), {}).get(f"{metric}_mean") for b in budgets]
                print(f"    {SHORT_POS.get(p, p):>5}: {_spark(series)}   "
                      + " ".join(f"{v:.2f}" if v is not None else " -  " for v in series))

            _print_extras(cells, focus, budgets, positions, metric)
        print()


def _print_extras(cells, focus, budgets, positions, metric) -> None:
    """Secondary metrics + length/depth deltas + cost signatures."""
    # Secondary metrics (averaged over all cells of this modality).
    for sec in SECONDARY.get(focus, []):
        if sec == metric:
            continue
        vals = [r.get(f"{sec}_mean") for r in cells.values()
                if r.get(f"{sec}_mean") is not None]
        if vals:
            print(f"  {sec}: {sum(vals)/len(vals):.3f} (avg over cells)")

    # Length effect: headline at longest vs shortest budget (avg over positions).
    def avg_at(budget):
        vs = [cells.get((budget, p), {}).get(f"{metric}_mean") for p in positions]
        vs = [v for v in vs if v is not None]
        return sum(vs) / len(vs) if vs else None

    lo, hi = avg_at(budgets[0]), avg_at(budgets[-1])
    if lo is not None and hi is not None and len(budgets) > 1:
        print(f"  length effect: {metric} {lo:.2f} @ {_human_budget(budgets[0])} "
              f"-> {hi:.2f} @ {_human_budget(budgets[-1])}  (Δ {hi-lo:+.2f})")

    # Depth effect at the longest budget: end vs start.
    if {"target_at_start", "target_at_end"} <= set(positions) and len(budgets):
        b = budgets[-1]
        s = cells.get((b, "target_at_start"), {}).get(f"{metric}_mean")
        e = cells.get((b, "target_at_end"), {}).get(f"{metric}_mean")
        if s is not None and e is not None:
            print(f"  depth effect @ {_human_budget(b)}: start {s:.2f} -> "
                  f"end {e:.2f}  (Δ {e-s:+.2f})")

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
                "budget_tokens": r["budget_tokens"], "position": r["position"],
                "metric": metric, "mean": mean, "std": r.get(f"{metric}_std"),
                "runs": r["runs"], "ok_runs": r["ok_runs"],
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

    runs = load_runs(args.results)
    summary = aggregate(runs)
    if not summary:
        raise SystemExit("error: no scored runs to analyze.")

    report(summary, args.metric)

    if not args.no_csv:
        n = write_curves_csv(summary, args.csv)
        print(f"Wrote {n} tidy rows -> {args.csv}")


if __name__ == "__main__":
    main()
