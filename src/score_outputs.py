"""score_outputs.py — parse model replies and score them against CUAD gold.

Pure functions: no network, no file I/O. run_models.py wires these into its
run loop. The scoring is unchanged from the original monolith:

  * extract_json  — tolerant JSON extraction from a reply -> (value, error)
  * token_f1      — SQuAD-style token F1 between a predicted and gold span
  * score_cell    — per-cell metrics from predictions keyed by qa_id
  * aggregate     — (model x cell) mean/stdev for the degradation curves
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import string
from collections import Counter
from pathlib import Path

_ARTICLES = {"a", "an", "the"}
_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")

_AGG_METRICS = ("span_f1", "exact_match", "abstention_rate",
                "hallucination_rate", "wrong_doc_rate", "answered_positive_rate")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def extract_json(text: str):
    """Parse a JSON object from a model reply. Returns (value, error_message)."""
    if not text:
        return None, "empty response"
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    try:
        return json.loads(t), None
    except json.JSONDecodeError as e:
        first_err = str(e)
    start = t.find("{")  # fall back to first balanced {...} block
    if start != -1:
        depth = 0
        for i in range(start, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i + 1]), None
                    except json.JSONDecodeError as e:
                        return None, f"JSON parse failed: {e}"
    return None, f"no JSON object found ({first_err})"


def results_of(parsed):
    """Pull the results list out of a parsed object, tolerant of shape."""
    if isinstance(parsed, dict):
        return parsed.get("results")
    if isinstance(parsed, list):
        return parsed
    return None


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _normalize(s: str) -> list[str]:
    s = _PUNCT_RE.sub(" ", (s or "").lower())
    return [t for t in s.split() if t not in _ARTICLES]


def token_f1(pred: str, gold: str) -> float:
    p, g = _normalize(pred), _normalize(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def tally_cell(cell: dict, pred_by_qid: dict, unscored: set) -> dict:
    """Raw per-question counts for one cell (the basis for all rate metrics).

    `unscored` holds qa_ids whose batch failed to parse — excluded entirely
    rather than miscounted as a (wrong) abstention. These counts are additive,
    so the standalone scorer can pool them across cells for micro-rates.
    """
    target_id = cell["target_document_id"]
    sum_f1 = 0.0
    sum_exact = n_pos = n_neg = 0
    answered_pos = wrong_doc = doc_id_present = 0
    correct_abstain = hallucinated = n_unscored = 0

    for q in cell["questions"]:
        qid = q["qa_id"]
        if qid in unscored:
            n_unscored += 1
            continue
        pred = pred_by_qid.get(qid)
        answer = str(pred.get("answer", "")).strip() if pred else ""
        present = bool(pred and pred.get("present") and answer)

        if not q["is_impossible"]:  # positive: gold span exists
            n_pos += 1
            golds = [a["text"] for a in q["answers"]]
            if present:
                answered_pos += 1
                sum_f1 += max(token_f1(answer, g) for g in golds)
                if any(_normalize(answer) == _normalize(g) for g in golds):
                    sum_exact += 1
                doc_id = pred.get("document_id")
                if doc_id is not None:
                    doc_id_present += 1
                    if doc_id != target_id:
                        wrong_doc += 1
            # a missed answerable clause contributes 0 to sum_f1 / sum_exact
        else:  # negative: clause genuinely absent
            n_neg += 1
            if present:
                hallucinated += 1
            else:
                correct_abstain += 1

    return {
        "n_positive": n_pos, "n_negative": n_neg, "n_unscored": n_unscored,
        "sum_f1": sum_f1, "sum_exact": sum_exact,
        "answered_positive": answered_pos, "wrong_doc": wrong_doc,
        "doc_id_present": doc_id_present,
        "correct_abstention": correct_abstain, "hallucinated": hallucinated,
    }


def score_cell(cell: dict, pred_by_qid: dict, unscored: set) -> dict:
    """Per-cell metrics (rates) derived from tally_cell."""
    t = tally_cell(cell, pred_by_qid, unscored)

    def rate(num, den):
        return num / den if den else None

    return {
        "parse_ok": t["n_unscored"] == 0,
        "scored": t["n_positive"] + t["n_negative"],
        "n_unscored": t["n_unscored"],
        "n_positive": t["n_positive"],
        "n_negative": t["n_negative"],
        "span_f1": rate(t["sum_f1"], t["n_positive"]),
        "exact_match": rate(t["sum_exact"], t["n_positive"]),
        "answered_positive_rate": rate(t["answered_positive"], t["n_positive"]),
        "wrong_doc_rate": rate(t["wrong_doc"], t["answered_positive"]),
        "abstention_rate": rate(t["correct_abstention"], t["n_negative"]),
        "hallucination_rate": rate(t["hallucinated"], t["n_negative"]),
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _mean_std(values: list):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    return (round(statistics.mean(vals), 4),
            round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0)


def aggregate(runs: list[dict]) -> list[dict]:
    """Group run records by (model, cell) -> mean/stdev for degradation curves."""
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        groups.setdefault((r["model"], r["cell_id"]), []).append(r)

    summary = []
    for (model, cell_id), rs in sorted(groups.items()):
        ok = [r for r in rs if r.get("metrics", {}).get("scored", 0) > 0]
        row = {
            "model": model, "model_slug": rs[0]["model_slug"], "cell_id": cell_id,
            "modality": rs[0]["modality"], "focus": rs[0]["focus"],
            "budget_tokens": rs[0]["budget_tokens"], "position": rs[0]["position"],
            "runs": len(rs), "ok_runs": len(ok),
        }
        for m in _AGG_METRICS:
            mean, std = _mean_std([r["metrics"].get(m) for r in ok])
            row[f"{m}_mean"], row[f"{m}_std"] = mean, std
        row["latency_mean"], _ = _mean_std([r.get("latency_s") for r in ok])
        row["prompt_tokens_mean"], _ = _mean_std(
            [r.get("usage", {}).get("prompt_tokens") for r in ok])
        row["cost_mean"], _ = _mean_std(
            [r.get("usage", {}).get("cost") for r in ok])
        summary.append(row)
    return summary


# =========================================================================== #
# Standalone scorer: re-score runs.jsonl from the logged raw outputs.
#
# Run as `python src/score_outputs.py`. This re-parses each run's raw_output
# (independent of the metrics recorded at run time), joins it to the gold cells
# from build_context.py, and reports six pooled (micro) metrics:
#   span token-F1, exact match, abstention rate, hallucination rate,
#   wrong-document rate (only if document_ids were returned), parse failure rate.
# =========================================================================== #
DEFAULT_RESULTS = Path("data/results/runs.jsonl")
DEFAULT_PREPARED = Path("data/prepared")

_ACC_FIELDS = ("n_positive", "n_negative", "n_unscored", "sum_f1", "sum_exact",
               "answered_positive", "wrong_doc", "doc_id_present",
               "correct_abstention", "hallucinated",
               "batches", "responded", "parse_failed", "noresp", "cells")


def load_runs(path: Path) -> list[dict]:
    if path.is_dir():
        path = path / "runs.jsonl"
    if not path.exists():
        raise SystemExit(f"error: {path} not found. Run run_models.py first.")
    runs = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    if not runs:
        raise SystemExit(f"error: {path} is empty.")
    return runs


def load_gold(prepared: Path) -> dict:
    """Map cell_id -> prepared cell (with gold answers), for joining to runs."""
    manifest = prepared / "manifest.json"
    if manifest.exists():
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        paths = [Path(c["path"]) for c in meta.get("cells", []) if c.get("path")]
    else:
        paths = sorted((prepared / "cells").glob("*.json"))
    cells = {}
    for p in paths:
        if p.exists():
            c = json.loads(p.read_text(encoding="utf-8"))
            cells[c["cell_id"]] = c
    if not cells:
        raise SystemExit(
            f"error: no gold cells under {prepared}. Run build_context.py first."
        )
    return cells


def predictions_from_run(run: dict) -> tuple[dict, set, int, int, int, int]:
    """Rebuild predictions from a run's raw_output, batch by batch.

    Returns (pred_by_qid, unscored_qids, batches, responded, parse_failed,
    no_response). raw_output is re-parsed so scoring does not depend on the
    metrics recorded at run time.
    """
    raws = run.get("raw_output") or []
    parseds = run.get("parsed_output") or []
    batch_qids = run.get("batch_qa_ids") or []
    n = len(batch_qids) or max(len(raws), len(parseds), 0)

    pred_by_qid: dict = {}
    unscored: set = set()
    responded = parse_failed = noresp = 0
    for i in range(n):
        qids = batch_qids[i] if i < len(batch_qids) else []
        raw = raws[i] if i < len(raws) else None
        if raw is None:  # no response (request error / empty) — not a parse fail
            results = results_of(parseds[i]) if i < len(parseds) else None
            if results is None:
                noresp += 1
                unscored.update(qids)
                continue
        else:
            parsed, _ = extract_json(raw)
            results = results_of(parsed)
        responded += 1
        if results is None:
            parse_failed += 1
            unscored.update(qids)
        else:
            for r in results:
                if isinstance(r, dict) and "qa_id" in r:
                    pred_by_qid[str(r["qa_id"]).strip()] = r
    return pred_by_qid, unscored, n, responded, parse_failed, noresp


def metrics_from_acc(acc: dict) -> dict:
    """Pooled (micro) rate metrics from accumulated counts."""
    def rate(num, den):
        return num / den if den else None

    return {
        "span_f1": rate(acc["sum_f1"], acc["n_positive"]),
        "exact_match": rate(acc["sum_exact"], acc["n_positive"]),
        "abstention_rate": rate(acc["correct_abstention"], acc["n_negative"]),
        "hallucination_rate": rate(acc["hallucinated"], acc["n_negative"]),
        # only meaningful if the model actually returned document_ids
        "wrong_doc_rate": (rate(acc["wrong_doc"], acc["answered_positive"])
                           if acc["doc_id_present"] else None),
        "parse_failure_rate": rate(acc["parse_failed"], acc["responded"]),
    }


def score_jsonl(runs: list[dict], gold: dict) -> dict:
    """Re-score every run, accumulating overall + per-model + per-modality."""
    overall = {k: 0 for k in _ACC_FIELDS}
    overall["missing_gold"] = 0
    by_model: dict = {}
    by_modality: dict = {}

    for run in runs:
        cell = gold.get(run.get("cell_id"))
        if cell is None:
            overall["missing_gold"] += 1
            continue
        pred, unscored, batches, responded, pfail, noresp = predictions_from_run(run)
        t = tally_cell(cell, pred, unscored)
        targets = [overall,
                   by_model.setdefault(run.get("model", "?"), {k: 0 for k in _ACC_FIELDS}),
                   by_modality.setdefault(run.get("modality", "?"), {k: 0 for k in _ACC_FIELDS})]
        for acc in targets:
            for k in ("n_positive", "n_negative", "n_unscored", "sum_f1",
                      "sum_exact", "answered_positive", "wrong_doc",
                      "doc_id_present", "correct_abstention", "hallucinated"):
                acc[k] += t[k]
            acc["batches"] += batches
            acc["responded"] += responded
            acc["parse_failed"] += pfail
            acc["noresp"] += noresp
            acc["cells"] += 1

    return {"overall": overall, "by_model": by_model, "by_modality": by_modality}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "  n/a"


def _print_overall(acc: dict) -> None:
    m = metrics_from_acc(acc)
    print(f"\nOverall ({acc['cells']} scored cell-runs, "
          f"{acc['n_positive']:,} positive + {acc['n_negative']:,} negative Qs):")
    print(f"  span token-F1        {_fmt(m['span_f1'])}   "
          f"(over {acc['n_positive']:,} positives)")
    print(f"  exact match          {_fmt(m['exact_match'])}")
    print(f"  abstention rate      {_fmt(m['abstention_rate'])}   "
          f"(over {acc['n_negative']:,} negatives)")
    print(f"  hallucination rate   {_fmt(m['hallucination_rate'])}")
    if acc["doc_id_present"]:
        print(f"  wrong-document rate  {_fmt(m['wrong_doc_rate'])}   "
              f"(over {acc['answered_positive']:,} answered positives; "
              f"{acc['doc_id_present']:,} carried a document_id)")
    else:
        print("  wrong-document rate  n/a  (no document_ids in outputs)")
    print(f"  parse failure rate   {_fmt(m['parse_failure_rate'])}   "
          f"({acc['parse_failed']}/{acc['responded']} responded batches"
          + (f"; {acc['noresp']} no-response" if acc['noresp'] else "") + ")")
    if acc["n_unscored"]:
        print(f"  ({acc['n_unscored']:,} questions left unscored by parse/response failures)")


def _print_breakdown(title: str, groups: dict) -> None:
    if not groups:
        return
    print(f"\nby {title}:")
    print(f"  {title:14} {'spanF1':>7} {'exact':>7} {'abst':>7} "
          f"{'halluc':>7} {'wrongD':>7} {'parseF':>7}  {'pos/neg':>13}")
    for name, acc in sorted(groups.items()):
        m = metrics_from_acc(acc)
        print(f"  {name:14} {_fmt(m['span_f1']):>7} {_fmt(m['exact_match']):>7} "
              f"{_fmt(m['abstention_rate']):>7} {_fmt(m['hallucination_rate']):>7} "
              f"{_fmt(m['wrong_doc_rate']):>7} {_fmt(m['parse_failure_rate']):>7}  "
              f"{str(acc['n_positive'])+'/'+str(acc['n_negative']):>13}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score runs.jsonl from logged raw outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                        help=f"runs.jsonl (or its dir) (default: {DEFAULT_RESULTS})")
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED,
                        help=f"gold cells dir (default: {DEFAULT_PREPARED})")
    parser.add_argument("--out", type=Path, default=None,
                        help="optional JSON file to write the scoreboard to")
    args = parser.parse_args()

    runs = load_runs(args.results)
    gold = load_gold(args.prepared)
    result = score_jsonl(runs, gold)

    print(f"Scored {len(runs)} run rows against {len(gold)} gold cells "
          f"(re-parsed from raw_output).")
    if result["overall"]["missing_gold"]:
        print(f"warning: {result['overall']['missing_gold']} run rows had no "
              f"matching gold cell in {args.prepared} (skipped).")
    _print_overall(result["overall"])
    _print_breakdown("model", result["by_model"])
    _print_breakdown("modality", result["by_modality"])

    if args.out:
        payload = {
            "overall": {**result["overall"], **metrics_from_acc(result["overall"])},
            "by_model": {k: {**v, **metrics_from_acc(v)}
                         for k, v in result["by_model"].items()},
            "by_modality": {k: {**v, **metrics_from_acc(v)}
                            for k, v in result["by_modality"].items()},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"\nWrote scoreboard -> {args.out}")


if __name__ == "__main__":
    main()
