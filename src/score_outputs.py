"""score_outputs.py — parse model replies and score them against CUAD gold.

Pure functions: no network, no file I/O. run_models.py wires these into its
run loop. The scoring is unchanged from the original monolith:

  * extract_json  — tolerant JSON extraction from a reply -> (value, error)
  * token_f1      — SQuAD-style token F1 between a predicted and gold span
  * score_cell    — per-cell metrics from predictions keyed by qa_id
  * aggregate     — (model x cell) mean/stdev for the degradation curves
"""

from __future__ import annotations

import json
import re
import statistics
import string
from collections import Counter

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


def score_cell(cell: dict, pred_by_qid: dict, unscored: set) -> dict:
    """Score merged predictions (keyed by qa_id) against the cell's gold.

    `unscored` holds qa_ids whose batch failed to parse — excluded from metrics
    rather than miscounted as a (wrong) abstention.
    """
    target_id = cell["target_document_id"]
    f1s, exacts = [], []
    answered_pos = wrong_doc = 0
    n_pos = n_neg = 0
    correct_abstain = hallucinated = 0
    n_unscored = 0

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
                f1s.append(max(token_f1(answer, g) for g in golds))
                exacts.append(any(_normalize(answer) == _normalize(g) for g in golds))
                if pred is not None and pred.get("document_id") not in (None, target_id):
                    wrong_doc += 1
            else:  # missed an answerable clause
                f1s.append(0.0)
                exacts.append(False)
        else:  # negative: clause genuinely absent
            n_neg += 1
            if present:
                hallucinated += 1
            else:
                correct_abstain += 1

    def rate(num, den):
        return num / den if den else None

    return {
        "parse_ok": n_unscored == 0,
        "scored": n_pos + n_neg,
        "n_unscored": n_unscored,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "span_f1": rate(sum(f1s), len(f1s)),
        "exact_match": rate(sum(exacts), len(exacts)),
        "answered_positive_rate": rate(answered_pos, n_pos),
        "wrong_doc_rate": rate(wrong_doc, answered_pos),
        "abstention_rate": rate(correct_abstain, n_neg),
        "hallucination_rate": rate(hallucinated, n_neg),
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
