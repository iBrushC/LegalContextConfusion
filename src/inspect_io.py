"""inspect_io.py — inspect the full prompt sent to a model and the answers it gave.

This is the per-cell, end-to-end view that sits between run_models.py's --dry-run
(which previews only the first prompt, truncated to 1800 chars) and
score_outputs.py (which reports aggregate rates only). For ONE prepared cell it
renders, for both CUAD and MAUD:

  * INPUT  — the EXACT prompt the model is shown: the system instruction plus the
             user message that carries the input documents (context) and the
             clause questions, rendered by run_models.build_messages so it is
             byte-for-byte what was sent (not a paraphrase).
  * GOLD   — each question with its category, gold answer(s) / spans, and whether
             it is a negative (CUAD: absent clause; MAUD: a safe-negative option).
  * ANSWERS— when a results run exists, the model's answer per qa_id, joined to
             the gold and marked correct/incorrect (span F1 + exact for CUAD;
             resolved option + correct for MAUD), one block per model/run.

Both datasets flow through the same renderer; the cell's focus ("labels" -> MAUD
multiple choice, else CUAD spans) selects the layout, exactly as run_models does.

The document context can be very large (up to the biggest budget). It is printed
in full by default; cap the printed prompt with --context-chars, drop it entirely
with --no-context, or use --json/--out for a structured dump that is easier to
page through than a 2 MB console spew.

Usage:
    python src/inspect_io.py --dataset maud --list                 # list cell ids
    python src/inspect_io.py --dataset maud --cell d003_baseline
    python src/inspect_io.py --dataset cuad --cell d000_baseline --model claude
    python src/inspect_io.py --dataset maud --cell d003_baseline --all-runs
    python src/inspect_io.py --dataset cuad --cell d000_baseline --no-context
    python src/inspect_io.py --dataset maud --cell d003_baseline --json --out cell.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Sibling modules (src/ is on sys.path when run directly; make it explicit).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_models import build_messages, is_mc_cell  # noqa: E402
from score_outputs import (  # noqa: E402
    _normalize, load_gold, load_runs, mc_is_correct, ms_score, option_label,
    predictions_from_run, prepared_dir, resolve_choice, resolve_choices,
    results_dir, token_f1,
)

CHARS_PER_TOKEN = 4  # matches build_context.py / run_models.py


# --------------------------------------------------------------------------- #
# Per-question analysis (gold + optional model answer), shared by text + JSON
# --------------------------------------------------------------------------- #
def _gold_letters(options: list[str], golds: list[str]) -> list[str]:
    """Option letters whose text matches a gold answer (MAUD MC gold key)."""
    return [option_label(i) for i, opt in enumerate(options)
            if any(_normalize(opt) == _normalize(g) for g in golds)]


def analyze_question(q: dict, pred: dict | None, has_run: bool) -> dict:
    """Structured view of one question: gold, and the model's answer if a run is
    selected. `pred` is the model's result dict for this qa_id (None = the model
    never returned this qa_id); `has_run` distinguishes "no run selected" from
    "run selected but this qa_id missing"."""
    rec: dict = {"qa_id": q["qa_id"], "category": q.get("category", ""),
                 "question": q.get("question", "")}
    answer = str(pred.get("answer", "")).strip() if pred else ""

    atype = q.get("answer_type")
    if atype in ("multiple_choice", "multi_select"):  # MAUD: closed-set
        options = q.get("answer_options") or q.get("gold_answers") or []
        golds = q.get("gold_answers") or [a["text"] for a in q.get("answers", [])]
        rec["type"] = atype
        rec["is_negative"] = bool(q.get("is_negative"))
        rec["options"] = [{"label": option_label(i), "text": o}
                          for i, o in enumerate(options)]
        rec["gold_answers"] = golds
        rec["gold_letters"] = _gold_letters(options, golds)
        if atype == "multi_select":  # select-all-that-apply: set match
            if has_run:
                rec["model_missing"] = pred is None
                rec["model_answer_raw"] = answer
                chosen = resolve_choices(answer, options) if pred else []
                rec["model_choices"] = chosen
                rec["model_letters"] = [option_label(options.index(c))
                                        for c in chosen]
                rec["model_document_id"] = pred.get("document_id") if pred else None
                exact, f1, _ = (ms_score(answer, options, golds) if pred
                                else (False, 0.0, []))
                rec["exact_match"], rec["set_f1"] = exact, f1
                rec["correct"] = bool(pred) and exact
            return rec
        if has_run:
            rec["model_missing"] = pred is None
            rec["model_answer_raw"] = answer
            rec["model_choice"] = resolve_choice(answer, options) if pred else None
            rec["model_document_id"] = pred.get("document_id") if pred else None
            rec["correct"] = bool(pred) and mc_is_correct(answer, options, golds)
        return rec

    # CUAD: extractive span QA
    rec["type"] = "span"
    rec["is_impossible"] = bool(q.get("is_impossible"))
    rec["gold_answers"] = [{"text": a.get("text", ""),
                            "answer_start": a.get("answer_start")}
                           for a in q.get("answers", [])]
    if has_run:
        present = bool(pred and pred.get("present") and answer)
        rec["model_missing"] = pred is None
        rec["model_present"] = present
        rec["model_answer"] = answer
        rec["model_document_id"] = pred.get("document_id") if pred else None
        if q.get("is_impossible"):  # absent clause -> correct iff the model abstained
            rec["correct"] = bool(pred) and not present
        else:
            gold_texts = [a.get("text", "") for a in q.get("answers", [])]
            rec["span_f1"] = (max((token_f1(answer, g) for g in gold_texts),
                                  default=0.0) if present else 0.0)
            rec["exact_match"] = bool(present and any(
                _normalize(answer) == _normalize(g) for g in gold_texts))
    return rec


# --------------------------------------------------------------------------- #
# Run selection
# --------------------------------------------------------------------------- #
def select_runs(runs: list[dict], cell_id: str, model: str | None,
                run_index: int | None) -> list[dict]:
    """Run rows for this cell. Default: every model at `run_index` (one row each,
    since (model, cell, run_index) is unique). --model NAME narrows to one model;
    run_index=None (--all-runs) keeps every run of every selected model."""
    rows = [r for r in runs if r.get("cell_id") == cell_id]
    if model:
        rows = [r for r in rows if r.get("model") == model]
    if run_index is not None:
        rows = [r for r in rows if r.get("run_index") == run_index]
    return sorted(rows, key=lambda r: (r.get("model", ""), r.get("run_index", 0)))


# --------------------------------------------------------------------------- #
# Structured (JSON) view
# --------------------------------------------------------------------------- #
def build_record(cell: dict, run_rows: list[dict], context_chars: int,
                 with_context: bool) -> dict:
    """Full structured view of a cell: meta, prompt, gold questions, answers."""
    msgs = build_messages(cell, cell["questions"])
    system, user = msgs[0]["content"], msgs[1]["content"]
    if not with_context:
        user = user.replace(cell["context"],
                            f"<<context omitted: {len(cell['context']):,} chars>>")
    elif context_chars and len(user) > context_chars:
        user = user[:context_chars] + "\n...[truncated]..."

    answers = []
    for run in run_rows:
        pred_by_qid, _unscored, *_ = predictions_from_run(run)
        answers.append({
            "model": run.get("model"),
            "model_slug": run.get("model_slug"),
            "run_index": run.get("run_index"),
            "error": run.get("error"),
            "metrics": run.get("metrics"),
            "questions": [analyze_question(q, pred_by_qid.get(q["qa_id"]), True)
                          for q in cell["questions"]],
        })

    return {
        "cell_id": cell["cell_id"],
        "dataset": cell.get("dataset"),
        "modality": cell.get("modality"),
        "focus": cell.get("focus"),
        "budget_tokens": cell.get("budget_tokens"),
        "is_baseline": cell.get("is_baseline"),
        "target_document_id": cell.get("target_document_id"),
        "target_document_title": cell.get("target_document_title"),
        "actual_tokens": cell.get("actual_tokens"),
        "actual_chars": cell.get("actual_chars"),
        "distractor_ids": cell.get("distractor_ids"),
        "prompt": {"system": system, "user": user},
        "gold": [analyze_question(q, None, False) for q in cell["questions"]],
        "answers": answers,
    }


# --------------------------------------------------------------------------- #
# Text view
# --------------------------------------------------------------------------- #
def _header(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _mark(ok: bool) -> str:
    return "[OK] " if ok else "[X]  "


def print_meta(cell: dict) -> None:
    mc = is_mc_cell(cell)
    n_neg = sum(1 for q in cell["questions"]
                if (q.get("is_negative") if mc else q.get("is_impossible")))
    n_q = len(cell["questions"])
    neg_word = "safe-negative" if mc else "absent"
    _header(f"{(cell.get('dataset') or '?').upper()} cell {cell['cell_id']}")
    print(f"  modality: {cell.get('modality')}   focus: {cell.get('focus')}   "
          f"budget: {cell.get('budget_tokens'):,} tok   "
          f"baseline: {cell.get('is_baseline')}")
    print(f"  target:   {cell.get('target_document_id')}"
          + (f"  ({cell['target_document_title']})"
             if cell.get('target_document_title') else ""))
    print(f"  context:  {cell.get('actual_chars', 0):,} chars "
          f"(~{cell.get('actual_tokens', 0):,} tok)   "
          f"distractors: {len(cell.get('distractor_ids', []))}")
    print(f"  questions:{n_q:>4}  ({n_q - n_neg} answerable, {n_neg} {neg_word})  "
          f"task: {'multiple-choice' if mc else 'extractive spans'}")


def print_prompt(cell: dict, context_chars: int, with_context: bool) -> None:
    msgs = build_messages(cell, cell["questions"])
    system, user = msgs[0]["content"], msgs[1]["content"]
    if not with_context:
        user = user.replace(cell["context"],
                            f"<<context omitted: {len(cell['context']):,} chars "
                            f"(--no-context); the questions follow it>>")
    _header("PROMPT - [system]")
    print(system)
    truncated = bool(context_chars and len(user) > context_chars)
    shown = user[:context_chars] if truncated else user
    _header(f"PROMPT - [user] ({len(user):,} chars"
            + (f"; showing first {context_chars:,}" if truncated else "") + ")")
    print(shown)
    if truncated:
        print("\n...[truncated - raise --context-chars or use --json for the rest]...")


def print_gold(cell: dict) -> None:
    _header("QUESTIONS + GOLD")
    for q in cell["questions"]:
        rec = analyze_question(q, None, False)
        if rec["type"] in ("multiple_choice", "multi_select"):
            multi = rec["type"] == "multi_select"
            kind = "SELECT ALL" if multi else "CHOOSE ONE"
            tag = "NEG" if rec["is_negative"] else "POS"
            print(f"\n[{rec['qa_id']}] ({tag}) [{kind}] category=\"{rec['category']}\"")
            print(f"  Q: {rec['question']}")
            for opt in rec["options"]:
                star = " *" if opt["label"] in rec["gold_letters"] else ""
                print(f"    ({opt['label']}) {opt['text']}{star}")
            print(f"  gold: {', '.join(rec['gold_letters']) or '?'}  "
                  f"= {', '.join(repr(g) for g in rec['gold_answers'])}")
        else:
            tag = "NEG/absent" if rec["is_impossible"] else "POS"
            print(f"\n[{rec['qa_id']}] ({tag}) category=\"{rec['category']}\"")
            print(f"  Q: {rec['question']}")
            if rec["is_impossible"]:
                print("  gold: (clause absent - correct answer is to abstain)")
            else:
                for g in rec["gold_answers"]:
                    print(f"  gold @{g['answer_start']}: {g['text']!r}")


def print_answers(cell: dict, run_rows: list[dict]) -> None:
    if not run_rows:
        _header("MODEL ANSWERS")
        print("  (no results run found for this cell — run run_models.py, or "
              "pass --results)")
        return
    for run in run_rows:
        pred_by_qid, _unscored, *_ = predictions_from_run(run)
        m = run.get("metrics", {})
        _header(f"MODEL ANSWERS - {run.get('model')} (run {run.get('run_index')})")
        if run.get("error"):
            print(f"  error: {run['error']}")
        for q in cell["questions"]:
            rec = analyze_question(q, pred_by_qid.get(q["qa_id"]), True)
            if rec.get("model_missing"):
                print(f"[{rec['qa_id']}] (no answer returned)")
                continue
            if rec["type"] == "multi_select":
                letters = rec.get("model_letters") or []
                shown = (", ".join(letters) if letters
                         else repr(rec["model_answer_raw"]) + " (unresolved)")
                print(f"{_mark(rec['correct'])}[{rec['qa_id']}] selected "
                      f"[{shown}]  set-F1={rec['set_f1']:.2f}"
                      f"   gold=[{', '.join(rec['gold_letters']) or '?'}]"
                      + _doc_note(cell, rec))
            elif rec["type"] == "multiple_choice":
                choice = rec["model_choice"]
                shown = (f"({_letter_of(rec)}) {choice}" if choice
                         else repr(rec["model_answer_raw"]) + " (unresolved)")
                print(f"{_mark(rec['correct'])}[{rec['qa_id']}] {shown}"
                      f"   gold={', '.join(rec['gold_letters']) or '?'}"
                      + _doc_note(cell, rec))
            elif rec["is_impossible"]:
                print(f"{_mark(rec['correct'])}[{rec['qa_id']}] "
                      + ("abstained" if not rec["model_present"]
                         else f"HALLUCINATED: {rec['model_answer']!r}"))
            else:
                em = " exact" if rec["exact_match"] else ""
                if rec["model_present"]:
                    print(f"{_mark(rec['span_f1'] >= 0.5)}[{rec['qa_id']}] "
                          f"F1={rec['span_f1']:.2f}{em}{_doc_note(cell, rec)}")
                    print(f"      A: {rec['model_answer']!r}")
                else:
                    print(f"[X]  [{rec['qa_id']}] missed (said absent)")
        _print_run_summary(cell, m)


def _letter_of(rec: dict) -> str:
    """The letter for the model's resolved MC choice (for display)."""
    for opt in rec["options"]:
        if opt["text"] == rec["model_choice"]:
            return opt["label"]
    return "?"


def _doc_note(cell: dict, rec: dict) -> str:
    """Flag a wrong-document attribution in the model's answer."""
    doc = rec.get("model_document_id")
    if doc is not None and doc != cell.get("target_document_id"):
        return f"   [wrong doc: {doc}]"
    return ""


def _print_run_summary(cell: dict, m: dict) -> None:
    if not m:
        return
    if is_mc_cell(cell):
        parts = [f"mc_accuracy={_num(m.get('mc_accuracy'))}"]
        if m.get("ms_f1") is not None or m.get("ms_exact_match") is not None:
            parts.append(f"ms_exact={_num(m.get('ms_exact_match'))}")
            parts.append(f"ms_f1={_num(m.get('ms_f1'))}")
    else:
        parts = [f"span_f1={_num(m.get('span_f1'))}",
                 f"exact={_num(m.get('exact_match'))}",
                 f"abstention={_num(m.get('abstention_rate'))}"]
    if m.get("wrong_doc_rate") is not None:
        parts.append(f"wrong_doc={_num(m['wrong_doc_rate'])}")
    if m.get("n_unscored"):
        parts.append(f"unscored={m['n_unscored']}")
    print("  summary: " + "  ".join(parts))


def _num(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"


# --------------------------------------------------------------------------- #
# Listing
# --------------------------------------------------------------------------- #
def list_cells(cells: dict, runs: list[dict]) -> None:
    have = {r.get("cell_id") for r in runs}
    _header(f"Cells ({len(cells)})")
    print(f"  {'cell_id':32} {'modality':14} {'budget':>10} {'~tok':>10}  runs")
    for cid, c in sorted(cells.items()):
        n_runs = sum(1 for r in runs if r.get("cell_id") == cid)
        print(f"  {cid:32} {str(c.get('modality')):14} "
              f"{c.get('budget_tokens', 0):>10,} {c.get('actual_tokens', 0):>10,}  "
              f"{n_runs if cid in have else '-'}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    # The full prompt embeds legal text with smart quotes / en-dashes that a
    # legacy console code page (Windows cp1252) cannot encode; print as UTF-8 and
    # replace anything unmappable rather than crashing mid-dump.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("cuad", "maud"), required=True,
                        help="which dataset's cells/results to read (a cell id is "
                             "dataset-specific, so exactly one is required)")
    parser.add_argument("--cell", default=None,
                        help="cell_id to inspect (e.g. d003_baseline); omit with "
                             "--list to see the available ids")
    parser.add_argument("--list", action="store_true",
                        help="list the available cell ids (+ run counts) and exit")
    parser.add_argument("--prepared", type=Path, default=None,
                        help="override the prepared cells dir "
                             "(default: data/prepared_<dataset>)")
    parser.add_argument("--results", type=Path, default=None,
                        help="override runs.jsonl or its dir "
                             "(default: data/results_<dataset>)")
    parser.add_argument("--model", default=None,
                        help="show answers from this model only (default: every "
                             "model that ran the cell, at --run-index)")
    parser.add_argument("--run-index", type=int, default=0,
                        help="which run to show (default: 0; use --all-runs for all)")
    parser.add_argument("--all-runs", action="store_true",
                        help="show every run index, not just one")
    parser.add_argument("--no-prompt", action="store_true",
                        help="skip the full prompt section (gold + answers only)")
    parser.add_argument("--no-context", action="store_true",
                        help="render the prompt with the document context elided "
                             "(keeps the instructions + questions; huge cells)")
    parser.add_argument("--context-chars", type=int, default=0,
                        help="cap the printed user prompt to N chars (0 = full)")
    parser.add_argument("--json", action="store_true",
                        help="emit one structured JSON record instead of text")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the output (text or JSON) to this file")
    args = parser.parse_args()

    prepared = args.prepared or prepared_dir(args.dataset)
    cells = load_gold(prepared)

    results = args.results or results_dir(args.dataset)
    runs_path = results if results.suffix == ".jsonl" else results / "runs.jsonl"
    runs = load_runs(runs_path) if runs_path.exists() else []

    if args.list:
        list_cells(cells, runs)
        return

    if not args.cell:
        raise SystemExit("error: pass --cell <cell_id> (or --list to see them).")
    cell = cells.get(args.cell)
    if cell is None:
        raise SystemExit(
            f"error: cell {args.cell!r} not found in {prepared}. "
            f"Use --list to see the {len(cells)} available cell ids.")

    run_index = None if args.all_runs else args.run_index
    run_rows = select_runs(runs, args.cell, args.model, run_index)
    with_context = not args.no_context

    if args.json:
        record = build_record(cell, run_rows, args.context_chars, with_context)
        text = json.dumps(record, ensure_ascii=False, indent=2)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
            print(f"Wrote structured record -> {args.out}")
        else:
            print(text)
        return

    # Text view. Tee to a file if --out is given, so console + file match.
    sinks = [sys.stdout]
    fh = None
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fh = args.out.open("w", encoding="utf-8")
        sinks.append(fh)
    try:
        orig_stdout = sys.stdout
        sys.stdout = _Tee(sinks)
        print_meta(cell)
        if not args.no_prompt:
            print_prompt(cell, args.context_chars, with_context)
        print_gold(cell)
        print_answers(cell, run_rows)
    finally:
        sys.stdout = orig_stdout
        if fh:
            fh.close()
            print(f"\nWrote inspection -> {args.out}")


class _Tee:
    """Write to several streams at once (console + --out file)."""

    def __init__(self, streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


if __name__ == "__main__":
    main()
