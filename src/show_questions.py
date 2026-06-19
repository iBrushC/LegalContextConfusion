"""show_questions.py — dump every unique MAUD question as the model is shown it.

A review aid: render each DISTINCT MAUD question in the EXACT prompt format the
model sees (qa_id | [CHOOSE ONE]/[SELECT ALL THAT APPLY] question + lettered
options), so the formatting can be eyeballed for mistakes in one place. The render
is shared with run_models (`mc_question_block`), so this preview never drifts from
the live prompt: no category is shown, and the same single-answer removal that
build_context applies is applied here too.

Source is the RAW MAUD CSV (not the sampled cells), so it covers every question in
the split regardless of which contracts were drawn as targets. Questions whose
dataset-wide option set has fewer than two choices (almost always a Y/N whose only
observed answer is "Yes") are DROPPED, exactly as build_context drops them; the
dropped ones are listed at the end so the removal itself can be sanity-checked.

Alongside the human-readable .txt preview this also emits a JSON sidecar (one
record per KEPT question: qa_id, category, the answer_type, the full option set,
and a single real clause EXCERPT from the dataset plus that excerpt's gold
answer). That JSON is the feedstock for rewriting MAUD's terse option-list
"questions" into natural-language prompts — the example clause grounds the rewrite
in what the text actually looks like.

Usage:
    python src/show_questions.py                          # -> data/maud/unique_questions.{txt,json}
    python src/show_questions.py --file data/maud/MAUD_test.csv
    python src/show_questions.py --data-type main         # which split of rows
    python src/show_questions.py --out -                  # print txt to stdout (json still written)
    python src/show_questions.py --json -                 # print json to stdout instead
    python src/show_questions.py --json ''                # skip the JSON sidecar
    python src/show_questions.py --example-chars 0        # keep full (untruncated) excerpts
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

# Sibling modules (src/ is on sys.path when run directly; make it explicit).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_maud import (  # noqa: E402
    DEFAULT_DATA_TYPE, DEFAULT_FILE, answer_options_map, load_maud_rows,
    multiselect_options_map,
)
from run_models import mc_question_block  # noqa: E402

DEFAULT_OUT = Path("data/maud/unique_questions.txt")
DEFAULT_JSON_OUT = Path("data/maud/unique_questions.json")
MIN_OPTIONS = 2  # a question needs >= 2 choices to be a real forced choice
# Clause excerpts run from a sentence to (rarely) hundreds of KB; cap the one we
# embed so the JSON stays a usable rewrite prompt. 0 (via --example-chars) keeps
# the full excerpt.
DEFAULT_EXAMPLE_CHARS = 1200


def example_text_map(rows: list[dict]) -> dict[str, dict]:
    """question -> one representative {text, answer} drawn from the dataset.

    The MAUD `text` column is the relevant clause EXCERPT for that (contract,
    question) row — exactly the kind of language a natural rewrite of the question
    should sound like. We pick the FIRST row (file order) whose `text` is non-empty
    so the choice is deterministic, and keep that row's gold `answer` so the
    excerpt is paired with what it was labelled. Questions whose every row has an
    empty `text` are simply absent from the map.
    """
    chosen: dict[str, dict] = {}
    for r in rows:
        q = r.get("question", "")
        if not q or q in chosen:
            continue
        text = (r.get("text") or "").strip()
        if text:
            chosen[q] = {"text": text, "answer": (r.get("answer") or "").strip()}
    return chosen


def collect_unique_questions(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """(kept, dropped) unique-question records in first-appearance order.

    Each record carries the question text, its category (for the dropped-report
    only — never rendered), the multi/single-select flag, the option set (derived
    dataset-wide exactly as build_context derives them), and one example clause
    excerpt + its gold answer (see `example_text_map`). `kept` records also carry a
    sequential `qa_id` + `answer_type` so they render via mc_question_block;
    `dropped` are those with fewer than MIN_OPTIONS options.
    """
    options = answer_options_map(rows)
    ms_options = multiselect_options_map(rows)
    examples = example_text_map(rows)

    seen: set[str] = set()
    kept: list[dict] = []
    dropped: list[dict] = []
    for r in rows:
        q = r.get("question", "")
        if not q or q in seen:
            continue
        seen.add(q)
        multi = q in ms_options
        opts = ms_options[q] if multi else options.get(q, [])
        example = examples.get(q, {})
        rec = {
            "question": q,
            "category": r.get("category", ""),
            "answer_type": "multi_select" if multi else "multiple_choice",
            "answer_options": opts,
            "example_answer": example.get("answer", ""),
            "example_text": example.get("text", ""),
        }
        if len(opts) >= MIN_OPTIONS:
            rec["qa_id"] = f"q{len(kept) + 1:02d}"
            kept.append(rec)
        else:
            dropped.append(rec)
    return kept, dropped


def render(kept: list[dict], dropped: list[dict], source: Path,
           data_type: str) -> str:
    """Assemble the full review text: header, every kept block, dropped list."""
    out: list[str] = []
    out.append(f"# Unique MAUD questions as shown to the model")
    out.append(f"# source: {source}  (data_type={data_type})")
    out.append(f"# {len(kept)} questions shown, {len(dropped)} dropped "
               f"(< {MIN_OPTIONS} options)")
    out.append("")
    out.append("=" * 72)
    out.append("QUESTIONS (exact model-facing format; category not shown)")
    out.append("=" * 72)
    out.append("")
    for rec in kept:
        out.append(mc_question_block(rec))
        out.append("")

    out.append("=" * 72)
    out.append(f"DROPPED — {len(dropped)} question(s) with fewer than "
               f"{MIN_OPTIONS} options")
    out.append("=" * 72)
    if dropped:
        for rec in dropped:
            opts = rec["answer_options"] or ["(none)"]
            out.append(f'- [{rec["answer_type"]}] category="{rec["category"]}"')
            out.append(f"    {rec['question']}")
            out.append(f"    only option(s): {', '.join(opts)}")
    else:
        out.append("(none)")
    out.append("")
    return "\n".join(out)


def build_json(kept: list[dict], source: Path, data_type: str,
               example_chars: int) -> str:
    """Serialize the kept questions (with one example excerpt each) as JSON.

    Shape is a small wrapper {meta, questions:[...]}; each question keeps the
    model-facing fields plus its example clause + the example's gold answer, with
    the excerpt truncated to `example_chars` (0 = full). This is what downstream
    rewriting reads to turn each terse option list into a natural-language prompt.
    """
    questions = []
    for rec in kept:
        text = rec["example_text"]
        if example_chars and len(text) > example_chars:
            text = textwrap.shorten(text, width=example_chars, placeholder=" …")
        questions.append({
            "qa_id": rec["qa_id"],
            "category": rec["category"],
            "answer_type": rec["answer_type"],
            "question": rec["question"],
            "answer_options": rec["answer_options"],
            "example_answer": rec["example_answer"],
            "example_text": text,
        })
    payload = {
        "source": str(source),
        "data_type": data_type,
        "question_count": len(questions),
        "questions": questions,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def main() -> None:
    # Legal text carries smart quotes / en-dashes a legacy console code page
    # (Windows cp1252) cannot encode; print as UTF-8, replacing the unmappable.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE,
                        help=f"MAUD CSV to read (default: {DEFAULT_FILE})")
    parser.add_argument("--data-type", default=DEFAULT_DATA_TYPE,
                        help=f"row split to use (default: {DEFAULT_DATA_TYPE}; "
                             f"'' for all)")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"output text file, or '-' for stdout "
                             f"(default: {DEFAULT_OUT})")
    parser.add_argument("--json", default=str(DEFAULT_JSON_OUT),
                        help=f"JSON sidecar with one example excerpt per question, "
                             f"'-' for stdout, or '' to skip "
                             f"(default: {DEFAULT_JSON_OUT})")
    parser.add_argument("--example-chars", type=int, default=DEFAULT_EXAMPLE_CHARS,
                        help=f"truncate each example excerpt to this many chars "
                             f"(0 = keep full; default: {DEFAULT_EXAMPLE_CHARS})")
    args = parser.parse_args()

    rows = load_maud_rows(args.file, args.data_type or None)
    kept, dropped = collect_unique_questions(rows)
    data_type = args.data_type or "(all)"
    text = render(kept, dropped, args.file, data_type)

    if args.out == "-":
        print(text)
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote {len(kept)} unique questions ({len(dropped)} dropped) "
              f"-> {out_path}")

    if args.json == "":
        return
    payload = build_json(kept, args.file, data_type, args.example_chars)
    n_examples = sum(1 for r in kept if r["example_text"])
    if args.json == "-":
        print(payload)
        return
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(payload, encoding="utf-8")
    print(f"Wrote {len(kept)} questions ({n_examples} with example text) "
          f"-> {json_path}")


if __name__ == "__main__":
    main()
