"""Inspect the CUAD dataset.

CUAD (Contract Understanding Atticus Dataset) ships in SQuAD-v2 / extractive-QA
form:

    data[]                          one contract
      title                         contract name
      paragraphs[].context          the full contract text
      paragraphs[].qas[]            one question per clause category
          question                  natural-language clause prompt
          id                        "<title>__<Clause Category>"
          is_impossible             True  -> clause absent (a NEGATIVE example)
          answers[]                 [{text, answer_start}]  gold span(s)

This script derisks that data before any sampling/eval is built. It reports the
schema, the positive/negative split, a per-category breakdown, span-length and
context-length distributions, and can dump one sampled contract with its fixed
question set (negatives flagged, since those drive the hallucination tests).

Usage:
    python src/inspect_cuad.py                          # summary of test.json
    python src/inspect_cuad.py --file data/cuad/CUADv1.json
    python src/inspect_cuad.py --categories             # per-category table
    python src/inspect_cuad.py --sample                 # dump one sampled contract
    python src/inspect_cuad.py --sample --seed 42       # reproducible pick
    python src/inspect_cuad.py --sample --doc-index 5   # pick a specific contract
    python src/inspect_cuad.py --all                    # every section
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import textwrap
from collections import Counter
from pathlib import Path

DEFAULT_FILE = Path("data/cuad/test.json")

# Rough chars-per-token ratio for English prose; good enough for a budget hint,
# not for real tokenization (that belongs in prepare_cuad.py).
CHARS_PER_TOKEN = 4


# --------------------------------------------------------------------------- #
# Loading / iteration
# --------------------------------------------------------------------------- #
def load_cuad(path: Path) -> dict:
    """Load a CUAD JSON file, with a friendly error if it is missing."""
    if not path.exists():
        raise SystemExit(
            f"error: {path} not found.\n"
            f"       Pass --file with a path to a CUAD JSON "
            f"(e.g. data/cuad/test.json)."
        )
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def iter_qas(data: list[dict]):
    """Yield (doc, paragraph, qa) for every question in the dataset."""
    for doc in data:
        for para in doc.get("paragraphs", []):
            for qa in para.get("qas", []):
                yield doc, para, qa


def category_of(qa: dict) -> str:
    """Clause category for a question, parsed from its id ('<title>__<cat>')."""
    qid = qa.get("id", "")
    if "__" in qid:
        return qid.rsplit("__", 1)[-1].strip()
    # Fallback: pull the quoted category out of the question text.
    question = qa.get("question", "")
    if 'related to "' in question:
        return question.split('related to "', 1)[1].split('"', 1)[0].strip()
    return "(unknown)"


def is_negative(qa: dict) -> bool:
    """True when the clause is genuinely absent (a native negative example)."""
    return bool(qa.get("is_impossible")) or not qa.get("answers")


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
def _header(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _stat_line(label: str, values: list[int]) -> None:
    """Print min / median / mean / max for a list of numbers."""
    if not values:
        print(f"  {label}: (none)")
        return
    mean = statistics.mean(values)
    median = statistics.median(values)
    print(
        f"  {label}: min={min(values):,}  median={median:,.0f}  "
        f"mean={mean:,.0f}  max={max(values):,}"
    )


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def report_summary(path: Path, raw: dict, data: list[dict]) -> None:
    """Top-level schema + positive/negative split + size distributions."""
    qas = list(iter_qas(data))
    total_q = len(qas)
    negatives = sum(1 for _, _, qa in qas if is_negative(qa))
    positives = total_q - negatives

    categories = {category_of(qa) for _, _, qa in qas}
    answers_per_pos = [
        len(qa["answers"]) for _, _, qa in qas if not is_negative(qa)
    ]
    span_lengths = [
        len(ans.get("text", ""))
        for _, _, qa in qas
        if not is_negative(qa)
        for ans in qa.get("answers", [])
    ]
    context_chars = [
        len(para.get("context", ""))
        for doc in data
        for para in doc.get("paragraphs", [])
    ]

    _header(f"CUAD summary — {path}")
    print(f"  version:          {raw.get('version', '(none)')}")
    print(f"  contracts:        {len(data):,}")
    print(f"  questions:        {total_q:,}")
    print(
        f"  answerable (pos): {positives:,} "
        f"({_pct(positives, total_q)})"
    )
    print(
        f"  absent (neg):     {negatives:,} "
        f"({_pct(negatives, total_q)})  <- feed hallucination tests"
    )
    print(f"  clause categories:{len(categories):>4}")

    _header("Answer spans (positives only)")
    _stat_line("answers / question", answers_per_pos)
    _stat_line("span length (chars)", span_lengths)

    _header("Context size per contract")
    _stat_line("context (chars)", context_chars)
    if context_chars:
        approx_tokens = [c // CHARS_PER_TOKEN for c in context_chars]
        _stat_line(f"context (~tokens, chars/{CHARS_PER_TOKEN})", approx_tokens)


def report_categories(data: list[dict]) -> None:
    """Per-category positive/negative counts, sorted by negative rate."""
    pos: Counter = Counter()
    neg: Counter = Counter()
    for _, _, qa in iter_qas(data):
        cat = category_of(qa)
        if is_negative(qa):
            neg[cat] += 1
        else:
            pos[cat] += 1

    cats = sorted(set(pos) | set(neg), key=lambda c: -_safe_div(neg[c], pos[c] + neg[c]))

    _header(f"Per-category breakdown ({len(cats)} categories)")
    print(f"  {'category':38} {'pos':>6} {'neg':>6} {'neg%':>6}")
    for cat in cats:
        p, n = pos[cat], neg[cat]
        print(f"  {cat[:38]:38} {p:>6} {n:>6} {_pct(n, p + n):>6}")


def report_sample(data: list[dict], seed: int, doc_index: int | None,
                  full_answers: bool) -> None:
    """Dump one contract: its context size and full fixed question set."""
    if doc_index is None:
        rng = random.Random(seed)
        doc_index = rng.randrange(len(data))
    if not 0 <= doc_index < len(data):
        raise SystemExit(
            f"error: --doc-index {doc_index} out of range (0..{len(data) - 1})"
        )

    doc = data[doc_index]
    qas = [qa for para in doc.get("paragraphs", []) for qa in para.get("qas", [])]
    context = "".join(
        para.get("context", "") for para in doc.get("paragraphs", [])
    )
    negatives = sum(1 for qa in qas if is_negative(qa))

    _header(f"Sampled contract  [index {doc_index}]")
    print(f"  title:    {doc.get('title', '(untitled)')}")
    print(f"  context:  {len(context):,} chars (~{len(context)//CHARS_PER_TOKEN:,} tokens)")
    print(f"  questions:{len(qas):>4}  ({len(qas) - negatives} answerable, "
          f"{negatives} absent)")

    _header("Question set (gold answers)")
    for qa in qas:
        cat = category_of(qa)
        if is_negative(qa):
            print(f"  [NEG] {cat}")
            continue
        answers = qa.get("answers", [])
        first = answers[0].get("text", "") if answers else ""
        extra = f"  (+{len(answers) - 1} more span(s))" if len(answers) > 1 else ""
        if full_answers:
            shown = " | ".join(a.get("text", "") for a in answers)
        else:
            shown = textwrap.shorten(first, width=90, placeholder=" …")
        print(f"  [POS] {cat}{extra}")
        print(f"        @ {qa.get('answers', [{}])[0].get('answer_start', '?')}: {shown}")


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #
def _safe_div(a: int, b: int) -> float:
    return a / b if b else 0.0


def _pct(part: int, whole: int) -> str:
    return f"{100 * _safe_div(part, whole):.0f}%"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE,
                        help=f"CUAD JSON to inspect (default: {DEFAULT_FILE})")
    parser.add_argument("--categories", action="store_true",
                        help="show per-category positive/negative table")
    parser.add_argument("--sample", action="store_true",
                        help="dump one sampled contract + its question set")
    parser.add_argument("--doc-index", type=int, default=None,
                        help="sample this contract index instead of a random one")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for reproducible sampling (default: 0)")
    parser.add_argument("--full-answers", action="store_true",
                        help="print full gold spans instead of truncating")
    parser.add_argument("--all", action="store_true",
                        help="show every section (summary + categories + sample)")
    args = parser.parse_args()

    raw = load_cuad(args.file)
    data = raw.get("data", [])
    if not data:
        raise SystemExit(f"error: no 'data' entries found in {args.file}")

    report_summary(args.file, raw, data)
    if args.categories or args.all:
        report_categories(data)
    if args.sample or args.all:
        report_sample(data, args.seed, args.doc_index, args.full_answers)


if __name__ == "__main__":
    main()
