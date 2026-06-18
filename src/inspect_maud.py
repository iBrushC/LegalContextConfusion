"""Inspect the MAUD dataset.

MAUD (Merger Agreement Understanding Dataset) is NOT in CUAD's SQuAD / extractive
form. It ships as CSV rows plus a directory of full merger-agreement texts:

    data/maud/MAUD_{dev,test,train}.csv   one row per (contract, question, label)
        data_type      main | abridged | rare_answers   (we use `main`)
        contract_name  "contract_13"  -> contracts/contract_13.txt
        text           the relevant clause EXCERPT (not the whole agreement)
        answer         the gold MULTIPLE-CHOICE answer, e.g. "All Cash"
        label          a numeric annotation index (NOT a stable global option id)
        question       the clause-type prompt, e.g. "Type of Consideration-Answer"
        subquestion    sub-aspect or "<NONE>"  (drives row duplication)
        category       one of 7 high-level groups
    data/maud/contracts/contract_N.txt    the FULL agreement text

Unlike CUAD, MAUD is a closed-set classification task: every question is
answerable and picks one option from a fixed per-question answer set. There are
no extractive spans and no SQuAD `is_impossible`. The closest thing to a native
negative is a "No" / "None" / "N/A" answer, which means the feature is ABSENT —
those are surfaced here as the safe negatives that feed the abstention tests.

This script derisks the data before any cell-building: it reports the schema, the
document/question counts, the label/answer format, the category breakdown, and
dumps one example agreement + question + gold answer.

Usage:
    python src/inspect_maud.py                          # summary of MAUD_dev.csv
    python src/inspect_maud.py --file data/maud/MAUD_test.csv
    python src/inspect_maud.py --data-type main         # which split of rows
    python src/inspect_maud.py --seed 42                # reproducible example pick
    python src/inspect_maud.py --contract contract_13   # example for a given doc
"""

from __future__ import annotations

import argparse
import csv
import random
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

# MAUD's `text` column holds whole clauses (hundreds of KB on some rows), well
# past csv's default field cap. 1e9 clears it without risking the OverflowError
# that sys.maxsize triggers on some 32-bit C longs.
csv.field_size_limit(10**9)

DEFAULT_FILE = Path("data/maud/MAUD_dev.csv")
DEFAULT_CONTRACTS = Path("data/maud/contracts")
DEFAULT_DATA_TYPE = "main"
CHARS_PER_TOKEN = 4  # same rough budgeting hint inspect_cuad uses

# A gold answer that normalizes to one of these means the feature is ABSENT.
# These are MAUD's only "safe negatives" and are what abstention scoring keys on.
NEGATIVE_MARKERS = {"none", "no", "n/a", "not applicable", "none of the above"}

# Header noise to skip when guessing a document title.
_TITLE_NOISE = ("exhibit", "execution version", "execution copy", "confidential",
                "page", "table of contents")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_maud_rows(path: Path, data_type: str | None = DEFAULT_DATA_TYPE) -> list[dict]:
    """Load MAUD CSV rows, optionally filtered to one data_type (e.g. 'main')."""
    if not path.exists():
        raise SystemExit(
            f"error: {path} not found.\n"
            f"       Pass --file with a MAUD CSV (e.g. data/maud/MAUD_dev.csv)."
        )
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"error: no rows found in {path}")
    if data_type:
        rows = [r for r in rows if r.get("data_type") == data_type]
        if not rows:
            kinds = sorted({r.get("data_type") for r in rows} or {"(none)"})
            raise SystemExit(
                f"error: no rows with data_type={data_type!r} in {path}. "
                f"Available: {kinds}."
            )
    return rows


def group_by_contract(rows: list[dict]) -> dict[str, list[dict]]:
    """contract_name -> its rows (skips the synthetic <RARE_ANSWERS> bucket)."""
    by_contract: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        name = r.get("contract_name", "")
        if name and not name.startswith("<"):
            by_contract[name].append(r)
    return by_contract


def answer_options_map(rows: list[dict]) -> dict[str, list[str]]:
    """question -> sorted distinct answers seen for it (the MC option set)."""
    opts: dict[str, set] = defaultdict(set)
    for r in rows:
        ans = (r.get("answer") or "").strip()
        if ans:
            opts[r.get("question", "")].add(ans)
    return {q: sorted(a) for q, a in opts.items()}


def is_negative_answer(answer: str) -> bool:
    """True when a gold answer means the feature is ABSENT (a safe negative)."""
    return (answer or "").strip().lower() in NEGATIVE_MARKERS


def build_questions(contract_rows: list[dict]) -> list[dict]:
    """Collapse a contract's rows into one record per question.

    MAUD `main` has exactly one distinct answer per (contract, question); the
    duplicate rows differ only in `subquestion`. We dedup to that single gold
    answer, keep the distinct labels/subquestions, and flag the answer as a
    negative (feature-absent) when it normalizes to a NEGATIVE_MARKER.
    """
    by_q: dict[str, dict] = {}
    for r in contract_rows:
        q = r.get("question", "")
        ans = (r.get("answer") or "").strip()
        slot = by_q.setdefault(q, {
            "question": q, "category": r.get("category", ""),
            "text_type": r.get("text_type", ""),
            "answers": set(), "labels": set(), "subquestions": set(),
        })
        if ans:
            slot["answers"].add(ans)
        if (lbl := (r.get("label") or "").strip()):
            slot["labels"].add(lbl)
        sub = (r.get("subquestion") or "").strip()
        if sub and sub != "<NONE>":
            slot["subquestions"].add(sub)

    questions = []
    for q, slot in by_q.items():
        golds = sorted(slot["answers"])
        negative = len(golds) == 1 and is_negative_answer(golds[0])
        questions.append({
            "question": q,
            "category": slot["category"],
            "text_type": slot["text_type"],
            "gold_answers": golds,
            "gold_labels": sorted(slot["labels"]),
            "subquestions": sorted(slot["subquestions"]),
            "is_negative": negative,
        })
    return questions


# --------------------------------------------------------------------------- #
# Contract text + title
# --------------------------------------------------------------------------- #
def read_contract_text(contracts_dir: Path, contract_name: str) -> str:
    """Full agreement text for a contract_name, BOM stripped."""
    path = contracts_dir / f"{contract_name}.txt"
    if not path.exists():
        raise SystemExit(f"error: contract text not found: {path}")
    return path.read_text(encoding="utf-8", errors="ignore").lstrip("﻿")


def derive_title(text: str, contract_name: str) -> str:
    """Best-effort agreement title from the header lines; falls back to the id."""
    lines = [ln.strip() for ln in text[:4000].splitlines() if ln.strip()]
    kept: list[str] = []
    for ln in lines:
        low = ln.lower()
        if low.startswith("table of contents"):
            break
        if any(low.startswith(n) for n in _TITLE_NOISE):
            continue
        # Stop once we hit numbered sections / recitals — header is over.
        if ln[:1].isdigit() and "." in ln[:6]:
            break
        kept.append(ln)
        if len(" ".join(kept)) > 140:
            break
    title = " ".join(kept).strip()
    return textwrap.shorten(title, width=140, placeholder=" …") if title else contract_name


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _header(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _pct(part: int, whole: int) -> str:
    return f"{(100 * part / whole) if whole else 0:.0f}%"


def report_summary(path: Path, data_type: str, rows: list[dict],
                   by_contract: dict[str, list[dict]]) -> None:
    total_rows = len(rows)
    questions = Counter(r.get("question", "") for r in rows)
    categories = Counter(r.get("category", "") for r in rows)
    options = answer_options_map(rows)
    distinct_answers = {a for opts in options.values() for a in opts}
    negatives = sum(1 for r in rows if is_negative_answer(r.get("answer", "")))

    # label sanity: is (question, answer) -> label a stable 1:1 map?
    qa_labels: dict[tuple, set] = defaultdict(set)
    for r in rows:
        qa_labels[(r.get("question"), r.get("answer"))].add(r.get("label"))
    inconsistent = sum(1 for v in qa_labels.values() if len(v) > 1)

    _header(f"MAUD summary — {path}  (data_type={data_type})")
    print(f"  documents (contracts): {len(by_contract):,}")
    print(f"  QA rows:               {total_rows:,}")
    print(f"  distinct questions:    {len(questions):,}")
    print(f"  distinct categories:   {len(categories):,}")
    print(f"  distinct answer values:{len(distinct_answers):>5,}")
    print(f"  safe negatives (No/None/N/A): {negatives:,} "
          f"({_pct(negatives, total_rows)} of rows)  <- feed abstention tests")

    _header("Label / answer format")
    print("  task:   closed-set MULTIPLE CHOICE (not extractive QA / not SQuAD)")
    print("  answer: gold option TEXT, e.g. " + ", ".join(
        repr(a) for a in list(distinct_answers)[:3]))
    print("  label:  numeric annotation index per row")
    print(f"          ({inconsistent} (question,answer) pairs map to >1 label — "
          "the label is NOT a stable option id; gold ANSWER text is authoritative)")
    qsizes = [len(o) for o in options.values()]
    if qsizes:
        print(f"  options/question: min={min(qsizes)} "
              f"max={max(qsizes)} mean={sum(qsizes) / len(qsizes):.1f}")

    _header("Categories")
    for cat, n in categories.most_common():
        nq = len({r.get("question") for r in rows if r.get("category") == cat})
        print(f"  {cat[:42]:42} {n:>6} rows  {nq:>3} questions")


def report_example(rows: list[dict], by_contract: dict[str, list[dict]],
                   contracts_dir: Path, seed: int, contract: str | None) -> None:
    options = answer_options_map(rows)
    names = sorted(by_contract)
    if contract is None:
        contract = random.Random(seed).choice(names)
    elif contract not in by_contract:
        raise SystemExit(f"error: contract {contract!r} not in this split. "
                         f"Try one of {names[:3]} …")

    text = read_contract_text(contracts_dir, contract)
    title = derive_title(text, contract)
    questions = build_questions(by_contract[contract])
    n_neg = sum(1 for q in questions if q["is_negative"])

    _header(f"Example agreement — {contract}")
    print(f"  title:      {title}")
    print(f"  text:       {len(text):,} chars (~{len(text) // CHARS_PER_TOKEN:,} tokens)")
    print(f"  questions:  {len(questions)}  ({n_neg} safe-negative answers)")
    print(f"  text head:  {textwrap.shorten(text[:600], width=180, placeholder=' …')}")

    _header("Example question + gold answer")
    # Prefer a negative example if one exists (it is the more interesting case).
    q = next((q for q in questions if q["is_negative"]), questions[0])
    opts = options.get(q["question"], q["gold_answers"])
    print(f"  category:     {q['category']}")
    print(f"  question:     {q['question']}")
    print(f"  answer_type:  multiple_choice ({len(opts)} options)")
    print(f"  options:      {', '.join(repr(o) for o in opts[:6])}"
          + (" …" if len(opts) > 6 else ""))
    print(f"  gold answer:  {', '.join(repr(a) for a in q['gold_answers'])}  "
          f"(label {', '.join(q['gold_labels']) or '?'})")
    print(f"  is_negative:  {q['is_negative']}"
          + ("  (feature absent -> correct behavior is to abstain)"
             if q["is_negative"] else ""))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE,
                        help=f"MAUD CSV to inspect (default: {DEFAULT_FILE})")
    parser.add_argument("--contracts-dir", type=Path, default=DEFAULT_CONTRACTS,
                        help=f"full agreement texts (default: {DEFAULT_CONTRACTS})")
    parser.add_argument("--data-type", default=DEFAULT_DATA_TYPE,
                        help="row split to use (default: main; '' for all)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the example pick (default: 0)")
    parser.add_argument("--contract", default=None,
                        help="show the example for this contract_name")
    args = parser.parse_args()

    rows = load_maud_rows(args.file, args.data_type or None)
    by_contract = group_by_contract(rows)
    if not by_contract:
        raise SystemExit(f"error: no contracts found in {args.file}")

    report_summary(args.file, args.data_type or "(all)", rows, by_contract)
    report_example(rows, by_contract, args.contracts_dir, args.seed, args.contract)


if __name__ == "__main__":
    main()
