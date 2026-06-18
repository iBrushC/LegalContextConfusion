"""build_maud_contexts.py — generate context-degradation cells from raw MAUD.

The MAUD counterpart to build_context.py. It compiles MAUD merger agreements into
the SAME prepared-cell format CUAD uses, so run_models.py / score_outputs.py can
consume both without changes. The CUAD builder is untouched; the few genuinely
generic helpers (token counter, deterministic seed, non-legal filler bank) are
imported from it, and the MAUD loaders come from inspect_maud.

  Init:
    1. Pick ONE merger agreement as the TARGET (full text from data/maud/contracts).
    2. Build its fixed question set from the MAUD CSV (one record per question,
       gold MULTIPLE-CHOICE answer + label). Prefer including >=1 safe negative.
    3. Target + question set are held FIXED; only the surrounding filler changes.

  Modalities (the same ones CUAD supports, where MAUD allows):
    * clean           -> target agreement only (no filler).
    * rot             -> target + NON-legal filler (length-only noise).
    * confusion       -> target + OTHER merger agreements (highly confusable).
    * missing_answer  -> target + other agreements, the safe-negative
                         condition: emitted ONLY when the chosen question set
                         contains a safe negative ("No"/"None"/"N/A" answer).
                         MAUD is forced-choice multiple choice, so the negative
                         is just one of the options (not a SQuAD is_impossible);
                         this modality stresses whether the model still picks
                         that option correctly amid distractors. Skipped with a
                         note when no safe negative is present.

  Documents are wrapped as <DOCUMENT id="..." title="..."> ... </DOCUMENT>. The
  target id/title are recorded so the eval can tell whether an answer came from
  the right agreement (the wrong-document signal).

  MAUD agreements are large (~85k tokens median), so at small budgets every
  document — including the target — is TRUNCATED to its budget share. Each cell
  records `target_fits` / `target_truncated` so that is visible downstream.

Output: one JSON record per cell under data/prepared_maud/cells/, plus
manifest.json. Each question carries MAUD-native fields (answer_type,
answer_options, gold_answers, gold_labels). On these `answer_type=multiple_choice`
cells (focus="labels"), run_models.py asks the multiple-choice prompt (the
lettered option set) and score_outputs.py scores by option match (mc_accuracy);
a SQuAD-shaped `answers` field is also kept so the CUAD-shaped helpers stay
happy. is_impossible is always False here — MAUD has no impossible questions.

Usage:
    python src/build_maud_contexts.py                                   # default grid
    python src/build_maud_contexts.py --budgets 4000 16000 --positions target_at_start target_at_end --max-questions 10 --seed 42
    python src/build_maud_contexts.py --modality clean confusion
    python src/build_maud_contexts.py --contract contract_13 --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# src/ is on sys.path when run directly; make it explicit for other invocations.
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Reuse the truly generic CUAD helpers (no CUAD data is touched at import time).
from build_context import (  # noqa: E402
    CHARS_PER_TOKEN, DOC_SEPARATOR, ROT_DIR, _NONLEGAL_SENTENCES,
    load_rot_corpus, make_token_counter, rot_pool_size, sample_rot_sections,
    stable_seed,
)
from inspect_maud import (  # noqa: E402
    answer_options_map, build_questions, derive_title, group_by_contract,
    load_maud_rows, read_contract_text,
)

DEFAULT_FILE = Path("data/maud/MAUD_dev.csv")
DEFAULT_CONTRACTS = Path("data/maud/contracts")
DEFAULT_OUT = Path("data/prepared_maud")
DEFAULT_DATA_TYPE = "main"

MODALITIES = ("clean", "rot", "confusion", "missing_answer")
DEFAULT_MODALITIES = MODALITIES
# clean/rot/confusion always work; missing_answer needs a safe negative present.
DEFAULT_BUDGETS = (4000, 16000, 64000)
POSITIONS = ("target_at_start", "target_at_middle", "target_at_end")
DEFAULT_POSITIONS = POSITIONS

# Smallest slice of the target we will ever keep when distractors share the budget.
MIN_TARGET_CHARS = 1000
# Fraction of the budget reserved for the target when distractors are present.
DEFAULT_TARGET_SHARE = 0.5
# Approx size of one synthetic non-legal "document" for the rot fallback bank
# (used only when the data/rot story corpus is missing/empty).
ROT_DOC_CHARS = 2000

FOCUS = "labels"  # MAUD cells score label/answer match, per the task spec.


# --------------------------------------------------------------------------- #
# Document wrapping (uppercase <DOCUMENT id title> tags, per the MAUD spec)
# --------------------------------------------------------------------------- #
def _open_tag(doc_id: str, title: str) -> str:
    safe = (title or "").replace('"', "'").replace("\n", " ").strip()
    return f'<DOCUMENT id="{doc_id}" title="{safe}">\n'


_CLOSE_TAG = "\n</DOCUMENT>"


def wrap_document(doc_id: str, title: str, text: str) -> str:
    return _open_tag(doc_id, title) + text + _CLOSE_TAG


def fit_block(doc_id: str, title: str, text: str, max_chars: int):
    """Wrap text so the whole <DOCUMENT> block fits in max_chars.

    Returns (block, truncated) or (None, False) if even the wrapper won't fit.
    """
    overhead = len(_open_tag(doc_id, title)) + len(_CLOSE_TAG)
    keep = max_chars - overhead
    if keep <= 0:
        return None, False
    truncated = len(text) > keep
    return wrap_document(doc_id, title, text[:keep]), truncated


# --------------------------------------------------------------------------- #
# Distractor pools
# --------------------------------------------------------------------------- #
def make_legal_pool(by_contract: dict, contracts_dir: Path, target_id: str):
    """Ordered list of other contract ids + a lazy (title, text) loader.

    Other merger agreements are the confusion / missing_answer filler. Texts are
    read on demand (they are large; a budget rarely needs more than one or two).
    """
    ids = [n for n in sorted(by_contract) if n != target_id]
    cache: dict[str, tuple[str, str]] = {}

    def load(doc_id: str) -> tuple[str, str]:
        if doc_id not in cache:
            text = read_contract_text(contracts_dir, doc_id)
            cache[doc_id] = (derive_title(text, doc_id), text)
        return cache[doc_id]

    return ids, load


def make_rot_pool(rot_dir: Path, max_filler_chars: int, seed: int):
    """Non-legal story excerpts (ids + loader + source label) for rot filler.

    Each "document" is a random-length section taken at a random offset from a
    randomly chosen story file under `rot_dir` (Project Gutenberg eBooks across
    genres, boilerplate stripped), so rot filler reads as varied prose rather
    than a repeated sentence bank. Nothing here is confusable with a merger
    agreement — it is length-only noise. The pool is sized to the largest budget
    so no section repeats. Falls back to the synthetic sentence bank only when
    the corpus directory is missing/empty.
    """
    corpus = load_rot_corpus(rot_dir)
    if not corpus:
        joined = " ".join(_NONLEGAL_SENTENCES)
        n_docs = 12
        big = (joined + " ") * (1 + (ROT_DOC_CHARS * n_docs) // max(len(joined), 1))
        docs = [big[i:i + ROT_DOC_CHARS]
                for i in range(0, ROT_DOC_CHARS * n_docs, ROT_DOC_CHARS)]
        table = {f"filler_{i + 1:03d}": d for i, d in enumerate(docs)}

        def load_synthetic(doc_id: str) -> tuple[str, str]:
            return "non-legal filler", table[doc_id]

        return list(table), load_synthetic, "synthetic-nonlegal"

    rng = random.Random(stable_seed(seed, "rot_sections"))
    sections = dict(sample_rot_sections(corpus, rot_pool_size(max_filler_chars), rng))

    def load(doc_id: str) -> tuple[str, str]:
        return "non-legal story excerpt", sections[doc_id]

    return list(sections), load, f"rot-sections:{rot_dir}"


# --------------------------------------------------------------------------- #
# Context assembly
# --------------------------------------------------------------------------- #
def build_cell_context(target: dict, pool_ids: list, load_pool, budget_chars: int,
                       position: str, target_share: float, sep: str, rng,
                       with_distractors: bool) -> dict:
    """Assemble one context: target + (optional) distractors at `position`."""
    if with_distractors and pool_ids:
        target_alloc = min(budget_chars, max(MIN_TARGET_CHARS,
                                              int(budget_chars * target_share)))
    else:
        target_alloc = budget_chars

    target_block, target_truncated = fit_block(
        target["id"], target["title"], target["text"], target_alloc)
    if target_block is None:  # budget smaller than the wrapper itself
        target_block = wrap_document(target["id"], target["title"], "")
        target_truncated = bool(target["text"])

    blocks, used_ids, filler_repeated = [], [], False
    if with_distractors and pool_ids:
        order = pool_ids[:]
        rng.shuffle(order)
        remaining = budget_chars - len(target_block)
        i = 0
        while remaining - len(sep) > 0:
            filler_repeated = filler_repeated or i >= len(order)
            doc_id = order[i % len(order)]
            title, text = load_pool(doc_id)
            block, _ = fit_block(doc_id, title, text, remaining - len(sep))
            if block is None:
                break
            blocks.append(block)
            used_ids.append(doc_id)
            remaining -= len(block) + len(sep)
            i += 1
            # rot filler is finite+repeatable; legal filler usually fills in one
            # block, so stop once we've cycled the whole pool without progress.
            if i >= len(order) and len(order) == 1:
                break

    if position == "target_at_start":
        idx = 0
    elif position == "target_at_end":
        idx = len(blocks)
    elif position in ("target_at_middle", "full"):
        idx = len(blocks) // 2
    else:
        raise SystemExit(f"error: unknown position {position!r}")

    ordered = blocks[:idx] + [target_block] + blocks[idx:]
    return {
        "context": sep.join(ordered),
        "distractor_ids": used_ids,
        "target_truncated": target_truncated,
        "target_fits": not target_truncated,
        "filler_repeated": filler_repeated,
    }


# --------------------------------------------------------------------------- #
# Target + question set
# --------------------------------------------------------------------------- #
def enrich_questions(raw_questions: list[dict]) -> list[dict]:
    """Attach opaque qa_ids + a SQuAD-shaped mirror for the shared run pipeline.

    MAUD is closed-set MULTIPLE CHOICE: every question is answerable and is
    scored by option match (score_outputs.tally_cell keys on answer_type), so
    `is_impossible` is always False — there are no SQuAD-style impossible
    questions here. A safe-negative answer ("No"/"None"/"N/A") is simply one of
    the options; `is_negative` is kept as an informative flag, and its presence
    is what lets the missing_answer modality build (see select_target).
    """
    enriched = []
    for i, q in enumerate(raw_questions, 1):
        # `answers` mirrors the gold option into the CUAD-shaped field; for MC
        # scoring `answer_options` + `gold_answers` are what actually matter.
        answers = [{"text": a} for a in q["gold_answers"]] or [{"text": ""}]
        enriched.append({
            "qa_id": f"q{i:02d}",
            "category": q["category"],
            "question": q["question"],
            "answer_type": "multiple_choice",
            "gold_answers": q["gold_answers"],
            "gold_labels": q["gold_labels"],
            "is_impossible": False,
            "is_negative": bool(q["is_negative"]),
            # extras (additive): real-run aids + CUAD-pipeline compatibility.
            "answer_options": q.get("answer_options", q["gold_answers"]),
            "subquestions": q["subquestions"],
            "source_question": q["question"],
            "answers": answers,
        })
    return enriched


def choose_questions(rng, questions: list[dict], max_questions: int | None,
                     min_negatives: int) -> list[dict]:
    """Subsample the question set, preferring to keep >= min_negatives negatives."""
    if max_questions is None or max_questions >= len(questions):
        return list(questions)
    chosen = rng.sample(questions, max_questions)
    negs = [q for q in questions if q["is_negative"]]
    have = sum(1 for q in chosen if q["is_negative"])
    want = min(min_negatives, len(negs))
    for nq in negs:
        if have >= want:
            break
        if nq not in chosen:
            for i, c in enumerate(chosen):
                if not c["is_negative"]:
                    chosen[i] = nq
                    have += 1
                    break
    return chosen


def select_target(by_contract: dict, all_rows: list[dict], contracts_dir: Path,
                  seed: int, contract: str | None, doc_index: int | None,
                  max_questions: int | None, min_negatives: int,
                  negatives_mode: str) -> dict:
    """Pick the target agreement and build its fixed, enriched question set."""
    names = sorted(by_contract)
    if contract is not None:
        if contract not in by_contract:
            raise SystemExit(f"error: contract {contract!r} not found. "
                             f"Try one of {names[:3]} …")
        name = contract
    elif doc_index is not None:
        if not 0 <= doc_index < len(names):
            raise SystemExit(f"error: --doc-index {doc_index} out of range "
                             f"(0..{len(names) - 1})")
        name = names[doc_index]
    else:
        rng = random.Random(stable_seed(seed, "select_target"))
        name = rng.choice(names)

    text = read_contract_text(contracts_dir, name)
    title = derive_title(text, name)

    raw_questions = build_questions(by_contract[name])
    # Attach the dataset-wide option set for each question (real-run aid).
    options = answer_options_map(all_rows)
    for q in raw_questions:
        q["answer_options"] = options.get(q["question"], q["gold_answers"])

    sel_rng = random.Random(stable_seed(seed, "choose_questions", name))
    chosen = choose_questions(sel_rng, raw_questions, max_questions, min_negatives)
    questions = enrich_questions(chosen)

    # Safe negatives gate the missing_answer modality. They are normal MC
    # options now (not impossible), so count them via is_negative; --negatives
    # none suppresses the gate (and thus the missing_answer modality).
    num_negatives = (sum(1 for q in questions if q["is_negative"])
                     if negatives_mode == "auto" else 0)

    return {
        "id": name,
        "doc_index": names.index(name),
        "title": title,
        "text": text,
        "text_chars": len(text),
        "questions": questions,
        "num_questions": len(questions),
        "num_negatives": num_negatives,
    }


# --------------------------------------------------------------------------- #
# Cell generation
# --------------------------------------------------------------------------- #
def make_cell(target: dict, modality: str, budget: int, position: str,
              count_tokens, legal_pool, rot_pool, target_share: float,
              seed: int, sep: str) -> dict:
    """Build one prepared MAUD cell."""
    rng = random.Random(stable_seed(seed, modality, budget, position))
    budget_chars = budget * CHARS_PER_TOKEN

    if modality == "clean":
        pool_ids, load_pool, source, with_distractors = [], None, "none", False
    elif modality == "rot":
        pool_ids, load_pool, source = rot_pool
        with_distractors = True
    else:  # confusion + missing_answer share the other-agreements pool
        pool_ids, load_pool = legal_pool
        source, with_distractors = "maud-other-agreements", True

    built = build_cell_context(target, pool_ids, load_pool, budget_chars,
                               position, target_share, sep, rng, with_distractors)
    context = built["context"]
    cell_id = (f"{modality}_b{budget}" if modality == "clean"
               else f"{modality}_b{budget}_{position}")

    return {
        "cell_id": cell_id,
        "dataset": "maud",
        "modality": modality,
        "focus": FOCUS,
        "budget_tokens": budget,
        "position": position,
        "actual_tokens": count_tokens(context),
        "actual_chars": len(context),
        "target_document_id": target["id"],
        "target_document_title": target["title"],
        "distractor_ids": built["distractor_ids"],
        "target_fits": built["target_fits"],
        "target_truncated": built["target_truncated"],
        "filler_source": source,
        "filler_repeated": built["filler_repeated"],
        "context": context,
        "questions": target["questions"],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE,
                        help=f"MAUD CSV to sample from (default: {DEFAULT_FILE})")
    parser.add_argument("--contracts-dir", type=Path, default=DEFAULT_CONTRACTS,
                        help=f"full agreement texts (default: {DEFAULT_CONTRACTS})")
    parser.add_argument("--rot-dir", type=Path, default=ROT_DIR,
                        help=f"dir/file of .txt non-legal story filler; rot draws "
                             f"random sections of random files (default: {ROT_DIR})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--data-type", default=DEFAULT_DATA_TYPE,
                        help="MAUD row split to use (default: main)")
    parser.add_argument("--modality", nargs="+", choices=MODALITIES,
                        default=list(DEFAULT_MODALITIES),
                        help="modalities to generate (default: all four)")
    parser.add_argument("--budgets", nargs="+", type=int,
                        default=list(DEFAULT_BUDGETS),
                        help="token budgets (default: 4000 16000 64000)")
    parser.add_argument("--positions", nargs="+", choices=POSITIONS,
                        default=list(DEFAULT_POSITIONS),
                        help="target positions (default: all three)")
    parser.add_argument("--contract", default=None,
                        help="use this contract_name as the target")
    parser.add_argument("--doc-index", type=int, default=None,
                        help="target the Nth contract (sorted); default: seeded random")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for sampling + filler shuffles (default: 0)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="subsample the question set (default: all)")
    parser.add_argument("--min-negatives", type=int, default=1,
                        help="prefer keeping >= this many safe negatives (default: 1)")
    parser.add_argument("--target-share", type=float, default=DEFAULT_TARGET_SHARE,
                        help="fraction of budget reserved for the target when "
                             "distractors are present (default: 0.5)")
    parser.add_argument("--negatives", choices=("auto", "none"), default="auto",
                        help="auto: map No/None/N/A answers to is_impossible "
                             "(enables missing_answer + abstention). none: treat "
                             "every question as answerable.")
    parser.add_argument("--tokenizer", choices=("approx", "tiktoken"),
                        default="approx", help="token counter (default: approx)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan without writing files")
    args = parser.parse_args()

    rows = load_maud_rows(args.file, args.data_type or None)
    by_contract = group_by_contract(rows)
    if not by_contract:
        raise SystemExit(f"error: no contracts found in {args.file}")

    count_tokens = make_token_counter(args.tokenizer)
    target = select_target(by_contract, rows, args.contracts_dir, args.seed,
                           args.contract, args.doc_index, args.max_questions,
                           args.min_negatives, args.negatives)

    print(f"Target: [{target['doc_index']}] {target['id']} — {target['title']}")
    print(f"  text: {target['text_chars']:,} chars "
          f"(~{count_tokens(target['text']):,} tokens)")
    print(f"  questions: {target['num_questions']} "
          f"({target['num_negatives']} safe-negative)")

    # missing_answer needs a safe negative in the chosen set; otherwise skip it.
    modalities = list(args.modality)
    if "missing_answer" in modalities and target["num_negatives"] == 0:
        note = ("MAUD is forced-choice; no safe negative in this question set"
                if args.negatives == "auto" else "--negatives none disables it")
        print(f"note: skipping 'missing_answer' ({note}).")
        modalities = [m for m in modalities if m != "missing_answer"]
    if not modalities:
        raise SystemExit("error: no modalities to generate.")

    legal_pool = make_legal_pool(by_contract, args.contracts_dir, target["id"])
    # Size the rot section pool to the largest budget so no section repeats.
    max_filler_chars = (max(args.budgets) if args.budgets else 0) * CHARS_PER_TOKEN
    rot_pool = make_rot_pool(args.rot_dir, max_filler_chars, args.seed)
    if "rot" in modalities:
        label = rot_pool[2]
        if label == "synthetic-nonlegal":
            print(f"  rot filler: synthetic placeholder "
                  f"(no .txt stories found at {args.rot_dir})")
        else:
            print(f"  rot filler: {len(rot_pool[0])} random story sections "
                  f"from {args.rot_dir}")

    cells_meta = []
    cells_dir = args.out / "cells"
    if not args.dry_run:
        cells_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating cells ({', '.join(modalities)}):")
    for modality in modalities:
        for budget in sorted(args.budgets):
            # clean is position-invariant (no filler) -> one cell per budget.
            positions = ["full"] if modality == "clean" else args.positions
            for position in positions:
                cell = make_cell(target, modality, budget, position, count_tokens,
                                 legal_pool, rot_pool, args.target_share,
                                 args.seed, DOC_SEPARATOR)
                flags = []
                if cell["target_truncated"]:
                    flags.append("target-truncated")
                if cell["filler_repeated"]:
                    flags.append("filler-repeated")
                flag_str = ("  [" + ", ".join(flags) + "]") if flags else ""
                print(f"  {cell['cell_id']:40} {cell['actual_tokens']:>9,} tok  "
                      f"{len(cell['distractor_ids']):>2} distractors{flag_str}")

                meta = {k: v for k, v in cell.items()
                        if k not in ("context", "questions")}
                if not args.dry_run:
                    path = cells_dir / f"{cell['cell_id']}.json"
                    path.write_text(json.dumps(cell, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
                    meta["path"] = str(path)
                cells_meta.append(meta)

    if args.dry_run:
        print(f"\n(dry run — {len(cells_meta)} cells planned, no files written)")
        return

    manifest = {
        "source_file": str(args.file),
        "contracts_dir": str(args.contracts_dir),
        "dataset": "maud",
        "data_type": args.data_type,
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "chars_per_token": CHARS_PER_TOKEN,
        "doc_separator": DOC_SEPARATOR,
        "negatives_mode": args.negatives,
        "target_share": args.target_share,
        "target": {k: v for k, v in target.items() if k != "text"},
        "modalities": modalities,
        "budgets": sorted(args.budgets),
        "positions": args.positions,
        "cells": cells_meta,
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"\nWrote {len(cells_meta)} cells + manifest to {args.out}/")


if __name__ == "__main__":
    main()
