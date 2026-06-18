"""build_maud_contexts.py — generate context-degradation cells from raw MAUD.

The MAUD counterpart to build_context.py. It compiles MAUD merger agreements into
the SAME prepared-cell format CUAD uses, so run_models.py / score_outputs.py can
consume both without changes. The CUAD builder is untouched; the few genuinely
generic helpers (token counter, deterministic seed, non-legal filler bank) are
imported from it, and the MAUD loaders come from inspect_maud.

  Init:
    1. Pick N merger agreements as the TARGETS (default 10, deterministic by
       seed; full text from data/maud/contracts).
    2. For each, build a capped fixed question set from the MAUD CSV (<=32 by
       default, one record per question, gold MULTIPLE-CHOICE answer + label),
       chosen deterministically per document. Prefer including >=1 safe negative.
    3. Each target + its question set are held FIXED; only the surrounding filler
       changes across that document's cells.

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

  Interference (length only), matching CUAD:
    Budgets are the amount of rot/confusion FILLER (in tokens) ADDED AROUND the
    target, NOT a cap on the window. The target agreement is ALWAYS kept whole
    and placed at the END of the window, after all the filler; only filler
    blocks are trimmed to fit the budget. (Earlier versions truncated the target
    to a budget share — that cutoff behaviour has been removed.)

Output: one JSON record per cell under data/prepared_maud/cells/, plus
manifest.json. Each question carries MAUD-native fields (answer_type,
answer_options, gold_answers, gold_labels). On these `answer_type=multiple_choice`
cells (focus="labels"), run_models.py asks the multiple-choice prompt (the
lettered option set) and score_outputs.py scores by option match (mc_accuracy);
a SQuAD-shaped `answers` field is also kept so the CUAD-shaped helpers stay
happy. is_impossible is always False here — MAUD has no impossible questions.

Usage:
    python src/build_maud_contexts.py                                   # 10 docs, default grid
    python src/build_maud_contexts.py --num-documents 4 --budgets 64000 128000 --seed 42
    python src/build_maud_contexts.py --modality clean confusion
    python src/build_maud_contexts.py --contracts contract_13 --dry-run
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
# Budgets mirror CUAD: filler tokens ADDED around the whole agreement.
DEFAULT_BUDGETS = (64_000, 128_000, 256_000, 512_000)
DEFAULT_NUM_DOCUMENTS = 10
DEFAULT_MAX_QUESTIONS = 32
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
def build_cell_context(target: dict, pool_ids: list, load_pool, filler_chars: int,
                       sep: str, rng, with_distractors: bool) -> dict:
    """Assemble one context: `filler_chars` of filler, then the WHOLE target.

    Mirrors CUAD: the target agreement is ALWAYS kept whole and placed at the
    END of the window, after all the filler. `filler_chars` is the amount of
    rot/confusion text ADDED before it (the independent variable), NOT a cap on
    the window. Only filler blocks are trimmed to fit the budget; the target is
    never truncated. filler_chars <= 0 (or no distractors) yields the bare
    target — the `clean` reference.
    """
    target_block = wrap_document(target["id"], target["title"], target["text"])

    blocks, used_ids, filler_repeated = [], [], False
    if with_distractors and pool_ids and filler_chars > 0:
        order = pool_ids[:]
        rng.shuffle(order)
        added = 0  # filler chars accumulated (target NOT counted)
        i = 0
        while True:
            budget_left = filler_chars - added
            if budget_left - len(sep) <= 0:
                break
            filler_repeated = filler_repeated or i >= len(order)
            doc_id = order[i % len(order)]
            title, text = load_pool(doc_id)
            block, _ = fit_block(doc_id, title, text, budget_left - len(sep))
            if block is None:
                break
            blocks.append(block)
            used_ids.append(doc_id)
            added += len(block) + len(sep)
            i += 1
            # rot filler is finite+repeatable; legal filler usually fills in one
            # block, so stop once we've cycled a single-doc pool without progress.
            if i >= len(order) and len(order) == 1:
                break

    ordered = blocks + [target_block]
    return {
        "context": sep.join(ordered),
        "distractor_ids": used_ids,
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
    is what lets the missing_answer modality build (see select_targets).
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


def select_targets(by_contract: dict, all_rows: list[dict], contracts_dir: Path,
                   seed: int, contracts: list[str] | None,
                   doc_indices: list[int] | None, num_documents: int,
                   max_questions: int | None, min_negatives: int,
                   negatives_mode: str) -> list[dict]:
    """Pick N agreements as targets and build each one's fixed question set.

    Documents are chosen deterministically from `seed`; explicit `contracts`
    (by name) or `doc_indices` override the random pick. Returns a list of
    target dicts. Unlike CUAD, MAUD selection has no min-negatives gate —
    missing_answer is decided per document at cell-generation time.
    """
    names = sorted(by_contract)
    if contracts:
        chosen_names = []
        for c in contracts:
            if c not in by_contract:
                raise SystemExit(f"error: contract {c!r} not found. "
                                 f"Try one of {names[:3]} …")
            chosen_names.append(c)
    elif doc_indices is not None:
        chosen_names = []
        for di in doc_indices:
            if not 0 <= di < len(names):
                raise SystemExit(f"error: --doc-indices entry {di} out of range "
                                 f"(0..{len(names) - 1})")
            chosen_names.append(names[di])
    else:
        rng = random.Random(stable_seed(seed, "select_docs"))
        order = list(range(len(names)))
        rng.shuffle(order)
        if num_documents > len(names):
            print(f"warning: only {len(names)} contracts available; "
                  f"proceeding with all of them.")
        indices = sorted(order[:min(num_documents, len(names))])
        chosen_names = [names[i] for i in indices]

    # The dataset-wide option set per question is identical across docs; build
    # it once and reuse it for every target.
    options = answer_options_map(all_rows)
    return [_build_target(by_contract, names, contracts_dir, seed, name, options,
                          max_questions, min_negatives, negatives_mode)
            for name in chosen_names]


def _build_target(by_contract: dict, names: list[str], contracts_dir: Path,
                  seed: int, name: str, options: dict,
                  max_questions: int | None, min_negatives: int,
                  negatives_mode: str) -> dict:
    """Build one target agreement's fixed, capped, enriched question set."""
    text = read_contract_text(contracts_dir, name)
    title = derive_title(text, name)

    raw_questions = build_questions(by_contract[name])
    for q in raw_questions:
        q["answer_options"] = options.get(q["question"], q["gold_answers"])

    # Each document draws its question subset from its OWN rng, keyed by name, so
    # documents don't share a draw and each subset is reproducible from --seed.
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
def make_cell(target: dict, modality: str, budget: int,
              count_tokens, legal_pool, rot_pool, seed: int, sep: str) -> dict:
    """Build one prepared MAUD cell.

    `budget` is the amount of rot/confusion FILLER (in tokens) added before the
    full target — the independent variable, not a cap on the window. The target
    is always kept whole and placed at the end. The cell id carries a
    `d{doc_index:03d}` prefix so cells stay unique across the target documents.
    """
    # Filler order is keyed by the target id so each document shuffles its own
    # pool independently yet reproducibly from --seed.
    rng = random.Random(stable_seed(seed, target["id"], modality, budget))

    if modality == "clean":
        pool_ids, load_pool, source, with_distractors = [], None, "none", False
        filler_chars = 0
    elif modality == "rot":
        pool_ids, load_pool, source = rot_pool
        with_distractors = True
        filler_chars = budget * CHARS_PER_TOKEN
    else:  # confusion + missing_answer share the other-agreements pool
        pool_ids, load_pool = legal_pool
        source, with_distractors = "maud-other-agreements", True
        filler_chars = budget * CHARS_PER_TOKEN

    built = build_cell_context(target, pool_ids, load_pool, filler_chars,
                               sep, rng, with_distractors)
    context = built["context"]
    cell_id = f"d{target['doc_index']:03d}_{modality}_b{budget}"

    return {
        "cell_id": cell_id,
        "dataset": "maud",
        "doc_index": target["doc_index"],
        "modality": modality,
        "focus": FOCUS,
        "budget_tokens": budget,
        "actual_tokens": count_tokens(context),
        "actual_chars": len(context),
        "target_document_id": target["id"],
        "target_document_title": target["title"],
        "distractor_ids": built["distractor_ids"],
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
                        help="rot/confusion filler tokens to add AROUND the whole "
                             "agreement (default: 64,000 128,000 256,000 512,000)")
    parser.add_argument("--num-documents", type=int, default=DEFAULT_NUM_DOCUMENTS,
                        help=f"number of target agreements to sample "
                             f"(default: {DEFAULT_NUM_DOCUMENTS})")
    parser.add_argument("--contracts", nargs="+", default=None,
                        help="explicit contract_name(s) to target "
                             "(overrides random selection)")
    parser.add_argument("--doc-indices", type=int, nargs="+", default=None,
                        help="explicit contract indices (sorted) to target "
                             "(overrides random selection)")
    parser.add_argument("--contract", default=None,
                        help="deprecated alias for a single --contracts entry")
    parser.add_argument("--doc-index", type=int, default=None,
                        help="deprecated alias for a single --doc-indices entry")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for sampling + filler shuffles (default: 0)")
    parser.add_argument("--max-questions", type=int, default=DEFAULT_MAX_QUESTIONS,
                        help=f"cap questions per document "
                             f"(default: {DEFAULT_MAX_QUESTIONS}; 0 = all)")
    parser.add_argument("--min-negatives", type=int, default=1,
                        help="prefer keeping >= this many safe negatives (default: 1)")
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

    # --contract / --doc-index are deprecated aliases for the plural flags.
    contracts = list(args.contracts) if args.contracts else None
    if args.contract is not None:
        print("note: --contract is deprecated; use --contracts.")
        contracts = (contracts or []) + [args.contract]
    doc_indices = args.doc_indices
    if args.doc_index is not None:
        print("note: --doc-index is deprecated; use --doc-indices.")
        if doc_indices is None:
            doc_indices = [args.doc_index]
        elif args.doc_index not in doc_indices:
            doc_indices = [args.doc_index] + list(doc_indices)
    # --max-questions 0 means "no cap" (keep the whole set).
    max_questions = None if args.max_questions in (0, None) else args.max_questions

    targets = select_targets(by_contract, rows, args.contracts_dir, args.seed,
                             contracts, doc_indices, args.num_documents,
                             max_questions, args.min_negatives, args.negatives)
    if not targets:
        raise SystemExit("error: no target documents selected.")

    print(f"Selected {len(targets)} target agreement(s):")
    for target in targets:
        print(f"  [{target['doc_index']:>3}] {target['id']} — {target['title']}  "
              f"~{count_tokens(target['text']):,} tok, "
              f"{target['num_questions']} q "
              f"({target['num_negatives']} safe-negative)")

    base_modalities = list(args.modality)
    # Size the rot section pool to the largest budget so no section repeats. The
    # rot pool is non-legal noise, so it is built ONCE and shared across docs.
    max_filler_chars = (max(args.budgets) if args.budgets else 0) * CHARS_PER_TOKEN
    rot_pool = make_rot_pool(args.rot_dir, max_filler_chars, args.seed)
    if "rot" in base_modalities:
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

    # Per document: missing_answer is decided per doc (needs a safe negative);
    # the legal distractor pool excludes the current target, so it is rebuilt
    # per document. The target always sits at the END of the window.
    print(f"\nGenerating cells for {len(targets)} document(s):")
    for target in targets:
        modalities = list(base_modalities)
        if "missing_answer" in modalities and target["num_negatives"] == 0:
            note = ("no safe negative in this question set"
                    if args.negatives == "auto" else "--negatives none disables it")
            print(f"  d{target['doc_index']:03d}: skipping 'missing_answer' ({note}).")
            modalities = [m for m in modalities if m != "missing_answer"]
        if not modalities:
            continue

        legal_pool = make_legal_pool(by_contract, args.contracts_dir, target["id"])
        print(f"  d{target['doc_index']:03d} {target['id']}: "
              f"{len(modalities)} modality(ies)")
        for modality in modalities:
            for budget in sorted(args.budgets):
                cell = make_cell(target, modality, budget, count_tokens,
                                 legal_pool, rot_pool, args.seed, DOC_SEPARATOR)
                flags = []
                if cell["filler_repeated"]:
                    flags.append("filler-repeated")
                flag_str = ("  [" + ", ".join(flags) + "]") if flags else ""
                print(f"    {cell['cell_id']:32} {cell['actual_tokens']:>9,} tok  "
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
        "num_documents": len(targets),
        "doc_indices": [t["doc_index"] for t in targets],
        "targets": [{k: v for k, v in t.items() if k != "text"}
                    for t in targets],
        "modalities": base_modalities,
        "budgets": sorted(args.budgets),
        "cells": cells_meta,
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"\nWrote {len(cells_meta)} cells + manifest to {args.out}/")


if __name__ == "__main__":
    main()
