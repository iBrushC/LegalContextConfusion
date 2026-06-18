"""build_context.py — generate context-degradation cells from raw CUAD.

Given the raw CUAD dataset, this builds the grid of degraded-context "cells"
that the models are run against. CUAD only for now (MAUD comes later).

  Init:
    1. Sample N contracts as the TARGETS (default 10, deterministic by seed).
    2. For each, fix a capped clause-category question set (<=32 by default,
       including >=1 negative category), chosen deterministically per document.
    3. Record gold spans + character offsets.
    Each target + its question set are held FIXED; only the surrounding filler
    (distractor documents) changes across that document's cells.

  Modalities:
    * rot              -> wrap the target in NON-legal filler (length-only).
    * confusion        -> wrap it in OTHER legal contracts (confusable).
    * missing_answer   -> confusable filler, scoring focused on the target's
                          native negatives (clause absent -> "not present").
    * missing_document -> ask about a document NOT in the window. NOT YET
                          implemented for CUAD (every CUAD question is tied to
                          its own contract); reserved for the MAUD work.

  Document wrappers:
    Every document is wrapped as <document id="..."> ... </document>. The
    target's id is recorded as `target_document_id`; the filler ids as
    `distractor_ids`, so the eval can tell whether an answer was extracted from
    the right document (the confusion signal).

  Interference (length only):
    * budgets    how much rot/confusion FILLER to add AROUND the probe, in
                 tokens (64k, 128k, ...). This is the independent variable
                 (the amount of interference), NOT a cap on the window: the
                 probe document is ALWAYS kept whole and the filler is added on
                 top of it. A 64k budget over a 16k probe -> a ~80k-token window.
    * placement  the target always sits at the END of the window, after all the
                 filler. (The old start/middle/end depth sweep has been removed.)

  Baseline:
    For the rot and confusion modalities a zero-filler BASELINE cell is also
    emitted (`<modality>_baseline`): the bare probe with no interference, the
    anchor every degradation curve is measured against. The abstention
    modalities (missing_answer/_document) have no interference axis -> no baseline.

Output: one JSON record per cell (per document: (modality x budget) +
per-modality baselines) under --out, plus a manifest.json describing the fixed
targets. Those records feed run_models.

Filler sources:
    * confusion / missing_answer filler is drawn from the OTHER contracts in the
      same CUAD file (confusable, available today).
    * rot filler is drawn from the NON-legal story corpus in data/rot — random
      sections of randomly chosen .txt files (Project Gutenberg eBooks across
      genres, boilerplate stripped), so the filler reads as varied prose, never
      confusable with a contract. Override the corpus with --rot-filler; if no
      .txt stories are found there it falls back to a synthetic sentence bank.

Usage:
    python src/build_context.py                                  # rot, 10 docs, default grid
    python src/build_context.py --modality rot confusion missing_answer
    python src/build_context.py --num-documents 4 --budgets 64000 128000
    python src/build_context.py --doc-indices 19 23 --seed 42
    python src/build_context.py --dry-run                        # plan only, no write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

# Reuse the loader / helpers from the sibling inspect script. When this file is
# run as `python src/build_context.py`, its own directory is on sys.path; make
# that explicit so the import also works under other invocations.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_cuad import category_of, is_negative, load_cuad  # noqa: E402

DEFAULT_FILE = Path("data/cuad/test.json")
DEFAULT_OUT = Path("data/prepared")
CHARS_PER_TOKEN = 4  # approximate; matches inspect_cuad's budgeting hint

# All modalities the schema knows about; only IMPLEMENTED ones generate cells.
MODALITIES = ("rot", "confusion", "missing_answer", "missing_document")
IMPLEMENTED_MODALITIES = ("rot", "confusion", "missing_answer")
# Modalities with an interference axis, so a zero-filler baseline makes sense.
BASELINE_MODALITIES = ("rot", "confusion")
DEFAULT_MODALITIES = ("rot",)
DEFAULT_BUDGETS = (64_000, 128_000, 256_000, 512_000)
DEFAULT_NUM_DOCUMENTS = 10
DEFAULT_MAX_QUESTIONS = 32

DOC_SEPARATOR = "\n\n"

# Default corpus of NON-legal story files (Project Gutenberg eBooks across
# genres) that rot filler is drawn from — random sections of random files, so
# the filler reads as varied prose rather than a repeated sentence bank.
ROT_DIR = Path("data/rot")
# Each rot filler "document" is a contiguous section of this many characters,
# drawn at a random offset from a randomly chosen story and snapped to nearby
# paragraph/word boundaries. A range (not a fixed size) keeps the blocks varied.
ROT_SECTION_MIN_CHARS = 1500
ROT_SECTION_MAX_CHARS = 4000
# Smallest pool we ever build, so even tiny budgets get some variety.
MIN_ROT_SECTIONS = 24
# When a snapped section starts/ends mid-paragraph, scan at most this far for a
# clean paragraph (then word) boundary before giving up and cutting as-is.
_SNAP_WINDOW = 400

# Synthetic fallback bank: neutral, deliberately NON-legal sentences used only
# when the rot corpus directory is missing/empty. No contract / merger
# vocabulary, so nothing here is confusable with CUAD.
_NONLEGAL_SENTENCES = (
    "The morning fog settled over the valley long before the hikers reached the ridge.",
    "She measured the flour twice before folding it into the warm batter.",
    "Migrating geese trace the same river south every autumn without a map.",
    "The old telescope needed a gentle cleaning before the comet became visible.",
    "He repotted the basil on the windowsill where the afternoon light was strongest.",
    "A quiet tide pulled the small boats back toward the harbor at dusk.",
    "The bakery on the corner sells out of sourdough by nine most mornings.",
    "Rain tapped against the greenhouse glass while the seedlings stretched upward.",
    "They followed the coastal trail until the lighthouse came into view.",
    "The chess club met in the library every Thursday after the last bell.",
    "Warm bread, sharp cheese, and a handful of olives made the whole picnic.",
    "The river otters played near the dam where the current slowed to a drift.",
    "A single violin carried the melody before the rest of the strings joined in.",
    "The cartographer sketched the coastline by hand, correcting it as she sailed.",
    "Frost outlined every leaf in the garden until the sun climbed past the fence.",
    "The train slowed through the mountain pass so passengers could watch the falls.",
    "He brewed the coffee a little stronger on the cold, dark winter mornings.",
    "Fireflies blinked across the meadow as the campfire settled into embers.",
    "The potter centered the clay, then opened it slowly with both thumbs.",
    "A kestrel hovered above the field, perfectly still against the moving clouds.",
)


def stable_seed(*parts) -> int:
    """Deterministic 64-bit seed from the parts (unlike Python's salted hash)."""
    key = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


# --------------------------------------------------------------------------- #
# Tokenizing (approximate by default; optional tiktoken)
# --------------------------------------------------------------------------- #
def make_token_counter(kind: str):
    """Return a function text -> token count.

    'approx' uses chars / CHARS_PER_TOKEN (no dependency). 'tiktoken' uses the
    cl100k encoding as a consistent reference if the package is installed.
    """
    if kind == "approx":
        return lambda text: math.ceil(len(text) / CHARS_PER_TOKEN)
    if kind == "tiktoken":
        try:
            import tiktoken
        except ImportError:
            raise SystemExit(
                "error: --tokenizer tiktoken requires `pip install tiktoken`; "
                "use --tokenizer approx to run without it."
            )
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text))
    raise SystemExit(f"error: unknown tokenizer {kind!r}")


# --------------------------------------------------------------------------- #
# Init: sample the fixed target + question set
# --------------------------------------------------------------------------- #
def _qualifies(doc: dict, min_negatives: int) -> bool:
    """True if `doc` has a usable paragraph with >= min_negatives negatives."""
    paragraphs = doc.get("paragraphs", [])
    if not paragraphs:
        return False
    qas = paragraphs[0].get("qas", [])
    return sum(1 for qa in qas if is_negative(qa)) >= min_negatives


def select_targets(data: list[dict], seed: int, doc_indices: list[int] | None,
                   num_documents: int, max_questions: int | None, balance: bool,
                   min_negatives: int) -> list[dict]:
    """Pick N contracts as targets and build each one's fixed question set.

    Documents are chosen deterministically from `seed`. Explicit `doc_indices`
    override the random pick (and are used verbatim). For the random path, a
    sampled doc that lacks >= min_negatives negatives is skipped and another is
    drawn from the remaining pool, so a short corpus never silently drops below
    the requested count; if fewer than N qualify we warn and proceed with what
    does.
    """
    n = len(data)
    if doc_indices is not None:
        indices = []
        for di in doc_indices:
            if not 0 <= di < n:
                raise SystemExit(
                    f"error: --doc-indices entry {di} out of range (0..{n - 1})")
            indices.append(di)
    else:
        rng = random.Random(stable_seed(seed, "select_docs"))
        order = list(range(n))
        rng.shuffle(order)
        indices, skipped = [], []
        for di in order:
            if len(indices) >= num_documents:
                break
            if _qualifies(data[di], min_negatives):
                indices.append(di)
            else:
                skipped.append(di)
        indices.sort()
        if skipped:
            print(f"note: skipped {len(skipped)} contract(s) lacking "
                  f">= {min_negatives} negative categories during selection.")
        if len(indices) < num_documents:
            print(f"warning: only {len(indices)} of {num_documents} requested "
                  f"documents qualify (>= {min_negatives} negatives); "
                  f"proceeding with {len(indices)}.")

    return [_build_target(data, di, seed, max_questions, balance, min_negatives)
            for di in indices]


def _build_target(data: list[dict], doc_index: int, seed: int,
                  max_questions: int | None, balance: bool,
                  min_negatives: int) -> dict:
    """Build the fixed, capped question set for one contract index."""
    doc = data[doc_index]
    paragraphs = doc.get("paragraphs", [])
    if not paragraphs:
        raise SystemExit(f"error: contract index {doc_index} has no paragraphs")
    if len(paragraphs) > 1:
        # CUAD contracts are single-paragraph; more would break answer offsets.
        print(f"warning: contract {doc_index} has {len(paragraphs)} paragraphs; "
              f"using the first only (gold offsets are paragraph-relative).")
    para = paragraphs[0]
    context = para.get("context", "")
    qas = para.get("qas", [])

    positives = [qa for qa in qas if not is_negative(qa)]
    negatives = [qa for qa in qas if is_negative(qa)]
    if len(negatives) < min_negatives:
        raise SystemExit(
            f"error: contract {doc_index} has {len(negatives)} negative "
            f"categories, need >= {min_negatives}. Try another --doc-indices."
        )

    doc_id = doc.get("title") or f"contract_{doc_index}"
    # Each document draws its question subset from its OWN rng, keyed by id, so
    # documents don't share a draw and each subset is reproducible from --seed.
    rng = random.Random(stable_seed(seed, "choose_questions", doc_id))
    chosen = _choose_questions(rng, positives, negatives, max_questions,
                               balance, min_negatives)

    # qa_id is an OPAQUE, stable per-question id (q01, q02, ...). It deliberately
    # does NOT embed the contract title the way CUAD's native id does, so showing
    # it in the prompt cannot leak which document is the target. The original
    # CUAD id is kept as `source_id` for traceability only (never shown to models).
    questions = [{
        "qa_id": f"q{i:02d}",
        "source_id": qa.get("id", ""),
        "category": category_of(qa),
        "question": qa.get("question", ""),
        "is_impossible": is_negative(qa),
        "answers": qa.get("answers", []),
    } for i, qa in enumerate(chosen, 1)]

    return {
        "doc_index": doc_index,
        "id": doc_id,
        "title": doc.get("title", ""),
        "context": context,
        "context_chars": len(context),
        "questions": questions,
        "num_positives": sum(1 for q in questions if not q["is_impossible"]),
        "num_negatives": sum(1 for q in questions if q["is_impossible"]),
    }


def _choose_questions(rng, positives, negatives, max_questions, balance,
                      min_negatives):
    """Select the question subset, guaranteeing >= min_negatives negatives."""
    if max_questions is None:
        return positives + negatives  # full fixed set
    if balance:
        half = max_questions // 2
        return (rng.sample(positives, min(half, len(positives)))
                + rng.sample(negatives, min(max_questions - half, len(negatives))))
    chosen = rng.sample(positives + negatives,
                        min(max_questions, len(positives) + len(negatives)))
    have_neg = sum(1 for qa in chosen if is_negative(qa))
    # Swap positives out for negatives until the minimum is met.
    for qa in negatives:
        if have_neg >= min_negatives:
            break
        if qa not in chosen:
            for i, c in enumerate(chosen):
                if not is_negative(c):
                    chosen[i] = qa
                    have_neg += 1
                    break
    return chosen


# --------------------------------------------------------------------------- #
# Distractor pools
# --------------------------------------------------------------------------- #
def legal_distractors(data: list[dict], target_index: int) -> list[tuple[str, str]]:
    """(id, text) for every OTHER contract — confusable filler."""
    pool = []
    for i, doc in enumerate(data):
        if i == target_index:
            continue
        for para in doc.get("paragraphs", []):
            ctx = para.get("context", "")
            if ctx:
                pool.append((doc.get("title") or f"contract_{i}", ctx))
    return pool


# Project Gutenberg wraps each story body in license boilerplate marked off by
# "*** START OF ... ***" / "*** END OF ... ***" lines. We sample only the body.
_GUTENBERG_START = re.compile(r"\*\*\*\s*START OF.*?\*\*\*", re.IGNORECASE)
_GUTENBERG_END = re.compile(r"\*\*\*\s*END OF.*?\*\*\*", re.IGNORECASE)


def strip_gutenberg(text: str) -> str:
    """Drop Project Gutenberg header/footer boilerplate, keeping the story body.

    Files without the markers are returned trimmed but otherwise unchanged.
    """
    m = _GUTENBERG_START.search(text)
    if m:
        text = text[m.end():]
    m = _GUTENBERG_END.search(text)
    if m:
        text = text[:m.start()]
    return text.strip()


def load_rot_corpus(rot_dir: Path) -> list[tuple[str, str]]:
    """(stem, story_text) for every .txt under rot_dir, boilerplate stripped."""
    if rot_dir.is_dir():
        files = sorted(rot_dir.rglob("*.txt"))
    elif rot_dir.is_file():
        files = [rot_dir]
    else:
        return []
    corpus: list[tuple[str, str]] = []
    for p in files:
        story = strip_gutenberg(p.read_text(encoding="utf-8", errors="ignore"))
        if story:
            corpus.append((p.stem, story))
    return corpus


def _snap_section(text: str, start: int, end: int) -> str:
    """Return text[start:end] nudged to nearby paragraph (else word) boundaries.

    Avoids starting/ending a filler block mid-word or mid-sentence: scans up to
    _SNAP_WINDOW chars for a paragraph break, falling back to a space.
    """
    n = len(text)
    if start > 0:
        para = text.find("\n\n", start, min(n, start + _SNAP_WINDOW))
        if para != -1:
            start = para + 2
        else:
            sp = text.find(" ", start, min(n, start + _SNAP_WINDOW))
            if sp != -1:
                start = sp + 1
    if end < n:
        lo = max(start, end - _SNAP_WINDOW)
        para = text.rfind("\n\n", lo, end)
        if para > start:
            end = para
        else:
            sp = text.rfind(" ", lo, end)
            if sp > start:
                end = sp
    return text[start:end].strip()


def sample_rot_sections(corpus: list[tuple[str, str]], n_sections: int,
                        rng: random.Random,
                        min_chars: int = ROT_SECTION_MIN_CHARS,
                        max_chars: int = ROT_SECTION_MAX_CHARS) -> list[tuple[str, str]]:
    """(id, text) random-length sections drawn from random files in `corpus`.

    Each section picks a random story, a random length in [min_chars, max_chars],
    and a random offset, then snaps to boundaries. ids embed the source stem plus
    an index so every block id is unique within a context.
    """
    sections: list[tuple[str, str]] = []
    for i in range(n_sections):
        stem, text = rng.choice(corpus)
        length = rng.randint(min_chars, max_chars)
        if len(text) <= length:
            seg = text
        else:
            start = rng.randint(0, len(text) - length)
            seg = _snap_section(text, start, start + length)
        if seg:
            sections.append((f"{stem}-{i:04d}", seg))
    return sections


def rot_pool_size(max_filler_chars: int, min_chars: int = ROT_SECTION_MIN_CHARS) -> int:
    """How many sections to pre-build so the largest budget never repeats one."""
    needed = math.ceil(max(0, max_filler_chars) / min_chars) + 8
    return max(MIN_ROT_SECTIONS, needed)


def nonlegal_distractors(rot_dir: Path, max_filler_chars: int,
                         seed: int) -> tuple[list[tuple[str | None, str]], str]:
    """(id, text) non-legal filler sections + a label describing their source.

    Draws random sections of random story files from `rot_dir`. Falls back to the
    synthetic sentence bank (id None -> generic filler id at assembly) only when
    the corpus directory is missing or empty.
    """
    corpus = load_rot_corpus(rot_dir)
    if not corpus:
        bank = list(_NONLEGAL_SENTENCES)
        paras = [" ".join(bank[i:i + 4]) for i in range(0, len(bank), 4)]
        return [(None, p) for p in paras], "synthetic-placeholder"
    rng = random.Random(stable_seed(seed, "rot_sections"))
    pool: list[tuple[str | None, str]] = [
        (sid, text)
        for sid, text in sample_rot_sections(corpus, rot_pool_size(max_filler_chars), rng)
    ]
    return pool, f"rot-sections:{rot_dir}"


# --------------------------------------------------------------------------- #
# Document wrapping + assembly
# --------------------------------------------------------------------------- #
def _open_tag(doc_id: str) -> str:
    return f'<document id="{doc_id}">\n'


_CLOSE_TAG = "\n</document>"


def wrap_document(doc_id: str, text: str) -> str:
    return _open_tag(doc_id) + text + _CLOSE_TAG


def build_cell_context(target_id: str, target_text: str,
                       pool: list[tuple[str | None, str]], filler_chars: int,
                       sep: str, rng: random.Random) -> dict:
    """Wrap the FULL target plus `filler_chars` of distractor filler before it.

    The target document is ALWAYS included verbatim and in full, and is placed
    at the END of the window, after all the filler; `filler_chars` is the amount
    of rot/confusion text added BEFORE it (the experiment's independent
    variable), NOT a cap on the total window. filler_chars <= 0 yields the bare
    target (the zero-interference baseline).

    Returns a dict with the assembled context, the char offset where the raw
    target text begins (so gold offsets map as offset + answer_start), the
    distractor ids used, and whether filler had to be repeated to reach the
    requested amount.
    """
    target_block = wrap_document(target_id, target_text)
    target_inner_offset = len(_open_tag(target_id))  # raw text start within block

    blocks: list[str] = []        # distractor blocks in fill order
    used_ids: list[str] = []
    repeated = False

    if pool and filler_chars > 0:
        order = pool[:]
        rng.shuffle(order)
        overhead_close = len(_CLOSE_TAG)
        added = 0                 # filler chars accumulated (target NOT counted)
        i = 0
        while True:
            budget_left = filler_chars - added
            if budget_left <= 0:
                break
            raw_id, text = order[i % len(order)]
            repeated = repeated or i >= len(order)
            did = raw_id if raw_id is not None else f"filler_{i + 1:03d}"
            block = wrap_document(did, text)
            if len(block) + len(sep) > budget_left:  # trim inner text so it fits
                keep = budget_left - len(sep) - len(_open_tag(did)) - overhead_close
                if keep <= 0:
                    break
                block = wrap_document(did, text[:keep])
            blocks.append(block)
            used_ids.append(did)
            added += len(block) + len(sep)
            i += 1

    # The target always sits at the END of the window, after all the filler.
    # With no filler (baseline) it simply stands alone.
    ordered = blocks + [target_block]
    chars_before = sum(len(b) for b in blocks) + len(sep) * len(blocks)
    context = sep.join(ordered)

    return {
        "context": context,
        "target_offset": chars_before + target_inner_offset,
        "distractor_ids": used_ids,
        "filler_repeated": repeated,
    }


# --------------------------------------------------------------------------- #
# Cell generation
# --------------------------------------------------------------------------- #
def cell_focus(modality: str) -> str:
    """Primary scoring focus per modality (recorded for the eval step)."""
    return "abstention" if modality in ("missing_answer", "missing_document") else "spans"


def make_cell(target: dict, modality: str, budget: int,
              count_tokens, legal_pool: list, nonlegal_pool: list,
              nonlegal_label: str, seed: int, sep: str,
              baseline: bool = False) -> dict:
    """Build one prepared-context record.

    `budget` is the amount of rot/confusion FILLER (in tokens) to add before the
    full target — the independent variable, not a cap on the window. The target
    is always kept whole and placed at the end. baseline=True forces zero filler
    (the bare-target reference point) and a `<modality>_baseline` cell id. The
    cell id carries a `d{doc_index:03d}` prefix so cells stay unique across the
    several target documents.
    """
    # Filler order is keyed by the target id so each document shuffles its own
    # pool independently yet reproducibly from --seed.
    rng = random.Random(stable_seed(seed, target["id"], modality, budget))
    filler_chars = 0 if baseline else budget * CHARS_PER_TOKEN

    if modality == "rot":
        pool, source = nonlegal_pool, nonlegal_label
    else:  # confusion + missing_answer use confusable legal filler
        pool, source = legal_pool, "cuad-other-contracts"

    built = build_cell_context(target["id"], target["context"], pool,
                               filler_chars, sep, rng)
    context = built["context"]
    target_offset = built["target_offset"]

    prefix = f"d{target['doc_index']:03d}"
    cell_id = (f"{prefix}_{modality}_baseline" if baseline
               else f"{prefix}_{modality}_b{budget}")
    return {
        "cell_id": cell_id,
        "doc_index": target["doc_index"],
        "modality": modality,
        "focus": cell_focus(modality),
        "budget_tokens": 0 if baseline else budget,  # rot/confusion filler tokens
        "is_baseline": baseline,
        "actual_tokens": count_tokens(context),
        "actual_chars": len(context),
        "target_document_id": target["id"],
        "distractor_ids": built["distractor_ids"],
        "target_offset": target_offset,
        "target_end": target_offset + len(target["context"]),
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
                        help=f"CUAD JSON to sample from (default: {DEFAULT_FILE})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--modality", nargs="+", choices=MODALITIES,
                        default=list(DEFAULT_MODALITIES),
                        help="modalities to generate (default: rot)")
    parser.add_argument("--budgets", nargs="+", type=int,
                        default=list(DEFAULT_BUDGETS),
                        help="rot/confusion filler tokens to add AROUND the full "
                             "probe (default: 64,000 128,000 256,000, 512,000); the zero-filler "
                             "case is emitted separately as the baseline cell")
    parser.add_argument("--num-documents", type=int, default=DEFAULT_NUM_DOCUMENTS,
                        help=f"number of target contracts to sample "
                             f"(default: {DEFAULT_NUM_DOCUMENTS})")
    parser.add_argument("--doc-indices", type=int, nargs="+", default=None,
                        help="explicit contract indices to target (overrides the "
                             "random, seed-based selection)")
    parser.add_argument("--doc-index", type=int, default=None,
                        help="deprecated alias for a single --doc-indices entry")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for sampling + filler shuffles (default: 0)")
    parser.add_argument("--max-questions", type=int, default=DEFAULT_MAX_QUESTIONS,
                        help=f"cap questions per document "
                             f"(default: {DEFAULT_MAX_QUESTIONS}; 0 = all)")
    parser.add_argument("--balance", action="store_true",
                        help="with --max-questions, pick ~equal pos/neg")
    parser.add_argument("--min-negatives", type=int, default=1,
                        help="minimum negative categories in the set (default: 1)")
    parser.add_argument("--rot-filler", type=Path, default=ROT_DIR,
                        help=f"dir/file of .txt non-legal story filler; rot draws "
                             f"random sections of random files (default: {ROT_DIR})")
    parser.add_argument("--tokenizer", choices=("approx", "tiktoken"),
                        default="approx", help="token counter (default: approx)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan without writing files")
    args = parser.parse_args()

    raw = load_cuad(args.file)
    data = raw.get("data", [])
    if not data:
        raise SystemExit(f"error: no 'data' entries found in {args.file}")

    # missing_document is reserved for later (needs a document absent from the
    # window — every CUAD question is tied to its own contract).
    modalities = []
    for m in args.modality:
        if m == "missing_document":
            print("note: 'missing_document' is not implemented for CUAD yet "
                  "(reserved for MAUD); skipping.")
            continue
        modalities.append(m)
    if not modalities:
        raise SystemExit("error: no implemented modalities requested.")

    count_tokens = make_token_counter(args.tokenizer)

    # --doc-index is a deprecated alias for a single --doc-indices entry.
    doc_indices = args.doc_indices
    if args.doc_index is not None:
        print("note: --doc-index is deprecated; use --doc-indices.")
        if doc_indices is None:
            doc_indices = [args.doc_index]
        elif args.doc_index not in doc_indices:
            doc_indices = [args.doc_index] + list(doc_indices)
    # --max-questions 0 means "no cap" (keep the whole set).
    max_questions = None if args.max_questions in (0, None) else args.max_questions

    targets = select_targets(data, args.seed, doc_indices, args.num_documents,
                             max_questions, args.balance, args.min_negatives)
    if not targets:
        raise SystemExit("error: no qualifying target documents selected.")

    print(f"Selected {len(targets)} target document(s):")
    for target in targets:
        print(f"  [{target['doc_index']:>3}] {target['id']}  "
              f"~{count_tokens(target['context']):,} tok, "
              f"{len(target['questions'])} q "
              f"({target['num_positives']} pos, {target['num_negatives']} neg)")

    # The zero-filler case is the baseline cell, not a budget; drop any <= 0.
    budgets = sorted(b for b in args.budgets if b > 0)
    dropped = [b for b in args.budgets if b <= 0]
    if dropped:
        print(f"note: ignoring non-positive budget(s) {dropped}; the zero-filler "
              f"case is emitted as the baseline cell instead.")

    # The rot section pool is non-legal noise (never confusable with a contract),
    # so it is built ONCE — sized to the largest budget — and shared across docs.
    max_filler_chars = (max(budgets) if budgets else 0) * CHARS_PER_TOKEN
    nonlegal_pool, nonlegal_label = nonlegal_distractors(
        args.rot_filler, max_filler_chars, args.seed)
    if "rot" in modalities:
        if nonlegal_label == "synthetic-placeholder":
            print(f"  rot filler: synthetic placeholder "
                  f"(no .txt stories found at {args.rot_filler})")
        else:
            print(f"  rot filler: {len(nonlegal_pool)} random story sections "
                  f"from {args.rot_filler}")

    cells_meta = []
    cells_dir = args.out / "cells"
    if not args.dry_run:
        cells_dir.mkdir(parents=True, exist_ok=True)

    # Per document: a zero-filler baseline per interference modality, then the
    # budget interference grid (target always at the END). The legal distractor
    # pool excludes the current target, so it is rebuilt per document.
    print(f"\nGenerating cells for {len(targets)} document(s):")
    for target in targets:
        legal_pool = legal_distractors(data, target["doc_index"])
        jobs = []  # (modality, budget, baseline)
        for modality in modalities:
            if modality in BASELINE_MODALITIES:
                jobs.append((modality, 0, True))
            for budget in budgets:
                jobs.append((modality, budget, False))

        print(f"  d{target['doc_index']:03d} {target['id']}: {len(jobs)} cells")
        for modality, budget, baseline in jobs:
            cell = make_cell(target, modality, budget, count_tokens,
                             legal_pool, nonlegal_pool, nonlegal_label,
                             args.seed, DOC_SEPARATOR, baseline=baseline)
            flags = []
            if cell["is_baseline"]:
                flags.append("baseline / zero-filler")
            if cell["filler_repeated"]:
                flags.append("filler-repeated")
            flag_str = ("  [" + ", ".join(flags) + "]") if flags else ""
            print(f"    {cell['cell_id']:32} "
                  f"{cell['actual_tokens']:>9,} tok  "
                  f"{len(cell['distractor_ids']):>3} distractors{flag_str}")

            meta = {k: v for k, v in cell.items()
                    if k not in ("context", "questions")}
            if not args.dry_run:
                path = cells_dir / f"{cell['cell_id']}.json"
                path.write_text(json.dumps(cell, ensure_ascii=False, indent=2),
                                encoding="utf-8")
                meta["path"] = str(path)
            cells_meta.append(meta)

    manifest = {
        "source_file": str(args.file),
        "dataset": "cuad",
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "chars_per_token": CHARS_PER_TOKEN,
        "doc_separator": DOC_SEPARATOR,
        "num_documents": len(targets),
        "doc_indices": [t["doc_index"] for t in targets],
        "targets": [{k: v for k, v in t.items() if k != "context"}
                    for t in targets],
        "modalities": modalities,
        "budgets": budgets,
        "baseline_modalities": [m for m in modalities if m in BASELINE_MODALITIES],
        "cells": cells_meta,
    }
    if args.dry_run:
        print(f"\n(dry run — {len(cells_meta)} cells planned, no files written)")
        return

    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"\nWrote {len(cells_meta)} cells + manifest to {args.out}/")


if __name__ == "__main__":
    main()
