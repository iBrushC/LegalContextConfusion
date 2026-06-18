"""build_context.py — generate context-degradation cells from raw CUAD and MAUD.

One builder for both datasets. Given the raw CUAD contracts and/or the raw MAUD
merger agreements, this emits the grid of degraded-context "cells" the models are
run against, in the SAME prepared-cell format so run_models.py / score_outputs.py
consume both without changes. `--dataset` selects which to build (default both —
they are almost always built together).

  Init (per dataset):
    1. Sample N documents as the TARGETS (default 10, deterministic by seed).
    2. For each, fix a capped question set (<=32 by default), chosen
       deterministically per document. CUAD draws SQuAD-style clause questions
       (with >=1 negative category); MAUD draws forced-choice multiple-choice
       questions (preferring >=1 safe negative).
    3. Each target + its question set are held FIXED; only the surrounding filler
       (distractor documents) changes across that document's cells.

  Modalities:
    * rot              -> wrap the target in NON-legal filler (length-only noise).
    * confusion        -> wrap it in OTHER legal documents (confusable).
    * missing_answer   -> confusable filler, scoring focused on the target's
                          negatives (CUAD: native is_impossible categories; MAUD:
                          the safe-negative MC option). MAUD emits it only when the
                          chosen question set contains a safe negative.
    * missing_document -> ask about a document NOT in the window. NOT YET
                          implemented for either dataset; requested -> skipped
                          with a note.

  Baseline:
    For rot and confusion a zero-filler BASELINE cell is also emitted
    (`<modality>_baseline`): the bare probe with no interference, the anchor every
    degradation curve is measured against. The abstention modalities have no
    interference axis -> no baseline.

  Interference (length only):
    Budgets are how much rot/confusion FILLER to add AROUND the probe, in tokens
    (64k, 128k, ...) — the independent variable, NOT a cap on the window. The
    probe document is ALWAYS kept whole and placed at the END of the window, after
    all the filler; only filler blocks are trimmed to fit the budget.

  Filler sources:
    * confusion / missing_answer filler is drawn from the OTHER documents in the
      same dataset (confusable).
    * rot filler is drawn from the NON-legal story corpus in data/rot — random
      sections of randomly chosen .txt files (Project Gutenberg eBooks, boilerplate
      stripped). Override with --rot-dir; with no .txt stories it falls back to a
      synthetic sentence bank.

WHERE CUAD AND MAUD GENUINELY DIFFER (kept dataset-specific on purpose):
    * source format     CUAD = one JSON file of contracts; MAUD = a CSV of
                        per-question rows + a directory of full agreement texts.
    * question model    CUAD = SQuAD spans with native is_impossible negatives;
                        MAUD = forced-choice multiple choice (focus="labels",
                        is_impossible always False, a safe-negative OPTION).
    * distractor pool   CUAD contracts are small -> loaded eagerly; MAUD agreements
                        are large -> loaded lazily, one or two per budget.
    Everything else (seeding, token counting, rot corpus, context assembly, the
    per-document cell grid, the cell schema, the manifest) is shared. If you find
    yourself adding a third per-dataset branch, that is the warning sign the two
    have drifted — reconcile rather than fork.

Output: one JSON record per cell (per document: (modality x budget) +
per-modality baselines) under each dataset's --out, plus a manifest.json. CUAD ->
data/prepared/ ; MAUD -> data/prepared_maud/ (the dirs run_models/score_outputs
default to).

Usage:
    python src/build_context.py                                  # both datasets, default grid
    python src/build_context.py --dataset cuad --modality rot confusion missing_answer
    python src/build_context.py --dataset maud --num-documents 4 --budgets 64000 128000
    python src/build_context.py --dataset cuad --doc-indices 19 23 --seed 42
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

# Reuse the loaders / helpers from the sibling inspect scripts. When this file is
# run as `python src/build_context.py`, its own directory is on sys.path; make
# that explicit so the imports also work under other invocations.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_cuad import category_of, is_negative, load_cuad  # noqa: E402
from inspect_maud import (  # noqa: E402
    answer_options_map, build_questions, derive_title, group_by_contract,
    load_maud_rows, read_contract_text,
)

CHARS_PER_TOKEN = 4  # approximate; matches inspect_cuad's budgeting hint

# CUAD defaults
CUAD_DEFAULT_FILE = Path("data/cuad/test.json")
CUAD_DEFAULT_OUT = Path("data/prepared")
# MAUD defaults
MAUD_DEFAULT_FILE = Path("data/maud/MAUD_dev.csv")
MAUD_DEFAULT_CONTRACTS = Path("data/maud/contracts")
MAUD_DEFAULT_OUT = Path("data/prepared_maud")
MAUD_DEFAULT_DATA_TYPE = "main"

# All modalities the schema knows about; only IMPLEMENTED ones generate cells.
MODALITIES = ("rot", "confusion", "missing_answer", "missing_document")
IMPLEMENTED_MODALITIES = ("rot", "confusion", "missing_answer")
# Modalities with an interference axis, so a zero-filler baseline makes sense.
BASELINE_MODALITIES = ("rot", "confusion")
DEFAULT_MODALITIES = IMPLEMENTED_MODALITIES
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
# vocabulary, so nothing here is confusable with a legal document.
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
# Rot (non-legal) filler corpus — shared by both datasets
# --------------------------------------------------------------------------- #
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


def build_rot_pool(rot_dir: Path, max_filler_chars: int, seed: int):
    """Non-legal filler as (entries, loader, label) — shared by both datasets.

    `entries` is a list of opaque section ids; `loader(id) -> (id, title, text)`.
    Draws random sections of random story files from `rot_dir`, sized to the
    largest budget so no section repeats. Falls back to the synthetic sentence
    bank only when the corpus directory is missing/empty. Nothing here is
    confusable with a legal document — it is length-only noise.
    """
    corpus = load_rot_corpus(rot_dir)
    if not corpus:
        bank = list(_NONLEGAL_SENTENCES)
        paras = [" ".join(bank[i:i + 4]) for i in range(0, len(bank), 4)]
        table = {f"filler_{i + 1:03d}": p for i, p in enumerate(paras)}

        def load_synthetic(did: str):
            return did, "non-legal filler", table[did]

        return list(table), load_synthetic, "synthetic-placeholder"

    rng = random.Random(stable_seed(seed, "rot_sections"))
    sections = dict(sample_rot_sections(corpus, rot_pool_size(max_filler_chars), rng))

    def load(did: str):
        return did, "non-legal story excerpt", sections[did]

    return list(sections), load, f"rot-sections:{rot_dir}"


# --------------------------------------------------------------------------- #
# Document wrapping + context assembly — shared by both datasets
# --------------------------------------------------------------------------- #
class DocWrapper:
    """Wrap a document body in the shared <DOCUMENT> tag.

    Both CUAD and MAUD use the same uppercase <DOCUMENT id="..." title="...">
    tag, so the assembly code below stays dataset-agnostic. (The tag name and
    whether the title is emitted are configurable should the two ever need to
    diverge again.)
    """

    def __init__(self, tag: str, with_title: bool):
        self.tag = tag
        self.with_title = with_title
        self.close = f"\n</{tag}>"

    def open(self, doc_id: str, title: str = "") -> str:
        if self.with_title:
            safe = (title or "").replace('"', "'").replace("\n", " ").strip()
            return f'<{self.tag} id="{doc_id}" title="{safe}">\n'
        return f'<{self.tag} id="{doc_id}">\n'

    def block(self, doc_id: str, title: str, text: str) -> str:
        return self.open(doc_id, title) + text + self.close


def build_cell_context(target_id: str, target_title: str, target_text: str,
                       entries: list, loader, filler_chars: int, sep: str,
                       rng: random.Random, wrapper: DocWrapper) -> dict:
    """Wrap the FULL target plus `filler_chars` of distractor filler before it.

    The target document is ALWAYS included verbatim and in full, and is placed at
    the END of the window, after all the filler; `filler_chars` is the amount of
    rot/confusion text added BEFORE it (the experiment's independent variable),
    NOT a cap on the total window. filler_chars <= 0 (or no entries) yields the
    bare target (the zero-interference baseline).

    `entries` are opaque distractor handles; `loader(handle) -> (id, title, text)`
    (eager for CUAD, lazy file reads for MAUD). Returns the assembled context, the
    char offset where the raw target text begins (gold offsets map as offset +
    answer_start), the distractor ids used, and whether filler had to be repeated.
    """
    target_block = wrapper.block(target_id, target_title, target_text)
    target_inner_offset = len(wrapper.open(target_id, target_title))  # raw text start

    blocks: list[str] = []        # distractor blocks in fill order
    used_ids: list[str] = []
    repeated = False

    if entries and filler_chars > 0:
        order = entries[:]
        rng.shuffle(order)
        overhead_close = len(wrapper.close)
        added = 0                 # filler chars accumulated (target NOT counted)
        i = 0
        while True:
            budget_left = filler_chars - added
            if budget_left <= 0:
                break
            handle = order[i % len(order)]
            repeated = repeated or i >= len(order)
            did, title, text = loader(handle)
            block = wrapper.block(did, title, text)
            if len(block) + len(sep) > budget_left:  # trim inner text so it fits
                keep = budget_left - len(sep) - len(wrapper.open(did, title)) - overhead_close
                if keep <= 0:
                    break
                block = wrapper.block(did, title, text[:keep])
            blocks.append(block)
            used_ids.append(did)
            added += len(block) + len(sep)
            i += 1
            # A single-document pool (e.g. MAUD's one huge agreement) can't add
            # variety by repeating, so stop once it has been used in full.
            if len(order) == 1 and i >= 1:
                break

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


# =========================================================================== #
# CUAD-specific: target selection + distractor pool
# =========================================================================== #
def _cuad_qualifies(doc: dict, min_negatives: int) -> bool:
    """True if `doc` has a usable paragraph with >= min_negatives negatives."""
    paragraphs = doc.get("paragraphs", [])
    if not paragraphs:
        return False
    qas = paragraphs[0].get("qas", [])
    return sum(1 for qa in qas if is_negative(qa)) >= min_negatives


def cuad_select_targets(data: list[dict], seed: int, doc_indices: list[int] | None,
                        num_documents: int, max_questions: int | None, balance: bool,
                        min_negatives: int) -> list[dict]:
    """Pick N contracts as targets and build each one's fixed question set.

    Documents are chosen deterministically from `seed`. Explicit `doc_indices`
    override the random pick. For the random path, a sampled doc that lacks
    >= min_negatives negatives is skipped and another is drawn, so a short corpus
    never silently drops below the requested count; if fewer than N qualify we
    warn and proceed with what does.
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
            if _cuad_qualifies(data[di], min_negatives):
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

    return [_cuad_build_target(data, di, seed, max_questions, balance, min_negatives)
            for di in indices]


def _cuad_build_target(data: list[dict], doc_index: int, seed: int,
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
    chosen = _cuad_choose_questions(rng, positives, negatives, max_questions,
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
        "text": context,
        "text_chars": len(context),
        "questions": questions,
        "num_positives": sum(1 for q in questions if not q["is_impossible"]),
        "num_negatives": sum(1 for q in questions if q["is_impossible"]),
    }


def _cuad_choose_questions(rng, positives, negatives, max_questions, balance,
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


def cuad_legal_pool(data: list[dict], target_index: int):
    """(entries, loader) for every OTHER contract — confusable filler (eager).

    CUAD contracts are small, so the whole pool is held in memory as (id, text)
    handles; the loader just unpacks (no per-block file reads).
    """
    entries: list[tuple[str, str]] = []
    for i, doc in enumerate(data):
        if i == target_index:
            continue
        for para in doc.get("paragraphs", []):
            ctx = para.get("context", "")
            if ctx:
                entries.append((doc.get("title") or f"contract_{i}", ctx))

    def load(handle):
        doc_id, text = handle
        return doc_id, "", text

    return entries, load


# =========================================================================== #
# MAUD-specific: target selection + distractor pool
# =========================================================================== #
def maud_legal_pool(by_contract: dict, contracts_dir: Path, target_id: str):
    """(entries, loader) for other agreements — confusable filler (lazy).

    MAUD agreements are large, so texts are read on demand (a budget rarely needs
    more than one or two). `entries` is the list of other contract ids;
    `loader(id) -> (id, title, text)` reads + caches.
    """
    ids = [n for n in sorted(by_contract) if n != target_id]
    cache: dict[str, tuple[str, str, str]] = {}

    def load(doc_id: str):
        if doc_id not in cache:
            text = read_contract_text(contracts_dir, doc_id)
            cache[doc_id] = (doc_id, derive_title(text, doc_id), text)
        return cache[doc_id]

    return ids, load


def maud_enrich_questions(raw_questions: list[dict]) -> list[dict]:
    """Attach opaque qa_ids + a SQuAD-shaped mirror for the shared run pipeline.

    MAUD is closed-set MULTIPLE CHOICE: every question is answerable and is scored
    by option match (score_outputs keys on answer_type), so `is_impossible` is
    always False. A safe-negative answer ("No"/"None"/"N/A") is simply one of the
    options; `is_negative` is kept as an informative flag, and its presence is
    what lets the missing_answer modality build.
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


def _maud_choose_questions(rng, questions: list[dict], max_questions: int | None,
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


def maud_select_targets(by_contract: dict, all_rows: list[dict], contracts_dir: Path,
                        seed: int, contracts: list[str] | None,
                        doc_indices: list[int] | None, num_documents: int,
                        max_questions: int | None, min_negatives: int,
                        negatives_mode: str) -> list[dict]:
    """Pick N agreements as targets and build each one's fixed question set.

    Documents are chosen deterministically from `seed`; explicit `contracts`
    (by name) or `doc_indices` override the random pick. Unlike CUAD, MAUD
    selection has no min-negatives gate — missing_answer is decided per document
    at cell-generation time.
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
    return [_maud_build_target(by_contract, names, contracts_dir, seed, name, options,
                               max_questions, min_negatives, negatives_mode)
            for name in chosen_names]


def _maud_build_target(by_contract: dict, names: list[str], contracts_dir: Path,
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
    chosen = _maud_choose_questions(sel_rng, raw_questions, max_questions, min_negatives)
    questions = maud_enrich_questions(chosen)

    # Safe negatives gate the missing_answer modality. They are normal MC options
    # now (not impossible), so count them via is_negative; --negatives none
    # suppresses the gate (and thus the missing_answer modality).
    num_negatives = (sum(1 for q in questions if q["is_negative"])
                     if negatives_mode == "auto" else 0)

    return {
        "id": name,
        "doc_index": names.index(name),
        "title": title,
        "text": text,
        "text_chars": len(text),
        "questions": questions,
        "num_positives": len(questions) - num_negatives,
        "num_negatives": num_negatives,
    }


# =========================================================================== #
# Dataset builders — the only per-dataset surface the shared driver touches
# =========================================================================== #
class Builder:
    """Per-dataset adapter. The shared driver (`run_build`) calls these; anything
    not overridden here is genuinely identical across CUAD and MAUD."""

    name: str
    wrapper: DocWrapper
    legal_source: str
    baseline_modalities = BASELINE_MODALITIES

    def file(self, args) -> Path: raise NotImplementedError
    def out(self, args) -> Path: raise NotImplementedError
    def load(self, args): raise NotImplementedError
    def select_targets(self, args, ctx, doc_indices, contracts) -> list[dict]: raise NotImplementedError
    def legal_pool(self, ctx, target): raise NotImplementedError

    def focus(self, modality: str) -> str:
        """Primary scoring focus per modality (recorded for the eval step)."""
        raise NotImplementedError

    def target_line(self, target: dict, count_tokens) -> str:
        """One-line description printed during selection."""
        return (f"  [{target['doc_index']:>3}] {target['id']}  "
                f"~{count_tokens(target['text']):,} tok, "
                f"{len(target['questions'])} q "
                f"({target['num_positives']} pos, {target['num_negatives']} neg)")

    def modalities_for(self, target: dict, requested: list[str]) -> tuple[list[str], str | None]:
        """(modalities, note) for this document. Default: build every requested
        modality. MAUD overrides to drop missing_answer with no safe negative."""
        return list(requested), None

    def manifest_extra(self, args) -> dict:
        """Dataset-specific manifest fields layered on the shared base."""
        return {}


class CuadBuilder(Builder):
    name = "cuad"
    wrapper = DocWrapper("DOCUMENT", with_title=True)
    legal_source = "cuad-other-contracts"

    def file(self, args): return args.cuad_file
    def out(self, args): return args.cuad_out

    def load(self, args):
        raw = load_cuad(args.cuad_file)
        data = raw.get("data", [])
        if not data:
            raise SystemExit(f"error: no 'data' entries found in {args.cuad_file}")
        return data

    def select_targets(self, args, ctx, doc_indices, contracts) -> list[dict]:
        if contracts:
            raise SystemExit("error: --contracts is MAUD-only (CUAD targets are "
                             "chosen by index; use --doc-indices).")
        return cuad_select_targets(ctx, args.seed, doc_indices, args.num_documents,
                                   args.max_questions, args.balance, args.min_negatives)

    def legal_pool(self, ctx, target):
        return cuad_legal_pool(ctx, target["doc_index"])

    def focus(self, modality):
        return "abstention" if modality in ("missing_answer", "missing_document") else "spans"


class MaudBuilder(Builder):
    name = "maud"
    wrapper = DocWrapper("DOCUMENT", with_title=True)
    legal_source = "maud-other-agreements"

    def file(self, args): return args.maud_file
    def out(self, args): return args.maud_out

    def load(self, args):
        rows = load_maud_rows(args.maud_file, args.maud_data_type or None)
        by_contract = group_by_contract(rows)
        if not by_contract:
            raise SystemExit(f"error: no contracts found in {args.maud_file}")
        return {"rows": rows, "by_contract": by_contract}

    def select_targets(self, args, ctx, doc_indices, contracts) -> list[dict]:
        return maud_select_targets(
            ctx["by_contract"], ctx["rows"], args.maud_contracts_dir, args.seed,
            contracts, doc_indices, args.num_documents, args.max_questions,
            args.min_negatives, args.maud_negatives)

    def legal_pool(self, ctx, target):
        # contracts_dir is stashed on ctx by run_build so the pool can be rebuilt
        # per document without threading args through the shared driver.
        return maud_legal_pool(ctx["by_contract"], ctx["contracts_dir"], target["id"])

    def focus(self, modality):
        return "labels"  # MAUD cells score label/answer match, per the task spec.

    def target_line(self, target, count_tokens):
        return (f"  [{target['doc_index']:>3}] {target['id']} — {target['title']}  "
                f"~{count_tokens(target['text']):,} tok, "
                f"{len(target['questions'])} q "
                f"({target['num_negatives']} safe-negative)")

    def modalities_for(self, target, requested):
        modalities = list(requested)
        if "missing_answer" in modalities and target["num_negatives"] == 0:
            note = (f"  d{target['doc_index']:03d}: skipping 'missing_answer' "
                    f"(no safe negative in this question set).")
            return [m for m in modalities if m != "missing_answer"], note
        return modalities, None

    def manifest_extra(self, args):
        return {
            "contracts_dir": str(args.maud_contracts_dir),
            "data_type": args.maud_data_type,
            "negatives_mode": args.maud_negatives,
        }


# =========================================================================== #
# Shared cell + driver
# =========================================================================== #
def make_cell(builder: Builder, target: dict, modality: str, budget: int,
              count_tokens, legal_pool, rot_pool, seed: int, sep: str,
              baseline: bool = False) -> dict:
    """Build one prepared-context record (schema shared across datasets).

    `budget` is the amount of rot/confusion FILLER (in tokens) to add before the
    full target — the independent variable, not a cap on the window. The target is
    always kept whole and placed at the end. baseline=True forces zero filler and
    a `<modality>_baseline` cell id. The cell id carries a `d{doc_index:03d}`
    prefix so cells stay unique across the several target documents.
    """
    # Filler order is keyed by the target id so each document shuffles its own
    # pool independently yet reproducibly from --seed.
    rng = random.Random(stable_seed(seed, target["id"], modality, budget))
    filler_chars = 0 if baseline else budget * CHARS_PER_TOKEN

    if modality == "rot":
        entries, loader, source = rot_pool
    else:  # confusion + missing_answer use confusable legal filler
        entries, loader = legal_pool
        source = builder.legal_source

    built = build_cell_context(target["id"], target.get("title", ""), target["text"],
                               entries, loader, filler_chars, sep, rng, builder.wrapper)

    prefix = f"d{target['doc_index']:03d}"
    cell_id = (f"{prefix}_{modality}_baseline" if baseline
               else f"{prefix}_{modality}_b{budget}")
    return {
        "cell_id": cell_id,
        "dataset": builder.name,
        "doc_index": target["doc_index"],
        "modality": modality,
        "focus": builder.focus(modality),
        "budget_tokens": 0 if baseline else budget,  # rot/confusion filler tokens
        "is_baseline": baseline,
        "actual_tokens": count_tokens(built["context"]),
        "actual_chars": len(built["context"]),
        "target_document_id": target["id"],
        "target_document_title": target.get("title", ""),
        "distractor_ids": built["distractor_ids"],
        "target_offset": built["target_offset"],
        "target_end": built["target_offset"] + len(target["text"]),
        "filler_source": source,
        "filler_repeated": built["filler_repeated"],
        "context": built["context"],
        "questions": target["questions"],
    }


def run_build(builder: Builder, args, count_tokens, requested: list[str],
              budgets: list[int], doc_indices, contracts) -> None:
    """Build + write one dataset's cells and manifest (the shared driver)."""
    out = builder.out(args)
    ctx = builder.load(args)
    # MAUD's legal pool needs the contracts dir at assembly time; expose it on ctx.
    if isinstance(ctx, dict):
        ctx["contracts_dir"] = args.maud_contracts_dir

    targets = builder.select_targets(args, ctx, doc_indices, contracts)
    if not targets:
        raise SystemExit(f"error: no qualifying target documents selected for "
                         f"{builder.name}.")

    print(f"\n=== {builder.name.upper()} — {builder.file(args)} ===")
    print(f"Selected {len(targets)} target document(s):")
    for target in targets:
        print(builder.target_line(target, count_tokens))

    # The rot section pool is non-legal noise (never confusable), so it is built
    # ONCE — sized to the largest budget — and shared across docs.
    max_filler_chars = (max(budgets) if budgets else 0) * CHARS_PER_TOKEN
    rot_pool = build_rot_pool(args.rot_dir, max_filler_chars, args.seed)
    if "rot" in requested:
        if rot_pool[2] == "synthetic-placeholder":
            print(f"  rot filler: synthetic placeholder "
                  f"(no .txt stories found at {args.rot_dir})")
        else:
            print(f"  rot filler: {len(rot_pool[0])} random story sections "
                  f"from {args.rot_dir}")

    cells_meta = []
    cells_dir = out / "cells"
    if not args.dry_run:
        cells_dir.mkdir(parents=True, exist_ok=True)

    # Per document: a zero-filler baseline per interference modality, then the
    # budget interference grid (target always at the END). The legal distractor
    # pool excludes the current target, so it is rebuilt per document.
    print(f"\nGenerating cells for {len(targets)} document(s):")
    for target in targets:
        modalities, note = builder.modalities_for(target, requested)
        if note:
            print(note)
        if not modalities:
            continue

        legal_pool = builder.legal_pool(ctx, target)
        jobs = []  # (modality, budget, baseline)
        for modality in modalities:
            if modality in builder.baseline_modalities:
                jobs.append((modality, 0, True))
            for budget in budgets:
                jobs.append((modality, budget, False))

        print(f"  d{target['doc_index']:03d} {target['id']}: "
              f"{len(modalities)} modality(ies), {len(jobs)} cells")
        for modality, budget, baseline in jobs:
            cell = make_cell(builder, target, modality, budget, count_tokens,
                             legal_pool, rot_pool, args.seed, DOC_SEPARATOR,
                             baseline=baseline)
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
        "source_file": str(builder.file(args)),
        "dataset": builder.name,
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "chars_per_token": CHARS_PER_TOKEN,
        "doc_separator": DOC_SEPARATOR,
        "num_documents": len(targets),
        "doc_indices": [t["doc_index"] for t in targets],
        "targets": [{k: v for k, v in t.items() if k != "text"} for t in targets],
        "modalities": requested,
        "budgets": budgets,
        "baseline_modalities": [m for m in requested if m in builder.baseline_modalities],
        "cells": cells_meta,
    }
    manifest.update(builder.manifest_extra(args))

    if args.dry_run:
        print(f"\n(dry run — {len(cells_meta)} {builder.name} cells planned, "
              f"no files written)")
        return

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"\nWrote {len(cells_meta)} {builder.name} cells + manifest to {out}/")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("cuad", "maud", "both"), default="both",
                        help="which dataset(s) to build (default: both — they are "
                             "almost always built together)")

    # Shared interference / sampling axes.
    parser.add_argument("--modality", nargs="+", choices=MODALITIES,
                        default=list(DEFAULT_MODALITIES),
                        help="modalities to generate (default: rot confusion "
                             "missing_answer)")
    parser.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS),
                        help="rot/confusion filler tokens to add AROUND the full "
                             "probe (default: 64,000 128,000 256,000 512,000); the "
                             "zero-filler case is emitted as the baseline cell")
    parser.add_argument("--num-documents", type=int, default=DEFAULT_NUM_DOCUMENTS,
                        help=f"number of target documents to sample per dataset "
                             f"(default: {DEFAULT_NUM_DOCUMENTS})")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for sampling + filler shuffles (default: 0)")
    parser.add_argument("--max-questions", type=int, default=DEFAULT_MAX_QUESTIONS,
                        help=f"cap questions per document "
                             f"(default: {DEFAULT_MAX_QUESTIONS}; 0 = all)")
    parser.add_argument("--min-negatives", type=int, default=1,
                        help="minimum/preferred negative categories in the set "
                             "(default: 1)")
    parser.add_argument("--rot-dir", "--rot-filler", type=Path, default=ROT_DIR,
                        dest="rot_dir",
                        help=f"dir/file of .txt non-legal story filler; rot draws "
                             f"random sections of random files (default: {ROT_DIR})")
    parser.add_argument("--tokenizer", choices=("approx", "tiktoken"),
                        default="approx", help="token counter (default: approx)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan without writing files")

    # Target selection (single-dataset only — index spaces differ per dataset).
    parser.add_argument("--doc-indices", type=int, nargs="+", default=None,
                        help="explicit document indices to target (overrides random "
                             "selection; requires a single --dataset)")
    parser.add_argument("--doc-index", type=int, default=None,
                        help="deprecated alias for a single --doc-indices entry")

    # CUAD-specific.
    parser.add_argument("--cuad-file", type=Path, default=CUAD_DEFAULT_FILE,
                        help=f"CUAD JSON to sample from (default: {CUAD_DEFAULT_FILE})")
    parser.add_argument("--cuad-out", type=Path, default=CUAD_DEFAULT_OUT,
                        help=f"CUAD output directory (default: {CUAD_DEFAULT_OUT})")
    parser.add_argument("--balance", action="store_true",
                        help="CUAD only: with --max-questions, pick ~equal pos/neg")

    # MAUD-specific.
    parser.add_argument("--maud-file", type=Path, default=MAUD_DEFAULT_FILE,
                        help=f"MAUD CSV to sample from (default: {MAUD_DEFAULT_FILE})")
    parser.add_argument("--maud-out", type=Path, default=MAUD_DEFAULT_OUT,
                        help=f"MAUD output directory (default: {MAUD_DEFAULT_OUT})")
    parser.add_argument("--maud-contracts-dir", type=Path, default=MAUD_DEFAULT_CONTRACTS,
                        help=f"MAUD full agreement texts "
                             f"(default: {MAUD_DEFAULT_CONTRACTS})")
    parser.add_argument("--maud-data-type", default=MAUD_DEFAULT_DATA_TYPE,
                        help=f"MAUD row split to use (default: {MAUD_DEFAULT_DATA_TYPE})")
    parser.add_argument("--maud-negatives", choices=("auto", "none"), default="auto",
                        help="MAUD only: auto maps No/None/N/A answers to safe "
                             "negatives (enables missing_answer); none treats every "
                             "question as answerable.")
    parser.add_argument("--contracts", nargs="+", default=None,
                        help="MAUD only: explicit contract_name(s) to target "
                             "(overrides random selection; requires --dataset maud)")
    parser.add_argument("--contract", default=None,
                        help="deprecated alias for a single --contracts entry")
    args = parser.parse_args()

    datasets = ("cuad", "maud") if args.dataset == "both" else (args.dataset,)

    # missing_document is reserved for later (needs a document absent from the
    # window — every question is currently tied to its own document).
    requested = []
    for m in args.modality:
        if m == "missing_document":
            print("note: 'missing_document' is not implemented yet "
                  "(reserved); skipping.")
            continue
        requested.append(m)
    if not requested:
        raise SystemExit("error: no implemented modalities requested.")

    # The zero-filler case is the baseline cell, not a budget; drop any <= 0.
    budgets = sorted(b for b in args.budgets if b > 0)
    dropped = [b for b in args.budgets if b <= 0]
    if dropped:
        print(f"note: ignoring non-positive budget(s) {dropped}; the zero-filler "
              f"case is emitted as the baseline cell instead.")

    # --doc-index / --contract are deprecated aliases for the plural flags.
    doc_indices = args.doc_indices
    if args.doc_index is not None:
        print("note: --doc-index is deprecated; use --doc-indices.")
        if doc_indices is None:
            doc_indices = [args.doc_index]
        elif args.doc_index not in doc_indices:
            doc_indices = [args.doc_index] + list(doc_indices)
    contracts = list(args.contracts) if args.contracts else None
    if args.contract is not None:
        print("note: --contract is deprecated; use --contracts.")
        contracts = (contracts or []) + [args.contract]

    # Explicit target selection is per-dataset (index spaces differ), so it is
    # only unambiguous when a single dataset is being built.
    if (doc_indices is not None or contracts is not None) and len(datasets) > 1:
        raise SystemExit("error: --doc-indices/--contracts target one dataset's "
                         "documents; pass --dataset cuad or --dataset maud.")

    # --max-questions 0 means "no cap" (keep the whole set).
    args.max_questions = None if args.max_questions in (0, None) else args.max_questions

    count_tokens = make_token_counter(args.tokenizer)
    builders = {"cuad": CuadBuilder(), "maud": MaudBuilder()}
    for ds in datasets:
        run_build(builders[ds], args, count_tokens, requested, budgets,
                  doc_indices, contracts)


if __name__ == "__main__":
    main()
