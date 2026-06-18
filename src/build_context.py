"""build_context.py — generate context-degradation cells from raw CUAD.

Given the raw CUAD dataset, this builds the grid of degraded-context "cells"
that the models are run against. CUAD only for now (MAUD comes later).

  Init:
    1. Sample ONE contract as the TARGET.
    2. Fix its clause-category question set, including >=1 negative category.
    3. Record gold spans + character offsets.
    The target + question set are held FIXED; only the surrounding filler
    (distractor documents) changes across cells.

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

  Interference & position sweep:
    * budgets    how much rot/confusion FILLER to add AROUND the probe, in
                 tokens (4k, 16k, 64k, ...). This is the independent variable
                 (the amount of interference), NOT a cap on the window: the
                 probe document is ALWAYS kept whole and the filler is added on
                 top of it. A 4k budget over a 16k probe -> a ~20k-token window.
    * positions  target_at_start (0%), target_at_middle (50%), target_at_end (100%).

  Baseline:
    For the rot and confusion modalities a zero-filler BASELINE cell is also
    emitted (`<modality>_baseline`): the bare probe with no interference, the
    anchor every degradation curve is measured against. The abstention
    modalities (missing_answer/_document) have no interference axis -> no baseline.

Output: one JSON record per cell ((modality x budget x position) + per-modality
baselines) under --out, plus a manifest.json describing the fixed target. Those
records feed run_models.

Filler sources:
    * confusion / missing_answer filler is drawn from the OTHER contracts in the
      same CUAD file (confusable, available today).
    * rot filler uses a synthetic NON-legal placeholder until the real rot
      corpora (Gutenberg / Wikipedia / news) are assembled; pass --rot-filler
      with a .txt file or directory to use real text instead.

Usage:
    python src/build_context.py                                  # rot, default grid
    python src/build_context.py --modality rot confusion missing_answer
    python src/build_context.py --budgets 4000 64000 --positions target_at_start target_at_end
    python src/build_context.py --doc-index 19 --seed 42
    python src/build_context.py --dry-run                        # plan only, no write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
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
DEFAULT_BUDGETS = (4000, 16000, 64000)
POSITIONS = ("target_at_start", "target_at_middle", "target_at_end")
DEFAULT_POSITIONS = POSITIONS

DOC_SEPARATOR = "\n\n"

# Neutral, deliberately NON-legal sentences for the rot placeholder. No
# contract / merger vocabulary, so nothing here is confusable with CUAD.
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
def select_target(data: list[dict], seed: int, doc_index: int | None,
                  max_questions: int | None, balance: bool,
                  min_negatives: int) -> dict:
    """Pick one contract as the target and build its fixed question set."""
    rng = random.Random(stable_seed(seed, "select_target"))
    if doc_index is None:
        doc_index = rng.randrange(len(data))
    if not 0 <= doc_index < len(data):
        raise SystemExit(
            f"error: --doc-index {doc_index} out of range (0..{len(data) - 1})"
        )

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
            f"categories, need >= {min_negatives}. Try another --doc-index."
        )

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
        "id": doc.get("title") or f"contract_{doc_index}",
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


def nonlegal_distractors(rot_filler: Path | None) -> tuple[list[tuple[str | None, str]], str]:
    """(id, text) non-legal filler pieces + a label describing their source.

    id is None for synthetic pieces (a generic filler id is assigned at
    assembly); for real files the filename stem is used.
    """
    if rot_filler is None:
        bank = list(_NONLEGAL_SENTENCES)
        paras = [" ".join(bank[i:i + 4]) for i in range(0, len(bank), 4)]
        return [(None, p) for p in paras], "synthetic-placeholder"

    files = []
    if rot_filler.is_dir():
        files = sorted(p for p in rot_filler.rglob("*.txt"))
    elif rot_filler.is_file():
        files = [rot_filler]
    if not files:
        raise SystemExit(f"error: no .txt filler found at {rot_filler}")
    pool: list[tuple[str | None, str]] = [(p.stem, p.read_text(encoding="utf-8", errors="ignore")) for p in files]
    return pool, f"files:{rot_filler}"


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
                       position: str, sep: str, rng: random.Random) -> dict:
    """Wrap the FULL target plus `filler_chars` of distractor filler around it.

    The target document is ALWAYS included verbatim and in full; `filler_chars`
    is the amount of rot/confusion text added AROUND it (the experiment's
    independent variable), NOT a cap on the total window. filler_chars <= 0
    yields the bare target (the zero-interference baseline).

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

    # Place the target among the distractors per position. With no filler
    # (baseline) the target stands alone, so position is moot -> index 0.
    if position in ("target_at_start", "baseline"):
        target_index = 0
    elif position == "target_at_end":
        target_index = len(blocks)
    elif position == "target_at_middle":
        target_index = len(blocks) // 2
    else:
        raise SystemExit(f"error: unknown position {position!r}")

    ordered = blocks[:target_index] + [target_block] + blocks[target_index:]
    chars_before = sum(len(b) for b in ordered[:target_index]) + len(sep) * target_index
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


def make_cell(target: dict, modality: str, budget: int, position: str,
              count_tokens, legal_pool: list, nonlegal_pool: list,
              nonlegal_label: str, seed: int, sep: str,
              baseline: bool = False) -> dict:
    """Build one prepared-context record.

    `budget` is the amount of rot/confusion FILLER (in tokens) to add around the
    full target — the independent variable, not a cap on the window. The target
    is always kept whole. baseline=True forces zero filler (the bare-target
    reference point) and a `<modality>_baseline` cell id.
    """
    rng = random.Random(stable_seed(seed, modality, budget, position))
    filler_chars = 0 if baseline else budget * CHARS_PER_TOKEN

    if modality == "rot":
        pool, source = nonlegal_pool, nonlegal_label
    else:  # confusion + missing_answer use confusable legal filler
        pool, source = legal_pool, "cuad-other-contracts"

    built = build_cell_context(target["id"], target["context"], pool,
                               filler_chars, position, sep, rng)
    context = built["context"]
    target_offset = built["target_offset"]

    cell_id = (f"{modality}_baseline" if baseline
               else f"{modality}_b{budget}_{position}")
    return {
        "cell_id": cell_id,
        "modality": modality,
        "focus": cell_focus(modality),
        "budget_tokens": 0 if baseline else budget,  # rot/confusion filler tokens
        "is_baseline": baseline,
        "position": position,
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
                             "probe (default: 4000 16000 64000); the zero-filler "
                             "case is emitted separately as the baseline cell")
    parser.add_argument("--positions", nargs="+", choices=POSITIONS,
                        default=list(DEFAULT_POSITIONS),
                        help="target positions (default: all three)")
    parser.add_argument("--doc-index", type=int, default=None,
                        help="target this contract index (default: random by seed)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for sampling + filler shuffles (default: 0)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="subsample the question set (default: all)")
    parser.add_argument("--balance", action="store_true",
                        help="with --max-questions, pick ~equal pos/neg")
    parser.add_argument("--min-negatives", type=int, default=1,
                        help="minimum negative categories in the set (default: 1)")
    parser.add_argument("--rot-filler", type=Path, default=None,
                        help="dir/file of .txt non-legal filler (default: synthetic)")
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
    target = select_target(data, args.seed, args.doc_index, args.max_questions,
                           args.balance, args.min_negatives)

    print(f"Target: [{target['doc_index']}] {target['id']}")
    print(f"  context: {target['context_chars']:,} chars "
          f"(~{count_tokens(target['context']):,} tokens)")
    print(f"  questions: {len(target['questions'])} "
          f"({target['num_positives']} pos, {target['num_negatives']} neg)")

    legal_pool = legal_distractors(data, target["doc_index"])
    nonlegal_pool, nonlegal_label = nonlegal_distractors(args.rot_filler)
    if "rot" in modalities and nonlegal_label == "synthetic-placeholder":
        print("  rot filler: synthetic placeholder "
              "(pass --rot-filler for real non-legal text)")

    cells_meta = []
    cells_dir = args.out / "cells"
    if not args.dry_run:
        cells_dir.mkdir(parents=True, exist_ok=True)

    # The zero-filler case is the baseline cell, not a budget; drop any <= 0.
    budgets = sorted(b for b in args.budgets if b > 0)
    dropped = [b for b in args.budgets if b <= 0]
    if dropped:
        print(f"note: ignoring non-positive budget(s) {dropped}; the zero-filler "
              f"case is emitted as the baseline cell instead.")

    # Job list: a zero-filler baseline per interference modality, then the
    # (budget x position) interference grid. (modality, budget, position, baseline)
    jobs = []
    for modality in modalities:
        if modality in BASELINE_MODALITIES:
            jobs.append((modality, 0, "baseline", True))
        for budget in budgets:
            for position in args.positions:
                jobs.append((modality, budget, position, False))

    print(f"\nGenerating {len(jobs)} cells:")
    for modality, budget, position, baseline in jobs:
        cell = make_cell(target, modality, budget, position, count_tokens,
                         legal_pool, nonlegal_pool, nonlegal_label,
                         args.seed, DOC_SEPARATOR, baseline=baseline)
        flags = []
        if cell["is_baseline"]:
            flags.append("baseline / zero-filler")
        if cell["filler_repeated"]:
            flags.append("filler-repeated")
        flag_str = ("  [" + ", ".join(flags) + "]") if flags else ""
        print(f"  {cell['cell_id']:42} "
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
        "target": {k: v for k, v in target.items() if k != "context"},
        "modalities": modalities,
        "budgets": budgets,
        "baseline_modalities": [m for m in modalities if m in BASELINE_MODALITIES],
        "positions": args.positions,
        "cells": cells_meta,
    }
    if args.dry_run:
        print("\n(dry run — no files written)")
        return

    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"\nWrote {len(cells_meta)} cells + manifest to {args.out}/")


if __name__ == "__main__":
    main()
