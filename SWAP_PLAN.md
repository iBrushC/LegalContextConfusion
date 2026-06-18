# SWAP_PLAN — dataset generation rework

Reworking how the degradation cells are generated and consumed. Four functional
changes plus the downstream fallout they create. This file is the map; each
**Step** below is meant to be tackled one at a time, in order. Steps are written
so a later step can assume the earlier ones landed.

## Goals (from the request)

1. **Multiple document bases.** Sample N random documents per dataset (default
   **10**, configurable), chosen deterministically — not a single target.
2. **Capped question count.** At most **32** questions per document, chosen
   deterministically.
3. **Placement = end only.** Drop the start/middle/end sweep. The source
   (target) document always sits at the **end** of the window, after all the
   rot/confusion filler.
4. **MAUD parity with CUAD.** MAUD budgets stop being cutoffs that truncate the
   agreement; they become the amount of rot/confusion filler **added around** a
   whole, untruncated target — exactly like CUAD. MAUD adopts CUAD's budget grid.

## Decisions locked in (from clarifying Q&A)

- **Cross-document aggregation:** pool the N documents into a single averaged
  curve per (model, modality, budget). Documents are treated as repeated trials;
  the band becomes cross-document + multirun variance.
- **MAUD budget grid:** match CUAD — `64000, 128000, 256000, 512000`. Budget is
  filler added on top of the (~85k-token) whole agreement.
- **Position cleanup:** fully remove the depth machinery — the position sweep,
  `--positions`, the `position` field on records, `degradation_depth.png`, and
  the depth dimension of the heatmap.

## Current architecture (baseline before changes)

- [src/build_context.py](src/build_context.py) — CUAD builder. Picks **one**
  target contract, builds its question set, emits a
  `(modality × budget × position)` grid + per-modality baselines to
  `data/prepared/`. Target always kept whole; budget = filler added around it.
- [src/build_maud_contexts.py](src/build_maud_contexts.py) — MAUD builder. Picks
  **one** target agreement. Budget is treated as a **cutoff**: the target gets a
  `target_share` of the budget and is **truncated**; distractors fill the rest.
  Emits to `data/prepared_maud/`.
- [src/run_models.py](src/run_models.py) — runs each cell against the models,
  writes `runs.jsonl` + `summary.{json,csv}`. Records carry `position`.
- [src/score_outputs.py](src/score_outputs.py) — scoring + `aggregate()` (groups
  by `(model, cell_id)`); standalone re-scorer.
- [src/analyze.py](src/analyze.py) — degradation curves over **length × depth**;
  `build_curves` keys by `(budget, position)`.
- [src/plot_results.py](src/plot_results.py) — `degradation_length.png`,
  `degradation_depth.png`, `heatmap.png` (length × depth).

## Shared design: determinism chain

One master `--seed` (default 0) drives everything via the existing
`stable_seed(*parts)` helper (sha256-based, unsalted):

```
seed
 ├─ document selection      stable_seed(seed, "select_docs")        -> pick N doc ids
 ├─ per-document questions   stable_seed(seed, "choose_questions", doc_id)
 ├─ per-cell filler shuffle  stable_seed(seed, doc_id, modality, budget)
 └─ filler pool sampling     stable_seed(seed, "rot_sections")      (unchanged)
```

Adding `doc_id` to the per-cell and per-question seeds is what keeps each
document's question subset and filler order independent yet reproducible.

## Shared design: cell identity & manifest

- **cell_id** gains a document prefix so cells stay unique and filename-safe
  across documents and across the dropped position axis:
  - interference cell: `d{doc_index:03d}_{modality}_b{budget}`
  - baseline cell:     `d{doc_index:03d}_{modality}_baseline`
    (MAUD's no-filler reference stays `d{doc_index:03d}_clean_b{budget}`)
- Each cell record keeps `target_document_id` and gains `doc_index` so the
  analysis layer can pool/group correctly. The `position` field is **removed**.
- **manifest.json**: `target` (single dict) → `targets` (list of per-doc dicts),
  plus top-level `num_documents` and `doc_indices`. `cells[]` (with `path`) is
  unchanged in shape, so `run_models.load_cells` / `score_outputs.load_gold`
  keep working — they only read `cells[].path`.

---

# Steps

Each step is independently reviewable. Do them in order; run the offline
pipeline (`--dry-run`, then `--mock`) after each builder change.

### Step 1 — CUAD builder: multi-document + question cap + end-only placement

File: [src/build_context.py](src/build_context.py)

This step folds changes 1, 2, and 3 into the CUAD builder (the simpler of the
two, and the reference the MAUD builder will mirror).

1. **Multi-document selection.** Replace `select_target(...)` (returns one) with
   `select_targets(data, seed, doc_indices|None, num_documents, max_questions,
   balance, min_negatives)` returning a **list** of target dicts.
   - New CLI: `--num-documents` (default `10`), `--doc-indices` (explicit list,
     overrides random selection; replaces today's single `--doc-index`). Keep
     `--doc-index` as a deprecated alias mapping to a 1-element `--doc-indices`.
   - Deterministic pick: `rng = Random(stable_seed(seed, "select_docs"))`, then
     `sorted(rng.sample(range(len(data)), k=min(num_documents, len(data))))`.
   - Each chosen doc must still satisfy `min_negatives`; if a sampled doc fails,
     **skip and resample** from the remaining pool (log which were skipped) so a
     short pool never silently drops below N. Hard-fail only if fewer than N
     qualifying docs exist (warn + proceed with what qualifies).
2. **Per-document question cap.** Default `--max-questions` from `None` → **32**.
   Each doc selects its subset with its own rng:
   `Random(stable_seed(seed, "choose_questions", doc["id"]))`, so docs don't
   share a draw. `_choose_questions` is unchanged except for taking that rng.
3. **End-only placement.** In `build_cell_context`, hard-code the target to the
   end: `target_index = len(blocks)`. Remove the `position` parameter and the
   start/middle branches. Remove `POSITIONS`, `DEFAULT_POSITIONS`, the
   `--positions` flag, and the `position` loop in `main`. The baseline cell is
   still "bare probe, no filler".
4. **Cell id + records.** Use the `d{idx:03d}_…` scheme above; add `doc_index`
   to each cell; drop `position` from the cell dict.
5. **Job list & manifest.** Outer loop over the N targets; inner loop builds
   `baseline` (for `BASELINE_MODALITIES`) + `(modality × budget)` cells. Write
   `targets` list + `num_documents` + `doc_indices` into the manifest.
6. **Filler pools** (`legal_distractors`, `nonlegal_distractors`) are built
   **per target** for the legal pool (it must exclude *that* doc), but the rot
   pool can be built once (sized to max budget) and shared across docs.

Acceptance: `python src/build_context.py --dry-run` prints N docs × (modalities ×
budgets + baselines) cells, no position text anywhere; cell_ids unique;
`--num-documents 2 --budgets 64000` produces the expected small grid.

### Step 2 — MAUD builder: added-filler semantics + CUAD budget grid + Steps 1's changes

File: [src/build_maud_contexts.py](src/build_maud_contexts.py)

Brings MAUD to parity (change 4) and applies the same multi-doc / cap / end-only
changes from Step 1.

1. **Added-filler assembly (the core of change 4).** Rewrite
   `build_cell_context` to mirror CUAD: the target agreement is **always kept
   whole** and placed last; `filler_chars = budget * CHARS_PER_TOKEN` of other
   agreements are added before it. Only **filler** blocks may be trimmed to fit
   the budget; the target never is.
   - Delete `target_share`, `MIN_TARGET_CHARS`, `DEFAULT_TARGET_SHARE`, the
     `--target-share` flag, and the target-truncation path in `fit_block`.
   - Remove `target_truncated` / `target_fits` from records (target is always
     whole now). Keep `filler_repeated`.
2. **Budget grid.** `DEFAULT_BUDGETS` → `(64000, 128000, 256000, 512000)`.
3. **Multi-document + question cap + end-only.** Same as Step 1: `select_targets`
   returning a list (`--num-documents` default 10, `--doc-indices`,
   `--max-questions` default 32), `d{idx:03d}_…` cell ids with `doc_index`,
   remove positions/`--positions`. MAUD already seeds question choice per
   contract name — extend that pattern to doc selection.
4. **`clean` modality** stays as the no-filler reference (the MAUD analogue of
   CUAD's baseline); cell id `d{idx:03d}_clean_b{budget}`. `missing_answer` still
   gated on a safe negative being present in that doc's question set (now decided
   per doc).
5. **Manifest** mirrors Step 1 (`targets`, `num_documents`, `doc_indices`); drop
   `target_share`.

Acceptance: `python src/build_maud_contexts.py --dry-run --num-documents 2`
shows whole (untruncated) targets, filler added on top, CUAD-sized budgets, no
`target-truncated` flags, no position text.

### Step 3 — Drop `position` from the run + score layer

Files: [src/run_models.py](src/run_models.py), [src/score_outputs.py](src/score_outputs.py)

1. **run_models record:** remove `"position": cell["position"]` from the run
   record. Everything else (resume key `(model, cell_id, run_index)`, cell
   loading) is already position-agnostic.
2. **score_outputs.aggregate:** remove `"position"` from the summary row. Add
   `modality` + `budget_tokens` are already present — keep them (the analysis
   layer pools on them). No scoring-math change.

Acceptance: `python src/run_models.py --mock --limit 4` then
`python src/score_outputs.py` runs clean; `summary.csv` has no `position`
column; metrics unchanged from before for an equivalent single-doc run.

### Step 4 — Pool documents + remove depth in the analysis layer

File: [src/analyze.py](src/analyze.py)

1. **Pooled aggregation.** The headline curves must average across the N
   documents. Add a curve-level aggregation keyed by
   `(model, modality, budget_tokens, is_baseline)` that pools **all runs** across
   docs (proper cross-doc + multirun mean/stdev) — either a new
   `aggregate_curves(runs)` in `score_outputs.py` or a `group_keys=` parameter on
   `aggregate`. `run_models`' per-cell `summary.csv` keeps using the existing
   `(model, cell_id)` grouping (useful for resume/debug).
2. **Remove depth.** Delete `POSITION_ORDER`, `SHORT_POS`, all depth columns and
   the "depth effect" block in `report`. `build_curves` keys by `budget` only
   (`{budget: row}` per `(model, modality)`). `write_curves_csv` drops the
   `position` column.
3. Console report becomes a length-only table + sparkline per (model, modality),
   with the ± now reflecting cross-document spread.

Acceptance: `python src/analyze.py` after a mock multi-doc run prints one row
per budget (no start/mid/end columns); `curves.csv` has no `position` column and
one row per `(model, modality, budget, metric)`.

### Step 5 — Charts: length-only

File: [src/plot_results.py](src/plot_results.py)

1. Keep `degradation_length.png` (now reads pooled, depth-free curves; the band
   is cross-document variance).
2. Delete `plot_depth` / `degradation_depth.png` and the `depth` choice in
   `--charts`.
3. **Heatmap:** with depth gone, the old length × depth heatmap is redundant.
   Either drop it, or repurpose to a `model × modality` grid of the headline
   metric vs budget. Default: **drop it** unless we want the compact overview;
   revisit after seeing Step 4 output.
4. Update imports from `analyze` (no more `POSITION_ORDER`, `SHORT_POS`).

Acceptance: `python src/plot_results.py --charts length` renders the length
chart from a mock multi-doc run with no import or key errors.

### Step 6 — Docs, defaults, and end-to-end check

1. Update the module docstrings / `Usage` blocks in both builders (remove
   position examples, document `--num-documents`, the 32-question default, the
   end-only placement, and MAUD's new added-filler semantics).
2. Update any README / board references to start/middle/end or single-target.
3. End-to-end smoke: build (small `--num-documents 2 --budgets 64000`) → `--mock`
   run → `analyze` → `plot` for **both** CUAD and MAUD.
4. Sanity on cost: cell count is now `N_docs × (modalities × budgets + baselines)`
   — no `×3` positions, but `×N_docs`. Re-check `run_models.py --estimate-cost`
   reads sensibly with the larger cell count before any real run.

## Open / deferred

- **`missing_document` modality** is still unimplemented for CUAD; out of scope
  here.
- **Heatmap fate** (drop vs repurpose) finalized in Step 5 after seeing curves.
- **Per-doc drill-down**: we chose pooled headline curves. If per-document
  inspection is later wanted, `curves.csv` can regain a `document` column without
  touching the builders.
