"""run_models.py — run models over the prepared CUAD/MAUD cells and score them.

This is the Run + Eval stage and the orchestrator. The pieces it coordinates
live in sibling modules:
  * openrouter_client.call_openrouter — the OpenRouter HTTP client.
  * score_outputs.{extract_json,score_cell,aggregate} — parsing + scoring.

For each prepared cell (from build_context.py) it asks every clause question and
scores the answers. Questions can be sent all in one prompt (default) or split
into smaller batches (--question-batch-size); the big context is re-sent per
batch, so fewer batches = cheaper.

Every (cell x model x run) is an independent request, so they are fanned out
across a thread pool (--max-concurrency, default 8) and served by OpenRouter in
parallel — the long-context / open-model sweeps that dominate runtime now overlap
instead of running one at a time. Results are scored and written as each request
returns; lower --max-concurrency (or set it to 1) if a provider rate-limits you.

Scoring (in score_outputs.py) feeds the eval nodes on the board:
  * span token-F1 / exact-match vs gold spans      (cuad-eval)
  * abstention vs hallucination on absent clauses   (abstention-eval)
  * wrong-document extraction rate                  (wrong-document-eval)
plus latency / token usage / cost. Multirun N -> mean +/- stdev per cell.

Answers are matched to questions by an opaque `qa_id` (not the category name and
not CUAD's native id, which would leak the target's title into the prompt).

Every JSONL run row records the model's `raw_output`, the `parsed_output`, and
any `json_parse_error` (per batch), so a reviewer can audit exactly what came
back. Runs are resumable: existing (model, cell_id, run_index) rows are skipped
unless --overwrite is passed.

Models go through OpenRouter. Override any of them with --models-config <json> (a {"friendly": "provider/slug"}
map) without editing this file.

Auth: set OPENROUTER_API_KEY in the environment for real runs. Use --mock to
exercise the full scoring pipeline offline (no API, no cost).

Each dataset is read from data/prepared_<dataset>/ (built by build_context.py)
and run into its OWN data/results_<dataset>/ stream — CUAD and MAUD are never
mixed (their cell ids are not namespaced and they score under a different focus).
--dataset selects which to run (default both, looped sequentially).

Usage:
    python src/run_models.py --mock                       # both datasets, offline
    python src/run_models.py --dataset cuad --dry-run     # plan + first prompt preview
    python src/run_models.py --dataset maud --prototype --limit 3   # cheap smoke test
    python src/run_models.py --models claude gemini --runs 3
    python src/run_models.py --question-batch-size 5      # 5 questions per call
    python src/run_models.py --max-concurrency 16         # fan out harder
    python src/run_models.py --max-concurrency 2          # throttle (rate limits)
    python src/run_models.py --overwrite                  # redo instead of resume
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Sibling modules (src/ is on sys.path when run directly; make it explicit).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openrouter_client import call_openrouter  # noqa: E402
from score_outputs import (  # noqa: E402
    _normalize, aggregate, extract_json, option_label, results_of, score_cell,
)

DATASETS = ("cuad", "maud")


def prepared_dir(dataset: str) -> Path:
    """Per-dataset prepared-cells dir written by build_context.py."""
    return Path(f"data/prepared_{dataset}")


def results_dir(dataset: str) -> Path:
    """Per-dataset results dir. CUAD and MAUD are kept in SEPARATE streams: cell
    ids are not dataset-namespaced (both emit d000_rot_b64000, ...) and the two
    datasets score under a different focus, so a shared runs.jsonl would collide
    on resume and mix incompatible cells downstream."""
    return Path(f"data/results_{dataset}")

# Friendly name -> OpenRouter slug. Verified against the live OpenRouter models
# list on 2026-06-17. Override any of them via --models-config.
MODEL_REGISTRY = {
    "claude":   "anthropic/claude-opus-4.8",                  # Opus 4.8
    "chatgpt":  "openai/gpt-5.5",                             # GPT 5.5
    "gemini":   "google/gemini-3.1-pro-preview",             # Gemini 3.1 Pro
    "deepseek": "deepseek/deepseek-v4-pro",                   # DeepSeek 4 Pro
    "nemotron": "nvidia/nemotron-3-ultra-550b-a55b",     # Nemotron 3 Ultra
    "mimo":     "xiaomi/mimo-v2.5-pro",                       # MiMo v2.5 Pro
    "glm":      "z-ai/glm-5.2",                               # GLM 5.2
}
ALIASES = {
    "gpt": "chatgpt", "openai": "chatgpt",
    "opus": "claude", "anthropic": "claude",
    "google": "gemini",
    "nemotron3": "nemotron", "nemotron-3-ultra": "nemotron",
    "mimo-v2.5-pro": "mimo", "mimo2.5": "mimo",
    "glm5": "glm", "glm-5.2": "glm",
}
# Cheapest model for prototyping (reuse-base-procedure card).
PROTOTYPE: tuple[str, str] = ("deepseek-flash", "deepseek/deepseek-v4-flash")
# PROTOTYPE: tuple[str, str] = ("mimo-2.5", "xiaomi/mimo-v2.5")
# PROTOTYPE: tuple[str, str] = ("nemotron3", "nvidia/nemotron-3-ultra-550b-a55b:free")
DEFAULT_MODELS = ("claude", "chatgpt", "gemini", "deepseek", "nemotron", "mimo", "glm")


# --------------------------------------------------------------------------- #
# Model resolution
# --------------------------------------------------------------------------- #
def resolve_models(names: list[str], config: Path | None) -> list[tuple[str, str]]:
    """Map friendly names (with aliases / config overrides) to (name, slug)."""
    registry = dict(MODEL_REGISTRY)
    if config is not None:
        if not config.exists():
            raise SystemExit(f"error: --models-config {config} not found")
        registry.update(json.loads(config.read_text(encoding="utf-8")))

    resolved = []
    for name in names:
        key = ALIASES.get(name.lower(), name.lower())
        if key not in registry:
            raise SystemExit(
                f"error: unknown model {name!r}. Known: {sorted(registry)}.\n"
                f"       Add it via --models-config '{{\"{name}\": \"provider/slug\"}}'."
            )
        resolved.append((key, registry[key]))
    return resolved


# --------------------------------------------------------------------------- #
# Prompt building
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a precise legal contract analyst. You are given one or more "
    "documents, each wrapped in <document id=\"...\"> tags. Exactly ONE of them "
    "is the TARGET document; its id is named in the user message. You are then "
    "given a list of clause questions, each labelled with a qa_id. Answer every "
    "qa_id ONLY with respect to the TARGET document: find the verbatim text span "
    "in the TARGET document that answers it. Treat every other document as "
    "unrelated background — even if a clause appears in one of them, it does NOT "
    "count for this task. If the clause is NOT present in the TARGET document, "
    "mark it absent — do not guess, fabricate, or borrow an answer from another "
    "document.\n\n"
    "Respond with ONLY a JSON object of this exact shape:\n"
    "{\"results\": [{\"qa_id\": str, \"present\": bool, "
    "\"answer\": str, \"document_id\": str|null}]}\n"
    "- Include every qa_id you were given, exactly as written.\n"
    "- present=false means the clause is absent from the TARGET document; then "
    "answer=\"\" and document_id=null.\n"
    "- answer must be copied verbatim from the TARGET document text.\n"
    "- document_id must be the id of the document the answer came from "
    "(this should be the TARGET document for any present=true answer)."
)


# MAUD is multiple choice, not extractive QA: the model is shown a fixed,
# lettered option set per question and must pick the single option that is
# correct for the TARGET agreement, returning its letter. (MAUD cells from
# build_context.py carry focus="labels"; CUAD cells use SYSTEM_PROMPT.)
MC_SYSTEM_PROMPT = (
    "You are a precise legal contract analyst answering MULTIPLE-CHOICE "
    "questions about merger agreements. You are given one or more documents, "
    "each wrapped in <DOCUMENT id=\"...\"> tags. Exactly ONE of them is the "
    "TARGET agreement; its id is named in the user message. Each question lists "
    "a fixed set of answer options labelled (A), (B), (C), … "
    "For each qa_id, decide which SINGLE option is correct for the TARGET "
    "agreement and return that option's LETTER. Base your answer ONLY on the "
    "TARGET agreement; ignore every other document in the window.\n\n"
    "Respond with ONLY a JSON object of this exact shape:\n"
    "{\"results\": [{\"qa_id\": str, \"answer\": str, "
    "\"document_id\": str|null}]}\n"
    "- Include every qa_id you were given, exactly as written.\n"
    "- answer MUST be exactly one option letter (e.g. \"A\"); do not paraphrase "
    "the option text.\n"
    "- Choose exactly one option per question — pick the best option even if "
    "you are uncertain; never leave it blank.\n"
    "- document_id must be the id of the TARGET <DOCUMENT> your answer is "
    "based on."
)


def is_mc_cell(cell: dict) -> bool:
    """True for MAUD multiple-choice cells (focus='labels')."""
    return cell.get("focus") == "labels"


def _span_user(cell: dict, questions: list[dict]) -> str:
    target = cell["target_document_id"]
    lines = [f'{q["qa_id"]} | category="{q["category"]}" | {q["question"]}'
             for q in questions]
    return (f'TARGET DOCUMENT id: "{target}"\n'
            f"Answer every question ONLY about this document; treat all other "
            f"documents as unrelated background.\n\n"
            f"DOCUMENTS:\n{cell['context']}\n\n"
            f'CLAUSE QUESTIONS (answer every qa_id about TARGET "{target}"):\n'
            + "\n".join(lines) + "\n\nReturn the JSON object now.")


def _mc_user(cell: dict, questions: list[dict]) -> str:
    target = cell["target_document_id"]
    blocks = []
    for q in questions:
        options = q.get("answer_options") or q.get("gold_answers") or []
        lines = [f'{q["qa_id"]} | category="{q["category"]}" | {q["question"]}']
        lines += [f"  ({option_label(i)}) {opt}" for i, opt in enumerate(options)]
        blocks.append("\n".join(lines))
    return (f'TARGET AGREEMENT id: "{target}"\n'
            f"Answer every question ONLY about this agreement; ignore all other "
            f"documents.\n\n"
            f"DOCUMENTS:\n{cell['context']}\n\n"
            f'MULTIPLE-CHOICE QUESTIONS (answer every qa_id about TARGET '
            f'"{target}" with one option letter):\n' + "\n\n".join(blocks) +
            "\n\nReturn the JSON object now.")


def build_messages(cell: dict, questions: list[dict]) -> list[dict]:
    """Render one prompt for a batch of questions over the cell's documents."""
    if is_mc_cell(cell):
        return [{"role": "system", "content": MC_SYSTEM_PROMPT},
                {"role": "user", "content": _mc_user(cell, questions)}]
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _span_user(cell, questions)}]


def chunk_questions(questions: list[dict], size: int) -> list[list[dict]]:
    """Split into batches of `size`; size<=0 or >=len means one batch (all)."""
    if not size or size <= 0 or size >= len(questions):
        return [list(questions)]
    return [questions[i:i + size] for i in range(0, len(questions), size)]


def _mock_mc_letter(q: dict) -> str:
    """The option letter of q's gold answer (exercises the letter->option path)."""
    options = q.get("answer_options") or q.get("gold_answers") or []
    golds = q.get("gold_answers") or [a["text"] for a in q.get("answers", [])]
    for i, opt in enumerate(options):
        if any(_normalize(opt) == _normalize(g) for g in golds):
            return option_label(i)
    return golds[0] if golds else ""


def mock_call(cell: dict, questions: list[dict]) -> dict:
    """Offline stand-in for call_openrouter: gold-perfect answers, no API."""
    mc = is_mc_cell(cell)
    results = []
    for q in questions:
        if mc:  # MAUD: return the gold option's letter
            results.append({"qa_id": q["qa_id"], "answer": _mock_mc_letter(q),
                            "document_id": cell["target_document_id"]})
        elif q["is_impossible"]:
            results.append({"qa_id": q["qa_id"], "present": False,
                            "answer": "", "document_id": None})
        else:
            results.append({"qa_id": q["qa_id"], "present": True,
                            "answer": q["answers"][0]["text"],
                            "document_id": cell["target_document_id"]})
    return {
        "content": json.dumps({"results": results}),
        "usage": {"prompt_tokens": cell.get("actual_tokens", 0),  # context re-sent
                  "completion_tokens": len(results) * 8, "cost": 0.0},
        "latency_s": 0.0,
        "error": None,
    }


def _fmt(v) -> str:
    """None-safe number formatting for console status lines."""
    return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)


def _reasoning_config(effort: str | None) -> dict | None:
    """Map the --reasoning-effort flag to OpenRouter's `reasoning` field.

    low/medium/high -> {"effort": ...}; "none" -> disable thinking entirely;
    None (flag omitted) -> send nothing, leaving the model's own default.
    """
    if not effort:
        return None
    if effort == "none":
        return {"enabled": False}
    return {"effort": effort}


# --------------------------------------------------------------------------- #
# Cost estimation
# --------------------------------------------------------------------------- #
MODELS_URL = "https://openrouter.ai/api/v1/models"
CHARS_PER_TOKEN = 4  # input-token approximation; matches build_context.py


def approx_tokens(text: str) -> int:
    """Approximate token count for prompt sizing (chars / CHARS_PER_TOKEN)."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def _to_float(v) -> float:
    """Coerce OpenRouter's string pricing fields to float (None/'' -> 0.0)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def fetch_model_pricing(slugs: list[str], timeout: float = 30.0) -> dict:
    """Map each slug -> per-token USD pricing from OpenRouter's public catalogue.

    Returns {slug: {"prompt": float, "completion": float, "request": float}};
    a slug OpenRouter doesn't list maps to None so the caller can flag it. No
    API key required — the /models endpoint is public.
    """
    req = urllib.request.Request(MODELS_URL, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8")).get("data", [])
    catalogue = {m.get("id"): (m.get("pricing") or {}) for m in data}

    pricing: dict = {}
    for slug in slugs:
        p = catalogue.get(slug)
        pricing[slug] = None if p is None else {
            "prompt": _to_float(p.get("prompt")),
            "completion": _to_float(p.get("completion")),
            "request": _to_float(p.get("request")),
        }
    return pricing


def estimate_token_plan(cells: list[dict], runs: int,
                        worst_output_per_call: int,
                        best_output_per_call: int) -> dict:
    """Input/request totals + best- and worst-case output totals for the run.

    Each cell is sent in ONE call with all of its questions at once (no
    batching — the question count is capped upstream at context-build time), so
    there is exactly one request per cell and the full context is sent once per
    cell. Output is bracketed PER CALL by a flat token count, independent of how
    many questions the call carries:
      * worst case  — the call saturates the cap (worst_output_per_call).
      * best case   — a short single response (best_output_per_call).
    Input tokens are approximated from the actually rendered prompts (system +
    context + questions) via chars/CHARS_PER_TOKEN. Within each run all questions
    are asked at once, so `runs` scales the whole plan linearly.
    """
    input_tokens = requests = output_worst = output_best = 0
    for cell in cells:
        questions = cell["questions"]
        msgs = build_messages(cell, questions)
        input_tokens += sum(approx_tokens(m["content"]) for m in msgs)
        output_worst += worst_output_per_call
        output_best += min(best_output_per_call, worst_output_per_call)
        requests += 1
    return {
        "input_tokens": input_tokens * runs,
        "output_worst": output_worst * runs,
        "output_best": output_best * runs,
        "requests": requests * runs,
    }


def print_cost_estimate(cells: list[dict], models: list[tuple[str, str]],
                        runs: int, max_output_tokens: int,
                        best_output_per_call: int) -> None:
    """Fetch live pricing and print a per-model best/worst-case cost table."""
    plan = estimate_token_plan(cells, runs, max_output_tokens,
                               best_output_per_call)
    print(f"\nCost estimate over {len(cells)} cells x {runs} run(s)")
    print(f"  one call per cell (all questions at once); "
          f"worst case: {max_output_tokens:,} output tokens/call; "
          f"best case: {best_output_per_call:,} output tokens/call")
    print(f"  Plan: {plan['requests']:,} requests; ~{plan['input_tokens']:,} "
          f"input tokens; output {plan['output_best']:,} (best) .. "
          f"{plan['output_worst']:,} (worst) "
          f"(input approx chars/{CHARS_PER_TOKEN}).")

    try:
        pricing = fetch_model_pricing([slug for _, slug in models])
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            OSError, ValueError) as e:
        raise SystemExit(f"error: could not fetch OpenRouter pricing: {e}")

    def _cost(p: dict, output_tokens: int) -> float:
        return (plan["input_tokens"] * p["prompt"]
                + output_tokens * p["completion"]
                + plan["requests"] * p["request"])

    print(f"\n  {'model':12} {'$/Mtok in':>10} {'$/Mtok out':>11} "
          f"{'best cost':>12} {'worst cost':>12}")
    total_best = total_worst = 0.0
    unknown: list[tuple[str, str]] = []
    for name, slug in models:
        p = pricing.get(slug)
        if p is None:
            unknown.append((name, slug))
            print(f"  {name:12} {'?':>10} {'?':>11} {'unknown':>12} {'unknown':>12}")
            continue
        best, worst = _cost(p, plan["output_best"]), _cost(p, plan["output_worst"])
        total_best += best
        total_worst += worst
        print(f"  {name:12} {p['prompt'] * 1e6:>10.3f} "
              f"{p['completion'] * 1e6:>11.3f} "
              f"{'$' + format(best, '.4f'):>12} {'$' + format(worst, '.4f'):>12}")
    print(f"  {'TOTAL':12} {'':>10} {'':>11} "
          f"{'$' + format(total_best, '.4f'):>12} "
          f"{'$' + format(total_worst, '.4f'):>12}")

    if unknown:
        listing = ", ".join(f"{n} ({s})" for n, s in unknown)
        print(f"\n  note: no OpenRouter pricing found for: {listing}\n"
              f"        verify the slug at https://openrouter.ai/models; "
              f"TOTALs exclude these.")


# --------------------------------------------------------------------------- #
# Per (cell x model x run) execution
# --------------------------------------------------------------------------- #
def run_cell_model(cell: dict, slug: str, batches: list[list[dict]],
                   api_key: str | None, args, mock: bool) -> dict:
    """Call the model over every question batch and score the merged result."""
    pred_by_qid: dict = {}
    unscored: set = set()
    raws, parseds, perrs, busages, blats, errors = [], [], [], [], [], []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
    total_latency = 0.0
    reasoning = _reasoning_config(getattr(args, "reasoning_effort", None))

    for qbatch in batches:
        if mock:
            resp = mock_call(cell, qbatch)
        else:
            assert api_key is not None
            resp = call_openrouter(slug, build_messages(cell, qbatch), api_key,
                                   args.max_tokens, args.temperature,
                                   args.timeout, args.retries, reasoning)
            # Optional inter-request pause (rate limiting). Units run
            # concurrently, so this only spaces a unit's own batches apart;
            # cap parallelism with --max-concurrency for account-wide limits.
            if args.sleep:
                time.sleep(args.sleep)

        raw = resp["content"]
        if raw is None:
            parsed, perr = None, resp["error"] or "empty response"
        else:
            parsed, perr = extract_json(raw)

        results = results_of(parsed)
        if results is None:
            if perr is None:
                perr = "no 'results' array in JSON"
            unscored.update(q["qa_id"] for q in qbatch)
        else:
            for r in results:
                if isinstance(r, dict) and "qa_id" in r:
                    pred_by_qid[str(r["qa_id"]).strip()] = r

        raws.append(raw)
        parseds.append(parsed)
        perrs.append(perr)
        busages.append(resp["usage"])
        blats.append(resp["latency_s"])
        if resp["error"]:
            errors.append(resp["error"])
        for k in total_usage:
            v = resp["usage"].get(k)
            if isinstance(v, (int, float)):
                total_usage[k] += v
        total_latency += resp["latency_s"] or 0

    return {
        "raw_output": raws,
        "parsed_output": parseds,
        "json_parse_error": perrs,
        "per_batch_usage": busages,
        "per_batch_latency_s": blats,
        "usage": total_usage,
        "latency_s": round(total_latency, 3),
        "error": "; ".join(errors) if errors else None,
        "metrics": score_cell(cell, pred_by_qid, unscored),
    }


# --------------------------------------------------------------------------- #
# Cell loading + resume
# --------------------------------------------------------------------------- #
def load_cells(prepared: Path, limit: int | None) -> list[Path]:
    manifest = prepared / "manifest.json"
    if manifest.exists():
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        paths = [Path(c["path"]) for c in meta.get("cells", []) if c.get("path")]
    else:
        paths = sorted((prepared / "cells").glob("*.json"))
    if not paths:
        raise SystemExit(
            f"error: no cells found under {prepared}. Run build_context.py first."
        )
    return paths[:limit] if limit else paths


def _clean_api_key(key: str) -> str:
    """Strip surrounding quotes and any invisible / non-ASCII contaminants.

    Pasting a key from a webpage often injects invisible characters (word
    joiners U+2060, narrow no-break spaces U+202F, zero-width spaces, smart
    quotes, BOM, ...) that are never part of a real key but crash urllib's
    latin-1 header encoding. A valid key is printable ASCII, so we keep only
    printable-ASCII characters and warn about anything removed.
    """
    raw = key.strip().strip("\"'")
    cleaned = "".join(c for c in raw if 33 <= ord(c) <= 126)
    dropped = [c for c in raw if not (33 <= ord(c) <= 126)]
    if dropped:
        codes = ", ".join(sorted({f"U+{ord(c):04X}" for c in dropped}))
        print(f"warning: removed {len(dropped)} hidden/non-ASCII char(s) "
              f"({codes}) from OPENROUTER_API_KEY — your .env has invisible "
              f"characters; consider re-typing the key.")
    if not cleaned:
        raise SystemExit("error: OPENROUTER_API_KEY is empty after cleaning.")
    return cleaned


def load_done(path: Path) -> tuple[set, list[dict]]:
    """Existing run records + the keys worth skipping on resume.

    Only SUCCESSFUL rows (scored > 0) mark a (model, cell_id, run_index) as
    done. Rows that errored or parse-failed are left out, so a plain re-run
    retries just those cells while keeping completed work.
    """
    done, records = set(), []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a half-written final line
        records.append(rec)
        if rec.get("metrics", {}).get("scored", 0) > 0:
            done.add((rec.get("model"), rec.get("cell_id"), rec.get("run_index")))
    return done, records


# --------------------------------------------------------------------------- #
# Per-dataset driver
# --------------------------------------------------------------------------- #
def process_dataset(dataset: str, args, models: list[tuple[str, str]],
                    api_key: str | None) -> None:
    """Plan + run one dataset, writing its results to data/results_<dataset>/."""
    prepared = args.prepared or prepared_dir(dataset)
    out = args.out or results_dir(dataset)
    cell_paths = load_cells(prepared, args.limit)
    total = len(models) * len(cell_paths) * args.runs

    print(f"\n=== {dataset.upper()} — {prepared} -> {out} ===")
    print(f"Models ({len(models)}):")
    for name, slug in models:
        print(f"  {name:12} {slug}")
    batch_desc = "all" if args.question_batch_size <= 0 else args.question_batch_size
    reason_desc = f"   Reasoning: {args.reasoning_effort}" if args.reasoning_effort else ""
    print(f"Cells: {len(cell_paths)}   Runs/cell: {args.runs}   "
          f"Questions/request: {batch_desc}   Total cell-runs: {total}{reason_desc}")

    if args.dry_run:
        first = json.loads(cell_paths[0].read_text(encoding="utf-8"))
        batches = chunk_questions(first["questions"], args.question_batch_size)
        msgs = build_messages(first, batches[0])
        user = msgs[1]["content"]
        kind = "multiple-choice" if is_mc_cell(first) else "spans"
        print(f"\n--- prompt preview | cell {first['cell_id']} | {kind} | "
              f"batch 1/{len(batches)} ({len(batches[0])} questions) ---")
        print("[system]")
        print(msgs[0]["content"])
        print(f"\n[user] ({len(user):,} chars; truncated to 1800)")
        print(user[:1800])
        if len(user) > 1800:
            print("...[truncated]...")
        print("\n(dry run — no requests sent)")
        return

    if args.estimate_cost:
        cells = [json.loads(p.read_text(encoding="utf-8")) for p in cell_paths]
        print_cost_estimate(cells, models, args.runs,
                            args.max_tokens, args.best_case_output_tokens)
        return

    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.jsonl"
    done, runs, mode = set(), [], "w"
    if runs_path.exists() and not args.overwrite:
        done, runs = load_done(runs_path)
        mode = "a"
        print(f"resume: {len(done)} existing run rows; skipping those "
              f"(use --overwrite to redo)")

    # Flatten every (cell x model x run) into an independent unit of work. Each
    # unit is one or more HTTP calls (one per question batch) and produces a
    # single run record; units share no state, so we fan them out across a
    # thread pool and let OpenRouter serve them in parallel. Cells are read
    # once here and reused across all their model/run units.
    units = []  # (key, name, slug, cell, run_idx, batches)
    for cell_path in cell_paths:
        cell = json.loads(cell_path.read_text(encoding="utf-8"))
        batches = chunk_questions(cell["questions"], args.question_batch_size)
        for name, slug in models:
            for run_idx in range(args.runs):
                key = (name, cell["cell_id"], run_idx)
                if key in done:
                    continue
                units.append((key, name, slug, cell, run_idx, batches))

    skipped = total - len(units)
    workers = max(1, args.max_concurrency)
    if skipped:
        print(f"resume: {skipped} cell-runs already done — skipping")
    print(f"\nDispatching {len(units)} cell-run(s) across {workers} "
          f"concurrent worker(s)...\n")

    with runs_path.open(mode, encoding="utf-8") as runs_fh, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        fut_to_unit = {}
        for unit in units:
            _key, _name, slug, cell, _run, batches = unit
            fut = pool.submit(run_cell_model, cell, slug, batches,
                              api_key, args, args.mock)
            fut_to_unit[fut] = unit

        # Assemble/score/write on the main thread as each unit returns; only
        # the I/O-bound model calls run in parallel, so runs.jsonl, `runs`, and
        # `done` are never touched from a worker thread.
        for i, fut in enumerate(as_completed(fut_to_unit), start=1):
            key, name, slug, cell, run_idx, batches = fut_to_unit[fut]
            try:
                res = fut.result()
            except Exception as e:  # never let one unit abort the whole sweep
                print(f"  [{i}/{len(units)}] {name:10} {cell['cell_id']:40} "
                      f"r{run_idx}  CRASH {type(e).__name__}: {e}")
                continue

            record = {
                "model": name, "model_slug": slug,
                "dataset": cell.get("dataset", dataset),
                "cell_id": cell["cell_id"], "modality": cell["modality"],
                "focus": cell["focus"], "budget_tokens": cell["budget_tokens"],
                "is_baseline": cell.get("is_baseline", False),
                "doc_index": cell.get("doc_index"), "run_index": run_idx,
                "reasoning_effort": args.reasoning_effort,
                "batch_size": (args.question_batch_size
                               if args.question_batch_size > 0
                               else len(cell["questions"])),
                "num_batches": len(batches),
                "batch_qa_ids": [[q["qa_id"] for q in b] for b in batches],
                "raw_output": res["raw_output"],
                "parsed_output": res["parsed_output"],
                "json_parse_error": res["json_parse_error"],
                "per_batch_usage": res["per_batch_usage"],
                "per_batch_latency_s": res["per_batch_latency_s"],
                "metrics": res["metrics"], "usage": res["usage"],
                "latency_s": res["latency_s"], "error": res["error"],
            }
            runs.append(record)
            done.add(key)
            runs_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            runs_fh.flush()

            m = res["metrics"]
            if m["scored"] == 0:
                status = "ERR" if res["error"] else "parse-fail"
            elif is_mc_cell(cell):
                status = f"acc={_fmt(m['mc_accuracy'])}"
                if m.get("wrong_doc_rate") is not None:
                    status += f" wrongD={_fmt(m['wrong_doc_rate'])}"
                if m["n_unscored"]:
                    status += f" unscored={m['n_unscored']}"
            else:
                status = (f"F1={_fmt(m['span_f1'])} "
                          f"abst={_fmt(m['abstention_rate'])}")
                if m["n_unscored"]:
                    status += f" unscored={m['n_unscored']}"
            print(f"  [{i}/{len(units)}] {name:10} {cell['cell_id']:40} "
                  f"r{run_idx}  {status}")

    summary = aggregate(runs)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if summary:
        with (out / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    errors = sum(1 for r in runs if r["error"])
    parse_fails = sum(1 for r in runs if r["metrics"].get("scored", 0) == 0)
    print(f"\nWrote {len(runs)} runs -> {runs_path}")
    print(f"Aggregated {len(summary)} (model x cell) rows -> "
          f"{out}/summary.json + summary.csv")
    if errors or parse_fails:
        print(f"  {errors} rows with request errors, {parse_fails} with no scored questions")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("cuad", "maud", "both"), default="both",
                        help="which dataset(s) to run (default: both — each is run "
                             "and scored into its own data/results_<dataset>/ stream)")
    parser.add_argument("--prepared", type=Path, default=None,
                        help="override the prepared cells dir (default: "
                             "data/prepared_<dataset>; requires a single --dataset)")
    parser.add_argument("--out", type=Path, default=None,
                        help="override the results output dir (default: "
                             "data/results_<dataset>; requires a single --dataset)")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                        help="friendly model names (default: full fleet)")
    parser.add_argument("--models-config", type=Path, default=None,
                        help="JSON {friendly: slug} to override/extend the registry")
    parser.add_argument("--prototype", action="store_true",
                        help=f"use only the cheap prototype model ({PROTOTYPE[1]})")
    parser.add_argument("--runs", type=int, default=1,
                        help="multiruns per cell for mean+/-stdev (default: 1)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap number of cells (cheap testing)")
    parser.add_argument("--question-batch-size", type=int, default=0,
                        help="questions per request (0 = all in one; e.g. 5)")
    parser.add_argument("--max-tokens", type=int, default=16000,
                        help="completion token CAP (default: 16000). It's a ceiling, "
                             "not a spend — but reasoning models burn it on hidden "
                             "thinking, so it must cover reasoning + the JSON answer "
                             "or you get an empty 'content'.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort",
                        choices=("low", "medium", "high", "none"), default=None,
                        help="control reasoning models' thinking budget via "
                             "OpenRouter (low/medium/high, or 'none' to disable). "
                             "Default: the model's own setting. Use 'low' to free "
                             "token budget for the answer and cut cost.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="seconds to wait between a unit's own batches "
                             "(rate limiting); use --max-concurrency to bound "
                             "account-wide load")
    parser.add_argument("--max-concurrency", type=int, default=8,
                        help="max cell-runs dispatched to OpenRouter at once "
                             "(default: 8). Lower it (e.g. 2) if a provider "
                             "rate-limits you; set 1 to run sequentially. "
                             "Free models (e.g. nemotron:free) have strict "
                             "per-minute caps — keep this low for those.")
    parser.add_argument("--overwrite", action="store_true",
                        help="redo all runs instead of resuming (skips by default)")
    parser.add_argument("--mock", action="store_true",
                        help="offline: gold-perfect answers, no API calls")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan + first rendered prompt, then exit")
    parser.add_argument("--estimate-cost", action="store_true",
                        help="estimate the $ cost of this run from live "
                             "OpenRouter pricing, bracketed best..worst case, "
                             "then exit")
    parser.add_argument("--best-case-output-tokens", type=int, default=256,
                        help="best-case output tokens PER CALL for "
                             "--estimate-cost (default: 256; worst case uses "
                             "the full --max-tokens cap per call)")
    args = parser.parse_args()

    models = [PROTOTYPE] if args.prototype else resolve_models(args.models,
                                                               args.models_config)

    datasets = DATASETS if args.dataset == "both" else (args.dataset,)
    # --prepared/--out name a single dir, so they only make sense for one dataset.
    if (args.prepared is not None or args.out is not None) and len(datasets) > 1:
        raise SystemExit("error: --prepared/--out override a single dataset's dir; "
                         "pass --dataset cuad or --dataset maud alongside them.")

    # A key is needed only for actual API calls: --mock/--dry-run/--estimate-cost
    # never hit OpenRouter (estimate uses the public, key-less pricing endpoint).
    api_key = None
    if not (args.mock or args.dry_run or args.estimate_cost):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit(
                "error: OPENROUTER_API_KEY not set. Export it for a real run, "
                "or pass --mock to test the pipeline offline."
            )
        api_key = _clean_api_key(api_key)
    if args.mock:
        print("[MOCK] gold-perfect answers, no API calls")

    for ds in datasets:
        process_dataset(ds, args, models, api_key)


if __name__ == "__main__":
    main()
