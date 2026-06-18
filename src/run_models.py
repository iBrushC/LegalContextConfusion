"""run_models.py — run models over the prepared CUAD cells and score them.

This is the Run + Eval stage and the orchestrator. The pieces it coordinates
live in sibling modules:
  * openrouter_client.call_openrouter — the OpenRouter HTTP client.
  * score_outputs.{extract_json,score_cell,aggregate} — parsing + scoring.

For each prepared cell (from build_context.py) it asks every clause question and
scores the answers. Questions can be sent all in one prompt (default) or split
into smaller batches (--question-batch-size); the big context is re-sent per
batch, so fewer batches = cheaper.

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

Models go through OpenRouter. The model SLUGS below are PLACEHOLDERS for several
future models and MUST be verified against https://openrouter.ai/models.
Override any of them with --models-config <json> (a {"friendly": "provider/slug"}
map) without editing this file.

Auth: set OPENROUTER_API_KEY in the environment for real runs. Use --mock to
exercise the full scoring pipeline offline (no API, no cost).

Usage:
    python src/run_models.py --mock                       # offline pipeline test
    python src/run_models.py --dry-run                    # plan + first prompt preview
    python src/run_models.py --prototype --limit 3        # cheap real smoke test
    python src/run_models.py --models claude gemini --runs 3
    python src/run_models.py --question-batch-size 5      # 5 questions per call
    python src/run_models.py --overwrite                  # redo instead of resume
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Sibling modules (src/ is on sys.path when run directly; make it explicit).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openrouter_client import call_openrouter  # noqa: E402
from score_outputs import aggregate, extract_json, results_of, score_cell  # noqa: E402

DEFAULT_PREPARED = Path("data/prepared")
DEFAULT_OUT = Path("data/results")

# Friendly name -> OpenRouter slug. SLUGS ARE PLACEHOLDERS — verify each at
# https://openrouter.ai/models and/or override via --models-config.
MODEL_REGISTRY = {
    "claude":   "anthropic/claude-opus-4.8",                  # VERIFY
    "chatgpt":  "openai/gpt-5.5",                         # VERIFY
    "gemini":   "google/gemini-3.1-pro",                      # VERIFY
    "deepseek": "deepseek/deepseek-v4-pro",                   # VERIFY
    "nemotron": "nvidia/nemotron-3-ultra-550b-a55b:free",     # VERIFY
    "mimo":     "xiaomi/mimo-v2.5-pro",                       # VERIFY
    "glm":      "z-ai/glm-5.2",                               # VERIFY
}
ALIASES = {
    "gpt": "chatgpt", "openai": "chatgpt",
    "opus": "claude", "anthropic": "claude",
    "google": "gemini",
    "nemotron3": "nemotron", "nemotron-3-ultra": "nemotron",
    "mimo-v2.5-pro": "mimo", "mimo2.5": "mimo",
    "glm5": "glm", "glm-5.2": "glm",
}
# Cheapest model for prototyping (reuse-base-procedure card). VERIFY.
PROTOTYPE = ("deepseek-flash", "deepseek/deepseek-v4-flash")
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
    "documents, each wrapped in <document id=\"...\"> tags, followed by a list "
    "of clause questions, each labelled with a qa_id. For each qa_id, find the "
    "verbatim text span in the documents that answers it. If the clause is NOT "
    "present in any document, mark it absent — do not guess or fabricate.\n\n"
    "Respond with ONLY a JSON object of this exact shape:\n"
    "{\"results\": [{\"qa_id\": str, \"present\": bool, "
    "\"answer\": str, \"document_id\": str|null}]}\n"
    "- Include every qa_id you were given, exactly as written.\n"
    "- present=false means the clause is absent; then answer=\"\" and "
    "document_id=null.\n"
    "- answer must be copied verbatim from the document text.\n"
    "- document_id must be the id of the <document> the answer came from."
)


def build_messages(cell: dict, questions: list[dict]) -> list[dict]:
    """Render one prompt for a batch of questions over the cell's documents."""
    lines = [f'{q["qa_id"]} | category="{q["category"]}" | {q["question"]}'
             for q in questions]
    user = (f"DOCUMENTS:\n{cell['context']}\n\n"
            f"CLAUSE QUESTIONS (answer every qa_id):\n" + "\n".join(lines) +
            "\n\nReturn the JSON object now.")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def chunk_questions(questions: list[dict], size: int) -> list[list[dict]]:
    """Split into batches of `size`; size<=0 or >=len means one batch (all)."""
    if not size or size <= 0 or size >= len(questions):
        return [list(questions)]
    return [questions[i:i + size] for i in range(0, len(questions), size)]


def mock_call(cell: dict, questions: list[dict]) -> dict:
    """Offline stand-in for call_openrouter: gold-perfect answers, no API."""
    results = []
    for q in questions:
        if q["is_impossible"]:
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

    for qbatch in batches:
        if mock:
            resp = mock_call(cell, qbatch)
        else:
            assert api_key is not None
            resp = call_openrouter(slug, build_messages(cell, qbatch), api_key,
                                   args.max_tokens, args.temperature,
                                   args.timeout, args.retries)
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


def load_done(path: Path) -> tuple[set, list[dict]]:
    """Existing (model, cell_id, run_index) keys + records, for resuming."""
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
        done.add((rec.get("model"), rec.get("cell_id"), rec.get("run_index")))
    return done, records


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED,
                        help=f"prepared cells dir (default: {DEFAULT_PREPARED})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"results output dir (default: {DEFAULT_OUT})")
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
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="completion token cap (default: 4096)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="seconds to wait between requests (rate limiting)")
    parser.add_argument("--overwrite", action="store_true",
                        help="redo all runs instead of resuming (skips by default)")
    parser.add_argument("--mock", action="store_true",
                        help="offline: gold-perfect answers, no API calls")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan + first rendered prompt, then exit")
    args = parser.parse_args()

    models = [PROTOTYPE] if args.prototype else resolve_models(args.models,
                                                               args.models_config)
    cell_paths = load_cells(args.prepared, args.limit)
    total = len(models) * len(cell_paths) * args.runs

    print(f"Models ({len(models)}):")
    for name, slug in models:
        flag = "  <- VERIFY slug" if slug in MODEL_REGISTRY.values() else ""
        print(f"  {name:12} {slug}{flag}")
    batch_desc = "all" if args.question_batch_size <= 0 else args.question_batch_size
    print(f"Cells: {len(cell_paths)}   Runs/cell: {args.runs}   "
          f"Questions/request: {batch_desc}   Total cell-runs: {total}")

    if args.dry_run:
        first = json.loads(cell_paths[0].read_text(encoding="utf-8"))
        batches = chunk_questions(first["questions"], args.question_batch_size)
        msgs = build_messages(first, batches[0])
        user = msgs[1]["content"]
        print(f"\n--- prompt preview | cell {first['cell_id']} | "
              f"batch 1/{len(batches)} ({len(batches[0])} questions) ---")
        print("[system]")
        print(SYSTEM_PROMPT)
        print(f"\n[user] ({len(user):,} chars; truncated to 1800)")
        print(user[:1800])
        if len(user) > 1800:
            print("...[truncated]...")
        print("\n(dry run — no requests sent)")
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not args.mock and not api_key:
        raise SystemExit(
            "error: OPENROUTER_API_KEY not set. Export it for a real run, "
            "or pass --mock to test the pipeline offline."
        )
    if args.mock:
        print("\n[MOCK] gold-perfect answers, no API calls")

    args.out.mkdir(parents=True, exist_ok=True)
    runs_path = args.out / "runs.jsonl"
    done, runs, mode = set(), [], "w"
    if runs_path.exists() and not args.overwrite:
        done, runs = load_done(runs_path)
        mode = "a"
        print(f"resume: {len(done)} existing run rows; skipping those "
              f"(use --overwrite to redo)")
    print()

    with runs_path.open(mode, encoding="utf-8") as runs_fh:
        for cell_path in cell_paths:
            cell = json.loads(cell_path.read_text(encoding="utf-8"))
            batches = chunk_questions(cell["questions"], args.question_batch_size)
            for name, slug in models:
                for run_idx in range(args.runs):
                    key = (name, cell["cell_id"], run_idx)
                    if key in done:
                        print(f"  {name:10} {cell['cell_id']:40} r{run_idx}  skip")
                        continue

                    res = run_cell_model(cell, slug, batches, api_key, args, args.mock)
                    record = {
                        "model": name, "model_slug": slug,
                        "cell_id": cell["cell_id"], "modality": cell["modality"],
                        "focus": cell["focus"], "budget_tokens": cell["budget_tokens"],
                        "position": cell["position"], "run_index": run_idx,
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
                    else:
                        status = (f"F1={_fmt(m['span_f1'])} "
                                  f"abst={_fmt(m['abstention_rate'])}")
                        if m["n_unscored"]:
                            status += f" unscored={m['n_unscored']}"
                    print(f"  {name:10} {cell['cell_id']:40} r{run_idx}  {status}")

    summary = aggregate(runs)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if summary:
        with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    errors = sum(1 for r in runs if r["error"])
    parse_fails = sum(1 for r in runs if r["metrics"].get("scored", 0) == 0)
    print(f"\nWrote {len(runs)} runs -> {runs_path}")
    print(f"Aggregated {len(summary)} (model x cell) rows -> "
          f"{args.out}/summary.json + summary.csv")
    if errors or parse_fails:
        print(f"  {errors} rows with request errors, {parse_fails} with no scored questions")


if __name__ == "__main__":
    main()
