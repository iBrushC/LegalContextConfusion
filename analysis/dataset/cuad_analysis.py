"""cuad_analysis.py — chart the shape of the CUAD dataset.

CUAD ships in SQuAD-v2 / extractive-QA form (see src/inspect_cuad.py for the
schema). This script turns that raw data into the figures we actually reason
about when designing the context-degradation grid: how big the input documents
are (the length sweep lives or dies on this), how many clause questions each
document carries, the answerable-vs-absent split (the native negatives are what
feed the hallucination / abstention tests), and how long the gold answer spans
are (span-F1 scoring is sensitive to this).

It reuses the loader + helpers from the sibling inspect_cuad module so the
notion of "category" / "negative" stays identical to the rest of the pipeline.

Figures written (PNG) under --out:
  * cuad_document_lengths.png    char + token length of each contract
  * cuad_questions_per_doc.png   clause questions per contract
  * cuad_answerable_vs_absent.png  overall pos/neg split + per-category neg rate
  * cuad_answer_spans.png        gold span length (chars/tokens) + answers/question

Token counts use tiktoken's cl100k_base when available (the project's reference
encoding); without tiktoken it falls back to the chars/4 approximation that the
rest of the codebase uses for budgeting, and the figures are labelled as such.

Usage:
    python analysis/dataset/cuad_analysis.py                       # test.json
    python analysis/dataset/cuad_analysis.py --file data/cuad/CUADv1.json
    python analysis/dataset/cuad_analysis.py --tokenizer approx    # skip tiktoken
    python analysis/dataset/cuad_analysis.py --out analysis/dataset/figures/cuad
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402

# Reuse the CUAD loader / category + negative logic from the sibling inspect
# script so "what counts as a negative" is defined in exactly one place. The
# repo root is two levels up from this file; put src/ on sys.path explicitly.
_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_SRC))
from inspect_cuad import category_of, is_negative, iter_qas, load_cuad  # noqa: E402

DEFAULT_FILE = Path("data/cuad/test.json")
DEFAULT_OUT = Path("analysis/dataset/figures/cuad")
CHARS_PER_TOKEN = 4  # fallback ratio, matches inspect_cuad / build_context


# --------------------------------------------------------------------------- #
# Tokenizing (tiktoken when present, chars/4 otherwise)
# --------------------------------------------------------------------------- #
def make_token_counter(kind: str):
    """Return (count_fn, label). label documents which method ended up used."""
    if kind == "tiktoken":
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return (lambda text: len(enc.encode(text)), "tiktoken cl100k_base")
        except ImportError:
            print("note: tiktoken not installed; falling back to chars/4 approx.")
            kind = "approx"
    if kind == "approx":
        return (lambda text: math.ceil(len(text) / CHARS_PER_TOKEN),
                f"approx (chars/{CHARS_PER_TOKEN})")
    raise SystemExit(f"error: unknown tokenizer {kind!r}")


# --------------------------------------------------------------------------- #
# Data extraction
# --------------------------------------------------------------------------- #
def collect(data: list[dict], count_tokens) -> dict:
    """Walk the dataset once and pull every distribution the figures need."""
    doc_chars: list[int] = []
    doc_tokens: list[int] = []
    questions_per_doc: list[int] = []

    span_chars: list[int] = []
    span_tokens: list[int] = []
    answers_per_q: list[int] = []

    pos_total = neg_total = 0
    pos_by_cat: Counter = Counter()
    neg_by_cat: Counter = Counter()

    for doc in data:
        # CUAD contracts are single-paragraph; join defensively if not.
        context = "".join(p.get("context", "") for p in doc.get("paragraphs", []))
        doc_chars.append(len(context))
        doc_tokens.append(count_tokens(context))
        questions_per_doc.append(
            sum(len(p.get("qas", [])) for p in doc.get("paragraphs", []))
        )

    for _doc, _para, qa in iter_qas(data):
        cat = category_of(qa)
        if is_negative(qa):
            neg_total += 1
            neg_by_cat[cat] += 1
            continue
        pos_total += 1
        pos_by_cat[cat] += 1
        answers = qa.get("answers", [])
        answers_per_q.append(len(answers))
        for ans in answers:
            text = ans.get("text", "")
            span_chars.append(len(text))
            span_tokens.append(count_tokens(text))

    return {
        "doc_chars": doc_chars,
        "doc_tokens": doc_tokens,
        "questions_per_doc": questions_per_doc,
        "span_chars": span_chars,
        "span_tokens": span_tokens,
        "answers_per_q": answers_per_q,
        "pos_total": pos_total,
        "neg_total": neg_total,
        "pos_by_cat": pos_by_cat,
        "neg_by_cat": neg_by_cat,
    }


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def _hist(ax, values: list[float], *, bins: int, color: str, title: str,
          xlabel: str, logx: bool = False) -> None:
    """Histogram with median/mean guide lines and a small stats caption."""
    if not values:
        ax.set_title(f"{title} (no data)")
        return
    if logx:
        lo = max(min(values), 1)
        edges = _log_bins(lo, max(values), bins)
        ax.hist(values, bins=edges, color=color, edgecolor="white", linewidth=0.4)
        ax.set_xscale("log")
    else:
        ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.4)

    med = statistics.median(values)
    mean = statistics.mean(values)
    ax.axvline(med, color="black", linestyle="--", linewidth=1,
               label=f"median {med:,.0f}")
    ax.axvline(mean, color="crimson", linestyle=":", linewidth=1,
               label=f"mean {mean:,.0f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.text(0.98, 0.97,
            f"n={len(values):,}\nmin {min(values):,.0f}\nmax {max(values):,.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.8))


def _log_bins(lo: float, hi: float, bins: int) -> list[float]:
    """Log-spaced bin edges; degrades to a single bin if lo==hi."""
    if hi <= lo:
        return [lo, lo + 1]
    step = (math.log10(hi) - math.log10(lo)) / bins
    return [10 ** (math.log10(lo) + i * step) for i in range(bins + 1)]


def _save(fig, out: Path, name: str, dpi: int) -> Path:
    path = out / name
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_document_lengths(d: dict, tok_label: str, out: Path, dpi: int) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    _hist(ax1, d["doc_chars"], bins=40, color="#4C72B0",
          title="CUAD — document length (characters)",
          xlabel="characters per contract", logx=True)
    _hist(ax2, d["doc_tokens"], bins=40, color="#55A868",
          title=f"CUAD — document length (tokens: {tok_label})",
          xlabel="tokens per contract", logx=True)
    fig.suptitle("Input document length distribution", fontweight="bold")
    _save(fig, out, "cuad_document_lengths.png", dpi)


def fig_questions_per_doc(d: dict, out: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    _hist(ax, d["questions_per_doc"], bins=30, color="#C44E52",
          title="CUAD — clause questions per contract",
          xlabel="questions per contract")
    _save(fig, out, "cuad_questions_per_doc.png", dpi)


def fig_answerable_vs_absent(d: dict, out: Path, dpi: int, top: int) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={"width_ratios": [1, 2]})

    # Left: overall positive / negative split.
    pos, neg = d["pos_total"], d["neg_total"]
    total = pos + neg or 1
    bars = ax1.bar(["answerable\n(positive)", "absent\n(negative)"], [pos, neg],
                   color=["#55A868", "#C44E52"])
    ax1.set_title("Overall question split")
    ax1.set_ylabel("questions")
    for b, v in zip(bars, [pos, neg]):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n{100*v/total:.0f}%",
                 ha="center", va="bottom", fontsize=9)

    # Right: per-category negative rate (the categories most often absent are the
    # richest source of native hallucination tests). Sort by neg rate, top N.
    cats = set(d["pos_by_cat"]) | set(d["neg_by_cat"])
    rows = []
    for c in cats:
        p, n = d["pos_by_cat"][c], d["neg_by_cat"][c]
        rows.append((c, n / (p + n) if (p + n) else 0.0, p + n))
    rows.sort(key=lambda r: r[1])
    rows = rows[-top:]  # highest neg-rate at the top of a horizontal bar chart

    # Numeric y positions (not the strings) so truncated category labels that
    # collide can't collapse or reorder the bars.
    labels = [r[0][:40] for r in rows]
    rates = [100 * r[1] for r in rows]
    y = range(len(labels))
    ax2.barh(y, rates, color="#C44E52")
    ax2.set_yticks(list(y))
    ax2.set_yticklabels(labels)
    ax2.set_title(f"Negative (clause-absent) rate by category — top {len(rows)}")
    ax2.set_xlabel("% of contracts where the clause is absent")
    ax2.set_xlim(0, 100)
    for i, (rate, tot) in enumerate(zip(rates, [r[2] for r in rows])):
        ax2.text(rate + 1, i, f"{rate:.0f}%  (n={tot})", va="center", fontsize=7)

    fig.suptitle("Answerable vs absent — native negatives feed hallucination tests",
                 fontweight="bold")
    _save(fig, out, "cuad_answerable_vs_absent.png", dpi)


def fig_answer_spans(d: dict, tok_label: str, out: Path, dpi: int) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))
    _hist(ax1, d["span_chars"], bins=40, color="#8172B3",
          title="CUAD — gold span length (characters)",
          xlabel="characters per span", logx=True)
    _hist(ax2, d["span_tokens"], bins=40, color="#CCB974",
          title=f"CUAD — gold span length (tokens: {tok_label})",
          xlabel="tokens per span", logx=True)
    # answers/question is small-integer; plot as integer-aligned bars.
    if d["answers_per_q"]:
        counts = Counter(d["answers_per_q"])
        xs = sorted(counts)
        ax3.bar(xs, [counts[x] for x in xs], color="#64B5CD", edgecolor="white")
        ax3.set_xticks(xs)
        med = statistics.median(d["answers_per_q"])
        ax3.axvline(med, color="black", linestyle="--", linewidth=1,
                    label=f"median {med:.0f}")
        ax3.legend(fontsize=8)
    ax3.set_title("CUAD — gold spans per answerable question")
    ax3.set_xlabel("answers per question")
    ax3.set_ylabel("count")
    fig.suptitle("Gold answer spans (answerable questions only)", fontweight="bold")
    _save(fig, out, "cuad_answer_spans.png", dpi)


# --------------------------------------------------------------------------- #
# Console summary (so the script is useful even without opening the PNGs)
# --------------------------------------------------------------------------- #
def _stat(label: str, v: list[float]) -> None:
    if not v:
        print(f"  {label}: (none)")
        return
    print(f"  {label}: n={len(v):,}  min={min(v):,.0f}  "
          f"median={statistics.median(v):,.0f}  mean={statistics.mean(v):,.0f}  "
          f"max={max(v):,.0f}")


def print_summary(d: dict, tok_label: str) -> None:
    print(f"\nCUAD distributions  (tokens via {tok_label})")
    print("-" * 48)
    _stat("document chars", d["doc_chars"])
    _stat("document tokens", d["doc_tokens"])
    _stat("questions / contract", d["questions_per_doc"])
    _stat("gold span chars", d["span_chars"])
    _stat("gold span tokens", d["span_tokens"])
    _stat("answers / question", d["answers_per_q"])
    total = d["pos_total"] + d["neg_total"] or 1
    print(f"  answerable: {d['pos_total']:,} ({100*d['pos_total']/total:.0f}%)  "
          f"absent: {d['neg_total']:,} ({100*d['neg_total']/total:.0f}%)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE,
                        help=f"CUAD JSON to analyze (default: {DEFAULT_FILE})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"figure output dir (default: {DEFAULT_OUT})")
    parser.add_argument("--tokenizer", choices=("tiktoken", "approx"),
                        default="tiktoken",
                        help="token counter (default: tiktoken cl100k_base)")
    parser.add_argument("--top-categories", type=int, default=25,
                        help="categories to show in the neg-rate chart (default: 25)")
    parser.add_argument("--dpi", type=int, default=140,
                        help="figure resolution (default: 140)")
    args = parser.parse_args()

    raw = load_cuad(args.file)
    data = raw.get("data", [])
    if not data:
        raise SystemExit(f"error: no 'data' entries found in {args.file}")

    count_tokens, tok_label = make_token_counter(args.tokenizer)
    print(f"Analyzing {len(data):,} contracts from {args.file} "
          f"(tokens via {tok_label})...")
    d = collect(data, count_tokens)
    print_summary(d, tok_label)

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting figures to {args.out}/")
    fig_document_lengths(d, tok_label, args.out, args.dpi)
    fig_questions_per_doc(d, args.out, args.dpi)
    fig_answerable_vs_absent(d, args.out, args.dpi, args.top_categories)
    fig_answer_spans(d, tok_label, args.out, args.dpi)
    print("\nDone.")


if __name__ == "__main__":
    main()
