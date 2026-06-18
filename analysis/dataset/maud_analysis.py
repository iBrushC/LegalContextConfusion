"""maud_analysis.py — chart the shape of the MAUD dataset.

MAUD (Merger Agreement Understanding Dataset) is multiple-choice reading
comprehension over merger agreements: ~92 ABA deal-point questions answered
against 152 agreements (~47k labels). Unlike CUAD, the input documents live as
plain-text files (data/maud/contracts/contract_*.txt); the labelled Q/A live in
the MAUD_*.csv splits, where each row is one (contract, question/subquestion,
answer) triple.

This script produces the figures we reason about when slotting MAUD into the
context-degradation grid: how big the agreements are (the length sweep), how
many deal-point questions each agreement carries, and the MCQ answer-option
balance (exact-match scoring on an imbalanced option set is easy to fool, so we
want to see the class skew before sampling a "balanced subset").

CSV notes:
  * Questions per document are counted from data_type == "main" only, and the
    synthetic "<RARE_ANSWERS>" pseudo-contract is dropped — it is an aggregation
    bucket, not a real agreement.
  * Document lengths are measured on the actual contract .txt files, not the CSV
    `text` column (which holds only the extracted supporting passage).

Token counts use tiktoken's cl100k_base when available; otherwise the chars/4
approximation, with figures labelled accordingly.

Figures written (PNG) under --out:
  * maud_document_lengths.png    char + token length of each agreement .txt
  * maud_questions_per_doc.png   deal-point questions per agreement
  * maud_answer_balance.png      answer-option skew + options/question + per-category

Usage:
    python analysis/dataset/maud_analysis.py
    python analysis/dataset/maud_analysis.py --csv data/maud/MAUD_test.csv
    python analysis/dataset/maud_analysis.py --tokenizer approx
    python analysis/dataset/maud_analysis.py --out analysis/dataset/figures/maud
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_CSV = Path("data/maud/MAUD_train.csv")
DEFAULT_CONTRACTS = Path("data/maud/contracts")
DEFAULT_OUT = Path("analysis/dataset/figures/maud")
CHARS_PER_TOKEN = 4  # fallback ratio, matches the rest of the codebase

# CSV sentinel: an aggregation bucket of rare answers, not a real agreement.
RARE_ANSWERS = "<RARE_ANSWERS>"


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
def collect_documents(contracts_dir: Path, count_tokens) -> dict:
    """Measure char + token length of every contract_*.txt agreement."""
    files = sorted(contracts_dir.glob("contract_*.txt"))
    if not files:
        raise SystemExit(
            f"error: no contract_*.txt files under {contracts_dir}. "
            f"Pass --contracts with the MAUD contracts directory."
        )
    doc_chars: list[int] = []
    doc_tokens: list[int] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        doc_chars.append(len(text))
        doc_tokens.append(count_tokens(text))
    return {"n_files": len(files), "doc_chars": doc_chars, "doc_tokens": doc_tokens}


def collect_questions(df: pd.DataFrame) -> dict:
    """Questions-per-agreement + MCQ answer balance from the labelled CSV."""
    real = df[df["contract_name"] != RARE_ANSWERS]
    main = real[real["data_type"] == "main"]
    if main.empty:  # some splits may not carry the 'main' tag — fall back
        main = real

    # Deal-point questions per agreement: distinct (question, subquestion) pairs,
    # since one deal-point question can fan out into several subquestions.
    per_doc = (main.groupby("contract_name")[["question", "subquestion"]]
               .apply(lambda g: g.drop_duplicates().shape[0]))
    questions_per_doc = per_doc.tolist()

    # Answer-option balance over the full real label set (skew is what matters).
    answer_counts = Counter(real["answer"].astype(str))

    # Options actually observed per deal-point question (MCQ branching factor).
    options_per_q = (real.groupby("question")["answer"]
                     .nunique().tolist())

    # Labels per top-level category.
    labels_per_cat = Counter(real["category"].astype(str))

    return {
        "n_contracts": main["contract_name"].nunique(),
        "n_questions": real["question"].nunique(),
        "n_labels": len(real),
        "questions_per_doc": questions_per_doc,
        "answer_counts": answer_counts,
        "options_per_q": options_per_q,
        "labels_per_cat": labels_per_cat,
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
def fig_document_lengths(docs: dict, tok_label: str, out: Path, dpi: int) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    _hist(ax1, docs["doc_chars"], bins=30, color="#4C72B0",
          title="MAUD — agreement length (characters)",
          xlabel="characters per agreement")
    _hist(ax2, docs["doc_tokens"], bins=30, color="#55A868",
          title=f"MAUD — agreement length (tokens: {tok_label})",
          xlabel="tokens per agreement")
    fig.suptitle("Input document length distribution "
                 f"({docs['n_files']} agreements)", fontweight="bold")
    _save(fig, out, "maud_document_lengths.png", dpi)


def fig_questions_per_doc(q: dict, out: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    _hist(ax, q["questions_per_doc"], bins=25, color="#C44E52",
          title="MAUD — deal-point questions per agreement",
          xlabel="distinct (question, subquestion) pairs per agreement")
    _save(fig, out, "maud_questions_per_doc.png", dpi)


def fig_answer_balance(q: dict, out: Path, dpi: int, top: int) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.5))

    # Left: most common answer options (MCQ class skew). Top N by frequency.
    # Use numeric y positions (not the strings) so truncated labels that collide
    # can't collapse or reorder the bars.
    common = q["answer_counts"].most_common(top)
    labels = [a[:38] for a, _ in common][::-1]
    counts = [c for _, c in common][::-1]
    y = range(len(labels))
    ax1.barh(y, counts, color="#8172B3")
    ax1.set_yticks(list(y))
    ax1.set_yticklabels(labels)
    ax1.set_title(f"Most common answer options — top {len(common)}")
    ax1.set_xlabel("labels")
    ax1.tick_params(axis="y", labelsize=7)

    # Middle: options observed per deal-point question (MCQ branching factor).
    if q["options_per_q"]:
        oc = Counter(q["options_per_q"])
        xs = sorted(oc)
        ax2.bar(xs, [oc[x] for x in xs], color="#CCB974", edgecolor="white")
        ax2.set_xticks(xs)
        med = statistics.median(q["options_per_q"])
        ax2.axvline(med, color="black", linestyle="--", linewidth=1,
                    label=f"median {med:.0f}")
        ax2.legend(fontsize=8)
    ax2.set_title("Answer options per question")
    ax2.set_xlabel("distinct answer options")
    ax2.set_ylabel("number of questions")

    # Right: labels per top-level category.
    cats = q["labels_per_cat"].most_common()
    clabels = [c[:34] for c, _ in cats][::-1]
    cvals = [v for _, v in cats][::-1]
    cy = range(len(clabels))
    ax3.barh(cy, cvals, color="#64B5CD")
    ax3.set_yticks(list(cy))
    ax3.set_yticklabels(clabels)
    ax3.set_title("Labels per category")
    ax3.set_xlabel("labels")
    ax3.tick_params(axis="y", labelsize=8)
    for i, v in enumerate(cvals):
        ax3.text(v, i, f" {v:,}", va="center", fontsize=7)

    fig.suptitle("MAUD answer-option balance "
                 "(class skew matters for exact-match scoring)", fontweight="bold")
    _save(fig, out, "maud_answer_balance.png", dpi)


# --------------------------------------------------------------------------- #
# Console summary
# --------------------------------------------------------------------------- #
def _stat(label: str, v: list[float]) -> None:
    if not v:
        print(f"  {label}: (none)")
        return
    print(f"  {label}: n={len(v):,}  min={min(v):,.0f}  "
          f"median={statistics.median(v):,.0f}  mean={statistics.mean(v):,.0f}  "
          f"max={max(v):,.0f}")


def print_summary(docs: dict, q: dict, tok_label: str) -> None:
    print(f"\nMAUD distributions  (tokens via {tok_label})")
    print("-" * 48)
    print(f"  agreements (.txt): {docs['n_files']:,}   "
          f"labelled contracts: {q['n_contracts']:,}   "
          f"deal-point questions: {q['n_questions']:,}   "
          f"labels: {q['n_labels']:,}")
    _stat("document chars", docs["doc_chars"])
    _stat("document tokens", docs["doc_tokens"])
    _stat("questions / agreement", q["questions_per_doc"])
    _stat("options / question", q["options_per_q"])
    top = q["answer_counts"].most_common(3)
    print(f"  top answer options: " +
          ", ".join(f"{a!r}={c:,}" for a, c in top))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help=f"MAUD labels CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS,
                        help=f"contract .txt dir (default: {DEFAULT_CONTRACTS})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"figure output dir (default: {DEFAULT_OUT})")
    parser.add_argument("--tokenizer", choices=("tiktoken", "approx"),
                        default="tiktoken",
                        help="token counter (default: tiktoken cl100k_base)")
    parser.add_argument("--top-answers", type=int, default=20,
                        help="answer options to show in the balance chart (default: 20)")
    parser.add_argument("--dpi", type=int, default=140,
                        help="figure resolution (default: 140)")
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"error: {args.csv} not found. Pass --csv with a MAUD split.")

    count_tokens, tok_label = make_token_counter(args.tokenizer)
    print(f"Loading agreements from {args.contracts} and labels from {args.csv} "
          f"(tokens via {tok_label})...")
    docs = collect_documents(args.contracts, count_tokens)
    df = pd.read_csv(args.csv)
    q = collect_questions(df)
    print_summary(docs, q, tok_label)

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting figures to {args.out}/")
    fig_document_lengths(docs, tok_label, args.out, args.dpi)
    fig_questions_per_doc(q, args.out, args.dpi)
    fig_answer_balance(q, args.out, args.dpi, args.top_answers)
    print("\nDone.")


if __name__ == "__main__":
    main()
