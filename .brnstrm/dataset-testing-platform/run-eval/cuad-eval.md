**Span scoring.** Compare predicted span to gold over character offsets:

- Jaccard / token-F1 overlap, plus **AUPR** (CUAD's own metric).
- Negative categories: credit only an explicit "no such clause."

Applies across all three modalities for CUAD cells.
