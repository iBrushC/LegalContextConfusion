The single independent variable, applied to every modality × dataset cell:

- **Length:** filler token budgets 64k → 128k → 256k → 512k. Budget is the
  amount of rot/confusion filler **added around** the whole probe document
  (never a cap that truncates it), for both CUAD and MAUD.

Placement is fixed: the probe document always sits at the **end** of the
window, after all the filler. (The old head/middle/tail depth sweep has been
removed.)

Instead of a single probe, **N documents** per dataset (default 10, chosen
deterministically by seed) are each run through the same length sweep, with a
capped question set (≤32 per document). Their cells are pooled into one curve
per (model, modality, budget), so the band reflects cross-document variance.
Each document's question set is held constant so the only thing changing across
that document's cells is the filler length.
