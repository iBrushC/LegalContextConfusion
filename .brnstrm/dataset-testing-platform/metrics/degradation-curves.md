Per **modality × dataset** cell, plot the eval score against:

- **context length** — the filler budget added around the probe (64k → 512k).

The probe always sits at the end of the window, so there is no depth axis. The
N sampled documents are pooled into one curve per (model, modality, budget),
treating documents as repeated trials.

Carry the base-procedure signatures too: time, tokens, cost. Multirun N and the
N documents together → report **mean ± stdev** (the band is cross-document +
multirun spread). The deliverable is the degradation curve, not a single
accuracy number.
