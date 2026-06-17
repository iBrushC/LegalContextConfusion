**Task:** span extraction.

- **Probe doc:** one commercial contract.
- **Question:** a clause category (Anti-Assignment, Change of Control, Governing Law, …; ~41 categories).
- **Gold answer:** the verbatim clause span(s), *or* "no such clause" for the native **negative** examples.

The built-in negatives (category absent from a contract) feed Hallucination testing directly — no synthetic construction needed.
