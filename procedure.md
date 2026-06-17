# Legal Context Confusion

This project tests how long-context LLMs handle legal Q&A when relevant legal evidence is surrounded by irrelevant, related, or missing context.

Failure modes:

1. Rot — unrelated filler degrades performance.
2. Confusion — related distractors cause wrong-source answers.
3. Hallucination — models answer when the document or answer is absent.

Dataset v0: CUAD.
Models v0: OpenAI, Claude, Gemini.
