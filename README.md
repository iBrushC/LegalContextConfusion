# Legal Context Confusion
LLMs increasingly advertise context windows of 1M+ tokens. A large window, however, is not the same as reliable retrieval across that window: model accuracy tends to degrade as relevant information moves deeper into a long context and as surrounding irrelevant content grows, regardless of the advertised maximum. This study measures that degradation specifically for legal document understanding, using the [MAUD](https://www.atticusprojectai.org/maud/) and [CUAD](https://www.atticusprojectai.org/cuad/) datasets, in realistic scenarios where the relevant document is surrounded by confusing or irrelevant context.

This benchmark **does not** test more complex systems. No document pre-processing, no RAG, no agent-based retrieval, no chunking. The purpose is to measure raw model performance on legal information processing in the naive scenario: a user (or a simple AI system) pastes a pile of text into a chat window and asks a question about one document in it.

## Testing Modalities
The procedure is split into three independent tests, each isolating a different failure mode. They are deliberately separated because a single "accuracy dropped" number conflates phenomena that behave differently and matter differently in legal work.

### Rot
Rot is context within the window which is **unrelated filler**. It is non-related, non-legal text which cannot be confused with the datasets we're using. This tests pure length-to-degredation, how much the size and position of the source material alone affects accuracy. To test, we surround the relevant context with unrelated stories or internet content.

### Confusion
Confusion is context within the window which is **related and confusable filler**. It contains legal or legal-adjacent material which could cause worse performance. This tests not only context-length, but also correct identfication within context. Is the model able to get the correct answer from the correct document? To test this, we surround the correct source material with other legal and finance data. This is most similar to how models will perform in basic chat windows and basic AI systems, since often times there are sequences of context documents and related questions.

### Hallucination
Hallucination is tested by giving the model long contexts of related materials (similar to confusoin) and asked to answer questions about content which is **not present in the context**. The correct behavior is always to state that the context isn't found. To test this, we feed in long confusion context which is completely absent of the correct source material for the questions given.

## Datasets Used
**Contract Understanding Atticus Dataset (CUAD)**: A span-extraction task: given a clause category (e.g. Anti-Assignment, Change of Control, Governing Law), locate the relevant clause text in a commercial contract. Spans across ~41 label categories, and importantly includes negative examples where no relevant clause exists.
**Merger Agreement Understanding Dataset (MAUD)**: A multiple-choice reading comprehension task based on the American Bar Association's 2021 Public Target Deal Point Study, with ~47,000 labels across 152 merger agreements covering 92 deal-point questions per agreement.

### Known Limitation, Training Contamination
Since both datasets use publicly available documents and have been out since 2023, there is a high possibilty that at least some of the testing dataset will have been used in parts of the training for flagship models. Since we are working with highly-specific facts from individual documents which would make up, at most, a tiny fraction of the training data, we will assume that it highly unlikely that most models would be able to accurately recall individual facts from individual documents within their training data.

