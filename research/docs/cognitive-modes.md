# Cognitive Processing Modes

For the first arXiv study, use a deliberately narrow taxonomy.

## Decision-oriented

An operation whose primary output is a bounded choice, score, ranking, judgment, or routing action over a known semantic space.

Candidate PaperQA examples:
- selecting the next tool/action,
- judging evidence relevance,
- selecting/ranking evidence,
- deciding whether additional evidence is needed.

## Generative

An operation whose primary output is an open-ended sequence constructed from context.

Candidate PaperQA examples:
- generating a search query,
- contextual summarization,
- generating the final cited answer.

## Mixed / unknown

Use only when the operation cannot be cleanly classified without changing its semantics.

## Annotation rule

Classify the **operation**, not the model used to execute it.

The purpose is to test whether different cognitive-processing requirements exhibit different model affinity. Do not infer the mode solely from the current upstream implementation.
