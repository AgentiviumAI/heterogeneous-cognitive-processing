# Dataset and Workflow Analysis

This document is owned initially by the workload-analysis task.

Please document:

1. How LitQA2 questions and gold answers are constructed.
2. Which split/subset is used in our pilot and why.
3. How PaperQA evaluates a completed episode.
4. Which PaperQA stages/tools are fixed by implementation.
5. Which actions/tool orders are selected dynamically by the agent.
6. A representative trajectory annotated as:
   - decision,
   - generation,
   - mixed/unknown.
7. Candidate decision points that can be replaced by Jev without changing the scientific task definition.

## Important distinction

The research question concerns cognitive-processing heterogeneity inside a realistic workflow. Do not redesign the workflow before we first characterize the upstream behavior.
