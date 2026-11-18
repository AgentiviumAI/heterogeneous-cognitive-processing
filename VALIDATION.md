# Release validation

Checked on 2026-11-18 before publication:

- Offline affinity tests: 4 passed (`tests/test_affinity.py`).
- Workflow metrics: 576 episodes, 96 unique IDs per configuration/base model. Completion counts, completed letter-match counts and all-attempt latency medians match the supplied preprint's workflow table.
- Python source syntax and project TOML parsing passed.
- A clean source snapshot built successfully into a PaperQA wheel using the declared setuptools build backend. The `0.0.0` fallback supports source archives without upstream Git tags.
- Staged files were checked for common credential patterns, environment files and raw trace directories.

Live model inference, GPU services, provider availability and the complete upstream PaperQA suite were not rerun. Upstream recorded HTTP cassettes are omitted. A wheel build does not validate installation of every optional dependency. Reproducing historical workflow trajectories requires the original article corpus and provider conditions; these are not supplied in this source release.
