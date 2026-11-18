# LitQA2 public questions

`train-00000-of-00001.parquet` is the unmodified public LitQA2 question file
from [Future House's Hugging Face dataset](https://huggingface.co/datasets/futurehouse/lab-bench/blob/5c77cec648430f30611808808861eb86f81d5eaa/LitQA2/train-00000-of-00001.parquet)
(revision `5c77cec648430f30611808808861eb86f81d5eaa`). Its SHA-256 is
`3e578fdaaa4728a77e884954dbf85ae0e3899791631a3be0838313b3e2334114`.
`litqa-v2-public.jsonl` is a row-preserving JSONL conversion of that Parquet
file so the pilot script needs no Parquet dependency. Its SHA-256 is
`ced2de20f90e954522a4bab16365e9355e79673f25fdb8bb6afd453a0577a248`.
Both files are redistributed under the upstream [CC BY-SA 4.0 license](LICENSE).
The license text was copied from the related
[Future House LAB-Bench repository](https://github.com/Future-House/LAB-Bench/blob/main/LICENSE).

We compared the Hugging Face file against the public
[GitHub JSONL release](https://github.com/Future-House/LAB-Bench/blob/998a8e0a40cf116c80e1b0e7a805ebb5fb9fa838/LitQA2/litqa-v2-public.jsonl):
all 199 IDs and all 10 shared fields match. The Hugging Face file adds
`source` and `subtask`.

The file has 199 questions with IDs, question text, an ideal answer,
distractors, source links, and a key passage. All 40 public `eval` question IDs
in [`docs/2024-10-16_litqa2-splits.json5`](../../../docs/2024-10-16_litqa2-splits.json5)
are present. The 49 `test` IDs are absent because that split is private.
Hugging Face calls its one public Parquet shard `train`; the `eval` designation
here comes from PaperQA's separate historical split file, not a Hugging Face
split.

## Preparing the shared pilot

`research/configs/pilot.yaml` currently specifies 5 questions from `eval`
with seed `20261003`, restricted to 19 questions whose source paper had open
full text at Europe PMC on 2026-10-05. The eligibility list and PMC IDs are
pinned in [`europe-pmc-open-fulltext.json`](europe-pmc-open-fulltext.json).
Run from the repository root:

```bash
.venv/bin/python research/datasets/litqa2/prepare_pilot.py --output-dir /tmp/litqa2-pilot
```

The script writes `questions.jsonl` (question ID, question, shuffled choices)
and `answer_key.jsonl` (question ID, correct letter) separately. The same
seed produces the same 5 questions and choice order. Give PaperQA only the
questions file; the answer key is for scoring after an answer is produced.
The `ideal`, `key-passage`, and source links are never included in the question
file. Keep both output files and `manifest.json` with each run for
reproducibility.

The question data does not contain paper full text. To download the five
selected open-access articles from Europe PMC and build a local corpus:

```bash
.venv/bin/python research/datasets/litqa2/prepare_corpus.py \
  --questions /tmp/litqa2-pilot/questions.jsonl \
  --output-dir /tmp/hcp-litqa2-fulltext-corpus
```

The downloader saves source XML in `source_xml/`, extracted article body text
for PaperQA, a manifest, and checksums in `sources.json`. The corpus stays
outside Git because article redistribution rights vary. Both pilot conditions
use the same local index. Do not feed `key-passage` to PaperQA: it contains
answer-bearing evidence from the benchmark. This open-access subset is an
integration pilot, not an exact reproduction of PaperQA's published benchmark
corpus.
