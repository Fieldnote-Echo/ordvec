# ordvec BEIR benchmark harness

Reproducible nDCG@10 evaluation of ordvec's rank/sign retrieval methods across
standard BEIR datasets, using Microsoft Harrier (harrier-oss-v1-0.6b, 1024-dim)
as the shared encoder.

## Claims discipline

The following two paragraphs reproduce the project's required claims policy
verbatim and govern every number produced by this harness:

> **Benchmark numbers in this repository reflect synthetic or user-runnable
> real-corpus experiments only.  No numbers are fabricated or cherry-picked.
> Every result file produced by `make benchmark-beir` is fully reproducible
> from the commands documented here, using publicly available BEIR datasets and
> the pinned encoder revision recorded in `embeddings.manifest.json`.**

> **FAISS FlatIP is a full-float dense retrieval baseline used for comparison
> purposes — it is NOT ground truth.  nDCG@10 is computed against the official
> BEIR qrels (human-annotated relevance judgements), not against FAISS results.
> ANN-recall-vs-FAISS (fraction of FAISS top-k recovered by an ANN method) is
> an optional diagnostic metric only; it does not substitute for qrel-based
> evaluation.**

## Dataset suite

| Dataset    | Domain          | #Queries | #Corpus |
|------------|-----------------|----------|---------|
| scifact    | Scientific claim verification | 300  | 5 183  |
| nfcorpus   | Biomedical IR   | 323      | 3 633  |
| fiqa       | Financial QA    | 648      | 57 638 |
| trec-covid | COVID-19 literature | 50   | 171 332 |

All datasets are downloaded automatically via the BEIR Python library on first
`make bench-beir-prepare` run.

## Encoder

**Harrier (harrier-oss-v1-0.6b)** — Microsoft's 600M-parameter bi-encoder
producing 1024-dimensional L2-normalised float32 embeddings.

- Documents receive no instruction prefix.
- Queries receive:
  `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "`
- Revision is pinned in `embeddings.manifest.json` per cache directory.

## Quick start

### 1. Install Python dependencies

```bash
make bench-beir-setup
```

Installs from `benchmarks/beir/requirements.txt`.

### 2. Smoke run (scifact only, ~5 min on GPU)

```bash
make benchmark-beir-smoke
```

Uses the `st` (sentence-transformers) provider with CUDA.  Encodes, runs all
ordvec methods, runs the FAISS baseline, then evaluates nDCG@{10,100}.

### 3. Full suite

**Sentence-transformers / CUDA lane (default):**

```bash
make benchmark-beir
```

Override encoder or device:

```bash
make benchmark-beir \
    ENCODER_PROVIDER=st \
    HARRIER_MODEL=microsoft/harrier-oss-v1-0.6b \
    DEVICE=cpu \
    ENCODE_BATCH=4
```

**Ollama lane (CPU, quantised, no GPU required):**

```bash
ollama pull hf.co/mradermacher/harrier-oss-v1-0.6b-GGUF:Q8_0
make bench-beir-prepare-ollama
make bench-beir-ordvec
make bench-beir-baselines
make bench-beir-eval
```

## Cache layout

One encoder run produces a directory per dataset/split:

```
.cache/ordvec-beir/<dataset>/<split>/encoder=<slug>/
    corpus.f32.npy          # float32, shape (n_docs, 1024), L2-normalised, C-order
    queries.f32.npy         # float32, shape (n_queries, 1024), L2-normalised, C-order
    corpus_ids.json         # list[str], sorted(corpus.keys())
    query_ids.json          # list[str], sorted(qrels.keys())
    qrels.json              # {qid: {doc_id: int_relevance}}
    texts.manifest.json     # reproducibility provenance for raw text
    embeddings.manifest.json# encoder provider/model/revision/dim/norm
    sha256s.json            # sha256 of each npy file
```

Encoder slug format: `<provider>__<model-path-components>__<revision-or-norev>`
with `/`, `:`, and other non-filesystem-safe characters replaced by `__`.

## Results layout

```
results/beir/<dataset>/
    <method>.topk.jsonl     # one JSON line per query
    <method>.summary.json   # aggregate latency + nDCG metrics
```

Top-k JSONL row schema:

```json
{"dataset":"scifact","split":"test","method":"ordvec-rq2",
 "qid_idx":0,"qid":"0","k":100,
 "doc_idxs":[42,7,...],"doc_ids":["abc","def",...],"scores":[0.91,0.88,...]}
```

Two-stage method names include parameters, e.g. `ordvec-bitmap-rq2-m500-b8`.

## Available methods

| Method          | Description |
|-----------------|-------------|
| `rq2`           | RankQuant (2 bits/dim), asymmetric float-query LUT scoring |
| `rq4`           | RankQuant (4 bits/dim), asymmetric float-query LUT scoring |
| `bitmap-rq2`    | Two-stage: Bitmap candidate gen + RankQuant-2 rerank |
| `sign-rq2`      | Two-stage: SignBitmap candidate gen + RankQuant-2 rerank |
| `faiss-flat`    | FAISS FlatIP full-float dense baseline (comparison, not ground truth) |

## `import ordvec` rule

The Python harness files in `benchmarks/beir/` **must not** contain
`import ordvec`.  This harness is an external consumer; it uses the installed
`ordvec` wheel.  The `bench-beir-guardrail` Make target (run automatically as
part of `benchmark-beir`) enforces this and fails with a clear error message if
any harness file violates it.

This rule preserves the reproducibility guarantee: anyone can clone this repo,
install the published wheel (`pip install ordvec`), and reproduce results
without needing the ordvec source tree.

## Clean up

```bash
make bench-beir-clean         # remove result files, keep embedding cache
make bench-beir-clean-cache   # remove embedding cache (re-encoding required)
```
