# ordvec-beir benchmark harness
# Reproduces nDCG@10 on standard BEIR datasets using ordvec's rank/sign retrieval
# methods plus FAISS FlatIP + HNSW dense baselines (for comparison, NOT ground truth).
#
# Usage:
#   make bench-beir-setup          # install Python deps
#   make benchmark-beir-smoke      # quick sanity run (scifact only)
#   make benchmark-beir            # full suite

# ── interpreter ──────────────────────────────────────────────────────────────
PY ?= python3

# ── paths ─────────────────────────────────────────────────────────────────────
CACHE_DIR  := .cache/ordvec-beir
RESULTS_DIR := results/beir

# ── dataset suite ─────────────────────────────────────────────────────────────
DATASETS       := scifact nfcorpus fiqa trec-covid
SMOKE_DATASETS := scifact
SPLIT          := test

# ── retrieval parameters ─────────────────────────────────────────────────────
TOPK       := 100
K_VALUES   := 10 100
BATCH      := 8
CANDIDATES := 500
SEED       := 1

# ── encoder ───────────────────────────────────────────────────────────────────
ENCODER_PROVIDER  := st
HARRIER_MODEL     := microsoft/harrier-oss-v1-0.6b
HARRIER_REVISION  :=
DEVICE            := cuda
ENCODE_BATCH      := 16

# ── ollama lane ───────────────────────────────────────────────────────────────
OLLAMA_URL          := http://localhost:11434
HARRIER_GGUF_MODEL  := hf.co/mradermacher/harrier-oss-v1-0.6b-GGUF:Q8_0

# ── baselines + ordvec methods ────────────────────────────────────────────────
ORDVEC_METHODS    := rq2,rq4,bitmap-rq2,sign-rq2
BASELINE_METHODS  := faiss-flat,hnswlib
HNSW_M            := 32
HNSW_EF_CONSTRUCT := 200
HNSW_EF_SEARCH    := 128

# ── phony ─────────────────────────────────────────────────────────────────────
.PHONY: benchmark-beir benchmark-beir-smoke benchmark-beir-bm25 \
        bench-beir-setup bench-beir-prepare bench-beir-prepare-ollama \
        bench-beir-ordvec bench-beir-baselines bench-beir-eval \
        bench-beir-guardrail bench-beir-clean bench-beir-clean-cache

# ── top-level targets ─────────────────────────────────────────────────────────

## Full benchmark run (guardrail → prepare → ordvec → baselines → eval)
benchmark-beir: bench-beir-guardrail bench-beir-prepare bench-beir-ordvec bench-beir-baselines bench-beir-eval

## Smoke run: scifact only, quick sanity check
benchmark-beir-smoke:
	$(MAKE) benchmark-beir \
		DATASETS=$(SMOKE_DATASETS) \
		TOPK=100 \
		ENCODE_BATCH=8

## Optional BM25 lane (placeholder — requires beir[bm25] extras)
benchmark-beir-bm25:
	@echo "BM25 lane: install beir[bm25] extras then run:"
	@echo "  $(PY) benchmarks/beir/beir_baselines.py --methods bm25 \\"
	@echo "      --datasets $(DATASETS) --split $(SPLIT) \\"
	@echo "      --cache-dir $(CACHE_DIR) --out-dir $(RESULTS_DIR) --top-k $(TOPK)"

# ── setup ─────────────────────────────────────────────────────────────────────

## Install Python benchmark dependencies
bench-beir-setup:
	$(PY) -m pip install -r benchmarks/beir/requirements.txt

# ── guardrail ─────────────────────────────────────────────────────────────────

## Fail loudly if any harness file imports the ordvec Python package directly.
## The harness is an EXTERNAL consumer — it must use the Rust crate at bench time,
## not the ordvec Python package. That coupling breaks the reproducibility claim.
bench-beir-guardrail:
	@if grep -R "import ordvec" benchmarks/beir 2>/dev/null; then \
		echo ""; \
		echo "ERROR: benchmarks/beir/ must not contain 'import ordvec'."; \
		echo "The benchmark hot path is the Rust crate, not the ordvec Python package."; \
		exit 1; \
	fi
	@echo "guardrail OK: no 'import ordvec' found in benchmarks/beir/"

# ── prepare ───────────────────────────────────────────────────────────────────

## Download datasets and encode with Harrier (sentence-transformers / CUDA lane)
bench-beir-prepare:
	$(PY) benchmarks/beir/beir_prepare.py \
		--datasets $(DATASETS) \
		--split $(SPLIT) \
		--provider $(ENCODER_PROVIDER) \
		--model "$(HARRIER_MODEL)" \
		$(if $(HARRIER_REVISION),--revision $(HARRIER_REVISION),) \
		--device "$(DEVICE)" \
		--batch-size $(ENCODE_BATCH) \
		--cache-dir "$(CACHE_DIR)" \
		--seed $(SEED)

## Encode with Harrier via Ollama (CPU/quantised lane — no GPU required)
bench-beir-prepare-ollama:
	$(PY) benchmarks/beir/beir_prepare.py \
		--datasets $(DATASETS) \
		--split $(SPLIT) \
		--provider ollama \
		--ollama-url "$(OLLAMA_URL)" \
		--model "$(HARRIER_GGUF_MODEL)" \
		--batch-size $(ENCODE_BATCH) \
		--cache-dir "$(CACHE_DIR)" \
		--seed $(SEED)

# ── ordvec retrieval ──────────────────────────────────────────────────────────

## Build the Rust beir_ordvec example binary and run all ordvec methods
bench-beir-ordvec:
	cargo build --release --example beir_ordvec
	@for dataset in $(DATASETS); do \
		$(CURDIR)/target/release/examples/beir_ordvec \
			--cache-dir "$(CACHE_DIR)" \
			--dataset "$$dataset" \
			--split $(SPLIT) \
			--top-k $(TOPK) \
			--batch $(BATCH) \
			--candidates $(CANDIDATES) \
			--methods $(ORDVEC_METHODS) \
			--out-dir "$(RESULTS_DIR)"; \
	done

# ── native dense baselines ────────────────────────────────────────────────────

## Run FAISS FlatIP + HNSW dense baselines (comparison references, NOT ground truth)
bench-beir-baselines:
	$(PY) benchmarks/beir/beir_baselines.py \
		--datasets $(DATASETS) \
		--split $(SPLIT) \
		--cache-dir "$(CACHE_DIR)" \
		--out-dir "$(RESULTS_DIR)" \
		--top-k $(TOPK) \
		--methods $(BASELINE_METHODS) \
		--hnsw-m $(HNSW_M) \
		--hnsw-ef-construction $(HNSW_EF_CONSTRUCT) \
		--hnsw-ef-search $(HNSW_EF_SEARCH) \
		--seed $(SEED)

# ── evaluation ────────────────────────────────────────────────────────────────

## Evaluate nDCG@10 etc. vs BEIR qrels + paired bootstrap deltas vs FAISS
bench-beir-eval:
	$(PY) benchmarks/beir/beir_eval.py \
		--datasets $(DATASETS) \
		--split $(SPLIT) \
		--cache-dir "$(CACHE_DIR)" \
		--runs-dir "$(RESULTS_DIR)" \
		--k-values $(K_VALUES) \
		--baseline faiss-flat \
		--bootstrap-iters 1000 \
		--seed $(SEED) \
		--out-dir "$(RESULTS_DIR)"

# ── cleanup ───────────────────────────────────────────────────────────────────

## Remove generated result files (keeps cache)
bench-beir-clean:
	find $(RESULTS_DIR) -name "*.topk.jsonl" -delete
	find $(RESULTS_DIR) -name "*.summary.json" -delete

## Remove the embedding cache (re-encoding will be required)
bench-beir-clean-cache:
	rm -rf $(CACHE_DIR)
