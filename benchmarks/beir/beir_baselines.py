"""
beir_baselines.py — native Python/C++ baselines for the ordvec-beir harness.

Methods
-------
faiss-flat
    Full-float inner-product exact search via faiss.IndexFlatIP on L2-normalised
    corpus vectors.  Inner product on unit vectors == cosine similarity.

hnswlib-m<M>-ef<ef_search>
    Approximate nearest-neighbour search via hnswlib's HNSW graph in cosine space.

Both methods consume the cached ``.npy`` arrays produced by ``beir_prepare.py``
and write results in the shared top-k JSONL + summary JSON formats defined by
``common.py``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

import numpy as np

# Allow `from common import ...` when run as a script from the repo root
# (the Makefile invokes `python3 benchmarks/beir/<script>.py`).
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from common import (
    find_encoder_dir,
    load_ids,
    load_npy_f32,
    slug_for_method,
    summary_json_path,
    topk_jsonl_path,
    write_topk_jsonl,
)

# ---------------------------------------------------------------------------
# Latency helpers
# ---------------------------------------------------------------------------


def _percentile(values: list[float], p: float) -> float:
    """Return the *p*-th percentile of *values* (0–100)."""
    if not values:
        return float("nan")
    arr = sorted(values)
    idx = (p / 100.0) * (len(arr) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(arr) - 1)
    frac = idx - lo
    return arr[lo] * (1.0 - frac) + arr[hi] * frac


# ---------------------------------------------------------------------------
# faiss-flat baseline
# ---------------------------------------------------------------------------


def run_faiss_flat(
    corpus: np.ndarray,
    queries: np.ndarray,
    corpus_ids: list[str],
    query_ids: list[str],
    dataset: str,
    split: str,
    top_k: int,
    out_dir: pathlib.Path,
) -> dict[str, Any]:
    """Run faiss.IndexFlatIP search and return a summary dict."""
    import faiss  # type: ignore[import]

    method = "faiss-flat"
    method_slug = slug_for_method(method, None, None)

    n_docs, dim = corpus.shape
    n_queries = queries.shape[0]

    # ------------------------------------------------------------------
    # Build index
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(dim)
    index.add(corpus)
    build_seconds = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Query — one-at-a-time to capture per-query latencies
    # ------------------------------------------------------------------
    rows: list[dict] = []
    query_times: list[float] = []

    for q_idx, qid in enumerate(query_ids):
        q_vec = queries[q_idx : q_idx + 1]  # shape (1, dim)
        qt0 = time.perf_counter()
        distances, indices = index.search(q_vec, top_k)
        query_times.append((time.perf_counter() - qt0) * 1_000.0)  # ms

        doc_idxs: list[int] = indices[0].tolist()
        doc_ids: list[str] = [corpus_ids[i] for i in doc_idxs]
        scores: list[float] = distances[0].tolist()

        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "method": method,
                "qid_idx": q_idx,
                "qid": qid,
                "k": top_k,
                "doc_idxs": doc_idxs,
                "doc_ids": doc_ids,
                "scores": scores,
            }
        )

    # ------------------------------------------------------------------
    # Write JSONL
    # ------------------------------------------------------------------
    jsonl_path = topk_jsonl_path(out_dir, dataset, method_slug)
    write_topk_jsonl(jsonl_path, rows)

    total_query_seconds = sum(query_times) / 1_000.0
    qps = n_queries / total_query_seconds if total_query_seconds > 0 else float("nan")

    summary: dict[str, Any] = {
        "method": method,
        "method_slug": method_slug,
        "dataset": dataset,
        "split": split,
        "n_docs": n_docs,
        "n_queries": n_queries,
        "top_k": top_k,
        "bytes_per_vector": dim * 4,  # float32
        "index_total_mib": (n_docs * dim * 4) / (1024 ** 2),
        "build_seconds": build_seconds,
        "query_latency_ms_p50": _percentile(query_times, 50),
        "query_latency_ms_p95": _percentile(query_times, 95),
        "query_latency_ms_p99": _percentile(query_times, 99),
        "queries_per_second": qps,
        "faiss_index_type": "IndexFlatIP",
    }

    sum_path = summary_json_path(out_dir, dataset, method_slug)
    sum_path.write_text(json.dumps(summary, indent=2))

    print(
        f"[faiss-flat] {dataset}/{split}: n_docs={n_docs}, n_queries={n_queries}, "
        f"build={build_seconds:.2f}s, p50={summary['query_latency_ms_p50']:.2f}ms, "
        f"QPS={qps:.1f}"
    )
    return summary


# ---------------------------------------------------------------------------
# hnswlib baseline
# ---------------------------------------------------------------------------


def run_hnswlib(
    corpus: np.ndarray,
    queries: np.ndarray,
    corpus_ids: list[str],
    query_ids: list[str],
    dataset: str,
    split: str,
    top_k: int,
    out_dir: pathlib.Path,
    hnsw_m: int,
    hnsw_ef_construction: int,
    hnsw_ef_search: int,
    seed: int,
) -> dict[str, Any]:
    """Run hnswlib HNSW search and return a summary dict."""
    import hnswlib  # type: ignore[import]

    method = f"hnswlib-m{hnsw_m}-ef{hnsw_ef_search}"
    method_slug = slug_for_method(method, None, None)

    n_docs, dim = corpus.shape
    n_queries = queries.shape[0]

    # ------------------------------------------------------------------
    # Build index
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(
        max_elements=n_docs,
        M=hnsw_m,
        ef_construction=hnsw_ef_construction,
        random_seed=seed,
    )
    index.add_items(corpus, num_threads=-1)
    index.set_ef(hnsw_ef_search)
    build_seconds = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Query — one-at-a-time for latency measurements
    # ------------------------------------------------------------------
    rows: list[dict] = []
    query_times: list[float] = []

    for q_idx, qid in enumerate(query_ids):
        q_vec = queries[q_idx : q_idx + 1]  # shape (1, dim)
        qt0 = time.perf_counter()
        labels, distances = index.knn_query(q_vec, k=min(top_k, n_docs))
        query_times.append((time.perf_counter() - qt0) * 1_000.0)  # ms

        # hnswlib returns cosine *distance* (1 - cosine_sim); convert to score
        doc_idxs: list[int] = labels[0].tolist()
        doc_ids: list[str] = [corpus_ids[i] for i in doc_idxs]
        # Score = 1 - distance (cosine similarity)
        scores: list[float] = [float(1.0 - d) for d in distances[0].tolist()]

        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "method": method,
                "qid_idx": q_idx,
                "qid": qid,
                "k": top_k,
                "doc_idxs": doc_idxs,
                "doc_ids": doc_ids,
                "scores": scores,
            }
        )

    # ------------------------------------------------------------------
    # Write JSONL
    # ------------------------------------------------------------------
    jsonl_path = topk_jsonl_path(out_dir, dataset, method_slug)
    write_topk_jsonl(jsonl_path, rows)

    total_query_seconds = sum(query_times) / 1_000.0
    qps = n_queries / total_query_seconds if total_query_seconds > 0 else float("nan")

    # hnswlib doesn't expose a clean bytes-per-vector figure;
    # report None and let the caller decide.
    summary: dict[str, Any] = {
        "method": method,
        "method_slug": method_slug,
        "dataset": dataset,
        "split": split,
        "n_docs": n_docs,
        "n_queries": n_queries,
        "top_k": top_k,
        "bytes_per_vector": None,  # HNSW graph overhead is graph-structure-dependent
        "index_total_mib": None,
        "build_seconds": build_seconds,
        "query_latency_ms_p50": _percentile(query_times, 50),
        "query_latency_ms_p95": _percentile(query_times, 95),
        "query_latency_ms_p99": _percentile(query_times, 99),
        "queries_per_second": qps,
        "hnsw_m": hnsw_m,
        "hnsw_ef_construction": hnsw_ef_construction,
        "hnsw_ef_search": hnsw_ef_search,
        "seed": seed,
    }

    sum_path = summary_json_path(out_dir, dataset, method_slug)
    sum_path.write_text(json.dumps(summary, indent=2))

    print(
        f"[{method}] {dataset}/{split}: n_docs={n_docs}, n_queries={n_queries}, "
        f"build={build_seconds:.2f}s, p50={summary['query_latency_ms_p50']:.2f}ms, "
        f"QPS={qps:.1f}"
    )
    return summary


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------


def run_dataset(
    dataset: str,
    split: str,
    cache_dir: pathlib.Path,
    out_dir: pathlib.Path,
    top_k: int,
    methods: set[str],
    hnsw_m: int,
    hnsw_ef_construction: int,
    hnsw_ef_search: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Load cached embeddings for *dataset* and run all requested methods."""
    enc_dir = find_encoder_dir(cache_dir, dataset, split)

    corpus_arr = load_npy_f32(enc_dir / "corpus.f32.npy")
    queries_arr = load_npy_f32(enc_dir / "queries.f32.npy")
    corpus_ids, query_ids = load_ids(enc_dir)

    # Sanity-check id-count vs embedding row-count
    if len(corpus_ids) != corpus_arr.shape[0]:
        raise ValueError(
            f"corpus_ids has {len(corpus_ids)} entries but corpus.f32.npy has "
            f"{corpus_arr.shape[0]} rows."
        )
    if len(query_ids) != queries_arr.shape[0]:
        raise ValueError(
            f"query_ids has {len(query_ids)} entries but queries.f32.npy has "
            f"{queries_arr.shape[0]} rows."
        )

    summaries: list[dict[str, Any]] = []

    if "faiss-flat" in methods:
        s = run_faiss_flat(
            corpus=corpus_arr,
            queries=queries_arr,
            corpus_ids=corpus_ids,
            query_ids=query_ids,
            dataset=dataset,
            split=split,
            top_k=top_k,
            out_dir=out_dir,
        )
        summaries.append(s)

    if "hnswlib" in methods:
        s = run_hnswlib(
            corpus=corpus_arr,
            queries=queries_arr,
            corpus_ids=corpus_ids,
            query_ids=query_ids,
            dataset=dataset,
            split=split,
            top_k=top_k,
            out_dir=out_dir,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
            seed=seed,
        )
        summaries.append(s)

    return summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_KNOWN_METHODS = {"faiss-flat", "hnswlib"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BEIR native baselines (faiss-flat, hnswlib).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        metavar="DATASET",
        help="One or more BEIR dataset names (e.g. nfcorpus scifact).",
    )
    p.add_argument(
        "--split",
        default="test",
        help="Corpus split to evaluate.",
    )
    p.add_argument(
        "--cache-dir",
        default="/home/ndspence/GitHub/ordvec-beir/.cache/ordvec-beir",
        help="Root of the embedding cache produced by beir_prepare.py.",
    )
    p.add_argument(
        "--out-dir",
        default="/home/ndspence/GitHub/ordvec-beir/results/beir",
        help="Root directory for top-k JSONL and summary JSON output.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of results to retrieve per query.",
    )
    p.add_argument(
        "--methods",
        default="faiss-flat,hnswlib",
        help="Comma-separated list of methods to run (faiss-flat, hnswlib).",
    )
    # HNSW params
    p.add_argument(
        "--hnsw-m",
        type=int,
        default=16,
        help="HNSW M parameter (number of bi-directional links per node).",
    )
    p.add_argument(
        "--hnsw-ef-construction",
        type=int,
        default=200,
        help="HNSW ef_construction parameter.",
    )
    p.add_argument(
        "--hnsw-ef-search",
        type=int,
        default=200,
        help="HNSW ef (search) parameter.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for hnswlib index construction.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    requested = {m.strip() for m in args.methods.split(",")}
    unknown = requested - _KNOWN_METHODS
    if unknown:
        raise ValueError(
            f"Unknown method(s): {sorted(unknown)}. "
            f"Supported: {sorted(_KNOWN_METHODS)}"
        )

    cache_dir = pathlib.Path(args.cache_dir)
    out_dir = pathlib.Path(args.out_dir)

    all_summaries: list[dict[str, Any]] = []
    for dataset in args.datasets:
        print(f"\n=== {dataset} ===")
        summaries = run_dataset(
            dataset=dataset,
            split=args.split,
            cache_dir=cache_dir,
            out_dir=out_dir,
            top_k=args.top_k,
            methods=requested,
            hnsw_m=args.hnsw_m,
            hnsw_ef_construction=args.hnsw_ef_construction,
            hnsw_ef_search=args.hnsw_ef_search,
            seed=args.seed,
        )
        all_summaries.extend(summaries)

    print(f"\nDone. {len(all_summaries)} run(s) completed.")


if __name__ == "__main__":
    main()
