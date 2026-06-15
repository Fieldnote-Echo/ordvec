//! OrdVec BEIR evaluation driver.
//!
//! Loads pre-computed embeddings from the shared cache layout, runs
//! OrdVec index methods over the BEIR query set, and writes per-query
//! top-k JSONL + per-method summary JSON to the results directory.
//!
//! BEIR metrics (NDCG@10, MAP, …) are NOT computed here; they are
//! evaluated offline in Python against the qrels files.
//!
//! Usage:
//!   cargo run --release --example beir_ordvec -- \
//!     --cache-dir .cache/ordvec-beir \
//!     --dataset scifact \
//!     --split test \
//!     --methods rq2,rq4,bitmap-rq2,sign-rq2 \
//!     --candidates 500 \
//!     --top-k 100 \
//!     --batch 8 \
//!     --out-dir results/beir
//!
//! Cache layout (one encoder per prepare run):
//!   <cache-dir>/<dataset>/<split>/encoder=<slug>/
//!     corpus.f32.npy   queries.f32.npy
//!     corpus_ids.json  query_ids.json  qrels.json
//!     texts.manifest.json  embeddings.manifest.json  sha256s.json
//!
//! Results layout:
//!   <out-dir>/<dataset>/<method-slug>.topk.jsonl
//!   <out-dir>/<dataset>/<method-slug>.summary.json

use ordvec::{Bitmap, CandidateBatch, RankQuant, SignBitmap, SubsetScratch};
use std::io::{BufWriter, Write};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

struct Config {
    cache_dir: String,
    dataset: String,
    split: String,
    top_k: usize,
    batch: usize,
    candidates: usize,
    methods: Vec<String>,
    out_dir: String,
}

fn parse_args() -> Config {
    let mut cache_dir = String::from(".cache/ordvec-beir");
    let mut dataset = String::new();
    let mut split = String::from("test");
    let mut top_k = 100usize;
    let mut batch = 8usize;
    let mut candidates = 500usize;
    let mut methods = vec![
        "rq2".to_string(),
        "rq4".to_string(),
        "bitmap-rq2".to_string(),
        "sign-rq2".to_string(),
    ];
    let mut out_dir = String::from("results/beir");

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--cache-dir" => cache_dir = args.next().expect("--cache-dir requires a value"),
            "--dataset" => dataset = args.next().expect("--dataset requires a value"),
            "--split" => split = args.next().expect("--split requires a value"),
            "--top-k" => {
                top_k = args
                    .next()
                    .expect("--top-k requires a value")
                    .parse()
                    .expect("--top-k must be an integer")
            }
            "--batch" => {
                batch = args
                    .next()
                    .expect("--batch requires a value")
                    .parse()
                    .expect("--batch must be an integer")
            }
            "--candidates" => {
                candidates = args
                    .next()
                    .expect("--candidates requires a value")
                    .parse()
                    .expect("--candidates must be an integer")
            }
            "--methods" => {
                methods = args
                    .next()
                    .expect("--methods requires a value")
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            }
            "--out-dir" => out_dir = args.next().expect("--out-dir requires a value"),
            other => panic!("unknown argument: {other}"),
        }
    }
    assert!(!dataset.is_empty(), "--dataset is required");
    assert!(batch >= 1, "--batch must be >= 1");
    assert!(top_k >= 1, "--top-k must be >= 1");
    assert!(candidates >= 1, "--candidates must be >= 1");

    Config {
        cache_dir,
        dataset,
        split,
        top_k,
        batch,
        candidates,
        methods,
        out_dir,
    }
}

// ---------------------------------------------------------------------------
// NumPy v1 reader (2-D LE f32, C-order) — adapted from bench_rank.rs
// ---------------------------------------------------------------------------

fn load_npy_f32(path: &str) -> (Vec<f32>, usize, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read npy {path}: {e}"));
    assert!(bytes.len() >= 10, "npy file too short: {path}");
    assert_eq!(&bytes[..6], b"\x93NUMPY", "not a numpy file: {path}");
    let major = bytes[6];
    let minor = bytes[7];
    assert!(
        major == 1 || major == 2,
        "unsupported npy version {major}.{minor}: {path}",
    );
    let (header_len, header_start) = if major == 1 {
        let hl = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        (hl, 10)
    } else {
        let hl = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        (hl, 12)
    };
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])
        .expect("npy header not utf-8");
    assert!(
        header.contains("'descr': '<f4'"),
        "expected <f4 dtype in {path}: {header}",
    );
    assert!(
        header.contains("'fortran_order': False"),
        "expected C order in {path}",
    );
    let shape_start = header.find("'shape':").expect("no shape in npy header");
    let after = &header[shape_start..];
    let open = after.find('(').unwrap();
    let close = after.find(')').unwrap();
    let dims: Vec<usize> = after[open + 1..close]
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();
    assert_eq!(
        dims.len(),
        2,
        "expected 2-D array in {path}, got {} dims",
        dims.len()
    );
    let n = dims[0];
    let dim = dims[1];
    let data_start = header_start + header_len;
    let n_floats = n * dim;
    assert_eq!(
        bytes.len() - data_start,
        n_floats * 4,
        "data length mismatch in {path}",
    );
    let mut out = vec![0.0f32; n_floats];
    for (i, chunk) in bytes[data_start..].chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    (out, n, dim)
}

// ---------------------------------------------------------------------------
// JSON helpers (no serde dep: manual string building)
// ---------------------------------------------------------------------------

fn load_json_string_array(path: &str) -> Vec<String> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    // Minimal parser: split on `"`, take even-indexed non-empty tokens between quotes.
    // Works for a JSON array of strings with no embedded quotes.
    let mut out = Vec::new();
    let mut in_str = false;
    let mut cur = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '"' {
            if in_str {
                out.push(cur.clone());
                cur.clear();
                in_str = false;
            } else {
                in_str = true;
            }
        } else if in_str {
            if c == '\\' {
                if let Some(next) = chars.next() {
                    match next {
                        '"' => cur.push('"'),
                        '\\' => cur.push('\\'),
                        'n' => cur.push('\n'),
                        't' => cur.push('\t'),
                        other => {
                            cur.push('\\');
                            cur.push(other);
                        }
                    }
                }
            } else {
                cur.push(c);
            }
        }
    }
    out
}

/// sha256 of a file using the system sha256sum / shasum / openssl.
///
/// Panics if none of those tools is available — the manifest hash is provenance
/// that must match the Python (`hashlib`) digest, so we never emit a non-SHA-256
/// value that merely looks like one.
fn sha256_file(path: &str) -> String {
    // Try sha256sum (Linux), then shasum -a 256 (macOS), then openssl.
    for (cmd, args) in &[
        ("sha256sum", vec![path]),
        ("shasum", vec!["-a", "256", path]),
        ("openssl", vec!["dgst", "-sha256", path]),
    ] {
        if let Ok(out) = std::process::Command::new(cmd).args(args).output() {
            if out.status.success() {
                let s = String::from_utf8_lossy(&out.stdout);
                // sha256sum: "<hex>  <file>"; shasum: same; openssl: "SHA256(...)= <hex>"
                for token in s.split_whitespace() {
                    if token.len() == 64 && token.chars().all(|c| c.is_ascii_hexdigit()) {
                        return token.to_string();
                    }
                }
            }
        }
    }
    panic!(
        "no SHA-256 tool available (tried sha256sum / shasum -a 256 / openssl) — \
         cannot compute the encoder-manifest digest for {path}"
    );
}

// ---------------------------------------------------------------------------
// Detect SIMD features
// ---------------------------------------------------------------------------

fn detected_simd() -> Vec<String> {
    #[cfg(target_arch = "x86_64")]
    {
        let mut v = Vec::new();
        if is_x86_feature_detected!("avx2") {
            v.push("avx2".to_string());
        }
        if is_x86_feature_detected!("fma") {
            v.push("fma".to_string());
        }
        if is_x86_feature_detected!("avx512f") {
            v.push("avx512f".to_string());
        }
        v
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// Percentile helper
// ---------------------------------------------------------------------------

fn percentile_ms(mut samples: Vec<u128>, p: f32) -> f64 {
    samples.sort_unstable();
    let i = ((samples.len() as f32 - 1.0) * p).round() as usize;
    // samples are nanoseconds; convert to milliseconds
    samples[i] as f64 / 1_000_000.0
}

// ---------------------------------------------------------------------------
// Top-k JSONL writer
// ---------------------------------------------------------------------------

/// Write one JSONL row per query to `writer`.
/// `indices` is a flat `nq * k` slice (global doc indices, i64; -1 = padding).
fn write_topk_jsonl<W: Write>(
    writer: &mut W,
    dataset: &str,
    split: &str,
    method: &str,
    k: usize,
    query_ids: &[String],
    corpus_ids: &[String],
    indices: &[i64],
    scores: &[f32],
) {
    let nq = query_ids.len();
    let n_corpus = corpus_ids.len();
    for qi in 0..nq {
        let row_indices = &indices[qi * k..(qi + 1) * k];
        // Build doc_idxs and doc_ids arrays (skip sentinels at the end).
        let mut doc_idxs_str = String::from("[");
        let mut doc_ids_str = String::from("[");
        let mut scores_str = String::from("[");
        let mut first = true;
        for (j, &di) in row_indices.iter().enumerate() {
            if di < 0 {
                break;
            }
            let di_usize = di as usize;
            if !first {
                doc_idxs_str.push(',');
                doc_ids_str.push(',');
                scores_str.push(',');
            }
            first = false;
            doc_idxs_str.push_str(&di_usize.to_string());
            let doc_id = if di_usize < n_corpus {
                corpus_ids[di_usize].as_str()
            } else {
                ""
            };
            doc_ids_str.push('"');
            doc_ids_str.push_str(doc_id);
            doc_ids_str.push('"');
            // Real per-doc score (results are rank-ordered; downstream eval ranks
            // by score). Non-finite guarded to keep the JSON valid.
            let sc = scores.get(qi * k + j).copied().unwrap_or(0.0);
            if sc.is_finite() {
                scores_str.push_str(&sc.to_string());
            } else {
                scores_str.push_str("0.0");
            }
        }
        doc_idxs_str.push(']');
        doc_ids_str.push(']');
        scores_str.push(']');

        writeln!(
            writer,
            r#"{{"dataset":"{dataset}","split":"{split}","method":"{method}","qid_idx":{qi},"qid":"{qid}","k":{k},"doc_idxs":{doc_idxs_str},"doc_ids":{doc_ids_str},"scores":{scores_str}}}"#,
            qid = query_ids[qi],
        )
        .expect("write topk jsonl");
    }
}

// ---------------------------------------------------------------------------
// Summary JSON writer
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn write_summary_json<W: Write>(
    writer: &mut W,
    dataset: &str,
    split: &str,
    method: &str,
    dim: usize,
    n_docs: usize,
    n_queries: usize,
    top_k: usize,
    bytes_per_vector: usize,
    index_total_mib: f64,
    build_seconds: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    queries_per_second: f64,
    candidate_m: usize,
    batch: usize,
    simd_detected: &[String],
    encoder_manifest_sha256: &str,
) {
    let simd_arr: String = {
        let parts: Vec<String> = simd_detected.iter().map(|s| format!("\"{s}\"")).collect();
        format!("[{}]", parts.join(","))
    };
    let rustc = rustc_version();
    let crate_ver = env!("CARGO_PKG_VERSION");
    writeln!(
        writer,
        r#"{{"dataset":"{dataset}","split":"{split}","method":"{method}","dim":{dim},"n_docs":{n_docs},"n_queries":{n_queries},"top_k":{top_k},"bytes_per_vector":{bytes_per_vector},"index_total_mib":{index_total_mib:.3},"build_seconds":{build_seconds:.4},"query_latency_ms_p50":{p50_ms:.4},"query_latency_ms_p95":{p95_ms:.4},"query_latency_ms_p99":{p99_ms:.4},"queries_per_second":{queries_per_second:.2},"candidate_m":{candidate_m},"batch":{batch},"cpu_arch":"{arch}","simd_detected":{simd_arr},"rustc":"{rustc}","crate_version":"{crate_ver}","encoder_manifest_sha256":"{encoder_manifest_sha256}"}}"#,
        arch = std::env::consts::ARCH,
    )
    .expect("write summary json");
}

fn rustc_version() -> String {
    if let Ok(out) = std::process::Command::new("rustc")
        .arg("--version")
        .output()
    {
        if out.status.success() {
            return String::from_utf8_lossy(&out.stdout).trim().to_string();
        }
    }
    "unknown".to_string()
}

// ---------------------------------------------------------------------------
// Output file helpers
// ---------------------------------------------------------------------------

fn open_output(out_dir: &str, dataset: &str, slug: &str, ext: &str) -> BufWriter<std::fs::File> {
    let dir = format!("{out_dir}/{dataset}");
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create_dir_all {dir}: {e}"));
    let path = format!("{dir}/{slug}.{ext}");
    let f = std::fs::File::create(&path).unwrap_or_else(|e| panic!("create {path}: {e}"));
    BufWriter::new(f)
}

// ---------------------------------------------------------------------------
// Validate embeddings
// ---------------------------------------------------------------------------

fn validate_embeddings(data: &[f32], n: usize, dim: usize, label: &str) {
    assert_eq!(dim, 1024, "{label}: embedding_dim must be 1024, got {dim}");
    assert_eq!(dim % 16, 0, "{label}: dim must be divisible by 16");
    for (i, row) in data.chunks_exact(dim).enumerate() {
        let norm2: f32 = row.iter().map(|x| x * x).sum();
        let norm = norm2.sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "{label} row {i}: L2 norm {norm:.6} not in [0.999, 1.001]",
        );
    }
    eprintln!("  {label}: validated {n} rows (dim={dim}, L2-normalised)");
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

/// Collect per-query latency samples (ns) for `search_fn` applied to each
/// query row. `warmup` queries are excluded from timing.
fn time_batched<F>(
    queries: &[f32],
    dim: usize,
    n_queries: usize,
    batch: usize,
    warmup: usize,
    mut search_fn: F,
) -> Vec<u128>
where
    F: FnMut(&[f32]) -> Vec<i64>,
{
    // Warmup: run a few batches without recording.
    let warmup_queries = warmup.min(n_queries);
    {
        let w_end = (warmup_queries + batch - 1) / batch * batch;
        let w_end = w_end.min(n_queries);
        for b_start in (0..w_end).step_by(batch) {
            let b_end = (b_start + batch).min(n_queries);
            let batch_q = &queries[b_start * dim..b_end * dim];
            let _ = search_fn(batch_q);
        }
    }

    let mut samples = Vec::with_capacity(n_queries);
    let mut b_start = 0usize;
    while b_start < n_queries {
        let b_end = (b_start + batch).min(n_queries);
        let b = b_end - b_start;
        let batch_q = &queries[b_start * dim..b_end * dim];
        let t0 = Instant::now();
        let _ = search_fn(batch_q);
        let elapsed_ns = t0.elapsed().as_nanos();
        let per_query_ns = elapsed_ns / b as u128;
        for _ in 0..b {
            samples.push(per_query_ns);
        }
        b_start = b_end;
    }
    samples
}

// ---------------------------------------------------------------------------
// Collect all predictions (flat nq*k indices)
// ---------------------------------------------------------------------------

fn collect_all_predictions<F>(
    queries: &[f32],
    dim: usize,
    n_queries: usize,
    top_k: usize,
    batch: usize,
    mut search_fn: F,
) -> (Vec<i64>, Vec<f32>)
where
    F: FnMut(&[f32]) -> (Vec<i64>, Vec<f32>),
{
    let mut out_i = Vec::with_capacity(n_queries * top_k);
    let mut out_s = Vec::with_capacity(n_queries * top_k);
    let mut b_start = 0usize;
    while b_start < n_queries {
        let b_end = (b_start + batch).min(n_queries);
        let batch_q = &queries[b_start * dim..b_end * dim];
        let (pi, ps) = search_fn(batch_q);
        out_i.extend_from_slice(&pi);
        out_s.extend_from_slice(&ps);
        b_start = b_end;
    }
    (out_i, out_s)
}

// ---------------------------------------------------------------------------
// Method runners
// ---------------------------------------------------------------------------

/// Build a Bitmap→CSR helper: convert `Vec<Vec<u32>>` (one row per query)
/// into CSR (offsets, concatenated candidates).
fn bitmap_vecs_to_csr(vecs: Vec<Vec<u32>>) -> (Vec<usize>, Vec<u32>) {
    let nq = vecs.len();
    let mut offsets = Vec::with_capacity(nq + 1);
    let mut candidates = Vec::new();
    offsets.push(0usize);
    for row in &vecs {
        candidates.extend_from_slice(row);
        offsets.push(candidates.len());
    }
    (offsets, candidates)
}

/// rq2 / rq4: full-scan asymmetric search.
fn run_full_scan(
    corpus: &[f32],
    queries: &[f32],
    dim: usize,
    n_queries: usize,
    top_k: usize,
    batch: usize,
    bits: u8,
    dataset: &str,
    split: &str,
    query_ids: &[String],
    corpus_ids: &[String],
    out_dir: &str,
    simd: &[String],
    encoder_sha: &str,
) {
    let method_slug = format!("ordvec-rq{bits}");
    eprintln!("building RankQuant b={bits} index ...");
    let mut idx = RankQuant::new(dim, bits);
    let t0 = Instant::now();
    idx.add(corpus);
    let build_seconds = t0.elapsed().as_secs_f64();
    eprintln!("  build done in {build_seconds:.2}s ({} docs)", idx.len());

    let n_docs = idx.len();
    let bytes_per_vector = idx.bytes_per_vec();
    let index_total_mib = idx.byte_size() as f64 / 1024.0 / 1024.0;

    // Warmup: min(5, n_queries) queries.
    let warmup = 5.min(n_queries);

    // Timing pass (batched, allocation-free after warmup).
    eprintln!("  timing {n_queries} queries (batch={batch}, warmup={warmup}) ...");
    let samples = time_batched(queries, dim, n_queries, batch, warmup, |batch_q| {
        let res = idx.search_asymmetric(batch_q, top_k);
        res.indices
    });

    let p50 = percentile_ms(samples.clone(), 0.50);
    let p95 = percentile_ms(samples.clone(), 0.95);
    let p99 = percentile_ms(samples, 0.99);
    let qps = 1_000.0 / p50.max(f64::EPSILON);

    // Collect predictions.
    eprintln!("  collecting predictions ...");
    let (pred_indices, pred_scores) =
        collect_all_predictions(queries, dim, n_queries, top_k, batch, |batch_q| {
            let res = idx.search_asymmetric(batch_q, top_k);
            (res.indices, res.scores)
        });

    // Write JSONL.
    let mut jsonl_writer = open_output(out_dir, dataset, &method_slug, "topk.jsonl");
    write_topk_jsonl(
        &mut jsonl_writer,
        dataset,
        split,
        &method_slug,
        top_k,
        query_ids,
        corpus_ids,
        &pred_indices,
        &pred_scores,
    );
    jsonl_writer.flush().expect("flush topk jsonl");

    // Write summary.
    let mut summary_writer = open_output(out_dir, dataset, &method_slug, "summary.json");
    write_summary_json(
        &mut summary_writer,
        dataset,
        split,
        &method_slug,
        dim,
        n_docs,
        n_queries,
        top_k,
        bytes_per_vector,
        index_total_mib,
        build_seconds,
        p50,
        p95,
        p99,
        qps,
        0, // candidate_m not applicable for full scan
        batch,
        simd,
        encoder_sha,
    );
    summary_writer.flush().expect("flush summary json");
    eprintln!("  {method_slug}: p50={p50:.3}ms p95={p95:.3}ms p99={p99:.3}ms qps={qps:.1}",);
}

/// bitmap-rq2: Bitmap candidate gen → RankQuant b=2 asymmetric rerank.
fn run_bitmap_rq2(
    corpus: &[f32],
    queries: &[f32],
    dim: usize,
    n_queries: usize,
    top_k: usize,
    batch: usize,
    candidates: usize,
    dataset: &str,
    split: &str,
    query_ids: &[String],
    corpus_ids: &[String],
    out_dir: &str,
    simd: &[String],
    encoder_sha: &str,
) {
    let method_slug = format!("ordvec-bitmap-rq2-m{candidates}-b{batch}");
    eprintln!("building Bitmap + RankQuant b=2 index (m={candidates}) ...");

    // n_top for the bitmap: mirror bench_rank.rs (dim/4 = top quarter, b=2-equivalent).
    let n_top = dim / 4;
    let mut bitmap = Bitmap::new(dim, n_top);
    let mut rq = RankQuant::new(dim, 2);
    let t0 = Instant::now();
    bitmap.add(corpus);
    rq.add(corpus);
    let build_seconds = t0.elapsed().as_secs_f64();
    eprintln!("  build done in {build_seconds:.2}s");

    let n_docs = rq.len();
    let bytes_per_vector = bitmap.bytes_per_vec() + rq.bytes_per_vec();
    let index_total_mib = (bitmap.byte_size() + rq.byte_size()) as f64 / 1024.0 / 1024.0;

    let out_k = top_k.min(candidates).min(n_docs);
    let warmup = 5.min(n_queries);

    // Allocate pooled scratch and output buffers — reused across all batches.
    let mut scratch = SubsetScratch::new();
    // Size the output buffers for a full batch (worst case = `batch` queries).
    let max_batch = batch;
    let mut out_scores_buf = vec![f32::NEG_INFINITY; max_batch * out_k];
    let mut out_indices_buf = vec![-1i64; max_batch * out_k];

    // Helper: run one query-batch through the two-stage pipeline.
    // Returns flat nq_batch * out_k indices (sentinel-padded).
    let two_stage_batch = |batch_q: &[f32],
                           bitmap: &Bitmap,
                           rq: &RankQuant,
                           scratch: &mut SubsetScratch,
                           out_scores_buf: &mut Vec<f32>,
                           out_indices_buf: &mut Vec<i64>|
     -> (Vec<i64>, Vec<f32>) {
        let nq_batch = batch_q.len() / dim;
        // Resize scratch output buffers if batch shrinks (last batch may be smaller).
        let needed = nq_batch * out_k;
        if out_scores_buf.len() != needed {
            out_scores_buf.resize(needed, f32::NEG_INFINITY);
            out_indices_buf.resize(needed, -1);
        }

        // Stage 1: bitmap candidate gen → Vec<Vec<u32>>.
        let cand_vecs = bitmap.top_m_candidates_batched(batch_q, candidates);
        // Convert to CSR for the batched subset rerank.
        let (offsets, cand_flat) = bitmap_vecs_to_csr(cand_vecs);

        // Stage 2: pooled subset rerank.
        rq.search_asymmetric_subset_batched_serial_into(
            batch_q,
            &offsets,
            &cand_flat,
            top_k,
            scratch,
            out_scores_buf,
            out_indices_buf,
        );

        // Map candidate-local indices (already global doc ids, since cands come from
        // bitmap which stores global doc ids) — indices are already global.
        // Pad per-query results to `top_k` with -1 sentinels if out_k < top_k.
        let mut result = vec![-1i64; nq_batch * top_k];
        let mut result_scores = vec![0.0f32; nq_batch * top_k];
        for qi in 0..nq_batch {
            let src = &out_indices_buf[qi * out_k..(qi + 1) * out_k];
            let src_s = &out_scores_buf[qi * out_k..(qi + 1) * out_k];
            let dst = &mut result[qi * top_k..(qi + 1) * top_k];
            let copy_len = src.len().min(dst.len());
            dst[..copy_len].copy_from_slice(&src[..copy_len]);
            result_scores[qi * top_k..qi * top_k + copy_len].copy_from_slice(&src_s[..copy_len]);
        }
        (result, result_scores)
    };

    eprintln!("  timing {n_queries} queries (batch={batch}, warmup={warmup}) ...");
    let samples = {
        // Warmup.
        {
            let w_end = (warmup + batch - 1) / batch * batch;
            let w_end = w_end.min(n_queries);
            for b_start in (0..w_end).step_by(batch) {
                let b_end = (b_start + batch).min(n_queries);
                let batch_q = &queries[b_start * dim..b_end * dim];
                let _ = two_stage_batch(
                    batch_q,
                    &bitmap,
                    &rq,
                    &mut scratch,
                    &mut out_scores_buf,
                    &mut out_indices_buf,
                );
            }
        }
        let mut s = Vec::with_capacity(n_queries);
        let mut b_start = 0usize;
        while b_start < n_queries {
            let b_end = (b_start + batch).min(n_queries);
            let b = b_end - b_start;
            let batch_q = &queries[b_start * dim..b_end * dim];
            let t0 = Instant::now();
            let _ = two_stage_batch(
                batch_q,
                &bitmap,
                &rq,
                &mut scratch,
                &mut out_scores_buf,
                &mut out_indices_buf,
            );
            let elapsed_ns = t0.elapsed().as_nanos();
            let per_query_ns = elapsed_ns / b as u128;
            for _ in 0..b {
                s.push(per_query_ns);
            }
            b_start = b_end;
        }
        s
    };

    let p50 = percentile_ms(samples.clone(), 0.50);
    let p95 = percentile_ms(samples.clone(), 0.95);
    let p99 = percentile_ms(samples, 0.99);
    let qps = 1_000.0 / p50.max(f64::EPSILON);

    eprintln!("  collecting predictions ...");
    let mut pred_indices = Vec::with_capacity(n_queries * top_k);
    let mut pred_scores: Vec<f32> = Vec::with_capacity(n_queries * top_k);
    let mut b_start = 0usize;
    while b_start < n_queries {
        let b_end = (b_start + batch).min(n_queries);
        let batch_q = &queries[b_start * dim..b_end * dim];
        let (preds, pred_s) = two_stage_batch(
            batch_q,
            &bitmap,
            &rq,
            &mut scratch,
            &mut out_scores_buf,
            &mut out_indices_buf,
        );
        pred_indices.extend_from_slice(&preds);
        pred_scores.extend_from_slice(&pred_s);
        b_start = b_end;
    }

    // Write outputs.
    let mut jsonl_writer = open_output(out_dir, dataset, &method_slug, "topk.jsonl");
    write_topk_jsonl(
        &mut jsonl_writer,
        dataset,
        split,
        &method_slug,
        top_k,
        query_ids,
        corpus_ids,
        &pred_indices,
        &pred_scores,
    );
    jsonl_writer.flush().expect("flush topk jsonl");

    let mut summary_writer = open_output(out_dir, dataset, &method_slug, "summary.json");
    write_summary_json(
        &mut summary_writer,
        dataset,
        split,
        &method_slug,
        dim,
        n_docs,
        n_queries,
        top_k,
        bytes_per_vector,
        index_total_mib,
        build_seconds,
        p50,
        p95,
        p99,
        qps,
        candidates,
        batch,
        simd,
        encoder_sha,
    );
    summary_writer.flush().expect("flush summary json");
    eprintln!("  {method_slug}: p50={p50:.3}ms p95={p95:.3}ms p99={p99:.3}ms qps={qps:.1}",);
}

/// sign-rq2: SignBitmap candidate gen → RankQuant b=2 asymmetric rerank.
fn run_sign_rq2(
    corpus: &[f32],
    queries: &[f32],
    dim: usize,
    n_queries: usize,
    top_k: usize,
    batch: usize,
    candidates: usize,
    dataset: &str,
    split: &str,
    query_ids: &[String],
    corpus_ids: &[String],
    out_dir: &str,
    simd: &[String],
    encoder_sha: &str,
) {
    let method_slug = format!("ordvec-sign-rq2-m{candidates}-b{batch}");
    eprintln!("building SignBitmap + RankQuant b=2 index (m={candidates}) ...");

    let mut sign = SignBitmap::new(dim);
    let mut rq = RankQuant::new(dim, 2);
    let t0 = Instant::now();
    sign.add(corpus);
    rq.add(corpus);
    let build_seconds = t0.elapsed().as_secs_f64();
    eprintln!("  build done in {build_seconds:.2}s");

    let n_docs = rq.len();
    let bytes_per_vector = sign.bytes_per_vec() + rq.bytes_per_vec();
    let index_total_mib = (sign.byte_size() + rq.byte_size()) as f64 / 1024.0 / 1024.0;

    let out_k = top_k.min(candidates).min(n_docs);
    let warmup = 5.min(n_queries);

    // Pooled scratch and output buffers — reused across batches (allocation-free after warmup).
    let mut scratch = SubsetScratch::new();
    let max_batch = batch;
    let mut out_scores_buf = vec![f32::NEG_INFINITY; max_batch * out_k];
    let mut out_indices_buf = vec![-1i64; max_batch * out_k];

    let sign_two_stage_batch = |batch_q: &[f32],
                                sign: &SignBitmap,
                                rq: &RankQuant,
                                scratch: &mut SubsetScratch,
                                out_scores_buf: &mut Vec<f32>,
                                out_indices_buf: &mut Vec<i64>|
     -> (Vec<i64>, Vec<f32>) {
        let nq_batch = batch_q.len() / dim;
        let needed = nq_batch * out_k;
        if out_scores_buf.len() != needed {
            out_scores_buf.resize(needed, f32::NEG_INFINITY);
            out_indices_buf.resize(needed, -1);
        }

        // Stage 1: SignBitmap → CSR CandidateBatch (SIMD XOR-popcount kernel).
        let cb: CandidateBatch = sign.top_m_candidates_batched_serial_csr(batch_q, candidates);

        // Stage 2: pooled subset rerank.
        rq.search_asymmetric_subset_batched_serial_into(
            batch_q,
            &cb.offsets,
            &cb.candidates,
            top_k,
            scratch,
            out_scores_buf,
            out_indices_buf,
        );

        // Pad per-query results to `top_k` with -1 sentinels if out_k < top_k.
        let mut result = vec![-1i64; nq_batch * top_k];
        let mut result_scores = vec![0.0f32; nq_batch * top_k];
        for qi in 0..nq_batch {
            let src = &out_indices_buf[qi * out_k..(qi + 1) * out_k];
            let src_s = &out_scores_buf[qi * out_k..(qi + 1) * out_k];
            let dst = &mut result[qi * top_k..(qi + 1) * top_k];
            let copy_len = src.len().min(dst.len());
            dst[..copy_len].copy_from_slice(&src[..copy_len]);
            result_scores[qi * top_k..qi * top_k + copy_len].copy_from_slice(&src_s[..copy_len]);
        }
        (result, result_scores)
    };

    eprintln!("  timing {n_queries} queries (batch={batch}, warmup={warmup}) ...");

    // Warmup via top_m_candidates_batched_serial_csr.
    {
        let w_end = warmup.min(n_queries);
        if w_end > 0 {
            let _ = sign.top_m_candidates_batched_serial_csr(&queries[..w_end * dim], candidates);
        }
    }

    let samples = {
        let mut s = Vec::with_capacity(n_queries);
        let mut b_start = 0usize;
        while b_start < n_queries {
            let b_end = (b_start + batch).min(n_queries);
            let b = b_end - b_start;
            let batch_q = &queries[b_start * dim..b_end * dim];
            let t0 = Instant::now();
            let _ = sign_two_stage_batch(
                batch_q,
                &sign,
                &rq,
                &mut scratch,
                &mut out_scores_buf,
                &mut out_indices_buf,
            );
            let elapsed_ns = t0.elapsed().as_nanos();
            let per_query_ns = elapsed_ns / b as u128;
            for _ in 0..b {
                s.push(per_query_ns);
            }
            b_start = b_end;
        }
        s
    };

    let p50 = percentile_ms(samples.clone(), 0.50);
    let p95 = percentile_ms(samples.clone(), 0.95);
    let p99 = percentile_ms(samples, 0.99);
    let qps = 1_000.0 / p50.max(f64::EPSILON);

    eprintln!("  collecting predictions ...");
    let mut pred_indices = Vec::with_capacity(n_queries * top_k);
    let mut pred_scores: Vec<f32> = Vec::with_capacity(n_queries * top_k);
    let mut b_start = 0usize;
    while b_start < n_queries {
        let b_end = (b_start + batch).min(n_queries);
        let batch_q = &queries[b_start * dim..b_end * dim];
        let (preds, pred_s) = sign_two_stage_batch(
            batch_q,
            &sign,
            &rq,
            &mut scratch,
            &mut out_scores_buf,
            &mut out_indices_buf,
        );
        pred_indices.extend_from_slice(&preds);
        pred_scores.extend_from_slice(&pred_s);
        b_start = b_end;
    }

    // Write outputs.
    let mut jsonl_writer = open_output(out_dir, dataset, &method_slug, "topk.jsonl");
    write_topk_jsonl(
        &mut jsonl_writer,
        dataset,
        split,
        &method_slug,
        top_k,
        query_ids,
        corpus_ids,
        &pred_indices,
        &pred_scores,
    );
    jsonl_writer.flush().expect("flush topk jsonl");

    let mut summary_writer = open_output(out_dir, dataset, &method_slug, "summary.json");
    write_summary_json(
        &mut summary_writer,
        dataset,
        split,
        &method_slug,
        dim,
        n_docs,
        n_queries,
        top_k,
        bytes_per_vector,
        index_total_mib,
        build_seconds,
        p50,
        p95,
        p99,
        qps,
        candidates,
        batch,
        simd,
        encoder_sha,
    );
    summary_writer.flush().expect("flush summary json");
    eprintln!("  {method_slug}: p50={p50:.3}ms p95={p95:.3}ms p99={p99:.3}ms qps={qps:.1}",);
}

// ---------------------------------------------------------------------------
// Cache resolution
// ---------------------------------------------------------------------------

/// Resolve the single `encoder=*` subdirectory under
/// `<cache_dir>/<dataset>/<split>/`. Panics if zero or multiple matches.
fn resolve_encoder_dir(cache_dir: &str, dataset: &str, split: &str) -> String {
    let parent = format!("{cache_dir}/{dataset}/{split}");
    let entries = std::fs::read_dir(&parent).unwrap_or_else(|e| panic!("read_dir {parent}: {e}"));
    let mut matches: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("encoder=") && e.path().is_dir())
        .map(|e| e.path().to_string_lossy().to_string())
        .collect();
    assert!(
        !matches.is_empty(),
        "no encoder=* subdirectory found under {parent}",
    );
    assert!(
        matches.len() == 1,
        "multiple encoder=* directories found under {parent}: {matches:?} — \
         specify exactly one encoder per dataset/split",
    );
    matches.remove(0)
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cfg = parse_args();

    eprintln!(
        "beir_ordvec: dataset={} split={} top_k={} batch={} candidates={} methods={:?}",
        cfg.dataset, cfg.split, cfg.top_k, cfg.batch, cfg.candidates, cfg.methods,
    );

    // Resolve encoder directory.
    let enc_dir = resolve_encoder_dir(&cfg.cache_dir, &cfg.dataset, &cfg.split);
    eprintln!("encoder dir: {enc_dir}");

    // Compute encoder_manifest_sha256 from embeddings.manifest.json.
    let manifest_path = format!("{enc_dir}/embeddings.manifest.json");
    let encoder_sha = sha256_file(&manifest_path);
    eprintln!("encoder_manifest_sha256: {encoder_sha}");

    // Load corpus embeddings.
    let corpus_npy = format!("{enc_dir}/corpus.f32.npy");
    eprintln!("loading corpus: {corpus_npy}");
    let t0 = Instant::now();
    let (corpus, n_corpus, dim) = load_npy_f32(&corpus_npy);
    eprintln!(
        "  loaded {n_corpus} corpus vectors in {:.2}s",
        t0.elapsed().as_secs_f64()
    );

    // Load query embeddings.
    let queries_npy = format!("{enc_dir}/queries.f32.npy");
    eprintln!("loading queries: {queries_npy}");
    let t0 = Instant::now();
    let (queries, n_queries, q_dim) = load_npy_f32(&queries_npy);
    assert_eq!(q_dim, dim, "query dim {q_dim} != corpus dim {dim}",);
    eprintln!(
        "  loaded {n_queries} query vectors in {:.2}s",
        t0.elapsed().as_secs_f64()
    );

    // Validate embeddings.
    validate_embeddings(&corpus, n_corpus, dim, "corpus");
    validate_embeddings(&queries, n_queries, q_dim, "queries");
    assert_eq!(
        n_corpus,
        corpus.len() / dim,
        "corpus id count / embedding count mismatch",
    );
    assert_eq!(
        n_queries,
        queries.len() / dim,
        "query id count / embedding count mismatch",
    );

    // Load corpus_ids and query_ids.
    let corpus_ids_path = format!("{enc_dir}/corpus_ids.json");
    let query_ids_path = format!("{enc_dir}/query_ids.json");
    let corpus_ids = load_json_string_array(&corpus_ids_path);
    let query_ids = load_json_string_array(&query_ids_path);
    assert_eq!(
        corpus_ids.len(),
        n_corpus,
        "corpus_ids length {} != n_corpus {n_corpus}",
        corpus_ids.len(),
    );
    assert_eq!(
        query_ids.len(),
        n_queries,
        "query_ids length {} != n_queries {n_queries}",
        query_ids.len(),
    );

    let simd = detected_simd();
    eprintln!("simd detected: {:?}", simd);
    eprintln!("dim={dim} n_corpus={n_corpus} n_queries={n_queries}");

    // Run each requested method.
    for method in &cfg.methods {
        eprintln!("\n--- method: {method} ---");
        match method.as_str() {
            "rq2" => run_full_scan(
                &corpus,
                &queries,
                dim,
                n_queries,
                cfg.top_k,
                cfg.batch,
                2,
                &cfg.dataset,
                &cfg.split,
                &query_ids,
                &corpus_ids,
                &cfg.out_dir,
                &simd,
                &encoder_sha,
            ),
            "rq4" => run_full_scan(
                &corpus,
                &queries,
                dim,
                n_queries,
                cfg.top_k,
                cfg.batch,
                4,
                &cfg.dataset,
                &cfg.split,
                &query_ids,
                &corpus_ids,
                &cfg.out_dir,
                &simd,
                &encoder_sha,
            ),
            "bitmap-rq2" => run_bitmap_rq2(
                &corpus,
                &queries,
                dim,
                n_queries,
                cfg.top_k,
                cfg.batch,
                cfg.candidates,
                &cfg.dataset,
                &cfg.split,
                &query_ids,
                &corpus_ids,
                &cfg.out_dir,
                &simd,
                &encoder_sha,
            ),
            "sign-rq2" => run_sign_rq2(
                &corpus,
                &queries,
                dim,
                n_queries,
                cfg.top_k,
                cfg.batch,
                cfg.candidates,
                &cfg.dataset,
                &cfg.split,
                &query_ids,
                &corpus_ids,
                &cfg.out_dir,
                &simd,
                &encoder_sha,
            ),
            other => panic!("unknown method '{other}'. Supported: rq2, rq4, bitmap-rq2, sign-rq2",),
        }
    }

    eprintln!("\ndone. Results in {}/{}", cfg.out_dir, cfg.dataset);
}
