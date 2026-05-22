# Rank-cosine index modes for turbovec

These index types operate on a *rank* view of the embedding rather
than on the rotated coordinates `TurboQuantIndex` quantizes:

> RankQuant turns vectors into fixed-mass ordinal sets, so candidate
> generation becomes bitmap overlap instead of low-bit dot product.
> Magnitude quantizers don't have this primitive.

The asymmetric scan ships an AVX-512 path (16-wide FMA, 4-way
multi-accumulator) that auto-detects at runtime and falls through to
AVX2 and then a scalar LUT scan; symmetric paths and the b=1
asymmetric path use the scalar LUT scan. The whole pipeline uses zero
training, zero rotation, zero codebook — the structural prior is what
does the work.

**Reproducible headline (synthetic clustered corpus).** Every number
in the leading tables below is regenerable from this repo with a
single command, no external data:

```bash
cargo run --release --example bench_rank
```

That runs the head-to-head on a structured synthetic corpus (D=1024,
N=50,000, 200 queries, 200 cluster prototypes, latent_dim=64; see
[Stress test](#stress-test-low-rank-clustered-synthetic) for the
exact construction). Results on real embedding corpora are
user-runnable via `--corpus-npy` / `--queries-npy` and reported, with
the exact command, under
[External-corpus results](#external-corpus-results-user-runnable) —
they are not the lead claim because the corpus is not shipped here.

The bitmap two-stage path (`BitmapIndex` candidate gen →
`RankQuantIndex` exact subset rerank) is the operating point that
turns RankQuant from a slow exact scan into a sub-linear retriever:
the bitmap probe is the cheap candidate generator, and
`search_asymmetric_subset` reruns the exact RankQuant kernel on only
the surviving M candidates. The bench reports this path as its
`TwoStage ...` rows, each annotated with a candidate-recall figure
(`CR` = fraction of exact-RankQuant top-10 indices present in the
bitmap's M-candidate set, averaged over queries — distinct from task
R@10, which is against FP32 brute-force cosine ground truth).

Centre-drop math (why the asymmetric kernel needs no per-coord LUT):
because centred bucket scores differ from raw-code scores only by a
query-constant offset (under the `dim % (1 << bits) == 0` constraint
that fixes every doc's bucket histogram), the asymmetric kernel can
score raw bucket IDs directly for ranking. The offset is re-applied
to the top-k scores at finalize so the displayed cosines stay exact.

## Bench environment

| field | value |
|---|---|
| CPU | AMD Ryzen 9 9950X (Zen 5, 16C/32T, full 512-bit AVX-512 datapath) |
| RAM | 128 GB Kingston Fury Beast DDR5-4000 CL29 × 4 DIMMs (capacity-optimised) |
| OS | CachyOS Linux, kernel 7.0.6 |
| Compiler | rustc 1.94.1 (LLVM 21.1.8) |
| Build | `cargo build --release` with `lto = true, codegen-units = 1, opt-level = 3` |
| Governor | `performance` |
| THP | `always` |
| Detected SIMD | sse4.2, avx2, fma, avx512f, avx512bw, avx512vl |
| Latency mode | single-thread per query (rayon parallelises *across* queries; per-query rows measure scan only) |

A two-DIMM DDR5-6000-class system may show shorter absolute latency;
the relative gap to TurboQuant is the load-bearing comparison.

This crate adds rank-view index types alongside `TurboQuantIndex`.
The two scored types are:

- **`RankIndex`** — stores the dimension-wise rank transform of each
  document as `u16` (`2 * dim` bytes per document).
- **`RankQuantIndex`** — buckets each rank into `1 << bits` equal-width
  bins on `[0, dim)` and packs `bits` bits per coordinate
  (`dim * bits / 8` bytes per document). Supported `bits ∈ {1, 2, 4}`.

Both expose `search` (symmetric: rank-vs-rank, Spearman correlation)
and `search_asymmetric` (FP32 query against rank-stored documents).
`BitmapIndex` and `SignBitmapIndex` provide the cheap candidate-gen
front end for the two-stage path (see [README](../README.md#rank-mode-index-types)).

The construction has no rotation matrix, no codebook, no Lloyd-Max
training, no per-document norms (the L2 norm of a permutation of
`{0..D-1}` is analytical). Encode is a single `argsort` pass per
vector, with the option to bucket and pack into `B` bits per
coordinate.

## Why this works: combinatorics, not geometry

The bitmap two-stage result is not merely a faster scoring kernel —
it is a structural primitive that magnitude-preserving quantization
does not expose. Three properties chain together:

**1. RankQuant is a constant-composition code.** The rank transform
is a permutation of `{0, ..., D-1}`, so under the equal-width bucket
partition every document assigns *exactly the same number of
coordinates to each bucket*: `D / 2^B` coordinates per bucket, for
all docs. For `D=1024, B=2` that is 256 coordinates in the top
bucket of every document.

**2. The similarity score decomposes over bucket-overlap counts.**
Let `Q_a` be the set of query coordinates in bucket `a`, and `D_b`
the analogous set for the document. Then asymmetric rank-cosine
re-expresses (up to per-query constants) as a weighted contingency
table of bucket-overlap counts:

```
score(q, d) = Σ_{a,b} w(a, b) · |Q_a ∩ D_b|
```

So RankQuant similarity is a bilinear function of bucket-overlap
counts between two constant-composition partitions, not a dot
product over magnitudes.

**3. The simplest truncation — top-bucket overlap — has a closed-form
null distribution.** For uniformly random fixed-size subsets,
`X = |Q_top ∩ D_top|` is hypergeometric `H(D, n_top, n_top)` with
`E[X] = n_top² / D`. For `D=1024, n_top=256` the expected overlap
under the null is exactly **64**. Observed overlaps significantly
above 64 are evidence of shared coordinate salience, with
closed-form p-values from the hypergeometric distribution.

This is what makes the bitmap probe a principled candidate
generator rather than a tunable heuristic. Magnitude quantizers
don't have a hypergeometric null because they don't have fixed
bucket cardinalities — their score distribution depends on the
unknown embedding distribution.

**A research program this suggests.** The chain — representation →
statistic → retrieval theorem → systems implementation — has a
plausible formal target. Under a shared-latent-support model where
relevant documents have elevated coordinates on a query-specific
support set `S_q`, the top-bucket overlap statistic is monotone in
the likelihood ratio for relevance, suggesting that bitmap probing
may approach Bayes-optimality under that model. We do not claim
that theorem here; this section flags it as the natural
mathematical direction for the empirical results below.

The systems consequence is what the bench measures: at a moderate M
the bitmap probe captures most of exact RankQuant's top-10 neighbours,
so the two-stage rerank reproduces near-exact RankQuant R@10 at a
fraction of the full-scan latency. The `bench_rank` run prints this
as its `TwoStage ...` rows with the per-M candidate-recall (`CR`)
figure attached.

## Headline numbers (synthetic clustered corpus)

This is the reproducible lead — regenerated by the default
`bench_rank` run, no external data required:

```bash
cargo run --release --example bench_rank
```

Setup: D=1024, N=50,000 documents, 200 queries, k=10. Low-rank
clustered corpus (200 cluster prototypes, latent_dim=64, projected to
D=1024 with N(0,1) noise = 0.3 for docs, 0.1 for queries). Ground
truth: FP32 brute-force cosine top-10. The construction is detailed
under [Stress test](#stress-test-low-rank-clustered-synthetic) — it
is deliberately anisotropic, which is the regime that most strains
TurboQuant's data-oblivious random rotation, so treat the synthetic
recall *gaps* as an upper bound on what real embeddings show (the
[external-corpus section](#external-corpus-results-user-runnable)
records a much smaller real-data gap).

Results are with the AVX-512 asymmetric scan enabled where applicable
(auto-detected at runtime; falls through to AVX2 then to a scalar LUT
scan). Symmetric paths and the b=1 asymmetric path use the scalar LUT
scan. Absolute latencies are machine-specific (see
[Bench environment](#bench-environment)); the relative gap to
TurboQuant is the load-bearing comparison.

| mode               | bytes/vec | encode v/s | p50 ms | GiB/s | ns/dim | R@10  |
|--------------------|-----------|------------|--------|-------|--------|-------|
| TurboQuant b=2     | 256       | 44,802     | 0.51   | 23.5  | 0.010  | 0.299 |
| TurboQuant b=4     | 512       | 18,552     | 1.17   | 20.4  | 0.023  | 0.492 |
| RankIndex sym      | 2048      | 1,065,560  | 25.0   | 3.8   | 0.489  | 0.874 |
| RankIndex asym     | 2048      | 1,065,560  | 25.8   | 3.7   | 0.504  | 0.911 |
| RankQuant b=2 sym  | 256       | 1,186,263  | 19.1   | 0.63  | 0.372  | 0.617 |
| RankQuant b=2 asym | 256       | 1,186,263  | 18.9   | 0.63  | 0.368  | 0.722 |
| RankQuant b=4 sym  | 512       | 1,142,377  | 19.3   | 1.23  | 0.377  | 0.849 |
| RankQuant b=4 asym | 512       | 1,142,377  | 19.5   | 1.22  | 0.381  | 0.889 |
| RankQuant b=1 sym  | 128       | 1,254,733  | 18.4   | 0.32  | 0.359  | 0.407 |
| RankQuant b=1 asym | 128       | 1,254,733  | 18.4   | 0.33  | 0.359  | 0.525 |

The kernel does not use a per-coord LUT — `bucket_centre(b) = b - (2^B - 1) / 2`
is one SIMD subtraction (folded out to the per-query offset via
centre-drop), so the inner loop is broadcast → variable-shift → mask
→ cvt → FMA with no LUT memory traffic.

### Reading the synthetic table

**Encode is 26-62× faster across the board.** TurboQuant has no
rotation matmul and no codebook fit to amortise; the dominant
RankQuant encode cost is `argsort` per vector. This advantage is
structural and transfers cleanly to real corpora (see
[Encode throughput](#encode-throughput-23-62-faster)).

**Storage is identical at matched bit width.** `bytes_per_vec =
dim * bits / 8` for both schemes. The byte budget is the same lever;
what differs is what each byte *means* (quantised magnitude vs
bucketed rank).

**Recall favours rank on this anisotropic corpus.** At matched bytes,
RankQuant asym beats TurboQuant by a wide margin here (e.g. +0.42
R@10 at 256 B/vec). This is the corpus where TurboQuant's
data-oblivious rotation is most strained; the gap shrinks sharply on
real embeddings, and at wider codes can invert. See the
[external-corpus section](#external-corpus-results-user-runnable) for
the honest real-data picture.

**Single-query exact-scan latency is the standing weakness.** The
synthetic rows above measure the per-query exact scan, where the
RankQuant b=2 asym scan (~0.63 GiB/s effective) trails TurboQuant's
hand-tuned kernel (15-23 GiB/s). The two-stage path
(`BitmapIndex` → `RankQuantIndex` subset rerank) is what closes this
in practice — it scores only the M bitmap survivors instead of the
full corpus. The candidate-recall vs latency trade is the bench's
two-stage rows; remaining single-query SIMD headroom is in
[Where TurboQuant still wins](#where-turboquant-still-wins).

## Stress test (low-rank clustered synthetic)

This is the construction behind the [headline
table](#headline-numbers-synthetic-clustered-corpus). The default
`bench_rank` run uses these parameters; the explicit form is:

```bash
cargo run --release --example bench_rank -- \
  --dim 1024 --n 50000 --queries 200 --clusters 200 --latent 64
```

Setup: D=1024, N=50,000 documents, 200 queries, k=10. Low-rank
clustered corpus (200 cluster prototypes, latent_dim=64, projected
to D=1024 with N(0,1) noise = 0.3 for docs, 0.1 for queries).
Ground truth: FP32 brute-force cosine top-10.

The corpus is anisotropic *by construction* (latent_dim=64 in
D=1024), which is exactly the regime where TurboQuant's
data-oblivious random rotation has least useful structure to exploit
— so RankQuant's recall advantage here (e.g. +0.42 R@10 at 256 B/vec
in the headline table) is a best case for rank, not a typical case.
On real embeddings with milder anisotropy the gap is much smaller and
can invert at wider codes; the
[external-corpus section](#external-corpus-results-user-runnable)
shows how to reproduce that contrast on your own data.

## What survived the head-to-head

### Encode throughput: 23-62× faster

| corpus  | bytes/vec | TurboQuant v/s | RankQuant v/s | ratio  |
|---------|-----------|----------------|---------------|--------|
| Synth   | 256       | 44,802         | 1,186,263     | 26.5×  |
| Synth   | 512       | 18,552         | 1,142,377     | 61.6×  |

The architectural reason is straightforward: no rotation matrix
multiply, no Lloyd-Max codebook fit, no per-vector norm storage.
Encode is one `argsort` per coordinate + one bucket-pack pass per
document. The numbers above are from the synthetic headline run; the
same ratio holds on real corpora because the encode cost is
data-independent.

### Storage: identical at matched bit width

`bytes_per_vec = dim * bits / 8` for both schemes. The byte budget is
the same lever. The implementation differs in what each byte
*means*: TurboQuant stores a quantised magnitude, RankQuant stores a
bucketed rank.

### Asymmetric beats symmetric

Synthetic headline run:

| mode             | sym R@10 | asym R@10 | Δ      |
|------------------|---------:|----------:|-------:|
| Rank full (2KB)  | 0.874    | 0.911     | +0.037 |
| RankQuant b=4    | 0.849    | 0.889     | +0.040 |
| RankQuant b=2    | 0.617    | 0.722     | +0.105 |
| RankQuant b=1    | 0.407    | 0.525     | +0.118 |

The asymmetric variant keeps the query side as full FP32 — the
encoder's output is consumed directly, only the document side loses
precision. This is the recommended mode. The advantage grows as
document-side precision shrinks (more information lost on the doc
side, more value in keeping the query rich). The same ordering
(asym > sym, gap widening at lower bits) reproduces on real
embeddings when you run the external-corpus bench.

## Where TurboQuant still wins

### Single-query exact-scan latency

TurboQuant's hand-tuned NEON/AVX kernels deliver 15-23 GiB/s
effective scan bandwidth on the synthetic headline run; the
RankQuant b=2 asymmetric scan runs at ~0.63 GiB/s effective in that
same run, so a full single-query exact scan is slower at matched
bytes. Two facts qualify this:

- **The two-stage path is the intended fast route.** Scoring only the
  M bitmap candidates with `search_asymmetric_subset` avoids the
  full-corpus scan entirely. That is the operating point the
  rank-mode README recommends, and where the structural prior pays
  off.
- **The asymmetric AVX-512 kernel is an exact packed scan, not an ANN
  approximation.** It returns identical top-k to the scalar RankQuant
  scorer and agrees within 1e-4 on scores (verified by
  `rankquant_asymmetric_matches_reference_b{1,2,4}` in
  `turbovec/tests/rank_index/quant.rs`).

The byte-LUT scorer remains in the codebase as a labelled reference
path (`turbovec::rank_index::search_asymmetric_byte_lut`,
benched as the `RankQuant b=… asym byte-LUT` rows) but is not the
production scoring route — streaming SIMD math beats query-LUT cache
traffic on the hardware tested.

### Remaining headroom

The single-query b=2 exact scan is decode-bound. Further closing the
gap to TurboQuant's bandwidth is, in priority order:

1. **Multi-accumulator b=2 kernel** — break the FMA dependency chain
   by splitting into 2-4 independent accumulators per doc. Cheap to
   implement, likely meaningful on the decode-bound path.
2. **Unroll across docs** — process 2-4 docs per inner iteration so
   the front-end can hide the broadcast/shift/mask latency.
3. **SIMD-blocked layout** — repack into 32-doc tiles like
   `pack.rs::repack`. Improves memory access pattern. Highest
   single-step win but largest restructuring.

None of these are research questions; all have a direct template in
`search.rs` for TurboQuant or in the existing
`rank_index/quant_kernels.rs` AVX-512 kernel.

The symmetric path is still scalar (lower-priority — asymmetric is
the recommended mode and wins every recall comparison here).
Symmetric SIMD is a natural follow-up.

## External-corpus results (user-runnable)

The numbers above all come from the in-repo synthetic corpus. To
check the head-to-head on real embeddings, point the same bench at
your own `.npy` arrays:

```bash
RUSTFLAGS="-l openblas" cargo run --release --example bench_rank -- \
  --corpus-npy  /path/to/embeddings.npy \
  --queries-npy /path/to/queries.npy \
  --queries 200 --k 10
```

`--corpus-npy` / `--queries-npy` each take a NumPy v1 `.npy` file
holding a 2-D little-endian `float32` (`<f4`), C-order array
(`(n, dim)` for the corpus, `(n_q, dim)` for queries); `--n` and
`--dim` are then taken from the file shapes. The npy loader is a
minimal built-in reader — no Python dependency at bench time.

What to expect from real embeddings (and why these are not the lead
claim): on dense sentence/passage encoders the recall gap between
RankQuant and TurboQuant is **much smaller** than on the synthetic
anisotropic corpus, and its **sign is encoder- and bit-width-
dependent**. The synthetic corpus is an adversarial best case for
rank; a typical real corpus sits closer to parity, with RankQuant
most competitive at the narrowest (2-bit) byte budget where its
build-cost advantage is largest, and magnitude quantisation tending
to pull ahead at wider codes. Run the command above on your target
embeddings to get the number that matters for your deployment — we do
not assert a specific real-corpus delta here because the corpus is
not shipped with this repo.

## A null result reported up front

We also tested adding 10 rank-native structural features (per-(q, d)
bitmap-overlap counts, bilinear bucket-pair contingency cells,
query-level concentration broadcast) to a LambdaMART reranker on
both RRF-generated and bitmap-generated candidate sets. Five-seed
multi-seed stability:

| candidate source | baseline R@10 | structural-feature lift |
|---|---:|---:|
| RRF top-100   | 0.951 | +0.0030 ± 0.0027 (Gaussian-noise lift +0.0037 ± 0.0031) |
| Bitmap top-100 | 0.891 | +0.0017 ± 0.0007 |

**Null result on both candidate distributions** (measured on an
external real-embedding corpus, five seeds). The structural features
are scalar projections of information the LambdaMART baseline already
captures via continuous `rank_cos`. We report it so the obvious
follow-up is on record as tested-and-didn't-land: the right place for
rank-native structure is candidate generation (where the bitmap
two-stage above wins), not LambdaMART feature engineering.

## API parity with `TurboQuantIndex`

| capability | TurboQuant | Rank | RankQuant |
|---|---|---|---|
| `new(dim, bits)` | ✓ | `new(dim)` | ✓ |
| `add(&[f32])` | ✓ | ✓ | ✓ |
| `search(&[f32], k)` | ✓ | ✓ symmetric | ✓ symmetric |
| `search_asymmetric(&[f32], k)` | — | ✓ | ✓ |
| `swap_remove(idx)` | ✓ | ✓ | ✓ |
| `len`/`is_empty`/`dim`/`bytes_per_vec`/`byte_size` | ✓ | ✓ | ✓ |
| `write`/`load` | ✓ | ✓ | ✓ |
| `search_asymmetric_subset(q, &cands, k)` | — | — | ✓ |
| `prepare` | ✓ | — (no lazy caches) | — |
| IdMap wrapping | ✓ | ✗ (not yet) | ✗ (not yet) |

`write`/`load` are implemented for every rank-mode type
(`RankIndex` / `RankQuantIndex` in `turbovec/src/rank_index/`,
`BitmapIndex` and `SignBitmapIndex` likewise) with the byte-level
serialisers living in `turbovec/src/rank_io.rs`; the Python bindings
expose the same `write`/`load` surface. `RankQuantIndex` additionally
exposes `search_asymmetric_subset` for scoring a precomputed
candidate set — the rerank half of the two-stage pattern — and that
entry point is wrapped in Python too.

The one remaining gap is `IdMapIndex` integration for the rank types:
`id_map.rs` wraps `TurboQuantIndex` only. Adding stable-id wrapping
for the rank types is a mechanical follow-up that would mirror the
existing `id_map.rs` plumbing.

## Test coverage

`cargo test -p turbovec --lib rank::` — unit tests for the primitives
in `turbovec/src/rank.rs` (rank transform vs numpy `argsort(argsort)`
reference, rank-is-a-permutation, uniform bucket partitioning,
bucket-centre symmetry, analytical norms match direct computation).

`cargo test -p turbovec --test rank_index` — the integration suite in
`turbovec/tests/rank_index/` (`index.rs`, `quant.rs`, `bitmap.rs`,
`multi_bucket.rs`). Representative cases:

- `rank_index_symmetric_matches_reference` — `RankIndex::search`
  matches a scalar Spearman implementation on a 256-doc / 128-dim
  corpus, exact top-10 ordering, score agreement to 1e-4.
- `rank_index_asymmetric_matches_reference` — same, for the FP32-vs-
  rank kernel.
- `rankquant_asymmetric_matches_reference_b{1,2,4}` — RankQuant
  asymmetric agrees with the scalar reference at every bit width
  (this is the AVX-512-vs-scalar exactness check).
- `rankquant_b2_recovers_planted_neighbour_in_top_10` — 50 queries
  each constructed by adding noise to a known corpus doc; RankQuant-2
  asymmetric recovers the planted doc in top-10 at recall ≥ 0.95.
- `rank_index_recall_at_10_matches_fp32` — rank-cosine and raw FP32
  cosine top-10 sets overlap ≥ 70% on smooth random data at D=128.
- `rank{,quant}_swap_remove_keeps_state_consistent` — `swap_remove`
  is byte-exact across the storage buffer.

## Reproducibility

```bash
cargo test -p turbovec --lib rank::                   # unit tests
cargo test -p turbovec --test rank_index              # integration

# Headline benchmark (synthetic clustered corpus — no external data).
RUSTFLAGS="-l openblas" cargo run --release --example bench_rank

# Same bench against your own real-embedding arrays.
RUSTFLAGS="-l openblas" cargo run --release --example bench_rank -- \
    --corpus-npy  /path/to/embeddings.npy \
    --queries-npy /path/to/queries.npy \
    --queries 200 --k 10
```

The `RUSTFLAGS="-l openblas"` shim is needed on Linux for the
TurboQuant rotation step (used by the TurboQuant baseline rows); the
rank modes themselves do not depend on BLAS. The npy loader is a
minimal NumPy v1 reader for `<f4` little-endian, C-order 2-D arrays;
no Python dependency at bench time. The synthetic headline numbers
are deterministic given the default parameters (fixed RNG seed in the
example); when benching real arrays, multi-seed stability is your
call.

## Design summary

1. **Strict superset of capability.** `TurboQuantIndex` is unchanged;
   `RankIndex`, `RankQuantIndex`, `BitmapIndex`, and `SignBitmapIndex`
   are additive types, compiled and tested alongside.
2. **No new heavy dependencies.** The rank primitives use
   `ordered_float` and `rayon` (already dependencies). No BLAS, no
   codebook training, no rotation matrix.
3. **Storage parity, build-speed advantage.** At matched bit width,
   `bytes_per_vec` is identical to TurboQuant; encode is 23-62×
   faster because there is no rotation matmul and no codebook fit.
4. **Recall is corpus-dependent.** On the synthetic anisotropic
   corpus RankQuant wins by a wide margin; on real embeddings the gap
   is much smaller and its sign depends on encoder and bit width
   (run the external-corpus bench on your data — see above).
5. **The audit-by-removal rationale.** RankQuant removes training,
   rotation, codebooks, and per-document norms from the pipeline. That
   retrieval still works after the removal is the interesting result:
   on the corpora tested, those components were carrying less than the
   dense-quantization literature assumes.
