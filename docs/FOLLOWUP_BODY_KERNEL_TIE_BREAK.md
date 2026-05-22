# Follow-up: deterministic tie-breaking for body bitmap candidate selection

`BitmapIndex::top_m_candidates` and `top_m_candidates_batched`
currently partition on bitmap overlap score alone. Boundary ties are
rarer than in count-fold because body scores have wider spread
(0..n_top = 0..256), but the structural nondeterminism is identical:
`select_nth_unstable_by` may choose different equal-scored docs at
the cutoff across runs or dispatch paths.

**Fix**: add composite-key ordering `(score desc, doc_id asc)` to
both the partition predicate (`select_nth_unstable_by`) and the
post-partition sort (`sort_unstable_by`), mirroring the rule the
count-fold spec mandates.

```rust
let mut cmp = |&a: &u32, &b: &u32| {
    scores[b as usize]
        .cmp(&scores[a as usize])
        .then_with(|| a.cmp(&b))
};
idx.select_nth_unstable_by(m_eff - 1, &mut cmp);
idx[..m_eff].sort_unstable_by(&mut cmp);
```

**Out of scope for the count-fold experiment.** Rolling this in
would contaminate benchmark attribution — if numbers move, we
wouldn't know whether count-fold helped or whether body-kernel
determinism changed the candidate set composition. File as a
separate PR after the count-fold experiment closes.

Surfaced in: `docs/SPEC_COUNT_FOLD_TIER.md` v1.4 revision history.
