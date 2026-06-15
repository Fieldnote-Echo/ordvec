//! libFuzzer target for the `.ovfs` / `OVFS` loader (the FastScan b=2
//! persistence format — new in the ordvec format, no legacy `TV*` magic),
//! driven through the public `ordvec::RankQuantFastscan::load` entry point.
//!
//! The low-level `rank_io::load_fastscan` parser is crate-internal
//! (`pub(crate)`), so the fuzzer exercises it through `RankQuantFastscan::load`
//! — which runs that exact loader (the full public load path). `load` takes a
//! `&Path` and the only public load entry points are path-based (issue #6), so
//! a shared process-local scratch file (see [`scratch`]) feeds the loader the
//! fuzz bytes without per-iteration `mkstemp`/`unlink` churn.
//!
//! Contract: on arbitrary bytes the loader must return `Ok(..)` or `Err(..)` —
//! never panic, abort, or read out of bounds. libFuzzer treats any panic/abort
//! as a crash, so simply letting the result drop is the assertion.

#![no_main]

use libfuzzer_sys::fuzz_target;

mod scratch;

fuzz_target!(|data: &[u8]| {
    scratch::with_scratch_file(data, |path| {
        // The only thing under test: arbitrary bytes -> Ok | Err, no panic.
        let _ = ordvec::RankQuantFastscan::load(path);
    });
});
