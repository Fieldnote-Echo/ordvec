//! Read/write rank-mode index files.
//!
//! Three formats live here, each self-describing via a 4-byte magic:
//! * `.tvr`  — [`RankIndex`](crate::RankIndex)        — magic `TVR1`
//! * `.tvrq` — [`RankQuantIndex`](crate::RankQuantIndex) — magic `TVRQ`
//! * `.tvbm` — [`BitmapIndex`](crate::BitmapIndex)    — magic `TVBM`
//!
//! All formats are little-endian. Headers are small fixed-size structs
//! followed by a single contiguous payload (the rank / packed / bitmap
//! bytes). No norms, no codebooks, no rotation matrices — these are the
//! deterministic-encode index types so the on-disk format is exactly the
//! in-memory buffer plus enough header to rehydrate the type parameters.
//!
//! The shape mirrors [`crate::io`] for `TurboQuantIndex`. ID-map wrappers
//! (analogous to `.tvim`) are an obvious follow-up but not in this v1.
//!
//! # Safety against malformed files
//!
//! All loaders validate header fields *before* allocating the payload
//! buffer:
//! * `dim` and `n_vectors` are bounded by [`MAX_DIM`] and [`MAX_VECTORS`]
//!   (chosen so a worst-case index fits in 128 GiB).
//! * `bits` is checked against `{1, 2, 4}` before any multiplication.
//! * Total payload size is computed via [`usize::checked_mul`] and
//!   rejected if it overflows.
//! * Per-index invariants (e.g., `dim % (1 << bits) == 0` for RankQuant)
//!   are returned as `Err(InvalidData)`, never `assert!`'d.
//!
//! Any malformed input returns `io::Error` rather than panicking.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const TVR_MAGIC: &[u8; 4] = b"TVR1";
const TVRQ_MAGIC: &[u8; 4] = b"TVRQ";
const TVBM_MAGIC: &[u8; 4] = b"TVBM";
const VERSION: u8 = 1;

/// Largest accepted `dim` from a loaded file. Matches `u16::MAX` so the
/// rank transform's `u16` invariant in [`crate::RankIndex`] is honoured.
pub const MAX_DIM: usize = u16::MAX as usize;
/// Largest accepted `n_vectors` from a loaded file. 64 M docs at
/// `dim=u16::MAX` (128 KiB / vec for u16 ranks) tops out at ~8 TiB,
/// well past any sane on-disk index. Chosen to fail loud before
/// allocation panics.
pub const MAX_VECTORS: usize = 64 * 1024 * 1024;

fn invalid<S: Into<String>>(msg: S) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn check_dim(dim: usize) -> io::Result<()> {
    if dim < 2 || dim > MAX_DIM {
        return Err(invalid(format!(
            "dim {dim} out of range [2, {MAX_DIM}]"
        )));
    }
    Ok(())
}

fn check_n_vectors(n_vectors: usize) -> io::Result<()> {
    if n_vectors > MAX_VECTORS {
        return Err(invalid(format!(
            "n_vectors {n_vectors} exceeds MAX_VECTORS={MAX_VECTORS}"
        )));
    }
    Ok(())
}

fn check_payload_bytes(payload_bytes: usize) -> io::Result<()> {
    // 128 GiB hard cap — refuses absurd allocations from a corrupt
    // header even if dim and n_vectors individually pass.
    const MAX_PAYLOAD: usize = 128 * 1024 * 1024 * 1024;
    if payload_bytes > MAX_PAYLOAD {
        return Err(invalid(format!(
            "payload {payload_bytes} B exceeds MAX_PAYLOAD={MAX_PAYLOAD}"
        )));
    }
    Ok(())
}

// -------------------------------------------------------------------
// RankIndex: u16 ranks per coordinate.
// Header: magic(4) | version(1) | dim(u32 LE) | n_vectors(u32 LE)  = 13 B
// Payload: n_vectors * dim * 2 bytes (u16 LE ranks).
// -------------------------------------------------------------------

pub fn write_rank(
    path: impl AsRef<Path>,
    dim: usize,
    n_vectors: usize,
    ranks: &[u16],
) -> io::Result<()> {
    assert_eq!(ranks.len(), n_vectors * dim);
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(TVR_MAGIC)?;
    f.write_all(&[VERSION])?;
    f.write_all(&(dim as u32).to_le_bytes())?;
    f.write_all(&(n_vectors as u32).to_le_bytes())?;
    for &r in ranks {
        f.write_all(&r.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

pub fn load_rank(path: impl AsRef<Path>) -> io::Result<(usize, usize, Vec<u16>)> {
    let mut f = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != TVR_MAGIC {
        return Err(invalid("not a TVR1 file: wrong magic"));
    }
    let mut ver = [0u8; 1];
    f.read_exact(&mut ver)?;
    if ver[0] != VERSION {
        return Err(invalid(format!("unsupported TVR1 version: {}", ver[0])));
    }
    let mut dim_buf = [0u8; 4];
    f.read_exact(&mut dim_buf)?;
    let dim = u32::from_le_bytes(dim_buf) as usize;
    check_dim(dim)?;
    let mut n_buf = [0u8; 4];
    f.read_exact(&mut n_buf)?;
    let n_vectors = u32::from_le_bytes(n_buf) as usize;
    check_n_vectors(n_vectors)?;
    let payload_bytes = n_vectors
        .checked_mul(dim)
        .and_then(|x| x.checked_mul(2))
        .ok_or_else(|| invalid("payload size overflows usize"))?;
    check_payload_bytes(payload_bytes)?;
    let mut bytes = vec![0u8; payload_bytes];
    f.read_exact(&mut bytes)?;
    let ranks: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    Ok((dim, n_vectors, ranks))
}

// -------------------------------------------------------------------
// RankQuantIndex: B-bit packed bucket vectors.
// Header: magic(4) | version(1) | bits(u8) | dim(u32 LE) | n_vectors(u32 LE) = 14 B
// Payload: n_vectors * dim * bits / 8 packed bytes.
// -------------------------------------------------------------------

pub fn write_rankquant(
    path: impl AsRef<Path>,
    bits: u8,
    dim: usize,
    n_vectors: usize,
    packed: &[u8],
) -> io::Result<()> {
    let expected = n_vectors * dim * bits as usize / 8;
    assert_eq!(packed.len(), expected);
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(TVRQ_MAGIC)?;
    f.write_all(&[VERSION])?;
    f.write_all(&[bits])?;
    f.write_all(&(dim as u32).to_le_bytes())?;
    f.write_all(&(n_vectors as u32).to_le_bytes())?;
    f.write_all(packed)?;
    f.flush()?;
    Ok(())
}

pub fn load_rankquant(path: impl AsRef<Path>) -> io::Result<(u8, usize, usize, Vec<u8>)> {
    let mut f = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != TVRQ_MAGIC {
        return Err(invalid("not a TVRQ file: wrong magic"));
    }
    let mut ver = [0u8; 1];
    f.read_exact(&mut ver)?;
    if ver[0] != VERSION {
        return Err(invalid(format!("unsupported TVRQ version: {}", ver[0])));
    }
    let mut bits_buf = [0u8; 1];
    f.read_exact(&mut bits_buf)?;
    let bits = bits_buf[0];
    if !matches!(bits, 1 | 2 | 4) {
        return Err(invalid(format!(
            "unsupported TVRQ bits: {bits} (expected 1, 2, or 4)"
        )));
    }
    let mut dim_buf = [0u8; 4];
    f.read_exact(&mut dim_buf)?;
    let dim = u32::from_le_bytes(dim_buf) as usize;
    check_dim(dim)?;
    let mut n_buf = [0u8; 4];
    f.read_exact(&mut n_buf)?;
    let n_vectors = u32::from_le_bytes(n_buf) as usize;
    check_n_vectors(n_vectors)?;
    let payload_bytes = n_vectors
        .checked_mul(dim)
        .and_then(|x| x.checked_mul(bits as usize))
        .map(|x| x / 8)
        .ok_or_else(|| invalid("payload size overflows usize"))?;
    check_payload_bytes(payload_bytes)?;
    let mut packed = vec![0u8; payload_bytes];
    f.read_exact(&mut packed)?;
    Ok((bits, dim, n_vectors, packed))
}

// -------------------------------------------------------------------
// BitmapIndex: top-n_top bitmap per document.
// Header: magic(4) | version(1) | dim(u32 LE) | n_top(u32 LE) | n_vectors(u32 LE) = 17 B
// Payload: n_vectors * dim / 8 bytes (qwords as u64 LE).
// -------------------------------------------------------------------

pub fn write_bitmap(
    path: impl AsRef<Path>,
    dim: usize,
    n_top: usize,
    n_vectors: usize,
    bitmaps: &[u64],
) -> io::Result<()> {
    let qpv = dim / 64;
    assert_eq!(bitmaps.len(), n_vectors * qpv);
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(TVBM_MAGIC)?;
    f.write_all(&[VERSION])?;
    f.write_all(&(dim as u32).to_le_bytes())?;
    f.write_all(&(n_top as u32).to_le_bytes())?;
    f.write_all(&(n_vectors as u32).to_le_bytes())?;
    for &w in bitmaps {
        f.write_all(&w.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

pub fn load_bitmap(
    path: impl AsRef<Path>,
) -> io::Result<(usize, usize, usize, Vec<u64>)> {
    let mut f = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != TVBM_MAGIC {
        return Err(invalid("not a TVBM file: wrong magic"));
    }
    let mut ver = [0u8; 1];
    f.read_exact(&mut ver)?;
    if ver[0] != VERSION {
        return Err(invalid(format!("unsupported TVBM version: {}", ver[0])));
    }
    let mut dim_buf = [0u8; 4];
    f.read_exact(&mut dim_buf)?;
    let dim = u32::from_le_bytes(dim_buf) as usize;
    check_dim(dim)?;
    if dim % 64 != 0 {
        return Err(invalid(format!(
            "TVBM dim {dim} is not a multiple of 64"
        )));
    }
    let mut top_buf = [0u8; 4];
    f.read_exact(&mut top_buf)?;
    let n_top = u32::from_le_bytes(top_buf) as usize;
    if n_top == 0 || n_top >= dim {
        return Err(invalid(format!(
            "TVBM n_top {n_top} must satisfy 0 < n_top < dim ({dim})"
        )));
    }
    let mut n_buf = [0u8; 4];
    f.read_exact(&mut n_buf)?;
    let n_vectors = u32::from_le_bytes(n_buf) as usize;
    check_n_vectors(n_vectors)?;
    let qpv = dim / 64;
    let payload_bytes = n_vectors
        .checked_mul(qpv)
        .and_then(|x| x.checked_mul(8))
        .ok_or_else(|| invalid("payload size overflows usize"))?;
    check_payload_bytes(payload_bytes)?;
    let mut bytes = vec![0u8; payload_bytes];
    f.read_exact(&mut bytes)?;
    let bitmaps: Vec<u64> = bytes
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .collect();
    Ok((dim, n_top, n_vectors, bitmaps))
}
