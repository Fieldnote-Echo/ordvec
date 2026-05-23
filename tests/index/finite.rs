//! Strict finite-input contract: public `add` / `search` / probe entry
//! points reject non-finite (NaN / ±Inf) inputs fail-fast via the shared
//! `assert_all_finite` guard. These spot-check one path per substrate
//! type; the panic message contains "non-finite".

use ordvec::{Bitmap, Rank, RankQuant, SignBitmap};

use crate::{make_corpus, D};

#[test]
#[should_panic(expected = "non-finite")]
fn rank_add_rejects_nan() {
    let mut idx = Rank::new(D);
    let mut v = make_corpus(7);
    v[3] = f32::NAN;
    idx.add(&v);
}

#[test]
#[should_panic(expected = "non-finite")]
fn rankquant_search_asymmetric_rejects_inf() {
    let mut idx = RankQuant::new(D, 2);
    idx.add(&make_corpus(8));
    let mut q = vec![0.1f32; D];
    q[0] = f32::INFINITY;
    let _ = idx.search_asymmetric(&q, 10);
}

#[test]
#[should_panic(expected = "non-finite")]
fn bitmap_top_m_candidates_rejects_nan() {
    let mut idx = Bitmap::new(D, D / 4);
    idx.add(&make_corpus(9));
    let mut q = vec![0.1f32; D];
    q[5] = f32::NAN;
    let _ = idx.top_m_candidates(&q, 16);
}

#[test]
#[should_panic(expected = "non-finite")]
fn sign_bitmap_build_query_rejects_neg_inf() {
    let idx = SignBitmap::new(D);
    let mut q = vec![0.1f32; D];
    q[1] = f32::NEG_INFINITY;
    let _ = idx.build_query_bitmap(&q);
}
