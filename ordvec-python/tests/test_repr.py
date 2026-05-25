"""``__repr__`` formatting for the four index classes.

Pins the human-readable representation (dim / bits / n_top / length) that
ML practitioners see when they ``print()`` an index in a notebook, so the
format does not drift silently.
"""

import numpy as np

import ordvec


def _rows(n: int, dim: int) -> np.ndarray:
    return np.arange(n * dim, dtype=np.float32).reshape(n, dim)


def test_rank_repr() -> None:
    idx = ordvec.Rank(8)
    assert repr(idx) == "Rank(dim=8, n=0)"
    idx.add(_rows(3, 8))
    assert repr(idx) == "Rank(dim=8, n=3)"


def test_rank_quant_repr() -> None:
    idx = ordvec.RankQuant(8, 2)
    assert repr(idx) == "RankQuant(dim=8, bits=2, n=0)"
    idx.add(_rows(2, 8))
    assert repr(idx) == "RankQuant(dim=8, bits=2, n=2)"


def test_bitmap_repr() -> None:
    idx = ordvec.Bitmap(64, 16)
    assert repr(idx) == "Bitmap(dim=64, n_top=16, n=0)"
    idx.add(_rows(5, 64))
    assert repr(idx) == "Bitmap(dim=64, n_top=16, n=5)"


def test_sign_bitmap_repr() -> None:
    idx = ordvec.SignBitmap(64)
    assert repr(idx) == "SignBitmap(dim=64, n=0)"
    idx.add(_rows(4, 64))
    assert repr(idx) == "SignBitmap(dim=64, n=4)"
