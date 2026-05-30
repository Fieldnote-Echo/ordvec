#!/usr/bin/env python3
"""Unit tests for release_pypi_canonical_dist.py."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


SCRIPT = Path(__file__).with_name("release_pypi_canonical_dist.py")
SPEC = importlib.util.spec_from_file_location("release_pypi_canonical_dist", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
canonical = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(canonical)


def write(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


class CanonicalPyPIDistTests(unittest.TestCase):
    def test_missing_pypi_release_uses_current_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            built = root / "built"
            out = root / "out"
            built.mkdir()
            write(built / "ordvec-0.3.0.tar.gz", b"fresh sdist")
            write(built / "ordvec-0.3.0-cp310-abi3-win_amd64.whl", b"fresh wheel")

            old_fetch = canonical.fetch_pypi_payload
            canonical.fetch_pypi_payload = lambda version: None
            try:
                with redirect_stdout(io.StringIO()):
                    canonical.canonicalize("0.3.0", built, out)
            finally:
                canonical.fetch_pypi_payload = old_fetch

            self.assertEqual((out / "ordvec-0.3.0.tar.gz").read_bytes(), b"fresh sdist")
            self.assertEqual((out / "ordvec-0.3.0-cp310-abi3-win_amd64.whl").read_bytes(), b"fresh wheel")

    def test_existing_pypi_release_uses_verified_remote_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            built = root / "built"
            remote = root / "remote"
            out = root / "out"
            built.mkdir()
            remote.mkdir()

            write(built / "ordvec-0.3.0.tar.gz", b"rebuilt sdist")
            write(built / "ordvec-0.3.0-cp310-abi3-win_amd64.whl", b"rebuilt wheel")
            sdist_sha = write(remote / "ordvec-0.3.0.tar.gz", b"pypi sdist")
            wheel_sha = write(remote / "ordvec-0.3.0-cp310-abi3-win_amd64.whl", b"pypi wheel")

            payload = {
                "urls": [
                    {
                        "filename": "ordvec-0.3.0.tar.gz",
                        "url": (remote / "ordvec-0.3.0.tar.gz").as_uri(),
                        "digests": {"sha256": sdist_sha},
                    },
                    {
                        "filename": "ordvec-0.3.0-cp310-abi3-win_amd64.whl",
                        "url": (remote / "ordvec-0.3.0-cp310-abi3-win_amd64.whl").as_uri(),
                        "digests": {"sha256": wheel_sha},
                    },
                ]
            }

            old_fetch = canonical.fetch_pypi_payload
            canonical.fetch_pypi_payload = lambda version: payload
            try:
                with redirect_stdout(io.StringIO()):
                    canonical.canonicalize("0.3.0", built, out)
            finally:
                canonical.fetch_pypi_payload = old_fetch

            self.assertEqual((out / "ordvec-0.3.0.tar.gz").read_bytes(), b"pypi sdist")
            self.assertEqual((out / "ordvec-0.3.0-cp310-abi3-win_amd64.whl").read_bytes(), b"pypi wheel")

    def test_existing_pypi_release_rejects_filename_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            built = root / "built"
            out = root / "out"
            built.mkdir()
            write(built / "ordvec-0.3.0.tar.gz", b"fresh sdist")

            payload = {
                "urls": [
                    {
                        "filename": "ordvec-0.3.0-cp310-abi3-win_amd64.whl",
                        "url": "file:///unused",
                        "digests": {"sha256": "0" * 64},
                    }
                ]
            }

            old_fetch = canonical.fetch_pypi_payload
            canonical.fetch_pypi_payload = lambda version: payload
            try:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    canonical.canonicalize("0.3.0", built, out)
            finally:
                canonical.fetch_pypi_payload = old_fetch


if __name__ == "__main__":
    unittest.main()
