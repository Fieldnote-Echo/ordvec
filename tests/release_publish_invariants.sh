#!/usr/bin/env bash
#
# Release-publish SBOM invariants — pinned in CI.
#
# release-crate.yml / release-python.yml are workflow_dispatch-only, so their
# "generate a CycloneDX SBOM, then publish" flow never runs in push/PR CI. A
# generated *.cdx.json SBOM once broke BOTH publish paths and would only have
# surfaced at a manual release:
#   * crate — the untracked SBOM dirtied the git tree, so `cargo publish` refused
#     it (and would otherwise bundle it into the published .crate);
#   * PyPI  — the SBOM artifact was downloaded into dist/, which twine rejects.
# This pins the fixes so a regression fails here, on every push/PR, instead of
# silently passing CI and only breaking at manual release time.
set -euo pipefail
fail() { echo "::error::release-publish invariant violated: $*"; exit 1; }

# (1) Both generated SBOMs must be gitignored. A tracked/untracked *.cdx.json
#     makes `cargo publish` refuse the (dirty) tree and would otherwise bundle
#     the SBOM into the .crate. (Verified end-to-end when this guard was added:
#     `cargo publish --dry-run` is clean with the SBOM present iff it stays
#     gitignored — so this check is the durable pin.)
for f in ordvec.cdx.json ordvec-python/ordvec-python.cdx.json; do
  git check-ignore -q -- "$f" || fail "$f is not gitignored (it is a generated SBOM artifact)"
done

# (2) The PyPI publish job must delete *.cdx.json from dist/ BEFORE the pypa
#     upload step — the merge-multiple artifact download pulls the SBOM into
#     dist/, and twine rejects a stray .cdx.json in the upload dir.
wf=".github/workflows/release-python.yml"
clean_line="$(grep -nE 'find .*cdx\.json.*-delete|rm .*cdx\.json' "$wf" | head -1 | cut -d: -f1 || true)"
pub_line="$(grep -n 'pypi-publish' "$wf" | head -1 | cut -d: -f1 || true)"
[ -n "$clean_line" ] || fail "$wf: publish job has no step deleting *.cdx.json from dist/"
[ -n "$pub_line" ]   || fail "$wf: no pypa publish step found"
[ "$clean_line" -lt "$pub_line" ] \
  || fail "$wf: the *.cdx.json cleanup (line $clean_line) must run BEFORE the pypa publish (line $pub_line)"

echo "OK: release-publish SBOM invariants hold."
