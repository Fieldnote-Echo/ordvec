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

# (2) In the PyPI publish job the step order must be:
#       actions/download-artifact  (pulls the SBOM into dist/)
#         -> delete *.cdx.json from dist/
#           -> pypa/gh-action-pypi-publish upload.
#     twine rejects a stray .cdx.json in dist/, so the cleanup must run AFTER the
#     download (otherwise it is a no-op for the downloaded SBOM) and BEFORE the
#     upload. The search is scoped to the `publish` job body, so a download step
#     in another job cannot satisfy the ordering; the delete is matched on its own
#     line, so it works for both `run: ... -delete` and a multi-line `run: |` block;
#     comment lines are skipped; and the publish step keys on the pinned action
#     name (not the bare string `pypi-publish`, which could match a job name).
wf=".github/workflows/release-python.yml"
[ -f "$wf" ] || fail "$wf: workflow file not found"

# Extract the `publish` job body: from its `  publish:` key to the next
# 2-space-indented job key, or EOF. Scoping here is what makes the ordering
# meaningful — the three steps must live in the SAME (publish) job.
pub_start="$(grep -nE '^  publish:[[:space:]]*$' "$wf" | head -1 | cut -d: -f1)"
[ -n "$pub_start" ] || fail "$wf: no 'publish:' job found"
pub_end="$(awk -v s="$pub_start" 'NR>s && /^  [A-Za-z0-9_-]+:/ {print NR-1; exit}' "$wf")"
[ -n "$pub_end" ] || pub_end="$(awk 'END{print NR}' "$wf")"
job="$(sed -n "${pub_start},${pub_end}p" "$wf")"

# First real (non-comment) line WITHIN the publish job matching the regex.
in_job() { printf '%s\n' "$job" | grep -nE "$1" | grep -vE '^[0-9]+:[[:space:]]*#' | head -1 | cut -d: -f1; }

dl_line="$(in_job 'uses:[[:space:]]*actions/download-artifact' || true)"
# Match the deletion command itself (not the `run:` key), so a multi-line
# `run: |` block works too. Still requires a real delete — `find ... -delete` or
# `rm ... *.cdx.json` — not a bare reference that would leave the SBOM in dist/.
clean_line="$(in_job '(find.*cdx\.json.*-delete|rm[[:space:]].*cdx\.json)' || true)"
pub_line="$(in_job 'uses:[[:space:]]*pypa/gh-action-pypi-publish' || true)"

[ -n "$dl_line" ]    || fail "$wf (publish job): no actions/download-artifact step found"
[ -n "$clean_line" ] || fail "$wf (publish job): no step deleting *.cdx.json from dist/ (need 'find ... -delete' or 'rm ... *.cdx.json')"
[ -n "$pub_line" ]   || fail "$wf (publish job): no pypa/gh-action-pypi-publish step found"

[ "$dl_line" -lt "$clean_line" ] \
  || fail "$wf (publish job): the *.cdx.json cleanup must run AFTER actions/download-artifact, else it is a no-op for the downloaded SBOM"
[ "$clean_line" -lt "$pub_line" ] \
  || fail "$wf (publish job): the *.cdx.json cleanup must run BEFORE the pypa publish"

echo "OK: release-publish SBOM invariants hold."
