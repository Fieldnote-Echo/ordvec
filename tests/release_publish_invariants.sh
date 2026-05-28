#!/usr/bin/env bash
#
# Release-publish SBOM invariants — pinned in CI.
#
# release.yml is the unified tag-triggered release pipeline; its publishes are
# gated behind GitHub Environments (Required reviewers), so the "generate a
# CycloneDX SBOM, then publish" flow runs only on a real release. A generated
# *.cdx.json SBOM once broke BOTH publish paths and would only have surfaced
# at the next release:
#   * crate — the untracked SBOM dirtied the git tree, so `cargo publish` refused
#     it (and would otherwise bundle it into the published .crate);
#   * PyPI  — the SBOM artifact was downloaded into dist/, which twine rejects.
# This pins the fixes so a regression fails here, on every push/PR, instead of
# silently passing CI and only breaking at release time.
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
#         -> delete *.cdx.json from dist/  (either explicit cdx.json delete OR
#            a keep-only-wheels/tar.gz find that excludes everything else)
#           -> pypa/gh-action-pypi-publish upload.
#     twine rejects a stray .cdx.json in dist/, so the cleanup must run AFTER the
#     download (otherwise it is a no-op for the downloaded SBOM) and BEFORE the
#     upload. The search is scoped to the `publish-pypi` job body, so a download
#     step in another job cannot satisfy the ordering; the delete is matched only
#     in an executing `run:` context (single-line or a `run: |` block), so a step
#     name or other non-executing text cannot satisfy it; comment lines are
#     skipped; and the publish step keys on the pinned action name (not the bare
#     string `pypi-publish`).
wf=".github/workflows/release.yml"
[ -f "$wf" ] || fail "$wf: workflow file not found"

# Extract the `publish-pypi` job body: from its `  publish-pypi:` key to the
# next 2-space-indented job key, or EOF. Scoping here is what makes the
# ordering meaningful — the three steps must live in the SAME job.
pub_start="$(grep -nE '^  publish-pypi:[[:space:]]*$' "$wf" | head -1 | cut -d: -f1)"
[ -n "$pub_start" ] || fail "$wf: no 'publish-pypi:' job found"
pub_end="$(awk -v s="$pub_start" 'NR>s && /^  [A-Za-z0-9_-]+:/ {print NR-1; exit}' "$wf")"
[ -n "$pub_end" ] || pub_end="$(awk 'END{print NR}' "$wf")"
job="$(sed -n "${pub_start},${pub_end}p" "$wf")"

# First real (non-comment) line WITHIN the publish-pypi job matching the regex.
in_job() { printf '%s\n' "$job" | grep -nE "$1" | grep -vE '^[0-9]+:[[:space:]]*#' | head -1 | cut -d: -f1; }

dl_line="$(in_job 'uses:[[:space:]]*actions/download-artifact' || true)"
# The cleanup must be a real delete in an EXECUTING `run:` context — either a
# single-line `run: ... -delete` or a line inside that step's `run: |`/`run: >`
# block. Matching the command text anywhere would also accept NON-executing text
# (a step `name:`, an `env:`/`with:` value, prose), so the delete only counts on
# a `run:` line or within a run block scalar. Accepts both forms:
#   (a) explicit cdx.json delete:                  `find ... cdx.json ... -delete` / `rm ... *.cdx.json`
#   (b) keep-only-wheels/tar.gz delete-everything: `find dist -type f ! -name '*.whl' ! -name '*.tar.gz' -delete`
# Either form removes the SBOM before the upload.
clean_line="$(printf '%s\n' "$job" | awk '
  function indent(s,  i){ i = match(s, /[^ ]/); return (i ? i - 1 : length(s)) }
  BEGIN {
    del_a = "find.*cdx\\.json.*-delete|rm[[:space:]].*cdx\\.json"
    del_b = "find.*-type[[:space:]]+f.*!.*-name.*whl.*!.*-name.*tar\\.gz.*-delete"
    del = del_a "|" del_b
  }
  { is_comment = ($0 ~ /^[[:space:]]*#/) }
  in_block {
    if ($0 ~ /^[[:space:]]*$/) next                  # blank line stays in block
    if (indent($0) > block_indent) {                 # block content (incl. shell # lines,
      if (!is_comment && $0 ~ del) { print NR; exit } # which are literal text here, not
      next                                            # YAML comments — stay in the block)
    }
    in_block = 0                                      # dedent ends block; re-test line
  }
  /^[[:space:]]*run:[[:space:]]*[|>]/ { in_block = 1; block_indent = indent($0); next }
  /^[[:space:]]*run:[[:space:]]/ && !is_comment { if ($0 ~ del) { print NR; exit } }
' || true)"
pub_line="$(in_job 'uses:[[:space:]]*pypa/gh-action-pypi-publish' || true)"

[ -n "$dl_line" ]    || fail "$wf (publish-pypi job): no actions/download-artifact step found"
[ -n "$clean_line" ] || fail "$wf (publish-pypi job): no step deleting *.cdx.json from dist/ (need 'find ... cdx.json ... -delete', 'rm ... *.cdx.json', or 'find ... ! -name *.whl ! -name *.tar.gz -delete')"
[ -n "$pub_line" ]   || fail "$wf (publish-pypi job): no pypa/gh-action-pypi-publish step found"

[ "$dl_line" -lt "$clean_line" ] \
  || fail "$wf (publish-pypi job): the *.cdx.json cleanup must run AFTER actions/download-artifact, else it is a no-op for the downloaded SBOM"
[ "$clean_line" -lt "$pub_line" ] \
  || fail "$wf (publish-pypi job): the *.cdx.json cleanup must run BEFORE the pypa publish"

echo "OK: release-publish SBOM invariants hold."
