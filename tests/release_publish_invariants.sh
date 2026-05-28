#!/usr/bin/env bash
#
# Release-publish SBOM invariants — pinned in CI.
#
# release.yml is the unified tag-triggered release pipeline. A generated
# *.cdx.json SBOM once broke BOTH publish paths:
#   * crate — the untracked SBOM dirtied the git tree, so `cargo publish`
#     refused it;
#   * PyPI  — the SBOM artifact was downloaded into dist/, which twine rejects.
# This guard keeps SBOMs ignored and ensures the PyPI publish job only downloads
# Python distributables into dist/ in the first place.
set -euo pipefail
fail() { echo "::error::release-publish invariant violated: $*"; exit 1; }

# (1) Both generated SBOMs must be gitignored. A tracked/untracked *.cdx.json
#     makes `cargo publish` refuse the dirty tree and would otherwise bundle
#     the SBOM into the .crate.
for f in ordvec.cdx.json ordvec-python/ordvec-python.cdx.json; do
  git check-ignore -q -- "$f" || fail "$f is not gitignored (it is a generated SBOM artifact)"
done

# (2) PyPI accepts only wheels + sdists. The publish-pypi job must therefore use
#     targeted artifact downloads for `wheels-*` and `sdist`; broad downloads
#     would also pull SBOM / provenance / Sigstore artifacts into dist/.
wf=".github/workflows/release.yml"
[ -f "$wf" ] || fail "$wf: workflow file not found"

pub_start="$(grep -nE '^  publish-pypi:[[:space:]]*$' "$wf" | head -1 | cut -d: -f1)"
[ -n "$pub_start" ] || fail "$wf: no 'publish-pypi:' job found"
pub_end="$(awk -v s="$pub_start" 'NR>s && /^  [A-Za-z0-9_-]+:/ {print NR-1; exit}' "$wf")"
[ -n "$pub_end" ] || pub_end="$(awk 'END{print NR}' "$wf")"
job="$(sed -n "${pub_start},${pub_end}p" "$wf")"

printf '%s\n' "$job" | grep -Eq 'uses:[[:space:]]*actions/download-artifact@' \
  || fail "$wf (publish-pypi job): no actions/download-artifact steps found"
printf '%s\n' "$job" | grep -Eq 'pattern:[[:space:]]*wheels-\*[[:space:]]*$' \
  || fail "$wf (publish-pypi job): no targeted wheels-* artifact download found"
printf '%s\n' "$job" | grep -Eq 'name:[[:space:]]*sdist[[:space:]]*$' \
  || fail "$wf (publish-pypi job): no targeted sdist artifact download found"
printf '%s\n' "$job" | grep -Eq 'uses:[[:space:]]*pypa/gh-action-pypi-publish@' \
  || fail "$wf (publish-pypi job): no pypa/gh-action-pypi-publish step found"

# Guard against reintroducing the old broad download shape:
#   uses: actions/download-artifact
#   with:
#     path: dist
#     merge-multiple: true
# and no `name:` or `pattern:` selector in that same step.
printf '%s\n' "$job" | awk '
  function flush() {
    if (in_download && has_dist && !has_selector) {
      print "broad"; exit 0
    }
    in_download = has_dist = has_selector = 0
  }
  /^[[:space:]]*-[[:space:]]/ { flush() }
  /uses:[[:space:]]*actions\/download-artifact@/ { in_download = 1 }
  in_download && /^[[:space:]]*path:[[:space:]]*dist[[:space:]]*$/ { has_dist = 1 }
  in_download && /^[[:space:]]*(name|pattern):[[:space:]]*/ { has_selector = 1 }
  END { flush() }
' | grep -q '^broad$' \
  && fail "$wf (publish-pypi job): broad artifact download into dist/ would include SBOM/provenance assets"

echo "OK: release-publish SBOM invariants hold."
