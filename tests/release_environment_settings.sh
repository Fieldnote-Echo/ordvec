#!/usr/bin/env bash
#
# Manual pre-tag audit for GitHub Environment release gates.
#
# This is intentionally not a normal CI check: it requires an authenticated
# gh token that can read repository environment settings.
set -euo pipefail

REPO="${REPO:-Project-Navi/ordvec}"
EXPECTED_REVIEWER_COUNT="${EXPECTED_REVIEWER_COUNT:-2}"
EXPECTED_REVIEWERS="${EXPECTED_REVIEWERS:-User:Fieldnote-Echo, User:toadkicker}"
EXPECTED_WAIT_TIMER="${EXPECTED_WAIT_TIMER:-30}"
EXPECTED_POLICY="${EXPECTED_POLICY:-v[0-9]*.[0-9]*.[0-9]*}"
ENVIRONMENTS=(crates-io pypi)

fail() {
  echo "::error::release environment settings audit failed: $*"
  exit 1
}

api_jq() {
  local path="$1"
  local filter="$2"
  local err output stderr

  if ! err="$(mktemp)"; then
    fail "could not create temporary file for gh api stderr"
  fi

  if ! output="$(gh api "$path" --jq "$filter" 2>"$err")"; then
    stderr="$(cat "$err")"
    rm -f "$err"
    fail "cannot read ${path}; authenticate with a token that can read ${REPO} repository environment settings. gh api: ${stderr}"
  fi
  rm -f "$err"

  printf '%s\n' "$output"
}

command -v gh >/dev/null 2>&1 \
  || fail "gh CLI not found; install GitHub CLI (gh) and authenticate before running this audit"

if ! gh auth status -h github.com; then
  fail "gh auth status failed; run gh auth login with an account/token that can read ${REPO} repository environment settings"
fi

check_environment() {
  local env="$1"
  local env_path="repos/${REPO}/environments/${env}"
  local policies_path="${env_path}/deployment-branch-policies?per_page=100"
  local env_data policies_data
  local env_name required_rule_count reviewer_count reviewer_summary prevent_self_review
  local custom_branch_policies protected_branches
  local wait_rule_count wait_timer
  local policy_total policy_summary policy_type policy_name

  echo "Auditing ${REPO} environment ${env}..."

  env_data="$(api_jq "$env_path" '[
    (.name // ""),
    ([.protection_rules[]? | select(.type == "required_reviewers")] | length | tostring),
    ([.protection_rules[]? | select(.type == "required_reviewers") | .reviewers[]?] | length | tostring),
    ([.protection_rules[]? | select(.type == "required_reviewers") | .reviewers[]? | "\(.type):\(.reviewer.login // .reviewer.slug // .reviewer.name // "unknown")"] | sort | join(", ")),
    ([.protection_rules[]? | select(.type == "required_reviewers") | .prevent_self_review] | first // false | tostring),
    ([.protection_rules[]? | select(.type == "wait_timer")] | length | tostring),
    ([.protection_rules[]? | select(.type == "wait_timer") | .wait_timer] | first // "" | tostring),
    (.deployment_branch_policy.custom_branch_policies | tostring),
    (.deployment_branch_policy.protected_branches | tostring)
  ] | @tsv')"
  IFS=$'\t' read -r env_name required_rule_count reviewer_count reviewer_summary prevent_self_review wait_rule_count wait_timer custom_branch_policies protected_branches <<< "$env_data"

  [ "$env_name" = "$env" ] \
    || fail "${env}: environment not found"

  [ "$required_rule_count" = "1" ] \
    || fail "${env}: expected exactly one required_reviewers protection rule; found ${required_rule_count}"

  [ "$reviewer_count" = "$EXPECTED_REVIEWER_COUNT" ] \
    || fail "${env}: expected ${EXPECTED_REVIEWER_COUNT} required reviewers (${EXPECTED_REVIEWERS}); found ${reviewer_count} (${reviewer_summary:-none})"
  [ "$reviewer_summary" = "$EXPECTED_REVIEWERS" ] \
    || fail "${env}: expected required reviewers ${EXPECTED_REVIEWERS}; found ${reviewer_summary:-none}"

  [ "$prevent_self_review" = "true" ] \
    || fail "${env}: expected required_reviewers.prevent_self_review == true; found ${prevent_self_review}"

  [ "$wait_rule_count" = "1" ] \
    || fail "${env}: expected exactly one wait_timer protection rule; found ${wait_rule_count}"
  [ "$wait_timer" = "$EXPECTED_WAIT_TIMER" ] \
    || fail "${env}: expected wait_timer ${EXPECTED_WAIT_TIMER} minutes; found ${wait_timer:-none}"

  [ "$custom_branch_policies" = "true" ] \
    || fail "${env}: expected deployment_branch_policy.custom_branch_policies == true; found ${custom_branch_policies}"

  [ "$protected_branches" = "false" ] \
    || fail "${env}: expected deployment_branch_policy.protected_branches == false; found ${protected_branches}"

  policies_data="$(api_jq "$policies_path" '[
    (.total_count | tostring),
    ([.branch_policies[]? | "\(.type):\(.name)"] | join(", ")),
    (.branch_policies[0].type // ""),
    (.branch_policies[0].name // "")
  ] | @tsv')"
  IFS=$'\t' read -r policy_total policy_summary policy_type policy_name <<< "$policies_data"

  [ "$policy_total" = "1" ] \
    || fail "${env}: expected exactly one deployment branch/tag policy tag:${EXPECTED_POLICY}; found ${policy_total} (${policy_summary:-none})"

  [ "$policy_type" = "tag" ] \
    || fail "${env}: expected deployment policy type tag; found ${policy_type:-none}"

  [ "$policy_name" = "$EXPECTED_POLICY" ] \
    || fail "${env}: expected deployment policy name ${EXPECTED_POLICY}; found ${policy_name:-none}"

  echo "OK: ${env} requires ${EXPECTED_REVIEWERS}, blocks self-review, waits ${EXPECTED_WAIT_TIMER} minutes, and only tag:${EXPECTED_POLICY}."
}

for env in "${ENVIRONMENTS[@]}"; do
  check_environment "$env"
done

echo "OK: release environment settings match the pre-tag policy."
