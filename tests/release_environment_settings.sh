#!/usr/bin/env bash
#
# Manual pre-tag audit for GitHub Environment release gates.
#
# This is intentionally not a normal CI check: it requires an authenticated
# gh token that can read repository environment settings.
set -euo pipefail

REPO="${REPO:-Fieldnote-Echo/ordvec}"
EXPECTED_REVIEWER="${EXPECTED_REVIEWER:-Fieldnote-Echo}"
EXPECTED_POLICY="${EXPECTED_POLICY:-v[0-9]*.[0-9]*.[0-9]*}"
ENVIRONMENTS=(crates-io pypi)

fail() {
  echo "::error::release environment settings audit failed: $*"
  exit 1
}

api_jq() {
  local path="$1"
  local filter="$2"
  local output

  if ! output="$(gh api "$path" --jq "$filter" 2>&1)"; then
    fail "cannot read ${path}; authenticate with a token that can read ${REPO} repository environment settings. gh api: ${output}"
  fi

  printf '%s\n' "$output"
}

if ! gh auth status; then
  fail "gh auth status failed; run gh auth login with an account/token that can read ${REPO} repository environment settings"
fi

api_jq "repos/${REPO}/environments?per_page=100" '.total_count' >/dev/null

check_environment() {
  local env="$1"
  local env_path="repos/${REPO}/environments/${env}"
  local policies_path="${env_path}/deployment-branch-policies?per_page=100"
  local env_name required_rule_count reviewer_count reviewer_summary
  local custom_branch_policies protected_branches
  local policy_total policy_summary policy_type policy_name

  echo "Auditing ${REPO} environment ${env}..."

  env_name="$(api_jq "$env_path" '.name // ""')"
  [ "$env_name" = "$env" ] \
    || fail "${env}: environment not found"

  required_rule_count="$(api_jq "$env_path" '[.protection_rules[]? | select(.type == "required_reviewers")] | length')"
  [ "$required_rule_count" = "1" ] \
    || fail "${env}: expected exactly one required_reviewers protection rule; found ${required_rule_count}"

  reviewer_count="$(api_jq "$env_path" '[.protection_rules[]? | select(.type == "required_reviewers") | .reviewers[]?] | length')"
  reviewer_summary="$(api_jq "$env_path" '[.protection_rules[]? | select(.type == "required_reviewers") | .reviewers[]? | "\(.type):\(.reviewer.login // .reviewer.slug // .reviewer.name // "unknown")"] | join(", ")')"
  [ "$reviewer_count" = "1" ] \
    || fail "${env}: expected exactly one required reviewer User:${EXPECTED_REVIEWER}; found ${reviewer_count} (${reviewer_summary:-none})"
  [ "$reviewer_summary" = "User:${EXPECTED_REVIEWER}" ] \
    || fail "${env}: expected required reviewer User:${EXPECTED_REVIEWER}; found ${reviewer_summary:-none}"

  custom_branch_policies="$(api_jq "$env_path" '.deployment_branch_policy.custom_branch_policies')"
  [ "$custom_branch_policies" = "true" ] \
    || fail "${env}: expected deployment_branch_policy.custom_branch_policies == true; found ${custom_branch_policies}"

  protected_branches="$(api_jq "$env_path" '.deployment_branch_policy.protected_branches')"
  [ "$protected_branches" = "false" ] \
    || fail "${env}: expected deployment_branch_policy.protected_branches == false; found ${protected_branches}"

  policy_total="$(api_jq "$policies_path" '.total_count')"
  policy_summary="$(api_jq "$policies_path" '[.branch_policies[]? | "\(.type):\(.name)"] | join(", ")')"
  [ "$policy_total" = "1" ] \
    || fail "${env}: expected exactly one deployment branch/tag policy tag:${EXPECTED_POLICY}; found ${policy_total} (${policy_summary:-none})"

  policy_type="$(api_jq "$policies_path" '.branch_policies[0].type // ""')"
  [ "$policy_type" = "tag" ] \
    || fail "${env}: expected deployment policy type tag; found ${policy_type:-none}"

  policy_name="$(api_jq "$policies_path" '.branch_policies[0].name // ""')"
  [ "$policy_name" = "$EXPECTED_POLICY" ] \
    || fail "${env}: expected deployment policy name ${EXPECTED_POLICY}; found ${policy_name:-none}"

  echo "OK: ${env} requires User:${EXPECTED_REVIEWER} and only tag:${EXPECTED_POLICY}."
}

for env in "${ENVIRONMENTS[@]}"; do
  check_environment "$env"
done

echo "OK: release environment settings match the pre-tag policy."
