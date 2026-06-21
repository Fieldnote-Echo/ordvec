#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

RECEIPT_SCHEMA = "ordvec.index_authority.v0.1"
POLICY_SCHEMA = "ordvec.index_authority.verifier_policy.v0.1"

VALID_DECISIONS = {
    "ALLOW_INDEX_FIRST",
    "REQUIRE_DENSE_FALLBACK",
    "REQUIRE_HNSW_COMPARISON",
    "DENY_UNSCOPED_CLAIM",
}

REQUIRED_TOP_LEVEL = [
    "schema",
    "subject",
    "baseline",
    "ifc",
    "evidence",
    "economics",
    "decision",
    "scope",
    "limitations",
]


def die(msg, code=2):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def load_json(path: Path, label: str):
    try:
        return json.loads(path.read_text())
    except Exception as e:
        die(f"cannot read {label}: {e}")


def sha(obj):
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(b).hexdigest()


def require_keys(obj, keys, label):
    if not isinstance(obj, dict):
        die(f"{label} must be an object")

    missing = [k for k in keys if k not in obj]
    if missing:
        die(f"{label} missing required field(s): {', '.join(missing)}")


def require_number(obj, key, label):
    value = obj.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        die(f"{label}.{key} must be a number")
    return float(value)


def require_list(obj, key, label):
    value = obj.get(key)
    if not isinstance(value, list):
        die(f"{label}.{key} must be a list")
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("receipt", type=Path)
    ap.add_argument(
        "--policy",
        type=Path,
        default=Path("policies/index-authority.default-policy.json"),
        help="Verifier-owned acceptance policy. Receipt policy fields are ignored.",
    )
    args = ap.parse_args()

    r = load_json(args.receipt, "receipt")
    policy = load_json(args.policy, "policy")

    require_keys(r, REQUIRED_TOP_LEVEL, "receipt")

    if r["schema"] != RECEIPT_SCHEMA:
        die(f"bad receipt schema: {r['schema']}")

    if policy.get("schema") != POLICY_SCHEMA:
        die(f"bad policy schema: {policy.get('schema')}")

    require_keys(
        policy,
        [
            "min_storage_reduction_x",
            "min_single_query_speedup_x",
            "max_quality_delta_loss",
            "require_scope",
            "require_limitations",
            "require_hnsw_comparison_for_parallel_claims",
        ],
        "policy",
    )

    e = r["evidence"]
    econ = r["economics"]
    base = r["baseline"]
    decision_obj = r["decision"]
    scope = r["scope"]
    limitations = r["limitations"]

    require_keys(e, ["candidate_score", "baseline_score", "delta_vs_baseline", "within_bootstrap_noise"], "evidence")
    require_keys(base, ["mode", "bytes_per_vector"], "baseline")
    require_keys(
        econ,
        ["candidate_bytes_per_vector", "storage_reduction_x", "single_query_latency_ms", "single_query_speedup_x"],
        "economics",
    )
    require_keys(econ["single_query_latency_ms"], ["baseline", "candidate"], "economics.single_query_latency_ms")
    require_keys(decision_obj, ["recommended"], "decision")
    require_keys(scope, ["applies_to", "does_not_claim"], "scope")

    recommended = decision_obj["recommended"]
    if recommended not in VALID_DECISIONS:
        die(f"invalid recommended decision: {recommended}")

    candidate_score = require_number(e, "candidate_score", "evidence")
    baseline_score = require_number(e, "baseline_score", "evidence")
    declared_delta = require_number(e, "delta_vs_baseline", "evidence")

    baseline_bytes = require_number(base, "bytes_per_vector", "baseline")
    candidate_bytes = require_number(econ, "candidate_bytes_per_vector", "economics")
    declared_storage = require_number(econ, "storage_reduction_x", "economics")

    latency = econ["single_query_latency_ms"]
    baseline_latency = require_number(latency, "baseline", "economics.single_query_latency_ms")
    candidate_latency = require_number(latency, "candidate", "economics.single_query_latency_ms")
    declared_speedup = require_number(econ, "single_query_speedup_x", "economics")

    if baseline_bytes <= 0 or candidate_bytes <= 0:
        die("bytes_per_vector values must be positive")
    if baseline_latency <= 0 or candidate_latency <= 0:
        die("latency values must be positive")

    expected_delta = candidate_score - baseline_score
    if abs(declared_delta - expected_delta) > 0.0001:
        die("delta_vs_baseline mismatch")

    expected_storage = baseline_bytes / candidate_bytes
    if abs(declared_storage - expected_storage) > 0.02:
        die("storage_reduction_x mismatch")

    expected_speedup = baseline_latency / candidate_latency
    if abs(declared_speedup - expected_speedup) > 0.02:
        die("single_query_speedup_x mismatch")

    applies_to = require_list(scope, "applies_to", "scope")
    does_not_claim = require_list(scope, "does_not_claim", "scope")

    if not isinstance(limitations, list):
        die("limitations must be a list")

    decision = "ALLOW_INDEX_FIRST"

    scope_missing = not applies_to or not does_not_claim
    limitations_missing = not limitations

    quality_loss = baseline_score - candidate_score
    quality_too_low = quality_loss > float(policy["max_quality_delta_loss"])
    outside_bootstrap_noise = e["within_bootstrap_noise"] is not True

    economics_too_weak = (
        declared_storage < float(policy["min_storage_reduction_x"])
        or declared_speedup < float(policy["min_single_query_speedup_x"])
    )

    claims_text = " ".join(str(x).lower() for x in applies_to)
    claims_parallel_or_production = any(
        marker in claims_text
        for marker in ["parallel", "threaded", "production", "prod", "serving", "online"]
    )

    has_hnsw_comparison = (
        e.get("compared_against_hnsw") is True
        or isinstance(e.get("hnsw_comparison"), dict)
    )

    if policy["require_scope"] and scope_missing:
        decision = "DENY_UNSCOPED_CLAIM"
    elif policy["require_limitations"] and limitations_missing:
        decision = "DENY_UNSCOPED_CLAIM"
    elif quality_too_low or outside_bootstrap_noise or economics_too_weak:
        decision = "REQUIRE_DENSE_FALLBACK"
    elif (
        policy["require_hnsw_comparison_for_parallel_claims"]
        and claims_parallel_or_production
        and not has_hnsw_comparison
    ):
        decision = "REQUIRE_HNSW_COMPARISON"

    print(f"decision: {decision}")
    print(f"mode: {r['subject'].get('mode')}")
    print(f"baseline: {base.get('mode')}")
    print(f"quality_within_bootstrap_noise: {str(e['within_bootstrap_noise']).lower()}")
    print(f"storage_reduction: {declared_storage}x")
    print(f"single_query_speedup: {declared_speedup}x")
    print(f"receipt_hash: {sha(r)}")
    print(f"policy_hash: {sha(policy)}")

    if decision != recommended:
        die(f"decision mismatch: receipt recommends {recommended}, verifier computed {decision}", code=3)

    print("verified: true")


if __name__ == "__main__":
    main()
