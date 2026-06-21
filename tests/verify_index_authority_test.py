import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "tools" / "verify_index_authority.py"
RECEIPT = ROOT / "examples" / "caif" / "trec-covid-sign-rq2.index-authority.json"
POLICY = ROOT / "policies" / "index-authority.default-policy.json"

def run_verify(path):
    return subprocess.run(
        [sys.executable, str(VERIFY), str(path), "--policy", str(POLICY)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

def test_valid_receipt_passes():
    result = run_verify(RECEIPT)
    assert result.returncode == 0, result.stderr + result.stdout

def test_missing_required_field_rejected(tmp_path):
    data = json.loads(RECEIPT.read_text())
    data.pop("evidence")
    bad = tmp_path / "missing-evidence.json"
    bad.write_text(json.dumps(data))
    result = run_verify(bad)
    assert result.returncode != 0

def test_metric_tampering_rejected(tmp_path):
    data = json.loads(RECEIPT.read_text())
    data["economics"]["storage_reduction_x"] = 999
    bad = tmp_path / "tampered.json"
    bad.write_text(json.dumps(data))
    result = run_verify(bad)
    assert result.returncode != 0

def test_decision_mismatch_exit_code_3(tmp_path):
    data = json.loads(RECEIPT.read_text())
    data["decision"]["recommended"] = "DENY_UNSCOPED_CLAIM"
    bad = tmp_path / "decision-mismatch.json"
    bad.write_text(json.dumps(data))
    result = run_verify(bad)
    assert result.returncode == 3
