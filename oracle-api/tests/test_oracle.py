"""
oracle-api/tests/test_oracle.py — Phase 7.4

End-to-end registration flow test.

Prerequisites:
  1. Start a local Hardhat node:        npx hardhat node   (in blockchain/)
  2. Deploy the contract:               npx hardhat run scripts/deploy.js --network localhost
  3. Set env vars or create oracle-api/.env:
       CONTRACT_ADDRESS=<from deployed.json>
       ABI_PATH=<absolute path to blockchain/deployed.json>
       PRIVATE_KEY=<any Hardhat test account private key, e.g. the first one>
       RPC_URL=http://127.0.0.1:8545
  4. Start the oracle service:          python oracle-api/app.py
  5. Run this test:                     python oracle-api/tests/test_oracle.py

The test verifies the complete registration flow:
   POST /validate-nin  →  NIN validated  →  registerIdentity tx broadcast  →
   on-chain state confirmed via web3.py direct contract call.
"""

import os
import sys
import json
import argparse
import urllib.request
import urllib.error
from pathlib import Path

BASE_URL = os.environ.get("ORACLE_API_URL", "http://localhost:5002")

# ── Sample data from nimc_directory.json ─────────────────────────────────────
VALID_NIN          = "12345678901"       # Adebayo Okafor
VALID_IPFS_CID     = "QmTestCIDplaceholder12345678901"

# A Hardhat default test account (account #1 — not the deployer/oracle, acts as subject)
SUBJECT_ADDRESS    = os.environ.get(
    "SUBJECT_ADDRESS",
    "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"  # Hardhat account #1
)

INVALID_NIN        = "00000000000"   # not in directory


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _post(endpoint: str, payload: dict) -> tuple[int, dict]:
    url  = f"{BASE_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _get(endpoint: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(f"{BASE_URL}{endpoint}", timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# ── Test runner ────────────────────────────────────────────────────────────────

def run_tests():
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}  ← {detail}")
            failed += 1

    print(f"\n── Oracle E2E Registration Flow ────────────────────────")
    print(f"  Server  : {BASE_URL}")
    print(f"  Subject : {SUBJECT_ADDRESS}\n")

    # ── 1. Health check ────────────────────────────────────────────────────────
    status, body = _get("/health")
    check("GET /health returns 200", status == 200, f"status={status}")
    check("NIMC directory loaded", body.get("nimc_entries", 0) > 0, str(body))

    # ── 2. Invalid NIN ─────────────────────────────────────────────────────────
    status, body = _post("/validate-nin", {
        "nin":             INVALID_NIN,
        "subject_address": SUBJECT_ADDRESS,
        "ipfs_cid":        VALID_IPFS_CID,
    })
    check("POST /validate-nin (invalid NIN) → 404", status == 404, f"status={status}")
    check("Response has valid=false", body.get("valid") is False, str(body))

    # ── 3. Valid NIN → on-chain registration ───────────────────────────────────
    status, body = _post("/validate-nin", {
        "nin":             VALID_NIN,
        "subject_address": SUBJECT_ADDRESS,
        "ipfs_cid":        VALID_IPFS_CID,
    })
    check("POST /validate-nin (valid NIN) → 200", status == 200, f"status={status}: {body}")
    check("Response has valid=true",  body.get("valid") is True,  str(body))
    check("Response has tx_hash",     "tx_hash" in body,          str(body))

    tx_hash = body.get("tx_hash", "")
    if tx_hash:
        print(f"  INFO  Transaction hash: {tx_hash}")
        print(f"  INFO  Block number:     {body.get('block', '?')}")

    # ── 4. On-chain state verification via web3.py direct call ─────────────────
    try:
        from web3 import Web3

        rpc_url          = os.environ.get("RPC_URL", "http://127.0.0.1:8545")
        abi_path         = os.environ.get("ABI_PATH", str(
            Path(__file__).parents[2] / "blockchain" / "deployed.json"
        ))
        contract_address = os.environ.get("CONTRACT_ADDRESS", "")

        w3 = Web3(Web3.HTTPProvider(rpc_url))
        with open(abi_path) as f:
            deployed = json.load(f)

        contract = w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=deployed["abi"],
        )
        is_registered = contract.functions.isRegistered(
            Web3.to_checksum_address(SUBJECT_ADDRESS)
        ).call()
        check("Subject is_registered on-chain == True", is_registered, f"isRegistered={is_registered}")

    except ImportError:
        print("  SKIP  On-chain check skipped (web3.py not installed)")
    except Exception as exc:
        check("On-chain isRegistered check", False, str(exc))

    # ── 5. Duplicate registration ───────────────────────────────────────────────
    status, body = _post("/validate-nin", {
        "nin":             VALID_NIN,
        "subject_address": SUBJECT_ADDRESS,
        "ipfs_cid":        VALID_IPFS_CID,
    })
    check(
        "POST /validate-nin (duplicate) → 500 (reverted on-chain)",
        status == 500,
        f"status={status}: {body}",
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n  {'─' * 50}")
    print(f"  {passed + failed} tests: {passed} passed, {failed} failed")
    if failed == 0:
        print("  ✅  End-to-end registration flow verified.\n")
    else:
        print("  ❌  Some tests failed — check output above.\n")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oracle E2E test")
    parser.add_argument("--url",     default=BASE_URL,        help="Oracle API base URL")
    parser.add_argument("--subject", default=SUBJECT_ADDRESS, help="Subject Ethereum address")
    args = parser.parse_args()
    BASE_URL        = args.url
    SUBJECT_ADDRESS = args.subject
    run_tests()
