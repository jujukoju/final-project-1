"""
oracle-api/app.py — Phase 7.2 & 7.3

Oracle / NIN Validation Service.

This service acts as the trusted intermediary between the frontend/ML backend
and the on-chain IdentityRegistry contract.  It:
  1. Receives a NIN + Ethereum subject address + IPFS CID (from the ML enroll flow).
  2. Validates the NIN against the simulated NIMC directory (nimc_directory.json).
  3. On success, signs and submits a `registerIdentity` transaction to the contract.

Endpoints:
    POST /validate-nin    — validate NIN, register on-chain if valid
    GET  /health          — liveness probe

Environment variables (set in oracle-api/.env or shell):
    CONTRACT_ADDRESS  — deployed IdentityRegistry address
    ABI_PATH          — path to blockchain/deployed.json (contains ABI)
    PRIVATE_KEY       — hex private key of the oracle account (no 0x prefix required)
    RPC_URL           — JSON-RPC endpoint (e.g. http://127.0.0.1:8545 for local Hardhat)

Run:
    pip install -r requirements.txt
    python oracle-api/app.py
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path

from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load .env from the oracle-api directory if present
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [oracle]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── NIMC directory ─────────────────────────────────────────────────────────────

_NIMC_DIR_PATH = Path(__file__).parent / "nimc_directory.json"

def _load_nimc_directory() -> dict[str, dict]:
    """Return a mapping NIN → record dict."""
    with open(_NIMC_DIR_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {entry["nin"]: entry for entry in data.get("entries", [])}

_NIMC_DIRECTORY: dict[str, dict] = _load_nimc_directory()
logger.info("Loaded NIMC directory: %d entries", len(_NIMC_DIRECTORY))

# ── Web3 / contract setup ──────────────────────────────────────────────────────

def _get_web3_and_contract():
    """
    Initialise web3 connection and contract instance.
    Called lazily so the service can start even if RPC is not yet running.
    """
    try:
        from web3 import Web3
    except ImportError:
        raise RuntimeError(
            "web3.py is not installed. Run: pip install web3"
        )

    rpc_url  = os.environ.get("RPC_URL", "http://127.0.0.1:8545")
    priv_key = os.environ.get("PRIVATE_KEY", "")
    contract_address = os.environ.get("CONTRACT_ADDRESS", "")
    abi_path = os.environ.get("ABI_PATH", str(Path(__file__).parents[1] / "blockchain" / "deployed.json"))

    if not priv_key:
        raise EnvironmentError("PRIVATE_KEY environment variable is not set.")
    if not contract_address:
        raise EnvironmentError("CONTRACT_ADDRESS environment variable is not set.")

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise ConnectionError(f"Cannot connect to JSON-RPC at {rpc_url}")

    with open(abi_path, encoding="utf-8") as f:
        deployed = json.load(f)
    abi = deployed["abi"]

    oracle_account = w3.eth.account.from_key(priv_key)
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(contract_address),
        abi=abi,
    )
    return w3, contract, oracle_account


def _hash_nin(nin: str) -> bytes:
    """Return a 32-byte keccak256 hash of the NIN (matching Solidity keccak256)."""
    try:
        from web3 import Web3
        return Web3.solidity_keccak(["string"], [nin])
    except ImportError:
        # Fallback: standard sha3-256 (not keccak256 — use only for testing without web3)
        return hashlib.sha3_256(nin.encode()).digest()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _json_error(message: str, status: int = 400):
    return jsonify({"error": message}), status


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "nimc_entries": len(_NIMC_DIRECTORY),
        "rpc_url": os.environ.get("RPC_URL", "http://127.0.0.1:8545"),
    })


@app.post("/validate-nin")
def validate_nin():
    """
    Validate a NIN and register the subject's identity on-chain if valid.

    Request JSON:
        {
            "nin":             "<11-digit NIN string>",
            "subject_address": "<0x Ethereum address of the subject>",
            "ipfs_cid":        "<IPFS CID of encrypted embedding>"
        }

    Response JSON (success):
        {
            "valid":    true,
            "nin":      "<NIN>",
            "name":     "<subject name from directory>",
            "tx_hash":  "<0x transaction hash>",
            "message":  "Identity registered on-chain."
        }

    Response JSON (invalid NIN):
        { "valid": false, "error": "NIN not found in directory." }
    """
    body = request.get_json(silent=True)
    if not body:
        return _json_error("Request body must be JSON.")

    nin             = str(body.get("nin", "")).strip()
    subject_address = str(body.get("subject_address", "")).strip()
    ipfs_cid        = str(body.get("ipfs_cid", "")).strip()

    if not nin:
        return _json_error("'nin' is required.")
    if not subject_address:
        return _json_error("'subject_address' is required.")
    if not ipfs_cid:
        return _json_error("'ipfs_cid' is required.")

    # ── Step 1: Validate NIN ──────────────────────────────────────────────────
    record = _NIMC_DIRECTORY.get(nin)
    if record is None:
        logger.warning("NIN validation failed — NIN not found: %s", nin[:4] + "***")
        return jsonify({"valid": False, "error": "NIN not found in directory."}), 404

    logger.info("NIN validated for subject: %s (subject_id=%s)", record["name"], record["subject_id"])

    # ── Step 2: Register on-chain ─────────────────────────────────────────────
    try:
        w3, contract, oracle_account = _get_web3_and_contract()
    except (EnvironmentError, ConnectionError, FileNotFoundError) as exc:
        logger.error("Blockchain not configured: %s", exc)
        return _json_error(f"Blockchain connection error: {exc}", 503)

    try:
        from web3 import Web3
        hash_nin = _hash_nin(nin)
        nonce    = w3.eth.get_transaction_count(oracle_account.address)

        tx = contract.functions.registerIdentity(
            Web3.to_checksum_address(subject_address),
            hash_nin,
            ipfs_cid,
        ).build_transaction({
            "from":  oracle_account.address,
            "nonce": nonce,
            "gas":   300_000,
            "gasPrice": w3.eth.gas_price,
        })

        signed  = oracle_account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

        if receipt.status != 1:
            return _json_error("Transaction reverted on-chain.", 500)

        tx_hash_hex = tx_hash.hex()
        logger.info(
            "Registered on-chain: subject=%s  tx=%s  block=%s",
            subject_address, tx_hash_hex, receipt.blockNumber,
        )

        return jsonify({
            "valid":       True,
            "nin":         nin,
            "name":        record["name"],
            "tx_hash":     tx_hash_hex,
            "block":       receipt.blockNumber,
            "message":     "Identity registered on-chain.",
        })

    except Exception as exc:
        logger.exception("registerIdentity transaction failed")
        return _json_error(f"On-chain registration failed: {exc}", 500)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
