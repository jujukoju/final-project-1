"""
ml-backend/app.py — Phase 5.3

Flask ML microservice for the fingerprint verification system.

Endpoints:
    GET  /health          — liveness probe
    POST /enroll          — fingerprint enrollment → encrypted embedding
    POST /verify          — live fingerprint vs stored encrypted embedding

Environment variables required:
    AES_KEY          — 64-char hex string (32 bytes for AES-256)
    CHECKPOINT_PATH  — path to best_siamese.pt (default: ml-backend/checkpoints/best_siamese.pt)

Run (development):
    $env:AES_KEY = "<your_hex_key>"
    python ml-backend/app.py
"""

import os
import sys
import base64
import logging
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_backend.embedding import EmbeddingExtractor
from ml_backend.crypto import encrypt_embedding, decrypt_embedding

# ── Config ────────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    str(Path(__file__).parent / "checkpoints" / "best_siamese.pt"),
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [flask]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Lazy-load model to keep startup fast during unit tests
_extractor: EmbeddingExtractor | None = None


def get_extractor() -> EmbeddingExtractor:
    global _extractor
    if _extractor is None:
        logger.info("Loading Siamese model from %s …", CHECKPOINT_PATH)
        _extractor = EmbeddingExtractor(CHECKPOINT_PATH)
    return _extractor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_image(b64_string: str) -> np.ndarray:
    """Decode a base64-encoded image string to a BGR numpy array."""
    try:
        img_bytes = base64.b64decode(b64_string)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("cv2.imdecode returned None (invalid image bytes)")
        return img
    except Exception as exc:
        raise ValueError(f"Failed to decode image: {exc}") from exc


def _json_error(message: str, status: int = 400):
    return jsonify({"error": message}), status


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """
    Liveness probe.
    Returns model load status without actually loading the model.
    """
    model_loaded = _extractor is not None
    return jsonify({
        "status": "ok",
        "model": "loaded" if model_loaded else "not_loaded",
        "checkpoint": CHECKPOINT_PATH,
    })


@app.post("/enroll")
def enroll():
    """
    Enroll a fingerprint.

    Request JSON:
        {
            "nin":             "<NIN string>",
            "fingerprint_b64": "<base64-encoded image (PNG/BMP/JPEG)>"
        }

    Response JSON (success):
        {
            "nin":                "<NIN>",
            "encrypted_embedding": "<base64url token>",
            "embedding_dim":       128,
            "ipfs_cid":           null   // placeholder until IPFS integration
        }
    """
    body = request.get_json(silent=True)
    if not body:
        return _json_error("Request body must be JSON.")

    nin = body.get("nin", "").strip()
    fingerprint_b64 = body.get("fingerprint_b64", "").strip()

    if not nin:
        return _json_error("'nin' field is required.")
    if not fingerprint_b64:
        return _json_error("'fingerprint_b64' field is required.")

    try:
        img = _decode_image(fingerprint_b64)
    except ValueError as exc:
        return _json_error(str(exc))

    try:
        extractor = get_extractor()
        embedding = extractor.from_array(img)
    except Exception as exc:
        logger.exception("Embedding generation failed")
        return _json_error(f"Embedding generation failed: {exc}", 500)

    try:
        token = encrypt_embedding(embedding)
    except EnvironmentError as exc:
        return _json_error(str(exc), 500)

    logger.info("Enrolled NIN=%s  embedding_dim=%d", nin, embedding.size)
    return jsonify({
        "nin": nin,
        "encrypted_embedding": token,
        "embedding_dim": int(embedding.size),
        "ipfs_cid": None,   # Phase 7+ will populate this
    })


@app.post("/verify")
def verify():
    """
    Verify a live fingerprint against a stored encrypted embedding.

    Request JSON:
        {
            "nin":                "<NIN string>",
            "fingerprint_b64":    "<base64-encoded live fingerprint>",
            "stored_embedding":   "<encrypted_embedding token from /enroll>",
            "threshold":          0.5    // optional, default 0.5
        }

    Response JSON (success):
        {
            "nin":      "<NIN>",
            "match":    true | false,
            "distance": 0.312,
            "threshold": 0.5
        }
    """
    body = request.get_json(silent=True)
    if not body:
        return _json_error("Request body must be JSON.")

    nin              = body.get("nin", "").strip()
    fingerprint_b64  = body.get("fingerprint_b64", "").strip()
    stored_token     = body.get("stored_embedding", "").strip()
    threshold        = float(body.get("threshold", 0.5))

    if not nin:
        return _json_error("'nin' is required.")
    if not fingerprint_b64:
        return _json_error("'fingerprint_b64' is required.")
    if not stored_token:
        return _json_error("'stored_embedding' is required.")

    # Decode live image
    try:
        live_img = _decode_image(fingerprint_b64)
    except ValueError as exc:
        return _json_error(str(exc))

    # Generate live embedding
    try:
        extractor = get_extractor()
        live_emb = extractor.from_array(live_img)
    except Exception as exc:
        logger.exception("Live embedding generation failed")
        return _json_error(f"Embedding generation failed: {exc}", 500)

    # Decrypt stored embedding
    try:
        stored_emb = decrypt_embedding(stored_token)
    except (ValueError, EnvironmentError) as exc:
        return _json_error(f"Could not decrypt stored embedding: {exc}", 400)

    # Compute distance and decision
    distance = extractor.distance(live_emb, stored_emb, metric="euclidean")
    match    = bool(distance < threshold)

    logger.info("Verified NIN=%s  distance=%.4f  match=%s", nin, distance, match)
    return jsonify({
        "nin":       nin,
        "match":     match,
        "distance":  round(float(distance), 6),
        "threshold": threshold,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pre-load model on startup in dev mode
    try:
        get_extractor()
    except FileNotFoundError:
        logger.warning(
            "Checkpoint not found at %s — model will load on first request. "
            "Train the model with: python ml-backend/train.py",
            CHECKPOINT_PATH,
        )
    app.run(host="0.0.0.0", port=5001, debug=True)
