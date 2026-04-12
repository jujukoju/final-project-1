"""
ml-backend/crypto.py — Phase 5.2

AES-256-GCM encryption/decryption utilities for fingerprint embeddings.

Key management (prototype):
    The 32-byte key is loaded from the AES_KEY environment variable,
    which must be a lowercase hex string of exactly 64 characters.

    Generate a key:
        python -c "import secrets; print(secrets.token_hex(32))"

Usage:
    from ml_backend.crypto import encrypt_embedding, decrypt_embedding
    token = encrypt_embedding(embedding_ndarray)   # → compact base64 string
    arr   = decrypt_embedding(token)               # → np.ndarray, shape (128,)
"""

import os
import base64
import logging
import struct

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

_AES_KEY_ENV = "AES_KEY"
_NONCE_SIZE  = 12   # 96-bit nonce, recommended for GCM
_DTYPE       = np.float32


def _get_key() -> bytes:
    """Load and validate the AES key from the environment."""
    hex_key = os.environ.get(_AES_KEY_ENV, "")
    if len(hex_key) != 64:
        raise EnvironmentError(
            f"Environment variable {_AES_KEY_ENV!r} must be a 64-char hex string "
            f"(32 bytes). Got length {len(hex_key)}. "
            f"Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    try:
        return bytes.fromhex(hex_key)
    except ValueError as exc:
        raise EnvironmentError(f"Invalid hex in {_AES_KEY_ENV!r}: {exc}") from exc


def encrypt_embedding(embedding: np.ndarray) -> str:
    """
    Encrypt a numpy float32 embedding with AES-256-GCM.

    Returns a compact base64url-encoded token:
        [4 bytes: embedding dimension] [12 bytes: nonce] [ciphertext & 16-byte tag]
    """
    key   = _get_key()
    aesgcm = AESGCM(key)

    emb_f32 = embedding.astype(_DTYPE)
    dim_header = struct.pack(">I", emb_f32.size)   # 4 bytes, big-endian uint32
    plaintext  = dim_header + emb_f32.tobytes()

    nonce      = os.urandom(_NONCE_SIZE)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)   # includes GCM tag

    token = base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii")
    logger.debug("encrypt_embedding: dim=%d, token_len=%d", emb_f32.size, len(token))
    return token


def decrypt_embedding(token: str) -> np.ndarray:
    """
    Decrypt a token produced by encrypt_embedding.

    Returns the original float32 numpy embedding.
    """
    key    = _get_key()
    aesgcm = AESGCM(key)

    raw = base64.urlsafe_b64decode(token.encode("ascii"))
    if len(raw) < _NONCE_SIZE + 4 + 16:
        raise ValueError("Token is too short to contain a valid encrypted embedding.")

    nonce      = raw[:_NONCE_SIZE]
    ciphertext = raw[_NONCE_SIZE:]

    try:
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    except Exception as exc:
        raise ValueError(f"Decryption failed (wrong key or corrupted token): {exc}") from exc

    dim = struct.unpack(">I", plaintext[:4])[0]
    emb_bytes = plaintext[4:]

    expected_bytes = dim * np.dtype(_DTYPE).itemsize
    if len(emb_bytes) != expected_bytes:
        raise ValueError(
            f"Decrypted payload size mismatch: expected {expected_bytes} bytes for "
            f"dim={dim}, got {len(emb_bytes)}."
        )

    embedding = np.frombuffer(emb_bytes, dtype=_DTYPE).copy()
    logger.debug("decrypt_embedding: dim=%d recovered", dim)
    return embedding


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Use a dummy key for the test
    os.environ[_AES_KEY_ENV] = "0" * 64
    test_emb = np.random.rand(128).astype(np.float32)
    token = encrypt_embedding(test_emb)
    recovered = decrypt_embedding(token)
    assert np.allclose(test_emb, recovered, atol=1e-6), "Round-trip mismatch!"
    print(f"✅  AES-256-GCM round-trip OK  |  token length = {len(token)} chars")
    sys.exit(0)
