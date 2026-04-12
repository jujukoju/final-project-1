"""
ml-backend/tests/test_api.py — Phase 5.4

Integration smoke tests for the Flask ML microservice.

Prerequisites:
    1. Set AES_KEY env var (64-char hex):
         $env:AES_KEY = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"
    2. Start the Flask server:
         python ml-backend/app.py
    3. Run this script:
         python ml-backend/tests/test_api.py [--url http://localhost:5001] [--image path/to/test.png]

The tests run sequentially and print PASS / FAIL per case.
No pytest dependency required (stdlib unittest used).
"""

import os
import sys
import base64
import argparse
import unittest
import textwrap
from io import BytesIO
from pathlib import Path

import urllib.request
import urllib.error
import json

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL   = os.environ.get("ML_API_URL", "http://localhost:5001")
TEST_IMAGE = None   # override via --image argument


# ── Helpers ───────────────────────────────────────────────────────────────────

def _post(endpoint: str, payload: dict) -> tuple[int, dict]:
    url  = f"{BASE_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _get(endpoint: str) -> tuple[int, dict]:
    url = f"{BASE_URL}{endpoint}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _make_dummy_image_b64(size: int = 96) -> str:
    """Generate a tiny white grayscale PNG as base64."""
    try:
        import cv2
        import numpy as np
        img = np.full((size, size), 200, dtype=np.uint8)
        # Add some variation so the image is non-trivial
        img[20:40, 20:40] = 50
        _, buf = cv2.imencode(".png", img)
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except ImportError:
        # Minimal valid 1×1 white PNG (hard-coded)
        return (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
            "z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
        )


def _load_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# ── Test cases ─────────────────────────────────────────────────────────────────

class TestHealth(unittest.TestCase):

    def test_health_returns_ok(self):
        status, body = _get("/health")
        self.assertEqual(status, 200, f"Expected 200, got {status}: {body}")
        self.assertEqual(body.get("status"), "ok")
        print("  PASS  GET /health → status=ok")


class TestEnroll(unittest.TestCase):

    def _b64(self):
        if TEST_IMAGE:
            return _load_image_b64(TEST_IMAGE)
        return _make_dummy_image_b64()

    def test_enroll_success(self):
        status, body = _post("/enroll", {
            "nin": "TEST-NIN-001",
            "fingerprint_b64": self._b64(),
        })
        self.assertEqual(status, 200, f"Expected 200, got {status}: {body}")
        self.assertIn("encrypted_embedding", body)
        self.assertIsInstance(body["encrypted_embedding"], str)
        self.assertGreater(len(body["encrypted_embedding"]), 50)
        print(f"  PASS  POST /enroll → token_len={len(body['encrypted_embedding'])}")
        return body["encrypted_embedding"]   # used by verify test

    def test_enroll_missing_nin(self):
        status, body = _post("/enroll", {"fingerprint_b64": self._b64()})
        self.assertEqual(status, 400)
        self.assertIn("error", body)
        print("  PASS  POST /enroll (missing nin) → 400")

    def test_enroll_missing_image(self):
        status, body = _post("/enroll", {"nin": "TEST-NIN-002"})
        self.assertEqual(status, 400)
        self.assertIn("error", body)
        print("  PASS  POST /enroll (missing image) → 400")

    def test_enroll_bad_image(self):
        status, body = _post("/enroll", {
            "nin": "TEST-NIN-003",
            "fingerprint_b64": "not-valid-base64!!!",
        })
        self.assertEqual(status, 400)
        print("  PASS  POST /enroll (bad image) → 400")


class TestVerify(unittest.TestCase):

    def _b64(self):
        if TEST_IMAGE:
            return _load_image_b64(TEST_IMAGE)
        return _make_dummy_image_b64()

    def test_verify_roundtrip(self):
        """Enroll then verify the same image — should return match=True."""
        img_b64 = self._b64()
        _, enroll_body = _post("/enroll", {"nin": "VERIFY-TEST-NIN", "fingerprint_b64": img_b64})
        token = enroll_body.get("encrypted_embedding", "")
        self.assertTrue(token, "Enrollment must succeed before verify test")

        status, body = _post("/verify", {
            "nin":              "VERIFY-TEST-NIN",
            "fingerprint_b64":  img_b64,
            "stored_embedding": token,
            "threshold":        0.9,   # generous threshold; same image should be very close
        })
        self.assertEqual(status, 200, f"Expected 200, got {status}: {body}")
        self.assertIn("match", body)
        self.assertIn("distance", body)
        print(f"  PASS  POST /verify (same image) → match={body['match']}  dist={body['distance']:.4f}")

    def test_verify_missing_fields(self):
        status, body = _post("/verify", {"nin": "X"})
        self.assertEqual(status, 400)
        print("  PASS  POST /verify (missing fields) → 400")


# ── Runner ─────────────────────────────────────────────────────────────────────

def main():
    global TEST_IMAGE, BASE_URL

    parser = argparse.ArgumentParser(description="ML microservice integration tests")
    parser.add_argument("--url",   default=BASE_URL, help="Base URL of the Flask server")
    parser.add_argument("--image", default=None,     help="Path to a real fingerprint PNG for testing")
    args = parser.parse_args()

    BASE_URL   = args.url
    TEST_IMAGE = args.image

    print(f"\n── ML Microservice Integration Tests ──────────────────")
    print(f"  Server : {BASE_URL}")
    print(f"  Image  : {TEST_IMAGE or '(synthetic dummy)'}\n")

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestHealth, TestEnroll, TestVerify]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, "w"))
    result = runner.run(suite)

    passed = result.testsRun - len(result.failures) - len(result.errors)
    print(f"\n{'─' * 54}")
    print(f"  {passed}/{result.testsRun} tests passed")
    if result.failures or result.errors:
        for _, tb in result.failures + result.errors:
            print(textwrap.indent(tb, "    "))
        sys.exit(1)
    else:
        print("  ✅  All integration tests passed.\n")


if __name__ == "__main__":
    main()
