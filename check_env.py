"""
check_env.py — Phase 2.3 local environment validation.

Run:
    python check_env.py

Checks:
    - Python version (≥3.9 required)
    - PyTorch import + CUDA/GPU availability
    - OpenCV import
    - NumPy import
    - Node.js / npm versions (via subprocess)
    - IPFS local gateway reachability (http://127.0.0.1:8080)
"""
import sys
import subprocess
import importlib
from urllib.request import urlopen
from urllib.error import URLError

GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def ok(msg):  return f"{GREEN}  ✅  {msg}{RESET}"
def fail(msg): return f"{RED}  ❌  {msg}{RESET}"
def warn(msg): return f"{YELLOW}  ⚠️  {msg}{RESET}"


results = []

# ── 1. Python version ──────────────────────────────────────────────────────────
vi = sys.version_info
ver_str = f"{vi.major}.{vi.minor}.{vi.micro}"
if (vi.major, vi.minor) >= (3, 9):
    results.append(ok(f"Python {ver_str}"))
else:
    results.append(fail(f"Python {ver_str}  (need ≥3.9)"))

# ── 2. PyTorch + GPU ──────────────────────────────────────────────────────────
try:
    import torch
    cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if cuda else "CPU only"
    results.append(ok(f"PyTorch {torch.__version__}  |  GPU: {device}"))
    if not cuda:
        results.append(warn("No CUDA GPU found — training will run on CPU (slower)"))
except ImportError as exc:
    results.append(fail(f"PyTorch not found: {exc}"))

# ── 3. OpenCV ─────────────────────────────────────────────────────────────────
try:
    import cv2
    results.append(ok(f"OpenCV {cv2.__version__}"))
except ImportError as exc:
    results.append(fail(f"OpenCV not found: {exc}"))

# ── 4. NumPy ──────────────────────────────────────────────────────────────────
try:
    import numpy as np
    results.append(ok(f"NumPy {np.__version__}"))
except ImportError as exc:
    results.append(fail(f"NumPy not found: {exc}"))

# ── 5. scikit-learn ───────────────────────────────────────────────────────────
try:
    import sklearn
    results.append(ok(f"scikit-learn {sklearn.__version__}"))
except ImportError:
    results.append(warn("scikit-learn not installed  (needed for evaluation metrics)"))

# ── 6. Flask ──────────────────────────────────────────────────────────────────
try:
    import flask
    results.append(ok(f"Flask {flask.__version__}"))
except ImportError:
    results.append(warn("Flask not installed  (needed for ml-backend and oracle-api)"))

# ── 7. cryptography ───────────────────────────────────────────────────────────
try:
    import cryptography
    results.append(ok(f"cryptography {cryptography.__version__}"))
except ImportError:
    results.append(warn("cryptography not installed  (needed for AES-256 embedding encryption)"))

# ── 8. web3 ──────────────────────────────────────────────────────────────────
try:
    import web3
    results.append(ok(f"web3.py {web3.__version__}"))
except ImportError:
    results.append(warn("web3.py not installed  (needed for oracle-api)"))

# ── 9. Node.js ────────────────────────────────────────────────────────────────
try:
    node_out = subprocess.check_output(["node", "--version"], stderr=subprocess.DEVNULL, timeout=5)
    node_ver = node_out.decode().strip()
    results.append(ok(f"Node.js {node_ver}"))
except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
    results.append(fail("Node.js not found  (required for Hardhat/blockchain)"))

# ── 10. npm ───────────────────────────────────────────────────────────────────
try:
    npm_out = subprocess.check_output(["npm", "--version"], stderr=subprocess.DEVNULL, timeout=5)
    npm_ver = npm_out.decode().strip()
    results.append(ok(f"npm {npm_ver}"))
except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
    results.append(fail("npm not found"))

# ── 11. IPFS local gateway ────────────────────────────────────────────────────
try:
    urlopen("http://127.0.0.1:8080/ipfs/", timeout=2)
    results.append(ok("IPFS local gateway reachable at http://127.0.0.1:8080"))
except URLError:
    results.append(warn("IPFS local gateway not reachable  (start IPFS Desktop or daemon if needed)"))

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  Environment Check — Decentralised Fingerprint Verification")
print("─" * 60)
for r in results:
    print(r)
print("─" * 60 + "\n")

any_fail = any("❌" in r for r in results)
if any_fail:
    print(f"{RED}Some checks failed — resolve the ❌ issues before proceeding.{RESET}\n")
    sys.exit(1)
else:
    print(f"{GREEN}All critical checks passed.{RESET}\n")
