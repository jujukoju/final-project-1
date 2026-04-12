"""
ml-backend/models/siamese.py

Siamese neural network for fingerprint verification.

Architecture:
    BaseCNN  — 4 conv blocks → GlobalAvgPool → FC → 128-D L2-normalised embedding
    SiameseNet — wraps two shared BaseCNN branches; outputs (emb1, emb2, distance)

Usage:
    from ml_backend.models.siamese import SiameseNet
    model = SiameseNet(embedding_dim=128, distance='euclidean')
    emb1, emb2, dist = model(img1, img2)   # img tensors: (B, 1, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Convolutional building block ──────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Base CNN feature extractor ────────────────────────────────────────────────

class BaseCNN(nn.Module):
    """
    4-stage convolutional feature extractor for grayscale fingerprint images.

    Input:  (B, 1, H, W)   — H=W=96 or 128 recommended
    Output: (B, embedding_dim) — L2-normalised embedding
    """

    def __init__(self, embedding_dim: int = 128, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1,   32, kernel=3, pool=2),   # → (B, 32,  H/2,  W/2)
            ConvBlock(32,  64, kernel=3, pool=2),   # → (B, 64,  H/4,  W/4)
            ConvBlock(64, 128, kernel=3, pool=2),   # → (B, 128, H/8,  W/8)
            ConvBlock(128, 256, kernel=3, pool=2),  # → (B, 256, H/16, W/16)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)          # → (B, 256, 1, 1)

        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised embedding of shape (B, embedding_dim)."""
        x = self.features(x)
        x = self.gap(x)
        x = self.embed(x)
        return F.normalize(x, p=2, dim=1)


# ── Siamese network ───────────────────────────────────────────────────────────

class SiameseNet(nn.Module):
    """
    Siamese network wrapping two shared BaseCNN branches.

    Args:
        embedding_dim : dimension of the output embedding (default 128)
        distance      : 'euclidean' or 'cosine' (default 'euclidean')
        dropout       : dropout rate inside BaseCNN (default 0.3)

    Forward:
        x1, x2  → emb1, emb2, distance
        x1 / x2 : (B, 1, H, W) float tensors in [0, 1]
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        distance: str = "euclidean",
        dropout: float = 0.3,
    ):
        super().__init__()
        assert distance in ("euclidean", "cosine"), \
            f"distance must be 'euclidean' or 'cosine', got {distance!r}"

        self.backbone = BaseCNN(embedding_dim=embedding_dim, dropout=dropout)
        self.distance_type = distance

    def _compute_distance(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        if self.distance_type == "euclidean":
            return torch.norm(emb1 - emb2, p=2, dim=1)          # (B,)
        # cosine distance = 1 - cosine_similarity
        return 1.0 - F.cosine_similarity(emb1, emb2, dim=1)     # (B,)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb1 = self.backbone(x1)
        emb2 = self.backbone(x2)
        dist = self._compute_distance(emb1, emb2)
        return emb1, emb2, dist

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method: return embedding for a single image batch."""
        return self.backbone(x)
