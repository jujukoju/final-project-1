"""
ml-backend/models/losses.py

Loss functions for Siamese fingerprint verification.

ContrastiveLoss  — primary training loss (Hadsell et al., 2006)
TripletLoss      — reference implementation for later experimentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for paired training.

    L = (1 - y) * 0.5 * d²
      + y       * 0.5 * max(0, margin - d)²

    where:
        d   = Euclidean/cosine distance between embeddings
        y   = 1 if genuine pair (same subject), 0 if impostor pair
        margin = minimum desired separation for impostor pairs

    Args:
        margin : float, default 1.0
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        distance: torch.Tensor,   # (B,) pairwise distance
        labels: torch.Tensor,     # (B,) 1=genuine, 0=impostor
    ) -> torch.Tensor:
        # Genuine pairs: penalise large distances
        loss_genuine = 0.5 * distance ** 2
        # Impostor pairs: penalise distances smaller than margin
        loss_impostor = 0.5 * torch.clamp(self.margin - distance, min=0.0) ** 2
        # labels: 1=genuine → use loss_genuine; 0=impostor → use loss_impostor
        loss = labels.float() * loss_genuine + (1 - labels.float()) * loss_impostor
        return loss.mean()


class TripletLoss(nn.Module):
    """
    Triplet loss for anchor-positive-negative training.

    L = max(0, d(a, p) - d(a, n) + margin)

    Args:
        margin : float, default 0.3
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,    # (B, D) embeddings
        positive: torch.Tensor,  # (B, D) embeddings
        negative: torch.Tensor,  # (B, D) embeddings
    ) -> torch.Tensor:
        d_pos = torch.norm(anchor - positive, p=2, dim=1)
        d_neg = torch.norm(anchor - negative, p=2, dim=1)
        loss = torch.clamp(d_pos - d_neg + self.margin, min=0.0)
        return loss.mean()
