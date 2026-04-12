"""
ml-backend/embedding.py — Phase 5.1

Embedding generation: wraps the trained Siamese BaseCNN so any preprocessed
fingerprint image can be converted to a 128-D normalised vector at inference time.

Usage:
    from ml_backend.embedding import EmbeddingExtractor
    extractor = EmbeddingExtractor("ml-backend/checkpoints/best_siamese.pt")
    emb = extractor.from_path("path/to/image.png")   # → np.ndarray, shape (128,)
    emb = extractor.from_array(img_uint8_gray)        # accept numpy H×W uint8
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """
    Thin inference wrapper around BaseCNN.

    Args:
        checkpoint : path to the `.pt` file saved by train.py
        device     : 'cuda', 'cpu', or None (auto-detect)
    """

    def __init__(self, checkpoint: str | Path, device: str | None = None):
        from ml_backend.models.siamese import SiameseNet   # local import avoids circular deps

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        saved_args = ckpt.get("args", {})

        self.embedding_dim = saved_args.get("embedding_dim", 128)
        self.img_size      = (saved_args.get("img_size", 96),) * 2
        distance_type      = saved_args.get("distance", "euclidean")

        model = SiameseNet(embedding_dim=self.embedding_dim, distance=distance_type)
        model.load_state_dict(ckpt["model_state_dict"])

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device).eval()

        logger.info(
            "EmbeddingExtractor loaded  |  checkpoint epoch=%s  |  device=%s",
            ckpt.get("epoch", "?"), self.device,
        )

    @torch.no_grad()
    def from_array(self, img: np.ndarray) -> np.ndarray:
        """
        Convert a grayscale uint8 numpy image (H×W) to a 128-D embedding.

        The image is expected to be already preprocessed (CLAHE + Gabor enhanced).
        """
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, self.img_size)
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float() / 255.0  # (1,1,H,W)
        tensor = tensor.to(self.device)
        emb = self.model.get_embedding(tensor)    # (1, D)
        return emb.squeeze(0).cpu().numpy()       # (D,)

    def from_path(self, path: str | Path) -> np.ndarray:
        """Load an image from disk and return its embedding."""
        path = str(path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return self.from_array(img)

    def distance(self, emb1: np.ndarray, emb2: np.ndarray, metric: str = "euclidean") -> float:
        """Compute distance between two embeddings."""
        if metric == "euclidean":
            return float(np.linalg.norm(emb1 - emb2))
        if metric == "cosine":
            return float(1.0 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))
        raise ValueError(f"Unknown metric: {metric!r}")
