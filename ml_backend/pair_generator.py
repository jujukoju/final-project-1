"""
ml-backend/pair_generator.py — Phase 4.2

Generates balanced genuine / impostor pairs from the SOCOFing metadata CSV.

Each dataset item is a tuple (img1, img2, label):
    label = 1  → same subject (genuine pair)
    label = 0  → different subjects (impostor pair)

The positive/negative ratio is exactly 1:1.
Pairs are reshuffled each epoch via the DataLoader's sampler.
"""

import random
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from split_dataset import load_metadata   # reuse the existing helper

logger = logging.getLogger(__name__)


class PairDataset(Dataset):
    """
    Args:
        metadata_csv_dir : directory that contains ``metadata.csv``
        processed_root   : root of pre-processed PNG images (mirrors SOCOFing tree)
        split            : 'train' | 'val' | 'test'
        img_size         : (H, W) to resize images to
        transform        : optional callable applied to each numpy grayscale image
                           before conversion to tensor, e.g. augmentation pipeline
        seed             : RNG seed for reproducible pair sampling
        pairs_per_epoch  : total pairs to materialise per epoch. Defaults to
                           2 × number of genuine images in the split so that
                           every real sample appears roughly once as an anchor.
    """

    def __init__(
        self,
        metadata_csv_dir: str | Path,
        processed_root: str | Path,
        split: str = "train",
        img_size: tuple[int, int] = (96, 96),
        transform=None,
        seed: int = 42,
        pairs_per_epoch: Optional[int] = None,
    ):
        self.processed_root = Path(processed_root)
        self.img_size = img_size
        self.transform = transform
        self.rng = random.Random(seed)

        # Load metadata for the requested split
        records = load_metadata(metadata_csv_dir, split=split)

        # Group image paths by subject_id
        self._subj_to_paths: dict[int, list[str]] = {}
        missing = 0
        for rec in records:
            p = Path(rec["filename"])
            # metadata stores raw BMP paths; look for processed PNG counterpart
            rel = p.relative_to(p.parts[0] + "/" + p.parts[1]) if len(p.parts) > 2 else p
            png_path = self.processed_root / Path(*p.parts[2:]).with_suffix(".png") \
                if len(p.parts) > 2 else self.processed_root / p.stem

            # Fallback: try the raw path if PNG not found
            if not png_path.exists():
                png_path = p

            if not png_path.exists():
                missing += 1
                continue
            sid = int(rec["subject_id"])
            self._subj_to_paths.setdefault(sid, []).append(str(png_path))

        if missing:
            logger.warning("PairDataset: %d files listed in metadata not found on disk", missing)

        self._subjects = sorted(self._subj_to_paths.keys())
        if len(self._subjects) < 2:
            raise RuntimeError("Need at least 2 subjects to form impostor pairs.")

        # Materialise pairs
        n_genuine = sum(len(v) for v in self._subj_to_paths.values())
        total_pairs = pairs_per_epoch if pairs_per_epoch else 2 * n_genuine
        self._pairs = self._build_pairs(total_pairs)

        logger.info(
            "PairDataset %s: %d subjects, %d pairs (%d genuine / %d impostor)",
            split, len(self._subjects), len(self._pairs),
            sum(1 for _, _, lbl in self._pairs if lbl == 1),
            sum(1 for _, _, lbl in self._pairs if lbl == 0),
        )

    # ── Pair construction ──────────────────────────────────────────────────────

    def _build_pairs(self, total: int) -> list[tuple[str, str, int]]:
        """Return a balanced list of (path1, path2, label) tuples."""
        n_each = total // 2
        pairs: list[tuple[str, str, int]] = []

        # Genuine pairs (label=1)
        for _ in range(n_each):
            sid = self.rng.choice(self._subjects)
            paths = self._subj_to_paths[sid]
            if len(paths) >= 2:
                p1, p2 = self.rng.sample(paths, 2)
            else:
                p1 = p2 = paths[0]
            pairs.append((p1, p2, 1))

        # Impostor pairs (label=0)
        for _ in range(n_each):
            s1, s2 = self.rng.sample(self._subjects, 2)
            p1 = self.rng.choice(self._subj_to_paths[s1])
            p2 = self.rng.choice(self._subj_to_paths[s2])
            pairs.append((p1, p2, 0))

        self.rng.shuffle(pairs)
        return pairs

    def reshuffle(self, seed: Optional[int] = None):
        """Re-generate pairs (call at the start of each epoch for variety)."""
        if seed is not None:
            self.rng = random.Random(seed)
        self._pairs = self._build_pairs(len(self._pairs))

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path1, path2, label = self._pairs[idx]
        img1 = self._load(path1)
        img2 = self._load(path2)
        return img1, img2, torch.tensor(label, dtype=torch.long)

    def _load(self, path: str) -> torch.Tensor:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Cannot read image: {path}")
        img = cv2.resize(img, self.img_size)
        if self.transform is not None:
            img = self.transform(img)
        # → (1, H, W) float32 in [0, 1]
        return torch.from_numpy(img).unsqueeze(0).float() / 255.0
