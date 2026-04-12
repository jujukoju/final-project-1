import cv2
import torch
import logging
import numpy as np
from pathlib import Path
import albumentations as A
from typing import Optional
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from split_dataset import (
    generate_and_save_splits,
    load_metadata,
)

logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    def __init__(
            self,
            img_size=(224, 224),
            gabor_ksize=21,
            gabor_sigma=5,
            gabor_thetas=None,
            gabor_lambda=10,
            gabor_gamma=0.5
    ):
        if gabor_thetas is None:
            gabor_thetas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

        self.img_size = img_size
        self.gabor_ksize = gabor_ksize
        self.gabor_sigma = gabor_sigma
        self.gabor_thetas = gabor_thetas
        self.gabor_lambda = gabor_lambda
        self.gabor_gamma = gabor_gamma

        self.augment = A.Compose([
            A.Rotate(limit=10, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.GaussNoise(p=0.3),
            A.RandomScale(scale_limit=0.1, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Resize(height=img_size[0], width=img_size[1]),
            ToTensorV2()
        ])


    def load_image(self, path):
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Couldn't read image: {path}")
        return img


    def grayscale_conversion(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img


    def extraction(self, img):
        _, thresh = cv2.threshold(img, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            roi = img[y:y+h, x:x+w]
        else:
            roi = img
        return roi


    def enhance_image(self, img):
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        return clahe.apply(img)


    def gabor_filter(self, img):

        responses = []
        for theta in self.gabor_thetas:

            kernel = cv2.getGaborKernel(
                (self.gabor_ksize, self.gabor_ksize),
                self.gabor_sigma,
                theta,
                self.gabor_lambda,
                self.gabor_gamma
            )

            filtered = cv2.filter2D(img, cv2.CV_64F, kernel)

            responses.append(filtered)

        enhanced = np.mean(responses, axis=0)

        enhanced = cv2.normalize(
            enhanced,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        return enhanced

    def resize_image(self, img):
        return cv2.resize(img, self.img_size)


    def normalize(self, img):
        return img.astype("float32") / 255.0


    def augment_image(self, img):
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        augmented = self.augment(image=img)
        return augmented["image"]


    def process(self, img, augment=False):

        img = self.grayscale_conversion(img)
        img = self.extraction(img)
        img = self.enhance_image(img)
        img = self.gabor_filter(img)

        img = self.resize_image(img)
        if augment:
            img = self.augment_image(img)
        img = self.normalize(img)
        return img


    def directory_processing(self, input_dir, output_dir, augment=False):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"{input_dir} does not exist")

        output_dir.mkdir(parents=True, exist_ok=True)
        valid_extensions = [".bmp", ".png", ".jpg", ".jpeg"]

        image_paths = [
            p for p in input_dir.rglob("*")
            if p.suffix.lower() in valid_extensions
        ]

        if not image_paths:
            logger.warning(f"No images found in {input_dir}")
            return {"processed": 0, "skipped": 0, "errors": []}

        logger.info("Found %d images in %s", len(image_paths), input_dir)

        processed = 0
        skipped = 0
        errors = []

        for img_path in image_paths:
            try:
                raw = self.load_image(img_path)

                result = self.process(raw, augment=augment)

                if isinstance(result, torch.Tensor):
                    arr = result.numpy()
                    if arr.ndim == 3:
                        arr = arr.squeeze(0)
                    img_to_save = (arr * 255).astype(np.uint8) \
                        if arr.max() <= 1.0 else arr.astype(np.uint8)
                else:
                    img_to_save = (result * 255).astype(np.uint8)

                relative = img_path.relative_to(input_dir)
                out_path = output_dir / relative.with_suffix(".png")
                out_path.parent.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(str(out_path), img_to_save)
                processed += 1
                logger.debug("Saved: %s", out_path)

            except Exception as exc:
                skipped += 1
                errors.append((str(img_path), str(exc)))
                logger.warning("Skipped %s — %s", img_path.name, exc)

        logger.info(
            "Done. Processed: %d | Skipped: %d", processed, skipped
        )
        return {"processed": processed, "skipped": skipped, "errors": errors}


class PalmDataset(Dataset):

    def __init__(
        self,
        root,
        augment: bool = False,
        img_size: tuple = (224, 224),
        pipeline=None,
        split: Optional[str] = None,
        metadata_dir: Optional[str | Path] = None,
    ):
        """Initialise the dataset.

        Parameters
        ----------
        root         : root directory of pre-processed PNG images
        augment      : apply albumentations augmentation pipeline
        img_size     : (H, W) to resize images to
        pipeline     : PreprocessingPipeline instance (for its augment transform)
        split        : 'train' | 'val' | 'test' — filters samples via metadata.
                       Ignored when *metadata_dir* is None.
        metadata_dir : path to the directory containing metadata.csv (produced by
                       split_dataset.py).  When supplied, samples are loaded from
                       metadata and filtered by *split*; otherwise the legacy
                       directory-scan fallback is used.
        """
        self.root = Path(root)
        self.augment = augment
        self.img_size = img_size
        self.split = split

        if pipeline is not None:
            self._aug = pipeline.augment
        else:
            self._aug = A.Compose([
                A.Rotate(limit=10, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.GaussNoise(p=0.3),
                A.RandomScale(scale_limit=0.1, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Resize(height=img_size[0], width=img_size[1]),
                ToTensorV2()
            ])

        self._base = transforms.Compose([
            transforms.ToTensor()
        ])

        if metadata_dir is not None:
            self.samples, self.class_to_idx, self.metadata = \
                self._load_from_metadata(Path(metadata_dir), split)
        else:
            self.samples, self.class_to_idx = self._discover(self.root)
            self.metadata = None

        if not self.samples:
            msg = f"No images found (split={split!r}, root={self.root})"
            raise RuntimeError(msg)

        logger.info(
            "PalmDataset: %d samples, %d classes, split=%s, augment=%s",
            len(self.samples), len(self.class_to_idx), split, augment
        )

    # ── helpers ───────────────────────────────────────────────────────────

    def _load_from_metadata(
        self, metadata_dir: Path, split: Optional[str]
    ) -> tuple[list, dict, list]:
        """Load sample paths and labels from metadata.csv.

        The subject_id is used as the class label so the dataset is always
        subject-identity aware, regardless of split.
        """
        records = load_metadata(metadata_dir, split=split)
        # Build a stable class index sorted by subject_id
        unique_subjects = sorted({r["subject_id"] for r in records})
        class_to_idx = {str(sid): idx for idx, sid in enumerate(unique_subjects)}

        samples = []
        missing = 0
        for rec in records:
            p = Path(rec["filename"])
            if not p.exists():
                missing += 1
                continue
            label = class_to_idx[str(rec["subject_id"])]
            samples.append((p, label))

        if missing:
            logger.warning(
                "_load_from_metadata: %d file(s) listed in metadata not found on disk",
                missing,
            )
        return samples, class_to_idx, records

    def _discover(self, root):
        png_paths = sorted(root.rglob("*.png"))

        subdirs = sorted({p.parent.name for p in png_paths if p.parent != root})

        if subdirs:
            class_to_idx = {name: idx for idx, name in enumerate(subdirs)}
            samples = []
            for p in png_paths:
                label_name = p.parent.name
                if label_name in class_to_idx:
                    samples.append((p, class_to_idx[label_name]))
        else:
            class_to_idx = {"default": 0}
            samples = [(p, 0) for p in png_paths]

        return samples, class_to_idx


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read processed image: {img_path}")

        img = cv2.resize(img, self.img_size)

        if self.augment:
            tensor = self._aug(image=img)["image"]  # → (1, H, W) float32
            if tensor.dtype == torch.uint8:
                tensor = tensor.float() / 255.0
        else:
            tensor = torch.from_numpy(img).unsqueeze(0).float() / 255.0

        return tensor, label

    @property
    def idx_to_class(self):
        return {v: k for k, v in self.class_to_idx.items()}



def get_dataloader(
        dataset_root,
        augment: bool = False,
        img_size: tuple = (224, 224),
        pipeline=None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        split: Optional[str] = None,
        metadata_dir: Optional[str | Path] = None,
):
    """Create a DataLoader for the given dataset root.

    Parameters
    ----------
    split        : 'train' | 'val' | 'test' — passed through to PalmDataset.
    metadata_dir : path to the metadata directory (from split_dataset.py).
                   When provided, samples are filtered by *split*.
    """
    dataset = PalmDataset(
        root=dataset_root,
        augment=augment,
        img_size=img_size,
        pipeline=pipeline,
        split=split,
        metadata_dir=metadata_dir,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    logger.info(
        "DataLoader ready — %d batches of size %d (split=%s, augment=%s)",
        len(loader), batch_size, split, augment
    )
    return loader

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ── 1. Pre-processing pipeline ────────────────────────────────────────
    pipeline = PreprocessingPipeline(img_size=(224, 224))
    print("Pipeline:", pipeline)

    # Pre-process raw images into PNGs (skip if already done)
    report = pipeline.directory_processing(
        input_dir="data/raw",
        output_dir="data/processed_png",
        augment=False,
    )
    print(report)

    # ── 2. Generate subject-wise splits from the SOCOFing *source* dataset ─
    SOCOFING_ROOT = "data/SOCOFing"
    METADATA_DIR  = "data/SOCOFing/metadata"

    print("\nGenerating subject-wise splits …")
    csv_path, json_path = generate_and_save_splits(
        dataset_root=SOCOFING_ROOT,
        output_dir=METADATA_DIR,
        train=0.70,
        val=0.15,
        test=0.15,
        seed=42,
    )
    print(f"  metadata.csv  → {csv_path}")
    print(f"  metadata.json → {json_path}")

    # ── 3. Create per-split DataLoaders ───────────────────────────────────
    # NOTE: PalmDataset expects pre-processed PNG files.  The 'filename'
    # entries in the metadata point to the *raw* SOCOFing .BMP files; adjust
    # 'dataset_root' to point to your processed PNG directory if needed.
    train_loader = get_dataloader(
        dataset_root="data/processed_png",
        augment=True,
        pipeline=pipeline,
        split="train",
        metadata_dir=METADATA_DIR,
        batch_size=32,
        shuffle=True,
    )

    val_loader = get_dataloader(
        dataset_root="data/processed_png",
        augment=False,
        pipeline=pipeline,
        split="val",
        metadata_dir=METADATA_DIR,
        batch_size=32,
        shuffle=False,
    )

    test_loader = get_dataloader(
        dataset_root="data/processed_png",
        augment=False,
        pipeline=pipeline,
        split="test",
        metadata_dir=METADATA_DIR,
        batch_size=32,
        shuffle=False,
    )

    print(f"\nLoaders — train: {len(train_loader)} batches "
          f"| val: {len(val_loader)} batches "
          f"| test: {len(test_loader)} batches")

    for images, labels in train_loader:
        print(f"Batch shape : {images.shape}")
        print(f"Pixel range : [{images.min():.3f}, {images.max():.3f}]")
        break