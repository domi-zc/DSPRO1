from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def imgname_to_labelpath(img_path: Path, lbl_dir: Path) -> Path:
    name_no_ext = img_path.name.rsplit(".", 1)[0]
    return lbl_dir / f"{name_no_ext}.txt"


def yolo_txt_to_boxes_xyxy(
    txt_path: Path,
    img_w: int,
    img_h: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a YOLO-format .txt file to (boxes, labels) in absolute xyxy.
    YOLO format: <cls> <cx> <cy> <w> <h> in normalized (0-1) coordinates.
    """
    boxes, labels = [], []

    if not txt_path.exists():
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
        )

    with txt_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, cx, cy, w, h = parts
            try:
                cls = int(float(cls))
                cx, cy, w, h = map(float, (cx, cy, w, h))
            except ValueError:
                continue

            x1 = (cx - w / 2.0) * img_w
            y1 = (cy - h / 2.0) * img_h
            x2 = (cx + w / 2.0) * img_w
            y2 = (cy + h / 2.0) * img_h

            # clamp to image bounds
            x1 = max(0.0, min(x1, img_w - 1))
            y1 = max(0.0, min(y1, img_h - 1))
            x2 = max(0.0, min(x2, img_w - 1))
            y2 = max(0.0, min(y2, img_h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(cls)  # keep 0-based class ids

    if boxes:
        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.int64),
        )

    return (
        torch.zeros((0, 4), dtype=torch.float32),
        torch.zeros((0,), dtype=torch.int64),
    )


class YOLODetectionDataset(Dataset):
    """
    Basic dataset for RetinaNet-style training, using YOLO txt label files.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        transforms: Optional[T.Compose] = None,
    ):
        self.img_dir = root / split / "images"
        self.lbl_dir = root / split / "labels"
        self.transforms = transforms

        self.image_paths: List[Path] = sorted(
            p for p in self.img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images in {self.img_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        lbl_path = imgname_to_labelpath(img_path, self.lbl_dir)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        boxes, labels = yolo_txt_to_boxes_xyxy(lbl_path, w, h)

        if self.transforms is not None:
            img_t = self.transforms(img)
        else:
            img_t = T.ToTensor()(img)

        target: Dict[str, Any] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
            * (boxes[:, 3] - boxes[:, 1]).clamp(min=0),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
        }

        return img_t, target


def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


class DetectionDataModule:
    """
    Small helper class that encapsulates:
      - dataset construction
      - transforms
      - dataloader creation

    so your training script stays clean.
    """

    def __init__(
        self,
        dataset_root: Path,
        train_split: str,
        val_split: str,
        batch_size: int,
        num_workers: int = 4,
        train_transforms: Optional[T.Compose] = None,
        val_transforms: Optional[T.Compose] = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Defaults: simple ToTensor; easy to extend later
        self.train_transforms = train_transforms or T.ToTensor()
        self.val_transforms = val_transforms or T.ToTensor()

        self.train_ds: Optional[YOLODetectionDataset] = None
        self.val_ds: Optional[YOLODetectionDataset] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

    def setup(self):
        self.train_ds = YOLODetectionDataset(
            self.dataset_root,
            self.train_split,
            transforms=self.train_transforms,
        )
        self.val_ds = YOLODetectionDataset(
            self.dataset_root,
            self.val_split,
            transforms=self.val_transforms,
        )

    def create_loaders(self, shuffle_train: bool = True):
        if self.train_ds is None or self.val_ds is None:
            self.setup()

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=shuffle_train,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
        )

        return self.train_loader, self.val_loader
