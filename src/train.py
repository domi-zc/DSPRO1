import datetime
import random
from pathlib import Path

import torch
from ultralytics import YOLO

CLASS_NAMES = ["plane"]

EPOCHS = 100
BATCH = 8
IMGSZ = 640
LR0 = 0.002
WEIGHT_DECAY = 5e-4
WORKERS = 0
SEED = 42

RUNS_ROOT = Path("./runs")
MODEL_YAML = Path("./models/yolo_plane_tiny.yaml")

def find_dataset_root(start: Path | None = None) -> Path:
    """
    Find a YOLO-style dataset root by searching upward from `start`
    (default: current working directory).

    Expected structure:
      dataset_root/
        train/images
        train/labels
        val|valid/images
        val|valid/labels
    """
    start = start or Path.cwd()

    def looks_like_dataset(p: Path) -> bool:
        train = p / "train"
        if not (train / "images").is_dir() or not (train / "labels").is_dir():
            return False

        for v in ("valid", "val"):
            if (p / v / "images").is_dir() and (p / v / "labels").is_dir():
                return True
        return False

    # check current dir and parents
    for p in [start, *start.parents]:
        if looks_like_dataset(p):
            return p

    # optional: check immediate subdirectories
    for p in start.iterdir():
        if p.is_dir() and looks_like_dataset(p):
            return p

    raise RuntimeError(
        "Could not auto-detect dataset root.\n"
        "Expected a folder containing:\n"
        "  train/images + train/labels\n"
        "  valid|val/images + valid|val/labels"
    )

dataset_root = find_dataset_root()


def require_gpu_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError("No GPU backend available (CUDA/MPS). Refusing to run on CPU.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def make_run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def detect_splits(dataset_root: Path) -> tuple[str, str]:
    train_images = dataset_root / "train" / "images"
    train_labels = dataset_root / "train" / "labels"
    if not train_images.is_dir() or not train_labels.is_dir():
        raise FileNotFoundError(
            f"Missing train folders. Expected:\n{train_images}\n{train_labels}"
        )

    candidates = ["valid", "val"]
    for v in candidates:
        val_images = dataset_root / v / "images"
        val_labels = dataset_root / v / "labels"
        if val_images.is_dir() and val_labels.is_dir():
            return "train/images", f"{v}/images"

    found = [p.name for p in dataset_root.iterdir() if p.is_dir()]
    raise FileNotFoundError(
        "Could not find validation split. Expected one of:\n"
        f"  {dataset_root}/valid/images + labels\n"
        f"  {dataset_root}/val/images + labels\n"
        f"Found split dirs: {found}"
    )


def write_dataset_yaml(path: Path, dataset_root: Path, class_names: list[str]) -> None:
    train_rel, val_rel = detect_splits(dataset_root)
    path.write_text(
        "\n".join(
            [
                f"path: {dataset_root.as_posix()}",
                f"train: {train_rel}",
                f"val: {val_rel}",
                f"nc: {len(class_names)}",
                f"names: {class_names}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    device = require_gpu_device()
    seed_everything(SEED)

    run_dir = make_run_dir(RUNS_ROOT)
    data_yaml = run_dir / "dataset.yaml"
    write_dataset_yaml(data_yaml, dataset_root, CLASS_NAMES)


    model = YOLO(str(MODEL_YAML))  # from scratch (YAML model definition)

    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        optimizer="AdamW",
        lr0=LR0,
        weight_decay=WEIGHT_DECAY,
        workers=WORKERS,
        seed=SEED,
        pretrained=False,
        project=str(run_dir),
        name="train",
        exist_ok=True,
        device=device,
    )

    print(f"Run saved in: {run_dir}")


if __name__ == "__main__":
    main()
