import json
from pathlib import Path
from typing import Optional, Dict, Any


class Config:
    """
    General project config manager.

    - Can write/read a config.json in an arbitrary directory (e.g. per-run folder).
    - Auto-finds YOLO-style dataset root when dataset_root is missing.
    """

    def __init__(
        self,
        start_dir: Optional[str] = None,
        config_name: str = "config.json",
        valid_split_names=("valid", "val"),
        defaults: Optional[Dict[str, Any]] = None,
    ):
        # start_dir is where we search for a dataset root, NOT where the config is stored.
        self.start_dir = Path(start_dir) if start_dir else Path.cwd()
        self.config_name = config_name
        self.valid_split_names = tuple(valid_split_names)

        self.defaults: Dict[str, Any] = {
            # dataset
            "dataset_root": None,
            "train_split": "train",
            "val_split": "valid",
            "class_names": ["plane"],

            # training hyperparams
            "batch_size": 4,
            "num_epochs": 5,
            "lr": 5e-4,
            "weight_decay": 5e-4,
            "num_workers": 0,
            "random_seed": 42,

            # checkpoints (no global path; per-run directory is used)
            "save_checkpoints": True,
        }
        if defaults:
            self.defaults.update(defaults)

    # ---- public one-call API ----
    def setup(
        self,
        dataset_root: Optional[Path] = None,
        config_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Load config if it exists; otherwise create it.
        Ensures dataset_root is present (auto-found if missing).
        Returns full config dict.

        config_dir: directory where the config.json for THIS RUN lives.
        """
        cfg_dir = Path(config_dir) if config_dir is not None else self.start_dir
        cfg_dir.mkdir(parents=True, exist_ok=True)

        cfg_path = cfg_dir / self.config_name

        if cfg_path.exists():
            cfg = self._read_json(cfg_path)
            cfg = self._merge_defaults(cfg)

            if not cfg.get("dataset_root"):
                root = dataset_root or self._find_dataset_root()
                cfg["dataset_root"] = str(root.resolve())
                self._write_json(cfg_path, cfg)

            return cfg

        # Create new config for this run
        root = dataset_root or self._find_dataset_root()
        cfg = dict(self.defaults)
        cfg["dataset_root"] = str(root.resolve())
        self._write_json(cfg_path, cfg)
        return cfg

    # ---- private helpers ----
    def _find_dataset_root(self) -> Path:
        """
        Search for dataset root starting from start_dir.
        Strategy:
          1) check start_dir and its parents
          2) if not found, scan descendants
        """
        for p in [self.start_dir, *self.start_dir.parents]:
            if self._looks_like_dataset_root(p):
                return p

        for p in self._iter_dirs(self.start_dir):
            if self._looks_like_dataset_root(p):
                return p

        raise FileNotFoundError(f"No dataset root found under {self.start_dir}")

    def _looks_like_dataset_root(self, root: Path) -> bool:
        train = root / "train"
        if not train.is_dir():
            return False

        valid = next(
            (root / n for n in self.valid_split_names if (root / n).is_dir()),
            None
        )
        if valid is None:
            return False

        need = [
            train / "images", train / "labels",
            valid / "images", valid / "labels",
        ]
        return all(x.is_dir() for x in need)

    def _iter_dirs(self, base: Path):
        """Yield directories under base, skipping hidden folders."""
        for p in base.rglob("*"):
            if p.is_dir() and not p.name.startswith("."):
                yield p

    def _read_json(self, path: Path) -> Dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_json(self, path: Path, data: Dict[str, Any]):
        path.write_text(json.dumps(data, indent=2))

    def _merge_defaults(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(self.defaults)
        merged.update(cfg or {})
        return merged