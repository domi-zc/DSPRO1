from device import setup_device_and_parallel
from config import Config
from data_module import DetectionDataModule
from custom_model import get_model
import time
from pathlib import Path
from tqdm.auto import tqdm

from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import retinanet_resnet50_fpn_v2
import datetime

CHECKPOINT_ROOT = Path("./checkpoints_custom_model")


def create_new_run_dir(checkpoint_root: Path) -> Path:
    """Ask for a run name, create a unique subfolder, and return it."""
    run_name = input("Enter a name for this training run (leave empty for timestamp): ").strip()
    if not run_name:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = checkpoint_root / run_name
    # Ensure uniqueness
    suffix = 1
    while run_dir.exists():
        run_dir = checkpoint_root / f"{run_name}_{suffix}"
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"[RUN] Created new run directory: {run_dir}")
    return run_dir


def select_run_dir(checkpoint_root: Path) -> tuple[Path, bool]:
    """
    Ask the user to either start a new run or continue an existing one.

    Returns:
        (run_dir, resume_flag)
    """
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    existing_runs = sorted([d for d in checkpoint_root.iterdir() if d.is_dir()])

    if not existing_runs:
        print("[RUN] No existing runs found. Starting a NEW training run.")
        run_dir = create_new_run_dir(checkpoint_root)
        return run_dir, False

    print("\n=== Training mode selection ===")
    print(" [0] Start NEW training run")
    for idx, d in enumerate(existing_runs, start=1):
        print(f" [{idx}] Continue run: {d.name}")
    choice = input("Select an option: ").strip()

    if choice == "" or choice == "0":
        run_dir = create_new_run_dir(checkpoint_root)
        return run_dir, False

    try:
        idx = int(choice)
    except ValueError:
        print("[WARN] Invalid input, defaulting to NEW run.")
        run_dir = create_new_run_dir(checkpoint_root)
        return run_dir, False

    if idx < 1 or idx > len(existing_runs):
        print("[WARN] Invalid selection, defaulting to NEW run.")
        run_dir = create_new_run_dir(checkpoint_root)
        return run_dir, False

    run_dir = existing_runs[idx - 1]
    print(f"[RUN] Continuing from existing run: {run_dir}")
    return run_dir, True


def save_checkpoint(run_dir: Path,
                    epoch: int,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    lr_scheduler,
                    best_val: float,
                    tag: str):
    """
    Save a full checkpoint (model + optimizer + scheduler + epoch + best_val).
    tag should be 'best' or 'last'.
    """
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "best_val": best_val,
    }
    path = run_dir / f"retinanet_{tag}.pth"
    torch.save(state, path)


def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    model.train()  # returns losses only in train mode when given targets
    loss_sum, iters = 0.0, 0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        losses = model(images, targets)
        loss = sum(losses.values()).item()
        loss_sum += loss
        iters += 1
    return loss_sum / max(1, iters)



def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=None):
    model.train()
    loss_sum, iters = 0.0, 0

    for i, (images, targets) in enumerate(data_loader, 1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        losses = model(images, targets)
        loss = sum(losses.values())

        if not torch.isfinite(loss):
            if print_freq is not None:
                print(f"[WARN] Non-finite loss at iter {i}: {loss.item()}, skipping")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        loss_sum += loss.item()
        iters += 1

        if print_freq is not None and (i % print_freq == 0):
            avg = loss_sum / max(1, iters)
            print(f"Epoch {epoch} | it {i}/{len(data_loader)} | running mean loss {avg:.4f}")

    return loss_sum / max(1, iters)



@torch.no_grad()
def evaluate_detection_metrics(model, data_loader, device, score_thresh: float = 0.05, iou_thresh: float = 0.50):
    """
    Returns dict: P, R, mAP50, mAP50-95

    Robust to model outputs being:
      - list[dict(boxes,scores,labels)]   (torchvision detection)
      - dict of batched tensors           (boxes[B,N,4], scores[B,N], labels[B,N])
      - tensor[B,N,>=6]                   (x1,y1,x2,y2,score,label, ...)
    """
    from torchvision.ops import box_iou
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    
    printed = False

    def _to_list_of_dicts(preds, batch_size: int):
        # Case 1: list of dicts (torchvision standard)
        if isinstance(preds, list) and (len(preds) == 0 or isinstance(preds[0], dict)):
            return preds

        # Case 2: dict of batched tensors
        if isinstance(preds, dict) and "boxes" in preds and "scores" in preds and "labels" in preds:
            boxes = preds["boxes"]
            scores = preds["scores"]
            labels = preds["labels"]

            # allow unbatched
            if torch.is_tensor(boxes) and boxes.dim() == 2:
                boxes = boxes.unsqueeze(0)
                scores = scores.unsqueeze(0)
                labels = labels.unsqueeze(0)

            out = []
            for b in range(batch_size):
                out.append({"boxes": boxes[b], "scores": scores[b], "labels": labels[b]})
            return out

        # Case 3: Tensor outputs (handles [N,D], [B,N,D], [G,B,N,D])
        if torch.is_tensor(preds):
            t = preds

            # If DataParallel-style: [G,B,N,D] -> flatten G and B into batch
            if t.dim() == 4:
                # [G,B,N,D] -> [G*B,N,D]
                G, B = t.shape[0], t.shape[1]
                t = t.reshape(G * B, t.shape[2], t.shape[3])

            # If single image: [N,D] -> [1,N,D]
            if t.dim() == 2:
                t = t.unsqueeze(0)

            # Now expect [B,N,D]
            if t.dim() == 3:
                B = t.shape[0]
                D = t.shape[2]

                # If batch size mismatches (can happen with DP), try to reconcile
                # Use min to avoid index errors and keep things moving.
                use_B = min(B, batch_size)

                # Interpret last dimension
                # Common conventions:
                #   D>=6: [x1,y1,x2,y2,score,label,(...)]
                #   D==5: [x1,y1,x2,y2,score] (no label) -> label=1
                #   D==4: [x1,y1,x2,y2] (no score/label) -> score=1, label=1
                if D >= 6:
                    boxes = t[..., 0:4]
                    scores = t[..., 4]
                    labels = t[..., 5].long()
                elif D == 5:
                    boxes = t[..., 0:4]
                    scores = t[..., 4]
                    labels = torch.ones((t.shape[0], t.shape[1]), dtype=torch.long, device=t.device)
                elif D == 4:
                    boxes = t[..., 0:4]
                    scores = torch.ones((t.shape[0], t.shape[1]), dtype=torch.float32, device=t.device)
                    labels = torch.ones((t.shape[0], t.shape[1]), dtype=torch.long, device=t.device)
                else:
                    raise TypeError(f"Tensor predictions have unsupported last-dim D={D}, shape={tuple(preds.shape)}")

                out = []
                for b in range(use_B):
                    out.append({"boxes": boxes[b], "scores": scores[b], "labels": labels[b]})
                return out

            raise TypeError(f"Tensor predictions have unsupported rank={t.dim()}, original shape={tuple(preds.shape)}")

        # Case 4: tuple/list container
        if isinstance(preds, (tuple, list)) and len(preds) > 0:
            return _to_list_of_dicts(preds[0], batch_size)

        raise TypeError(f"Unsupported prediction type/shape: {type(preds)}")

    model.eval()

    map_metric = MeanAveragePrecision(iou_type="bbox")
    tp = fp = fn = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        batch_size = len(images)

        raw_preds = model(images)
        preds_list = _to_list_of_dicts(raw_preds, batch_size)
        
        if not printed:
            p0, t0 = preds_list[0], targets[0]
            print("IMG size:", images[0].shape[-2:], "dtype:", images[0].dtype)
            print("GT boxes min/max:",
                float(t0["boxes"].min().item()) if t0["boxes"].numel() else None,
                float(t0["boxes"].max().item()) if t0["boxes"].numel() else None,
                "shape:", tuple(t0["boxes"].shape))
            print("Pred boxes min/max:",
                float(p0["boxes"].min().item()) if p0["boxes"].numel() else None,
                float(p0["boxes"].max().item()) if p0["boxes"].numel() else None,
                "num:", int(p0["boxes"].shape[0]))
            print("Pred score max:",
                float(p0["scores"].max().item()) if p0["scores"].numel() else 0.0)
            print("GT labels unique:", torch.unique(t0["labels"]).tolist() if "labels" in t0 else None)
            print("Pred labels unique:", torch.unique(p0["labels"]).tolist() if p0["labels"].numel() else [])
            printed = True


        # Move to CPU for metrics + our matching
        preds_cpu_for_map = []
        targets_cpu_for_map = []

        for p, t in zip(preds_list, targets):
            pb = p["boxes"].detach().cpu()
            ps = p["scores"].detach().cpu()
            pl = p["labels"].detach().cpu()

            gt_boxes = t["boxes"].detach().cpu()
            gt_labels = t["labels"].detach().cpu()

            preds_cpu_for_map.append({"boxes": pb, "scores": ps, "labels": pl})
            targets_cpu_for_map.append({"boxes": gt_boxes, "labels": gt_labels})

            # ---- Simple P/R matching (score-filtered) ----
            keep = ps >= score_thresh
            pb_f = pb[keep]
            ps_f = ps[keep]
            pl_f = pl[keep]

            if pb_f.numel() == 0:
                fn += int(gt_boxes.shape[0])
                continue
            if gt_boxes.numel() == 0:
                fp += int(pb_f.shape[0])
                continue

            order = torch.argsort(ps_f, descending=True)
            pb_f = pb_f[order]
            pl_f = pl_f[order]

            matched_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)

            ious = box_iou(pb_f, gt_boxes)
            for pi in range(pb_f.shape[0]):
                same_cls = (gt_labels == pl_f[pi]) & (~matched_gt)
                if not torch.any(same_cls):
                    fp += 1
                    continue

                ious_pi = ious[pi].clone()
                ious_pi[~same_cls] = -1.0
                gi = int(torch.argmax(ious_pi).item())

                if ious_pi[gi].item() >= iou_thresh:
                    tp += 1
                    matched_gt[gi] = True
                else:
                    fp += 1

            fn += int((~matched_gt).sum().item())

        map_metric.update(preds_cpu_for_map, targets_cpu_for_map)

    out = map_metric.compute()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "P": float(precision),
        "R": float(recall),
        "mAP50": float(out["map_50"].item()),
        "mAP50-95": float(out["map"].item()),
    }



def main():
    print("[VERSIONS] torch:", torch.__version__, "| torchvision:", torchvision.__version__)

    # --- Checkpoint root and run directory selection ---
    checkpoint_root = CHECKPOINT_ROOT
    run_dir, resume = select_run_dir(checkpoint_root)

    # --- Per-run config handling ---
    cfg_manager = Config()
    cfg_path = run_dir / cfg_manager.config_name

    if resume:
        cfg = cfg_manager.setup(config_dir=run_dir)
        print(f"[CONFIG] Loaded config from: {cfg_path}")
    else:
        cfg = cfg_manager.setup(config_dir=run_dir)
        print(f"[CONFIG] Created new config for this run at: {cfg_path}")
        print("[CONFIG] You can now edit this file (e.g. lr, batch_size, num_epochs, etc.).")
        input("[CONFIG] When you're done editing and have saved the file, press Enter to start training...")
        cfg = cfg_manager.setup(config_dir=run_dir)

    dataset_root = Path(cfg["dataset_root"])
    print(f"[INFO] Using dataset_root: '{dataset_root}'")

    seed_everything(cfg["random_seed"])

    # Data
    data_module = DetectionDataModule(
        dataset_root=dataset_root,
        train_split=cfg["train_split"],
        val_split=cfg["val_split"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        train_transforms=T.ToTensor(),
        val_transforms=T.ToTensor(),
    )
    train_loader, val_loader = data_module.create_loaders()

    num_classes = len(cfg["class_names"])
    print(
        f"[INFO] Train images: {len(data_module.train_ds)} | "
        f"Val images: {len(data_module.val_ds)} | Classes: {num_classes}"
    )

    # --- Model / device / optimizer / scheduler setup ---
    model = get_model(num_classes)
    device, model, autocast_dtype = setup_device_and_parallel(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=4,
        gamma=0.1,
    )

    # --- Resume logic ---
    best_val = float("inf")
    start_epoch_offset = 0

    if resume:
        ckpt_path_last = run_dir / "retinanet_last.pth"
        ckpt_path_best = run_dir / "retinanet_best.pth"

        if ckpt_path_last.exists():
            ckpt_path = ckpt_path_last
        elif ckpt_path_best.exists():
            ckpt_path = ckpt_path_best
        else:
            ckpt_path = None

        if ckpt_path is None:
            print("[WARN] No checkpoint file found in the selected run. Training will start from scratch.")
        else:
            print(f"[CKPT] Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)

            if isinstance(checkpoint, dict) and "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
                if "optimizer" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if "lr_scheduler" in checkpoint:
                    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                best_val = checkpoint.get("best_val", float("inf"))
                start_epoch_offset = checkpoint.get("epoch", 0)
                print(f"[CKPT] Resumed from epoch {start_epoch_offset}, best_val={best_val:.4f}")
            else:
                model.load_state_dict(checkpoint)
                print("[CKPT] Loaded model weights only (old checkpoint format).")

    # --- Training loop (tqdm over GLOBAL epochs) ---
    if cfg.get("save_checkpoints", True):
        print(f"[INFO] Checkpoints for this run will be stored in: {run_dir}")

    num_epochs = int(cfg["num_epochs"])
    start_epoch = start_epoch_offset + 1
    end_epoch = start_epoch_offset + num_epochs  # inclusive

    pbar = tqdm(
        range(start_epoch, end_epoch + 1),
        total=end_epoch,          # makes it show .../end_epoch (global)
        initial=start_epoch - 1,  # shows resume progress (e.g. 2 already done)
        dynamic_ncols=True,
        leave=True,
        desc="Epochs",
    )

    for epoch in pbar:
        # IMPORTANT: your train_one_epoch must return mean train loss and accept print_freq=None
        _train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=None)

        val_loss = evaluate_loss(model, val_loader, device)
        lr_scheduler.step()

        # optional: keep your "best val loss" checkpointing logic unchanged
        if cfg.get("save_checkpoints", True) and val_loss < best_val:
            best_val = val_loss
            save_checkpoint(run_dir, epoch, model, optimizer, lr_scheduler, best_val, tag="best")

        # compute detection metrics for display
        #metrics = evaluate_detection_metrics(model, val_loader, device, score_thresh=0.05, iou_thresh=0.50)
        #metrics = evaluate_detection_metrics(model, val_loader, device, score_thresh=0.2, iou_thresh=0.50)
        metrics = evaluate_detection_metrics(model, val_loader, device, score_thresh=0.01, iou_thresh=0.50)


        # overwrite tqdm line with ONLY what you want
        pbar.set_postfix(
            val=f"{val_loss:.4f}",
            P=f"{metrics['P']:.3f}",
            R=f"{metrics['R']:.3f}",
            mAP50=f"{metrics['mAP50']:.3f}",
            mAP50_95=f"{metrics['mAP50-95']:.3f}",
        )

    if cfg.get("save_checkpoints", True):
        save_checkpoint(run_dir, end_epoch, model, optimizer, lr_scheduler, best_val, tag="last")


if __name__ == "__main__":
    main()
