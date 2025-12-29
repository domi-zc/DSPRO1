import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


class TinyBackbone(nn.Module):
    def __init__(self, in_channels=3, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # /2
            nn.Conv2d(base, base * 2, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # /4
            nn.Conv2d(base * 2, base * 4, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # /8
            nn.Conv2d(base * 4, base * 8, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # /16
        )
        self.out_channels = base * 8

    def forward(self, x):
        return self.net(x)


class SimplePlaneDetector(nn.Module):
    """
    Minimal detector for ONE class (airplane), multiple boxes per image.

    Per grid cell predicts:
      - objectness logit
      - tx, ty (offset within cell)  -> sigmoid
      - tw, th (log size relative to cell stride) -> exp when decoding

    Training:
      model(images, targets) -> dict of losses, compatible with your loop.
    """
    def __init__(self, in_channels=3, base=32, lambda_box=5.0):
        super().__init__()
        self.backbone = TinyBackbone(in_channels=in_channels, base=base)
        self.head = nn.Conv2d(self.backbone.out_channels, 5, kernel_size=1)
        self.lambda_box = float(lambda_box)

    def forward(self, images, targets=None):
        # Your dataloader gives list[Tensor(C,640,640)]
        if isinstance(images, (list, tuple)):
            batch = torch.stack(images, dim=0)  # assumes all same size
        else:
            batch = images

        pred = self.head(self.backbone(batch))  # (B,5,Gh,Gw)

        if self.training and targets is not None:
            H, W = batch.shape[-2], batch.shape[-1]
            return self._losses(pred, targets, H, W)

        # Return decoded detections for metrics / inference
        H, W = batch.shape[-2], batch.shape[-1]
        #return self.decode(pred, H, W, score_thresh=0.05, iou_thresh=0.5)
        #return self.decode(pred, H, W, score_thresh=0.2, iou_thresh=0.5, topk=100)
        return self.decode(pred, H, W, score_thresh=0.01, iou_thresh=0.5, topk=200)




    def _losses(self, pred, targets, H, W):
        device = pred.device
        B, _, Gh, Gw = pred.shape
        stride_y = H / float(Gh)
        stride_x = W / float(Gw)

        obj_logit = pred[:, 0]      # (B,Gh,Gw)
        box_raw = pred[:, 1:5]      # (B,4,Gh,Gw)

        # Targets per cell
        obj_t = torch.zeros((B, Gh, Gw), device=device)
        tx_t = torch.zeros((B, Gh, Gw), device=device)
        ty_t = torch.zeros((B, Gh, Gw), device=device)
        tw_t = torch.zeros((B, Gh, Gw), device=device)
        th_t = torch.zeros((B, Gh, Gw), device=device)
        pos_mask = torch.zeros((B, Gh, Gw), dtype=torch.bool, device=device)

        # For collision handling (multiple boxes in same cell), keep largest area
        best_area = torch.zeros((B, Gh, Gw), device=device)

        for b in range(B):
            boxes = targets[b].get("boxes", None)
            if boxes is None or boxes.numel() == 0:
                continue

            boxes = boxes.to(device)  # (M,4) xyxy

            for i in range(boxes.shape[0]):
                x1, y1, x2, y2 = boxes[i]
                bw = (x2 - x1).clamp(min=1.0)
                bh = (y2 - y1).clamp(min=1.0)
                area = bw * bh

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                gx = int(torch.clamp((cx / stride_x).long(), 0, Gw - 1).item())
                gy = int(torch.clamp((cy / stride_y).long(), 0, Gh - 1).item())

                # keep the largest plane if multiple map to same cell
                if area <= best_area[b, gy, gx]:
                    continue
                best_area[b, gy, gx] = area

                cell_x = gx * stride_x
                cell_y = gy * stride_y

                obj_t[b, gy, gx] = 1.0
                pos_mask[b, gy, gx] = True

                tx_t[b, gy, gx] = ((cx - cell_x) / stride_x).clamp(0.0, 1.0)
                ty_t[b, gy, gx] = ((cy - cell_y) / stride_y).clamp(0.0, 1.0)
                tw_t[b, gy, gx] = torch.log((bw / stride_x).clamp(min=1e-6))
                th_t[b, gy, gx] = torch.log((bh / stride_y).clamp(min=1e-6))

        # Objectness loss over all cells (simple & readable; can be imbalanced but works)
        obj_loss = F.binary_cross_entropy_with_logits(obj_logit, obj_t)

        # Box loss only on positive cells
        if pos_mask.any():
            tx_pred = torch.sigmoid(box_raw[:, 0])
            ty_pred = torch.sigmoid(box_raw[:, 1])
            tw_pred = box_raw[:, 2]
            th_pred = box_raw[:, 3]

            box_loss = (
                F.smooth_l1_loss(tx_pred[pos_mask], tx_t[pos_mask]) +
                F.smooth_l1_loss(ty_pred[pos_mask], ty_t[pos_mask]) +
                F.smooth_l1_loss(tw_pred[pos_mask], tw_t[pos_mask]) +
                F.smooth_l1_loss(th_pred[pos_mask], th_t[pos_mask])
            )
        else:
            box_loss = torch.tensor(0.0, device=device)

        return {
            "loss_objectness": obj_loss,
            "loss_box_reg": self.lambda_box * box_loss,
        }
    
    @torch.no_grad()
    def decode(self, pred, H, W, score_thresh=0.05, iou_thresh=0.5, topk=200):
        """
        Convert raw grid preds (B,5,Gh,Gw) into torchvision-style detections.
        Returns: list of dicts with boxes (xyxy pixels), scores, labels.
        """
        device = pred.device
        B, C, Gh, Gw = pred.shape
        assert C == 5, f"Expected 5 channels, got {C}"

        stride_y = H / float(Gh)
        stride_x = W / float(Gw)

        obj_logit = pred[:, 0]          # (B,Gh,Gw)
        tx_raw = pred[:, 1]             # (B,Gh,Gw)
        ty_raw = pred[:, 2]
        tw_raw = pred[:, 3]
        th_raw = pred[:, 4]

        obj = torch.sigmoid(obj_logit)
        tx = torch.sigmoid(tx_raw)
        ty = torch.sigmoid(ty_raw)

        # grid indices
        gy = torch.arange(Gh, device=device).view(1, Gh, 1).expand(B, Gh, Gw)
        gx = torch.arange(Gw, device=device).view(1, 1, Gw).expand(B, Gh, Gw)

        # cell top-left in pixels
        cell_x = gx * stride_x
        cell_y = gy * stride_y

        # center in pixels
        cx = cell_x + tx * stride_x
        cy = cell_y + ty * stride_y

        # size in pixels (inverse of your training encoding)
        bw = torch.exp(tw_raw) * stride_x
        bh = torch.exp(th_raw) * stride_y

        x1 = (cx - bw / 2.0).clamp(0, W - 1)
        y1 = (cy - bh / 2.0).clamp(0, H - 1)
        x2 = (cx + bw / 2.0).clamp(0, W - 1)
        y2 = (cy + bh / 2.0).clamp(0, H - 1)

        detections = []
        for b in range(B):
            scores = obj[b].reshape(-1)  # (Gh*Gw,)
            keep = scores >= score_thresh
            if keep.sum().item() == 0:
                detections.append({
                    "boxes": torch.zeros((0, 4), device=device),
                    "scores": torch.zeros((0,), device=device),
                    "labels": torch.zeros((0,), dtype=torch.long, device=device),
                })
                continue

            boxes = torch.stack([
                x1[b].reshape(-1),
                y1[b].reshape(-1),
                x2[b].reshape(-1),
                y2[b].reshape(-1),
            ], dim=1)

            boxes = boxes[keep]
            scores_k = scores[keep]

            # optional: keep only topk before NMS for speed
            if boxes.shape[0] > topk:
                idx = torch.argsort(scores_k, descending=True)[:topk]
                boxes = boxes[idx]
                scores_k = scores_k[idx]

            # single class => labels all 0
            labels = torch.zeros((boxes.shape[0],), dtype=torch.long, device=device)

            # NMS
            keep_nms = nms(boxes, scores_k, iou_thresh)
            boxes = boxes[keep_nms]
            scores_k = scores_k[keep_nms]
            labels = labels[keep_nms]

            detections.append({"boxes": boxes, "scores": scores_k, "labels": labels})

        return detections



def get_model(num_classes: int = 1, in_channels: int = 3, base: int = 32):
    if num_classes != 1:
        raise ValueError("This SimplePlaneDetector is for ONE class (airplane). Set num_classes=1.")
    return SimplePlaneDetector(in_channels=in_channels, base=base)
