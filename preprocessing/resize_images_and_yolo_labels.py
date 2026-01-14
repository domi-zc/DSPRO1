import os
from pathlib import Path
from PIL import Image, ImageOps

INPUT_DIR = Path(
    "/Users/noahzemljic/Library/CloudStorage/OneDrive-HochschuleLuzern/2025:26/3. Semester/DSPRO1/Datasets/Google Maps/Images with Annotations/images"
)

LABELS_DIR = Path("/Users/noahzemljic/Library/CloudStorage/OneDrive-HochschuleLuzern/2025:26/3. Semester/DSPRO1/Datasets/Google Maps/Images with Annotations/labels/")

OUTPUT_DIR = INPUT_DIR / "resized"
OUT_IMAGES = OUTPUT_DIR / "images"
OUT_LABELS = OUTPUT_DIR / "labels"
TARGET_W, TARGET_H = 640, 640
BACKGROUND_COLOR = (0, 0, 0)  # black padding; change to (255,255,255) for white

EXTS = (".jpg", ".png")

OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_LABELS.mkdir(parents=True, exist_ok=True)

def pad_resize_keep_aspect(img: Image.Image, tw: int, th: int):
    """Return (canvas, scale, dx, dy, new_w, new_h)."""
    # Honor EXIF rotation
    img = ImageOps.exif_transpose(img)

    iw, ih = img.size
    scale = min(tw / iw, th / ih)
    new_w = max(1, int(round(iw * scale)))
    new_h = max(1, int(round(ih * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    has_alpha = resized.mode in ("RGBA", "LA") or (resized.mode == "P" and "transparency" in resized.info)
    canvas_mode = "RGBA" if has_alpha else "RGB"

    canvas = Image.new(canvas_mode, (tw, th), (0, 0, 0, 0) if has_alpha else BACKGROUND_COLOR)
    dx = (tw - new_w) // 2
    dy = (th - new_h) // 2
    canvas.paste(resized, (dx, dy), resized if has_alpha else None)
    return canvas, scale, dx, dy, new_w, new_h, iw, ih

def clamp01(x):  # keep within [0,1]
    return max(0.0, min(1.0, x))

def quad_to_yolo(line, img_w, img_h):
    parts = line.strip().split()
    cls = parts[0]
    # 4 points
    xs = list(map(float, parts[1::2]))
    ys = list(map(float, parts[2::2]))

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    w = xmax - xmin
    h = ymax - ymin
    xc = xmin + w / 2.0
    yc = ymin + h / 2.0

    # normalize
    xc /= img_w
    yc /= img_h
    w  /= img_w
    h  /= img_h

    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

def remap_yolo_line(line, iw, ih, scale, dx, dy, tw, th):
    """
    Supports:
      - YOLO box: class cx cy w h
      - YOLO segmentation: class x1 y1 x2 y2 ... (normalized points)
    Returns remapped line string or None if invalid.
    """
    parts = line.strip().split()
    if not parts:
        return None
    try:
        cls = int(float(parts[0]))
    except:
        return None

    nums = [float(x) for x in parts[1:]]
    # Decide if it's a box or polygon: box has exactly 4 numbers; seg has 6+ (even count)
    if len(nums) == 4:
        # Box: cx, cy, w, h (normalized to original)
        cx, cy, w, h = nums
        # Convert to original pixels
        cx_px = cx * iw
        cy_px = cy * ih
        w_px  = w  * iw
        h_px  = h  * ih
        # Apply scale + padding offset
        cx_new = cx_px * scale + dx
        cy_new = cy_px * scale + dy
        w_new  = w_px  * scale
        h_new  = h_px  * scale
        # Normalize to target size
        cx_n = clamp01(cx_new / tw)
        cy_n = clamp01(cy_new / th)
        w_n  = clamp01(w_new  / tw)
        h_n  = clamp01(h_new  / th)
        # Filter degenerate boxes
        if w_n <= 0 or h_n <= 0:
            return None
        return f"{cls} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}"
    else:
        # Segmentation polygon: x1 y1 x2 y2 ...
        if len(nums) % 2 != 0 or len(nums) < 6:
            # not a valid polygon
            return None
        remapped = [cls]
        for i in range(0, len(nums), 2):
            x_n, y_n = nums[i], nums[i+1]  # normalized to original
            # to original pixels
            x_px = x_n * iw
            y_px = y_n * ih
            # scale + pad
            x_new = x_px * scale + dx
            y_new = y_px * scale + dy
            # normalize to target
            x_out = clamp01(x_new / tw)
            y_out = clamp01(y_new / th)
            remapped.extend([x_out, y_out])
        return " ".join([str(remapped[0])] + [f"{v:.6f}" for v in remapped[1:]])

def process_one(src_path: Path):
    # image
    with Image.open(src_path) as im:
        canvas, scale, dx, dy, new_w, new_h, iw, ih = pad_resize_keep_aspect(im, TARGET_W, TARGET_H)

    # save image (preserve extension)
    out_img = OUT_IMAGES / src_path.name
    # If saving JPEG, ensure RGB
    if out_img.suffix.lower() in (".jpg", ".jpeg", ".png") and canvas.mode not in ("RGB",):
        canvas = canvas.convert("RGB")
    canvas.save(out_img, quality=95, optimize=True) if out_img.suffix.lower() in (".jpg",".jpeg") else canvas.save(out_img, optimize=True)

    # labels
    label_path = LABELS_DIR / (src_path.stem + ".txt")  # change here if your labels are in a separate folder
    out_lbl = OUT_LABELS / (src_path.stem + ".txt")

    if not label_path.exists():
        # no labels; just skip
        return

    new_lines = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            if out_img.suffix.lower() == "jpg":
                mapped = remap_yolo_line(line, iw, ih, scale, dx, dy, TARGET_W, TARGET_H)
            elif out_img.suffix.lower() == ".png":
                mapped = quad_to_yolo(line, iw, ih)


            if mapped:
                new_lines.append(mapped)

    if new_lines:
        with open(out_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
    else:
        # If everything was invalid (rare), write empty file so trainer won’t crash expecting labels
        open(out_lbl, "w").close()

def main():
    files = [p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
    print(f"📂 Input: {INPUT_DIR}")
    print(f"🖼️ Images found: {len(files)}")
    for i, src in enumerate(files, 1):
        try:
            process_one(src)
            print(f"✅ [{i}/{len(files)}] {src.name}")
        except Exception as e:
            print(f"❌ [{i}/{len(files)}] {src.name} — {e}")

    print(f"\n🎯 Done. Images → {OUT_IMAGES}\n📝 Labels → {OUT_LABELS}")

if __name__ == "__main__":
    main()
