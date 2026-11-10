"""Render a short synthetic sequence and X-AnyLabeling JSON. No real imagery."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def ground_truth_box(index: int, n_frames: int = 60) -> np.ndarray:
    """Smooth drone-like motion: translation plus a small scale change."""
    t = index / max(1, n_frames - 1)
    cx = 48 + 210 * t + 12 * np.sin(2 * np.pi * t)
    cy = 90 + 28 * np.sin(2 * np.pi * t) - 10 * t
    w = 46 + 10 * np.sin(np.pi * t)
    h = 22 + 4 * np.cos(np.pi * t)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=float)


def render_frame(box, size=(320, 180)) -> Image.Image:
    image = Image.new("RGB", size, (16, 20, 28))
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    # Thin arms and a body. The box is the axis-aligned extent of this shape.
    arm = max(2, int(round((y2 - y1) * 0.18)))
    draw.line([(x1, cy), (x2, cy)], fill=(210, 214, 220), width=arm)
    draw.line([(cx, y1), (cx, y2)], fill=(186, 190, 198), width=arm)
    body = [cx - (x2 - x1) * 0.16, cy - (y2 - y1) * 0.28, cx + (x2 - x1) * 0.16, cy + (y2 - y1) * 0.28]
    draw.ellipse(body, fill=(240, 244, 248))
    return image


def jitter_box(box, rng, scale=2.5) -> np.ndarray:
    noise = rng.normal(0, scale, size=4)
    out = np.asarray(box, dtype=float) + noise
    # Keep a valid rectangle inside a loose pad; the writer clips to the image.
    if out[2] <= out[0] + 4:
        out[2] = out[0] + 4
    if out[3] <= out[1] + 4:
        out[3] = out[1] + 4
    return out


def write_sequence(root: Path, n_frames: int = 60, seed: int = 0) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    width, height = 320, 180
    clean = []
    annotated = []
    for i in range(n_frames):
        box = ground_truth_box(i, n_frames)
        box[0::2] = np.clip(box[0::2], 0, width - 1)
        box[1::2] = np.clip(box[1::2], 0, height - 1)
        clean.append(box)
        image_path = root / f"{i:05d}.jpg"
        render_frame(box, (width, height)).save(image_path, quality=90)
        # Keyframes stay exact. Other frames carry annotation jitter.
        if i % 10 == 0 or i == n_frames - 1:
            used = box.copy()
        else:
            used = jitter_box(box, rng)
            used[0::2] = np.clip(used[0::2], 0, width - 1)
            used[1::2] = np.clip(used[1::2], 0, height - 1)
        annotated.append(used)
        doc = {
            "version": "4.0.0-beta.7",
            "flags": {},
            "shapes": [
                {
                    "label": "drone",
                    "group_id": 1,
                    "shape_type": "rectangle",
                    "points": [
                        [float(used[0]), float(used[1])],
                        [float(used[2]), float(used[1])],
                        [float(used[2]), float(used[3])],
                        [float(used[0]), float(used[3])],
                    ],
                    "flags": {},
                    "attributes": {},
                }
            ],
            "imagePath": image_path.name,
            "imageData": None,
            "imageWidth": width,
            "imageHeight": height,
        }
        image_path.with_suffix(".json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    gt_path = root / "ground_truth_xyxy.json"
    gt_path.write_text(json.dumps({"image_size": [width, height], "boxes": np.asarray(clean).tolist()}), encoding="utf-8")
    return {"root": str(root), "frames": n_frames, "ground_truth": str(gt_path)}


if __name__ == "__main__":
    here = Path(__file__).resolve().parent / "synthetic"
    print(write_sequence(here))
