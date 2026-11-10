"""Run interval smoothing on the synthetic sequence. SAM2 is optional and skipped when absent."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.make_synthetic_sequence import render_frame, write_sequence
from xal_temporal.benchmark import (
    hold_previous,
    keyframe_indices,
    library_refine,
    linear_interpolate,
    summarize,
)
from xal_temporal.smoothing import build_preview


def _draw_box(image, box, color, width=2):
    draw = ImageDraw.Draw(image)
    draw.rectangle([float(box[0]), float(box[1]), float(box[2]), float(box[3])], outline=color, width=width)
    return image


def main():
    out = Path(__file__).resolve().parent / "synthetic"
    if out.exists():
        shutil.rmtree(out)
    write_sequence(out)
    gt = np.asarray(json.loads((out / "ground_truth_xyxy.json").read_text())["boxes"], dtype=float)
    images = sorted(out.glob("*.jpg"))
    boxes = []
    for image in images:
        doc = json.loads(image.with_suffix(".json").read_text(encoding="utf-8"))
        pts = np.asarray(doc["shapes"][0]["points"], dtype=float)
        boxes.append([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
    boxes = np.asarray(boxes, dtype=float)
    keys = keyframe_indices(len(boxes), 10)
    size = (320, 180)
    limit = float(max(size))
    interp = linear_interpolate(gt, keys)
    # Product path: existing (jittered) boxes, human keyframes locked.
    smoothed = library_refine(boxes, keys, size, method="smooth", limit=6.0, strength=4.0)
    seeded = hold_previous(gt, keys)
    filled = library_refine(seeded, keys, size, method="interpolate", limit=limit, strength=4.0)
    print("synthetic non-keyframe metrics vs rendered ground truth")
    for name, pred in (
        ("jittered annotations", boxes),
        ("linear interpolation of keyframes", interp),
        ("interval smoothing (limit 6px on jittered boxes)", smoothed),
        ("library interpolate from hold-last seeds", filled),
    ):
        stats = summarize(pred, gt, keys)["non_keyframes"]
        print(f"  {name}: IoU {stats['mean_iou']:.4f}  center {stats['mean_center_error_px']:.3f}px  "
              f"IoU>0.5 {stats['pct_iou_over_0_5']:.1f}%  n={stats['frames']}")

    paths = [p.with_suffix(".json") for p in images]
    rows = build_preview([str(p) for p in images], [str(p) for p in paths], ("drone", 1), set(keys),
                         strength=4.0, limit=6.0, mode="both", method="smooth")
    preview_boxes = np.asarray([r["new_box"] for r in rows])
    # Preview must match the library call on the same inputs.
    if np.max(np.abs(preview_boxes - smoothed)) > 1e-6:
        raise SystemExit("build_preview disagreed with refine_boxes")

    figures = ROOT / "docs" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in (0, 15, 29, 45, 59):
        base = render_frame(gt[i]).convert("RGB")
        before = _draw_box(base.copy(), boxes[i], (239, 68, 68))
        after = _draw_box(base.copy(), smoothed[i], (37, 198, 194))
        pair = Image.new("RGB", (before.width * 2 + 8, before.height), (0, 0, 0))
        pair.paste(before, (0, 0))
        pair.paste(after, (before.width + 8, 0))
        frames.append(pair)
    gif_path = figures / "synthetic_before_after.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=600, loop=0)
    png_path = figures / "synthetic_before_after.png"
    frames[2].save(png_path)
    print(f"wrote {png_path}")
    print(f"wrote {gif_path}")

    # AI fill is a separate optional process. Do not pretend it ran.
    try:
        import sam2  # noqa: F401
        import torch  # noqa: F401
        checkpoint = Path.cwd() / "runtime" / "tracking_runtime.json"
        if checkpoint.is_file():
            print("SAM2 imports succeeded and runtime/tracking_runtime.json exists.")
            print("Launch AI fill from the X-AnyLabeling dialog; this demo does not download weights.")
        else:
            print("SAM2 is importable, but runtime/tracking_runtime.json is missing. AI fill skipped.")
    except Exception as error:
        print(f"AI fill skipped (SAM2/torch unavailable: {error}). Interval smoothing does not need it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
