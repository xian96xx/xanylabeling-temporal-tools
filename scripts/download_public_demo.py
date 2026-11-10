#!/usr/bin/env python3
"""Download 1–2 short DAVIS 2017 sequences and score box filling against the masks.

The dataset is not committed. Files land in ``data/davis2017/`` (gitignored).

Why DAVIS, not VisDrone / UAV123 / Anti-UAV / LaSOT / GOT-10k:
those releases are research-only or non-commercial (or, for UAV123, have no
clear open license and mix third-party footage). DAVIS 2017 TrainVal is
published under the BSD license by the DAVIS organizers; the 2017 challenge
page also describes the new annotations as CC BY 4.0. Both allow this demo.
SeaDronesSee is CC0 and UAV-related, but the official image archive is one
30 GB zip without working HTTP range requests, so it is not the default.

Boxes are the axis-aligned extent of the official palette masks (pixel
boundaries: xmin, ymin, xmax+1, ymax+1). One object id is chosen per
sequence: the id that occupies the most frames, ties broken by total area.
Default sequences are two official validation clips: ``car-shadow`` (40
frames) and ``motocross-jump`` (40 frames). The TrainVal clip named ``drone``
is not used: it is a multi-object trade-show shot, and the largest mask is a
person rather than a UAV.

SAM2 is not downloaded. If ``runtime/tracking_runtime.json`` and the SAM2
packages are missing, AI fill is reported as skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import urllib.request
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xal_temporal.benchmark import (
    hold_previous,
    keyframe_indices,
    library_refine,
    linear_interpolate,
    summarize,
)

ZIP_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
DEFAULT_SEQUENCES = ("car-shadow", "motocross-jump")
# Official 2017 validation split (ImageSets/2017/val.txt).
SPLIT = {"car-shadow": "val", "motocross-jump": "val"}


def _request(url, range_header=None):
    headers = {"User-Agent": "xal-temporal-public-demo"}
    if range_header:
        headers["Range"] = range_header
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=120)


def _content_length(url) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "xal-temporal-public-demo"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return int(response.headers["Content-Length"])


def _read_range(url, start, end) -> bytes:
    # end is inclusive, matching HTTP Range.
    with _request(url, f"bytes={start}-{end}") as response:
        data = response.read()
    return data


def _central_directory(url):
    size = _content_length(url)
    tail = _read_range(url, max(0, size - 6_000_000), size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("ZIP 中央目录未找到。")
    cd_size, cd_off = struct.unpack_from("<II", tail, eocd + 12)
    tail_start = size - len(tail)
    rel = cd_off - tail_start
    blob = tail[rel : rel + cd_size]
    entries = []
    pos = 0
    while pos + 46 <= len(blob):
        if blob[pos : pos + 4] != b"PK\x01\x02":
            break
        method, = struct.unpack_from("<H", blob, pos + 10)
        csize, usize = struct.unpack_from("<II", blob, pos + 20)
        nlen, elen, clen = struct.unpack_from("<HHH", blob, pos + 28)
        local_off, = struct.unpack_from("<I", blob, pos + 42)
        name = blob[pos + 46 : pos + 46 + nlen].decode("utf-8")
        entries.append({"name": name, "method": method, "csize": csize, "usize": usize, "offset": local_off})
        pos += 46 + nlen + elen + clen
    return entries


def _decompress(method, payload, usize):
    if method == 0:
        return payload
    if method == 8:
        return zlib.decompress(payload, -15)
    raise RuntimeError(f"不支持的 zip 压缩方法 {method}")


def _extract_span(url, entries):
    """Download one contiguous local-header span and return {name: bytes}."""
    if not entries:
        return {}
    entries = sorted(entries, key=lambda item: item["offset"])
    start = entries[0]["offset"]
    # Local header + name + extra is well under 512 bytes for these paths.
    end = entries[-1]["offset"] + 30 + 512 + entries[-1]["csize"]
    blob = _read_range(url, start, end)
    found = {}
    for entry in entries:
        rel = entry["offset"] - start
        if blob[rel : rel + 4] != b"PK\x03\x04":
            raise RuntimeError(f"本地头无效: {entry['name']}")
        nlen, elen = struct.unpack_from("<HH", blob, rel + 26)
        method, = struct.unpack_from("<H", blob, rel + 8)
        data_off = rel + 30 + nlen + elen
        payload = blob[data_off : data_off + entry["csize"]]
        if len(payload) != entry["csize"]:
            raise RuntimeError(f"下载截断: {entry['name']}")
        found[entry["name"]] = _decompress(method, payload, entry["usize"])
    return found


def mask_box(mask, object_id):
    ys, xs = np.where(mask == object_id)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def choose_object(masks):
    scores = {}
    for mask in masks:
        for object_id in np.unique(mask):
            if int(object_id) == 0:
                continue
            scores.setdefault(int(object_id), [0, 0])
            area = int((mask == object_id).sum())
            if area:
                scores[int(object_id)][0] += 1
                scores[int(object_id)][1] += area
    if not scores:
        raise RuntimeError("掩码里没有前景。")
    return max(scores, key=lambda key: (scores[key][0], scores[key][1]))


def load_sequence(url, catalog, sequence, dest: Path):
    images = [e for e in catalog if f"/JPEGImages/480p/{sequence}/" in e["name"] and e["name"].endswith(".jpg")]
    masks = [e for e in catalog if f"/Annotations/480p/{sequence}/" in e["name"] and e["name"].endswith(".png")]
    images.sort(key=lambda e: e["name"])
    masks.sort(key=lambda e: e["name"])
    if len(images) < 3 or len(images) != len(masks):
        raise RuntimeError(f"{sequence} 帧数异常: jpg {len(images)} png {len(masks)}")
    if [Path(e["name"]).stem for e in images] != [Path(e["name"]).stem for e in masks]:
        raise RuntimeError(f"{sequence} 的图像和掩码帧号不一致。")
    raw_images = _extract_span(url, images)
    raw_masks = _extract_span(url, masks)
    seq_dir = dest / sequence
    seq_dir.mkdir(parents=True, exist_ok=True)
    mask_arrays = []
    image_size = None
    for image_entry, mask_entry in zip(images, masks):
        path = seq_dir / Path(image_entry["name"]).name
        path.write_bytes(raw_images[image_entry["name"]])
        with Image.open(path) as im:
            image_size = im.size
        mask = np.array(Image.open(__import__("io").BytesIO(raw_masks[mask_entry["name"]])))
        mask_arrays.append(mask)
        path.with_suffix(".png").write_bytes(raw_masks[mask_entry["name"]])
    object_id = choose_object(mask_arrays)
    boxes = []
    for index, mask in enumerate(mask_arrays):
        box = mask_box(mask, object_id)
        if box is None:
            raise RuntimeError(f"{sequence} 第 {index} 帧没有对象 {object_id}，无法当作连续单目标。")
        boxes.append(box)
    meta = {
        "sequence": sequence,
        "split": SPLIT.get(sequence, "unknown"),
        "frames": len(boxes),
        "image_size": list(image_size),
        "object_id": object_id,
        "boxes_xyxy": boxes,
    }
    (seq_dir / "ground_truth.json").write_text(json.dumps(meta), encoding="utf-8")
    _write_xlabel(seq_dir, meta)
    return meta


def link_or_copy(src: Path, dest: Path) -> str:
    """Point dest at src. Symlink, then hardlink, then a full copy.

    Windows without Developer Mode raises WinError 1314 on symlink.
    """
    src, dest = Path(src), Path(dest)
    try:
        dest.symlink_to(src)
        return "symlink"
    except OSError:
        pass
    try:
        os.link(src, dest)
        return "hardlink"
    except OSError:
        shutil.copyfile(src, dest)
        return "copy"


def _write_xlabel(seq_dir: Path, meta):
    """Full GT JSON plus a sparse copy that keeps only every 10th frame."""
    width, height = meta["image_size"]
    sparse = seq_dir / "sparse_keyframes"
    sparse.mkdir(exist_ok=True)
    keys = set(keyframe_indices(meta["frames"], 10))
    for index, box in enumerate(meta["boxes_xyxy"]):
        name = f"{index:05d}.jpg"
        shape = {
            "label": "object",
            "group_id": 1,
            "shape_type": "rectangle",
            "points": [
                [box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]],
            ],
            "flags": {},
        }
        doc = {
            "version": "4.0.0-beta.7",
            "flags": {},
            "shapes": [shape],
            "imagePath": name,
            "imageData": None,
            "imageWidth": width,
            "imageHeight": height,
        }
        (seq_dir / f"{index:05d}.json").write_text(json.dumps(doc), encoding="utf-8")
        if index in keys:
            # Sparse folder references the same frames; only keyframe JSON is written.
            target = sparse / name
            if not target.exists():
                link_or_copy(seq_dir / name, target)
            (sparse / f"{index:05d}.json").write_text(json.dumps(doc), encoding="utf-8")


def sam2_status() -> str:
    config = Path.cwd() / "runtime" / "tracking_runtime.json"
    try:
        import sam2  # noqa: F401
        import torch  # noqa: F401
    except Exception as error:
        return f"skipped: SAM2/torch not importable ({error.__class__.__name__})"
    if not config.is_file():
        return "skipped: runtime/tracking_runtime.json not present (weights are not downloaded by this script)"
    return "available but not run: this script does not launch a multi-minute GPU/CPU propagation"


def _clip_box(box, size):
    width, height = size
    out = np.array(box, dtype=float)
    out[0::2] = np.clip(out[0::2], 0, width - 1)
    out[1::2] = np.clip(out[1::2], 0, height - 1)
    if out[2] <= out[0] + 1:
        out[2] = min(width, out[0] + 1)
    if out[3] <= out[1] + 1:
        out[3] = min(height, out[1] + 1)
    return out


def _try_method(name, fn):
    try:
        pred = fn()
    except Exception as error:
        return {"error": f"{error.__class__.__name__}: {error}"}
    return pred


def evaluate_sequence(meta) -> dict:
    gt = np.asarray(meta["boxes_xyxy"], dtype=float)
    keys = keyframe_indices(len(gt), 10)
    size = tuple(meta["image_size"])
    limit = float(max(size))
    seed = hold_previous(gt, keys)
    rng = np.random.default_rng(0)
    noisy = gt.copy()
    for index in range(len(gt)):
        if index in keys:
            continue
        noisy[index] = _clip_box(gt[index] + rng.normal(0, 3.0, size=4), size)
    specs = {
        "linear_interpolation": lambda: linear_interpolate(gt, keys),
        "library_interpolate_on_hold_last": lambda: library_refine(
            seed, keys, size, method="interpolate", limit=limit
        ),
        "library_smooth_on_hold_last": lambda: library_refine(
            seed, keys, size, method="smooth", limit=limit, strength=4.0
        ),
        "jittered_boxes_sigma3": lambda: noisy,
        "library_smooth_on_jitter_limit8": lambda: library_refine(
            noisy, keys, size, method="smooth", limit=8.0, strength=4.0
        ),
    }
    rows = {}
    for name, fn in specs.items():
        pred = _try_method(name, fn)
        if isinstance(pred, dict) and "error" in pred:
            rows[name] = pred
            continue
        stats = summarize(pred, gt, keys)
        rows[name] = stats["non_keyframes"]
        rows[name]["all_frames_mean_iou"] = stats["all_frames"]["mean_iou"]
    return {
        "sequence": meta["sequence"],
        "split": meta["split"],
        "frames": meta["frames"],
        "image_size": meta["image_size"],
        "object_id": meta["object_id"],
        "keyframe_step": 10,
        "keyframes": keys,
        "non_keyframe_metrics": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=ROOT / "data" / "davis2017")
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    args = parser.parse_args()
    print("reading DAVIS 2017 TrainVal 480p catalog (range request, not the whole zip)")
    catalog = _central_directory(ZIP_URL)
    results = []
    for sequence in args.sequences:
        print(f"downloading {sequence}")
        meta = load_sequence(ZIP_URL, catalog, sequence, args.dest)
        print(f"  frames={meta['frames']} object_id={meta['object_id']} size={meta['image_size']}")
        results.append(evaluate_sequence(meta))
    payload = {
        "dataset": "DAVIS 2017 TrainVal 480p",
        "url": ZIP_URL,
        "license": "BSD (DAVIS organizers). The 2017 challenge page additionally describes new annotations as CC BY 4.0.",
        "citation": "Pont-Tuset et al., The 2017 DAVIS Challenge on Video Object Segmentation, arXiv:1704.00675. Perazzi et al., A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation, CVPR 2016.",
        "box_definition": "axis-aligned mask extent, xyxy with exclusive max (xmin, ymin, xmax+1, ymax+1)",
        "protocol": "Keyframes are ground-truth boxes every 10 frames plus the last frame. Metrics below are non-keyframe frames only. linear_interpolation independently interpolates the four edges. library_*_on_hold_last copies each keyframe forward to seed missing boxes, then calls xal_temporal.smoothing.refine_boxes with limit equal to the longer image side (strength 4 for smooth). library_smooth_on_jitter_limit8 adds N(0, 3px) noise to non-keyframes, keeps keyframes exact, and smooths with the product's per-edge limit of 8 px. That second setup is de-jitter, not gap filling.",
        "sam2": sam2_status(),
        "sequences": results,
    }
    out = args.dest / "benchmark_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
