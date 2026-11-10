"""Metrics for sparse-keyframe box filling. No dataset and no GUI."""
from __future__ import annotations

import numpy as np

from .smoothing import refine_boxes


def pairwise_iou(pred, gt) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    wh = np.maximum(0, np.minimum(pred[:, 2:], gt[:, 2:]) - np.maximum(pred[:, :2], gt[:, :2]))
    inter = wh[:, 0] * wh[:, 1]
    area_p = np.maximum(0, pred[:, 2] - pred[:, 0]) * np.maximum(0, pred[:, 3] - pred[:, 1])
    area_g = np.maximum(0, gt[:, 2] - gt[:, 0]) * np.maximum(0, gt[:, 3] - gt[:, 1])
    union = area_p + area_g - inter
    out = np.zeros(len(pred), dtype=float)
    ok = union > 0
    out[ok] = inter[ok] / union[ok]
    return out


def center_error(pred, gt) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    pc = (pred[:, :2] + pred[:, 2:]) / 2
    gc = (gt[:, :2] + gt[:, 2:]) / 2
    return np.linalg.norm(pc - gc, axis=1)


def keyframe_indices(n_frames: int, step: int = 10) -> list[int]:
    if n_frames < 3:
        raise ValueError("至少三帧才能比较区间平滑。")
    keys = list(range(0, n_frames, step))
    if keys[-1] != n_frames - 1:
        keys.append(n_frames - 1)
    return keys


def linear_interpolate(gt, keys) -> np.ndarray:
    """Independent linear interpolation of the four edges. Keyframes stay exact."""
    gt = np.asarray(gt, dtype=float)
    keys = np.asarray(sorted(set(keys)), dtype=int)
    out = np.empty_like(gt)
    index = np.arange(len(gt))
    for edge in range(4):
        out[:, edge] = np.interp(index, keys, gt[keys, edge])
    return out


def hold_previous(gt, keys) -> np.ndarray:
    """Copy each keyframe forward until the next one. Endpoints match ``gt``."""
    gt = np.asarray(gt, dtype=float)
    seed = gt.copy()
    keys = sorted(set(keys))
    for start, stop in zip(keys, keys[1:]):
        seed[start + 1 : stop] = gt[start]
    return seed


def library_refine(seed, keys, image_size, method: str, limit: float, strength: float = 4.0):
    out, _, _, _ = refine_boxes(
        np.asarray(seed, dtype=float),
        set(keys),
        image_size,
        strength=strength,
        limit=limit,
        mode="both",
        method=method,
    )
    return out


def summarize(pred, gt, keys) -> dict:
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    ious = pairwise_iou(pred, gt)
    centers = center_error(pred, gt)
    mask = np.ones(len(gt), dtype=bool)
    mask[list(keys)] = False
    if not mask.any():
        raise ValueError("没有非关键帧可评估。")

    def pack(sel):
        return {
            "frames": int(sel.sum()),
            "mean_iou": float(ious[sel].mean()),
            "mean_center_error_px": float(centers[sel].mean()),
            "pct_iou_over_0_5": float((ious[sel] > 0.5).mean() * 100.0),
        }

    return {"non_keyframes": pack(mask), "all_frames": pack(np.ones(len(gt), dtype=bool))}
