import numpy as np

from xal_temporal.benchmark import (
    center_error,
    hold_previous,
    keyframe_indices,
    linear_interpolate,
    pairwise_iou,
    summarize,
)
from xal_temporal.smoothing import refine_boxes


def test_iou_and_center_of_identical_boxes():
    boxes = np.array([[0, 0, 10, 10], [2, 2, 6, 8]], float)
    np.testing.assert_allclose(pairwise_iou(boxes, boxes), [1, 1])
    np.testing.assert_allclose(center_error(boxes, boxes), [0, 0])


def test_linear_interpolation_locks_keyframes_and_matches_edges():
    gt = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [9, 3, 19, 13], [9, 3, 19, 13]], float)
    keys = keyframe_indices(len(gt), step=3)
    assert keys == [0, 3]
    out = linear_interpolate(gt, keys)
    np.testing.assert_allclose(out[[0, -1]], gt[[0, -1]])
    np.testing.assert_allclose(out[1], [3, 1, 13, 11])
    stats = summarize(out, out, keys)
    assert stats["non_keyframes"]["mean_iou"] == 1


def test_hold_previous_and_library_interpolate_stay_finite():
    gt = np.stack([np.array([10 + i, 20, 40 + i, 50]) for i in range(12)], axis=0).astype(float)
    keys = keyframe_indices(len(gt), 10)
    seed = hold_previous(gt, keys)
    assert np.allclose(seed[1], gt[0])
    out = refine_boxes(seed, set(keys), (200, 100), limit=200, method="interpolate")[0]
    np.testing.assert_allclose(out[keys], gt[keys])
    assert np.all(out[:, 2:] > out[:, :2])
