"""Hooks that X-AnyLabeling's label widget calls. No change to annotation files until the dialog applies a preview."""
import os.path as osp
from pathlib import Path


def label_path_for_image(widget, image_file):
    """Same rule as upstream ``get_label_file``: sibling ``.json``, or ``output_dir``."""
    label_file = osp.splitext(str(image_file))[0] + ".json"
    output_dir = getattr(widget, "output_dir", None)
    if output_dir:
        label_file = osp.join(output_dir, osp.basename(label_file))
    return label_file


def _sequence_images(widget, suffixes):
    root = Path(widget.image_path or widget.filename).resolve().parent
    images = []
    for raw in widget.image_list:
        path = Path(raw).resolve()
        if path.parent != root:
            continue
        if path.suffix.lower() not in suffixes:
            continue
        images.append(str(path))
    return images


def _identities(widget):
    return sorted(
        {
            (shape.label, shape.group_id)
            for shape in widget.canvas.shapes
            if shape.shape_type == "rectangle" and shape.group_id is not None
        },
        key=lambda identity: (str(identity[1]), identity[0]),
    )


def _selected_identity(widget):
    selected = widget.canvas.selected_shapes
    if len(selected) == 1:
        return (selected[0].label, selected[0].group_id)
    return None


def open_temporal_smoothing(widget):
    """Open interval smoothing for the current image folder."""
    from PyQt6.QtWidgets import QMessageBox

    from xal_temporal.qt.smoothing_dialog import TemporalSmoothingDialog

    if not getattr(widget, "filename", None):
        widget.status("请先打开一个带矩形标注的序列。")
        return
    if getattr(widget, "dirty", False):
        QMessageBox.information(widget, "请先保存当前帧", "请保存当前人工修改，再打开区间平滑。")
        return
    images = _sequence_images(widget, {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if len(images) < 3:
        widget.status("当前文件夹至少需要三张连续视频帧。")
        return
    identities = _identities(widget)
    if not identities:
        widget.status("当前帧没有带组 ID 的矩形目标，请先设置目标 ID。")
        return
    current = str(Path(widget.image_path or widget.filename).resolve())
    index = images.index(current) if current in images else 0
    dialog = TemporalSmoothingDialog(
        widget,
        images,
        [label_path_for_image(widget, p) for p in images],
        identities,
        index,
        _selected_identity(widget),
    )
    dialog.exec()
    if dialog.applied:
        widget.load_file(widget.filename)
        widget.status("区间平滑已更新；可在区间平滑窗口撤销批量修改。")


def open_temporal_tracking(widget):
    """Open AI gap filling for the current image folder."""
    from PyQt6.QtWidgets import QMessageBox

    from xal_temporal.qt.tracking_dialog import TemporalTrackingDialog

    if not getattr(widget, "filename", None):
        widget.status("请先打开视频帧序列，画一个带组 ID 的矩形框。")
        return
    if getattr(widget, "dirty", False):
        QMessageBox.information(widget, "请先保存当前帧", "请保存当前人工修改，再打开 AI 补齐。")
        return
    images = _sequence_images(widget, {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    identities = _identities(widget)
    if not images or not identities:
        widget.status("请在当前帧画一个矩形，设置组 ID 并保存，再打开 AI 补齐。")
        return
    current = str(Path(widget.image_path or widget.filename).resolve())
    index = images.index(current) if current in images else 0
    try:
        dialog = TemporalTrackingDialog(
            widget,
            images,
            [label_path_for_image(widget, p) for p in images],
            identities,
            index,
            _selected_identity(widget),
        )
        dialog.exec()
        if dialog.applied:
            widget.load_file(widget.filename)
            widget.status("AI 补齐已更新；可在 AI 补齐窗口撤销批量修改。")
    except Exception as error:
        QMessageBox.warning(widget, "AI 补齐未打开", str(error))
