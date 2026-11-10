# 接入 X-AnyLabeling v4.0.0-beta.7

支持的接法是整份补丁，不是只加两个按钮。补丁相对标签 `v4.0.0-beta.7` 的 `label_widget.py`、`widgets/__init__.py`、`app_info.py`，并带上分叉实际 import 的新文件。

```bash
git clone --branch v4.0.0-beta.7 --depth 1 https://github.com/CVHub520/X-AnyLabeling.git
cd X-AnyLabeling
patch -p1 --dry-run < /path/to/xal-temporal/integration/x-anylabeling-v4.0.0-beta.7-temporal.patch
patch -p1 < /path/to/xal-temporal/integration/x-anylabeling-v4.0.0-beta.7-temporal.patch
```

`patch -p1 --dry-run` 已在该标签的干净检出上通过。

## 补丁里的行为

功能说明见 [`docs/CUSTOM_VIDEO_FEATURES_CN.md`](../docs/CUSTOM_VIDEO_FEATURES_CN.md)。清单：

1. 同级文件夹导航（名称、复制、上一个/下一个、Alt+↑ / Alt+↓）
2. 主界面 BBX 线宽
3. 所有帧 80% 显示缩放
4. 区间平滑
5. AI 补齐
6. 缺失框继承
7. 后续 N 帧矩形同步
8. 对象勾选作用于整个视频目录（只影响显示）
9. 删除从当前帧传播到结尾，而不是整段视频
10. 画框时空格平移提示
11. 图像分类入口，以及视频分类入口
12. 窗口标题后缀 `[区间平滑测试版]`

区间平滑和 AI 补齐的源文件按分叉原来的路径放进树里：

- `anylabeling/views/labeling/temporal_smoothing.py`
- `anylabeling/views/labeling/temporal_tracking.py`
- `anylabeling/views/labeling/temporal_tracking_masks.py`
- `anylabeling/views/labeling/widgets/temporal_smoothing_dialog.py`
- `anylabeling/views/labeling/widgets/temporal_tracking_dialog.py`
- `tools/temporal_tracking_*.py`

`xal_temporal` 包是同一套平滑和跟踪逻辑的可测试副本（导入已改成包内路径，运行时目录用 `XAL_TEMPORAL_HOME`）。补丁里的副本保持分叉原来的 `tools.` 导入和 `PROJECT_ROOT`，这样和 `label_widget.py` 里的调用一致。

## 视频分类文件从哪来

`widgets/__init__.py` 增加了 `VideoClassifierDialog`。v4.0.0-beta.7 和上传的压缩包里都没有这个模块。补丁附带的是上游 **v4.0.0-beta.8** 的 `video_classifier/` 与 `widgets/video_classifier_dialog.py`，未改内容。没有它们，打过补丁的 `label_widget` 无法导入。

## AI 权重

补丁不包含 `.pt`。AI 补齐仍读取运行目录下的 `runtime/tracking_runtime.json`，示例见 `examples/tracking_runtime.example.json`。

## 只想在较新的 label_widget 上挂两个按钮

`xal_temporal.integration.open_temporal_smoothing` / `open_temporal_tracking` 只打开这两个对话框。它们不包含文件夹导航、80% 缩放、缺失框继承或 N 帧同步。那些行为只在这份完整补丁里。

## English

Apply `x-anylabeling-v4.0.0-beta.7-temporal.patch` with `patch -p1` on a clean `v4.0.0-beta.7` tree. It is the full fork of `label_widget.py`, plus `widgets/__init__.py`, `app_info.py`, the temporal modules at their original paths, and the video-classifier package copied unchanged from upstream `v4.0.0-beta.8` (required by the fork's import, absent from beta.7 and from the upload). The feature list is in `docs/CUSTOM_VIDEO_FEATURES_CN.md`. `xal_temporal` remains the importable, tested copy of smoothing and tracking only.
