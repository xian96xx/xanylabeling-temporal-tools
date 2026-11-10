# 视频标注增强（相对 X-AnyLabeling v4.0.0-beta.7）

这些行为都在 `integration/x-anylabeling-v4.0.0-beta.7-temporal.patch` 里。在 v4.0.0-beta.7 的源码根目录执行 `patch -p1` 即可。窗口标题改为 `X-AnyLabeling [区间平滑测试版]`（`anylabeling/app_info.py`）。

## 1. 同级文件夹切换

- 顶部显示当前文件夹名，可复制。复制的是文件夹名，不是完整路径。
- 「▲上个」「▼下个」，快捷键 Alt+↑ / Alt+↓。
- 只在同一父目录的非隐藏子文件夹之间切换，按自然顺序排序。

## 2. 主界面 BBX 线宽

- 「BBX线宽」数值一改就应用到当前画布上的框，并写回配置。
- 范围 0.1–20。不改图片，也不改坐标。

## 3. 所有帧 80% 显示

- 只改当前序列的显示缩放，并记到每一帧的手动缩放里，避免切帧时回到 Fit Window。
- 不缩放原图，不改标注坐标。

## 4. 区间平滑

- 顶部「区间平滑」。用已有矩形和人工锁定帧做有界去抖或插值。
- 先预览。应用时在标注目录写 `.temporal_smoothing_backups`，可整段撤销。
- 不看像素。细节见仓库 README。

## 5. AI 补齐

- 顶部「AI 补齐」。人工关键帧经 SAM 2.1 传播到中间帧。
- 先预览再写入，备份在 `.temporal_tracking_backups`。
- 需要本机的 SAM 2.1 Small 权重和 `runtime/tracking_runtime.json`。补丁不包含权重。

## 6. 缺失框继承

- 「缺失框继承」默认关闭。打开后，若当前帧缺少上一帧里的某个已跟踪目标，则继承上一帧的矩形并保存。
- 开关写入配置项 `inherit_missing_from_prev_frame`。

## 7. 后续 N 帧同步

- 对象列表标题栏「后续N帧」。`N=0` 关闭。
- `N>0` 时，移动或缩放当前帧矩形后，把左、上、右、下的变化量同步到后面 N 帧。
- 匹配顺序：`target_id`、`global_track_id`、`category_track_id`、`track_id`，否则 `label + group_id`。
- 后续帧没有 JSON 或没有对应对象时跳过。同一次回调不会重复叠加。

## 8. 勾选状态作用于整个视频

- 勾选或取消某个对象时，同一视频目录里该对象在所有帧显示或隐藏。
- 只影响界面，不删除 JSON。状态按视频目录保存在本次运行中。
- 标签过滤和组 ID 过滤走同一套可见性规则。列表文字显示 `标签 (组 ID)`。

## 9. 删除只传播到视频结尾

- 删除一个带跟踪 ID 或组 ID 的矩形：当前帧和之后的对应框删除，之前的帧保留。
- 没有跟踪 ID 也没有组 ID 的对象只删当前帧。
- 当前帧若保存失败，后面的帧不动。

## 10. 画框时的空格平移提示

- 正在画且滚动条可滚动时，状态栏提示按住空格拖动平移。

## 11. 图像分类与视频分类入口

- 「图像分类」沿用原来的分类对话框，并在打开前隐藏主窗口。原来的 `open_classifier` 转到这个入口。
- 「视频分类」打开 `VideoClassifierDialog`。关闭主窗口时会先关这个对话框。
- v4.0.0-beta.7 没有视频分类模块。上传的源码里也没有这些文件，但 `widgets/__init__.py` 和 `label_widget.py` 会导入它们。补丁因此带上 **v4.0.0-beta.8** 的 `anylabeling/views/labeling/video_classifier/` 和 `widgets/video_classifier_dialog.py`，内容未改。

## 不在补丁里的说明

原先笔记里的 Windows JSON 句柄修复和 `repair_json_annotations.py` 没有出现在这次上传的文件里，所以没有放进补丁。不要对某个私有目录运行任何修复脚本；本仓库不包含这类脚本。
