# 接入 X-AnyLabeling 4.0.0-beta.7

本仓库不附带上游 `label_widget.py`（约 300 KB）。区间平滑和 AI 补齐的计算在 `xal_temporal` 里，界面在 `xal_temporal.qt`。上游主窗口只需要两个按钮和两次调用。

固定版本：[X-AnyLabeling v4.0.0-beta.7](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v4.0.0-beta.7)。补丁按该标签的 `anylabeling/views/labeling/label_widget.py` 生成。

```bash
git clone --branch v4.0.0-beta.7 --depth 1 https://github.com/CVHub520/X-AnyLabeling.git
cd X-AnyLabeling
patch -p1 < /path/to/xal-temporal/integration/x-anylabeling-v4.0.0-beta.7-temporal.patch
pip install -e /path/to/xal-temporal
pip install "PyQt6>=6.6"
```

补丁做两件事：

1. 在标注说明下方加「区间平滑」和「AI 补齐」按钮。
2. 增加 `open_temporal_smoothing` / `open_temporal_tracking`，转调 `xal_temporal.integration`。

标注路径沿用上游规则：与图片同名的 `.json`，若设置了 `output_dir` 则写到该目录。不依赖后续版本里才有的私有辅助函数。

AI 补齐还需要一份运行配置，放在环境变量 `XAL_TEMPORAL_HOME`（默认当前工作目录）下：

```text
runtime/tracking_runtime.json
```

字段见 `examples/tracking_runtime.example.json`。`python` 是装有 PyTorch 和 SAM 2 的解释器，`checkpoint` 指向你自己下载的 `sam2.1_hiera_small.pt`。本仓库不包含权重。未配置时，区间平滑仍可使用，打开 AI 补齐会提示环境未就绪。

会话、提示点和涂抹缓存在 `$XAL_TEMPORAL_HOME/runtime/`，不写进图片目录。应用到标注时仍在标注目录生成 `.temporal_smoothing_backups` 或 `.temporal_tracking_backups`。

## 这个补丁不包含的本地改动

原先那份完整 `label_widget.py` 里还有文件夹切换、80% 缩放、缺失框继承、后续 N 帧同步等改动。它们和这两项功能绑在同一个 300 KB 文件里，没有单独的可测试模块，因此没有放进本仓库。若要上游那些行为，需要对照 v4.0.0-beta.7 另做补丁。

## English

Pin X-AnyLabeling at tag `v4.0.0-beta.7`, apply `x-anylabeling-v4.0.0-beta.7-temporal.patch`, and install this package with PyQt6. The patch adds two buttons and delegates to `xal_temporal.integration`. SAM 2.1 Small is optional: point `runtime/tracking_runtime.json` at a local checkpoint. Do not commit weights.
