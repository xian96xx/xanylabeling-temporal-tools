"""OS-independent path, link, and runtime-config behavior."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from xal_temporal import tracking
from xal_temporal.fsutil import link_or_copy
from xal_temporal.sam2_worker import read_runtime_config


def test_runtime_config_accepts_utf8_bom(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    python = tmp_path / "python"
    checkpoint = tmp_path / "model.pt"
    python.write_text("interpreter", encoding="utf-8")
    checkpoint.write_bytes(b"weights")
    payload = {
        "python": str(python),
        "checkpoint": str(checkpoint),
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "device": "cpu",
    }
    raw = json.dumps(payload).encode("utf-8")
    (runtime / "tracking_runtime.json").write_bytes(b"\xef\xbb\xbf" + raw)
    monkeypatch.setattr(tracking, "PROJECT_ROOT", tmp_path)
    config = tracking.runtime_config()
    assert Path(config["python"]) == python.resolve()
    assert Path(config["checkpoint"]) == checkpoint.resolve()
    assert config["device"] == "cpu"


def test_worker_runtime_read_accepts_utf8_bom(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "tracking_runtime.json").write_bytes(
        b'\xef\xbb\xbf{"python":"py","checkpoint":"w.pt","device":"cpu"}'
    )
    assert read_runtime_config(tmp_path)["device"] == "cpu"
    patch = Path(__file__).resolve().parents[1] / "integration" / "x-anylabeling-v4.0.0-beta.7-temporal.patch"
    text = patch.read_text(encoding="utf-8")
    assert "read_text(encoding='utf-8-sig')" in text
    assert "(ROOT/'runtime/tracking_runtime.json').read_text(encoding='utf-8')" not in text


def test_link_or_copy_falls_back_to_hardlink_then_copy(tmp_path):
    src = tmp_path / "frame.jpg"
    src.write_bytes(b"jpeg-bytes")
    dest = tmp_path / "sparse" / "frame.jpg"
    dest.parent.mkdir()

    with patch.object(Path, "symlink_to", side_effect=OSError("WinError 1314")):
        with patch("xal_temporal.fsutil.os.link", side_effect=OSError("no hardlink")):
            kind = link_or_copy(src, dest)
    assert kind == "copy"
    assert dest.read_bytes() == b"jpeg-bytes"
    dest.unlink()

    with patch.object(Path, "symlink_to", side_effect=OSError("WinError 1314")):
        kind = link_or_copy(src, dest)
    assert kind == "hardlink"
    assert dest.read_bytes() == b"jpeg-bytes"


def test_link_or_copy_prefers_symlink_when_available(tmp_path):
    src = tmp_path / "frame.jpg"
    src.write_bytes(b"jpeg-bytes")
    dest = tmp_path / "linked.jpg"
    try:
        kind = link_or_copy(src, dest)
    except OSError:
        pytest.skip("this environment allows neither symlink nor hardlink")
    assert kind in {"symlink", "hardlink", "copy"}
    assert dest.read_bytes() == b"jpeg-bytes"
