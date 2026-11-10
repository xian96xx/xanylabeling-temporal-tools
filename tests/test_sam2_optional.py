import pytest


def test_sam2_worker_imports_when_installed():
    torch = pytest.importorskip("torch")
    pytest.importorskip("sam2")
    assert torch is not None
    import xal_temporal.sam2_inspect  # noqa: F401
    import xal_temporal.sam2_refine  # noqa: F401


def test_public_dataset_benchmark_is_opt_in():
    """CI does not download DAVIS. Run scripts/download_public_demo.py locally."""
    pytest.skip("opt-in public download: python scripts/download_public_demo.py")
