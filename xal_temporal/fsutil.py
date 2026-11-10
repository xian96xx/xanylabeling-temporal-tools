"""Small filesystem helpers with Windows fallbacks."""
import os
import shutil
from pathlib import Path


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
