"""Resolve the foldr/fitr/batchr console-script binaries.

Shared by worker.py (foldr, fitr) and run.py (batchr). Neither fitr nor
batchr ships a `__main__.py` (only foldr does), so `python -m fitr ...`
fails with ModuleNotFoundError — we resolve every tool binary the same
way instead of relying on `-m`.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


class ToolNotFoundError(RuntimeError):
    """Raised when foldr/fitr/batchr isn't installed in this environment."""


def tool_command(name: str) -> str:
    """Resolve `name`'s console-script path, independent of PATH/cwd.

    Looks next to the current interpreter first (works whether or not the
    caller activated a venv), then falls back to PATH.
    """
    sibling = Path(sys.executable).parent / name
    if sibling.exists():
        return str(sibling)
    found = shutil.which(name)
    if found:
        return found
    raise ToolNotFoundError(
        f"{name!r} executable not found. Install with: "
        f"pip install git+https://github.com/nikhilcherry/{name}"
    )
