"""Versioned MOSS cache metadata and non-destructive replacement."""

import os
from pathlib import Path
import tempfile

import numpy as np

CACHE_VERSION = 2


def backup_cache(path):
    path = Path(path)
    if not path.exists():
        return
    backup = Path(str(path) + ".legacy-v1.bak")
    count = 0
    while backup.exists():
        count += 1
        backup = Path(str(path) + f".legacy-v1.bak.{count}")
    path.rename(backup)


def write_cache(path, values, *, backup=True):
    """Encode first; retain the old bytes and atomically install the completed cache."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as file:
            temporary = file.name
            np.savez(file, **values)
        if backup:
            backup_cache(path)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
