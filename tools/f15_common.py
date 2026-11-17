#!/usr/bin/env python3
"""Helpers shared by the F15 material-validation tools (stdlib only, no numpy)."""
from __future__ import annotations

import hashlib
import pathlib


def sha256_file(path) -> str:
    """Hex SHA-256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
