"""Adapters from trajectory file formats to the step model."""

from __future__ import annotations

from pathlib import Path

from trajlens.adapters.generic import load_generic
from trajlens.model import Trajectory

FORMATS = ("generic",)


def load(path: str | Path, fmt: str = "auto") -> Trajectory:
    """Load one trajectory file. Only the generic JSONL format is implemented."""
    if fmt == "auto":
        fmt = "generic"
    if fmt == "generic":
        return load_generic(path)
    raise ValueError(f"unknown format {fmt!r}; known: {', '.join(FORMATS)}")


__all__ = ["FORMATS", "load", "load_generic"]
