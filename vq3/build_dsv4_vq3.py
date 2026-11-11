#!/usr/bin/env python3
"""Forwarding wrapper to scripts/build_dsv4_vq3.py."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "scripts" / "build_dsv4_vq3.py"
    runpy.run_path(str(target), run_name="__main__")
