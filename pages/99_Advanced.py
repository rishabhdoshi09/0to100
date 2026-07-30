"""Advanced engineering and research console.

The former monolithic QuantTerm application is preserved here so no diagnostic,
research or operator surface is deleted while the retail experience becomes the
default entry point.
"""
from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "advanced_app.py"), run_name="__main__")
