#!/usr/bin/env python3
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "data_tools" / "standardize_dataset_actions.py"),
    run_name="__main__",
)
