import sys
from pathlib import Path

import matplotlib

# Use a non-interactive backend for matplotlib during tests
matplotlib.use("Agg")

# Ensure source directory is on the import path for direct module imports
ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
