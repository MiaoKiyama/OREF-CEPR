"""Resolve release-local model sources without an installed research checkout."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = Path(__file__).resolve().parent / 'legacy'
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))

