"""Entry point for the original, unchanged training implementation."""
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    os.execv(sys.executable, [sys.executable, str(ROOT / 'training_original/launch.py'), 'oref', *sys.argv[1:]])
