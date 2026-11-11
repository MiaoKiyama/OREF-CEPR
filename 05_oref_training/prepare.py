"""Seal inputs using the original OREF experiment preparation."""
from pathlib import Path
import runpy
import sys
ROOT = Path(__file__).resolve().parents[1] / 'training_original/oref'
sys.path.insert(0, str(ROOT))
if __name__ == '__main__':
    runpy.run_path(str(ROOT / 'native24_object_graph_prepare.py'), run_name='__main__')
