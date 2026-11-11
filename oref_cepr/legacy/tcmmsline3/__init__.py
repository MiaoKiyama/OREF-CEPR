"""MS-guided region-level optical-SAR semantic segmentation."""

from .config import load_config
from .model import MSGuidedRegionFusion

__all__ = ["MSGuidedRegionFusion", "load_config"]

