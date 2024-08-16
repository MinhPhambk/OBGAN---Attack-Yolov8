# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.2.70"

import os

# Set ENV Variables (place before imports)
os.environ["OMP_NUM_THREADS"] = "1"  # reduce CPU utilization during training

from yolov8.data.explorer.explorer import Explorer
from yolov8.models import NAS, RTDETR, SAM, SAM2, YOLO, FastSAM, YOLOWorld
from yolov8.utils import ASSETS, SETTINGS
from yolov8.utils.checks import check_yolo as checks
from yolov8.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "SAM2",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
)
