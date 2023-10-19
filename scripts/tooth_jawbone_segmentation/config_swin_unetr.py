from pathlib import Path
import tempfile
import subprocess

from monai.transforms import ScaleIntensityRanged
from scripts import MLFLOW_TRACKING_URI

work_dir = Path(__file__).parent.resolve().joinpath("logs").joinpath("swin_unetr").joinpath("35")

SPACING = (0.25, 0.25, 0.25)  # 数据预处理
IMAGE_SIZE = (96, 96, 96)  # 数据训练size

scale_intensity_range = ScaleIntensityRanged(
    keys=["image"],
    a_min=-500,
    a_max=4000,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)

CLASS_COUNT = 3  # 分类类别，0-背景 1-上颌骨 2-下颌骨
CACHE_DIR = Path("/home/yujiannan/Projects/MONAI/data/temp/jawbone_segmentation_swin_unetr")
LOAD_FROM = Path("/home/yujiannan/下载/model_swinvit.pt")
# LOAD_FROM = None

