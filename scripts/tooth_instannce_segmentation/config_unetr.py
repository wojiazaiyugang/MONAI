from pathlib import Path

from monai.transforms import ScaleIntensityRanged
from scripts import get_model

work_dir = Path(__file__).parent.resolve().joinpath("logs").joinpath("unetr").joinpath("1")

IMAGE_SIZE = (160, 160, 160)  # 数据训练size

scale_intensity_range = ScaleIntensityRanged(
    keys=["image"],
    a_min=0,
    a_max=4000,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)

CLASS_COUNT = 1 + 32  # 分类类别，0-背景 1-牙齿
CACHE_DIR = "/home/yujiannan/Projects/MONAI/data/temp/unetr_tooth_instance_segmentation"
