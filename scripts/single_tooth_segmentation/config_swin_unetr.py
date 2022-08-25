from pathlib import Path

from monai.transforms import ScaleIntensityRanged
from scripts import get_model

work_dir = Path(__file__).parent.resolve().joinpath("logs").joinpath("swin_unetr").joinpath("7")

IMAGE_SIZE = (96, 96, 96)  # 数据训练size
# IMAGE_SIZE = (192, 192, 192)  # 数据训练size
scale_intensity_range = ScaleIntensityRanged(
    keys=["image"],
    a_min=0,
    a_max=4000,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)

CLASS_COUNT = 2  # 分类类别，0-背景 1-牙齿
