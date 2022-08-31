from pathlib import Path

from monai.transforms import ScaleIntensityRanged

work_dir = Path(__file__).parent.resolve().joinpath("logs").joinpath("swin_unetr").joinpath("8")

# IMAGE_SIZE = (96, 96, 96)  # 数据训练size
IMAGE_SIZE = (160, 160, 160)  # 数据训练size
scale_intensity_range = ScaleIntensityRanged(
    keys=["image"],
    a_min=0,
    a_max=4000,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)

CLASS_COUNT = 2  # 分类类别，0-背景 1-牙齿
CACHE_DIR = "/home/yujiannan/Projects/MONAI/data/temp/swin_unetr_single_tooth_segmentation"
