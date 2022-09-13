from pathlib import Path

from monai.transforms import ScaleIntensityRanged

WORK_DIR = Path(__file__).parent.resolve().joinpath("logs").joinpath("deepedit").joinpath("3")

IMAGE_SIZE = [96, 96, 96]  # 数据训练size
CACHE_DIR = "/home/yujiannan/Projects/MONAI/data/temp/deepedit_tooth_instance_segmentation/"

scale_intensity_range = ScaleIntensityRanged(
    keys=["image"],
    a_min=0,
    a_max=4000,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)

