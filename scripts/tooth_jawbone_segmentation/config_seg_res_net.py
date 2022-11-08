from pathlib import Path

from monai.transforms import ScaleIntensityRanged

work_dir = Path(__file__).parent.resolve().joinpath("logs").joinpath("seg_res_net").joinpath("3")

IMAGE_SIZE = (160, 160, 160)  # 数据训练size

scale_intensity_range = ScaleIntensityRanged(
    keys=["image"],
    a_min=0,
    a_max=4000,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)

CLASS_COUNT = 5  # 分类类别，0-背景 1 2 颌骨 3 4 牙齿
CACHE_DIR = Path("/home/yujiannan/Projects/MONAI/data/temp/tooth_jawbone_segmentation_seg_res_net")