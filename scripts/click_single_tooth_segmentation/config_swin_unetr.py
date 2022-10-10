from pathlib import Path

from monai.transforms import ScaleIntensityRanged

log_dir = Path(__file__).parent.resolve().joinpath("logs").joinpath("swin_unetr")
work_dir = log_dir.joinpath("13")

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

CLASS_COUNT = 3  # 分类类别，0-背景 1-牙齿
DATASET = Path("/media/3TB/data/xiaoliutech/20220923")
CACHE_DIR = "/home/yujiannan/Projects/MONAI/data/temp/click_single_tooth_segmentation"
LOAD_FROM = None  # log_dir.joinpath("11").joinpath("best_metric_model.pth")
LOAD_FROM_DICE = 0  # 0.985
