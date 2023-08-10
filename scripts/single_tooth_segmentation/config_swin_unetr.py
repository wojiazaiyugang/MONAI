from pathlib import Path

from monai.transforms import ScaleIntensityRanged
from scripts import get_data_dir, get_cache_data_dir

work_dir = Path(__file__).parent.resolve().joinpath("logs").joinpath("swin_unetr").joinpath("7")

IMAGE_SIZE = (96, 96, 96)  # 数据训练size
scale_intensity_range = ScaleIntensityRanged(
    keys=["image"],
    a_min=0,
    a_max=4000,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)

CLASS_COUNT = 2  # 分类类别，0-背景 1-牙齿
DATASET_DIR = get_data_dir().joinpath("tooth_instannce_segmentation")
CACHE_DIR = get_cache_data_dir().joinpath("swin_unetr_single_tooth_segmentation")