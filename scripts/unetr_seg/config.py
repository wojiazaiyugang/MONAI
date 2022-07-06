from monai.transforms import ScaleIntensityRanged

from scripts import get_log_dir

work_dir = get_log_dir().joinpath("unetr_seg").joinpath("2")

SPACING = (0.5, 0.5, 0.5)  # 数据预处理
IMAGE_SIZE = (96, 96, 96)  # 数据训练size

scale_intensity_range = ScaleIntensityRanged(
    keys=["image"],
    a_min=0,
    a_max=1000,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)

CLASS_COUNT = 4  # 分类类别，0-背景 1-牙齿 2-下颌骨 3-上颌骨
