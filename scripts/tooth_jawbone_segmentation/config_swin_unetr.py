from pathlib import Path

from monai.transforms import ScaleIntensityRanged

work_dir = Path(__file__).parent.resolve().joinpath("logs").joinpath("temp")

SPACING = (0.25, 0.25, 0.25)  # 数据预处理
IMAGE_SIZE = (96, 96, 96)  # 数据训练size

scale_intensity_range = ScaleIntensityRanged(
    keys=["image"],
    a_min=-500,
    a_max=3300,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)

CLASS_COUNT = 5  # 分类类别，0-背景 1-牙齿 2-下颌骨 3-上颌骨
