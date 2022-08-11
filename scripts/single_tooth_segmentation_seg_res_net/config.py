from pathlib import Path

from monai.transforms import ScaleIntensityRanged

work_dir = Path(__file__).parent.resolve().joinpath("logs").joinpath("3")

scale_intensity_range = ScaleIntensityRanged(
    keys=["image"],
    a_min=0,
    a_max=4000,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)

CLASS_COUNT = 1  # 分类类别，只有一类，不包括背景
