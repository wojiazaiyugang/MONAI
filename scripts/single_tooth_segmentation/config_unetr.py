from pathlib import Path

from monai.transforms import ScaleIntensityRanged
from scripts import get_model

work_dir = Path(__file__).parent.resolve().joinpath("logs").joinpath("6")

IMAGE_SIZE = (96, 96, 96)  # 数据训练size

scale_intensity_range = ScaleIntensityRanged(
    keys=["image"],
    a_min=300,
    a_max=3300,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)

CLASS_COUNT = 2  # 分类类别，0-背景 1-牙齿
PRETRAINED_MODEL = get_model("vitautoenc_weights.pt")  # 预训练模型
