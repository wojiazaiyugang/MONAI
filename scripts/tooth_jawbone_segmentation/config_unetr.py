from pathlib import Path

from monai.transforms import ScaleIntensityRanged
from scripts import get_model

WORK_DIR = Path(__file__).resolve().parent.joinpath("logs").joinpath("4")

SPACING = (0.25, 0.25, 0.25)  # 数据预处理
IMAGE_SIZE = (96, 96, 96)  # 数据训练size

SCALE_INTENSITY_RANGE = ScaleIntensityRanged(
    keys=["image"],
    a_min=0,
    a_max=1000,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)

CLASS_COUNT = 4  # 分类类别，0-背景 1-牙齿 2-下颌骨 3-上颌骨
PRETRAINED_MODEL = get_model("vitautoenc_weights.pt")  # 预训练模型
