from pathlib import Path

from monai.transforms import ScaleIntensityRanged
from scripts import get_model

WORK_DIR = Path(__file__).resolve().parent.joinpath("logs").joinpath("unetr").joinpath("8")

SPACING = (0.25, 0.25, 0.25)  # 数据预处理
IMAGE_SIZE = (160, 160, 160)  # 数据训练size

SCALE_INTENSITY_RANGE = ScaleIntensityRanged(
    keys=["image"],
    a_min=0,
    a_max=4000,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)

CLASS_COUNT = 5  # 分类类别，0-背景 1-上颌骨 2-下颌骨 3-上颌骨牙齿 4-下颌骨牙齿
CACHE_DIR = Path("/home/yujiannan/Projects/MONAI/data/temp/tooth_jawbone_seg")
PRETRAINED_MODEL = Path("/home/yujiannan/Projects/MONAI/scripts/pretrain_unert/logs/2/best_model.pt")  # 预训练模型
