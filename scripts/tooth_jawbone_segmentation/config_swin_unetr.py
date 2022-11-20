from pathlib import Path

from monai.transforms import ScaleIntensityRanged

work_dir = Path(__file__).parent.resolve().joinpath("logs").joinpath("swin_unetr").joinpath("26")

SPACING = (0.25, 0.25, 0.25)  # 数据预处理
IMAGE_SIZE = (128, 128, 128)  # 数据训练size

scale_intensity_range = ScaleIntensityRanged(
    keys=["image"],
    a_min=0,
    a_max=4000,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)

CLASS_COUNT = 3  # 分类类别，0-背景 1-上颌骨 2-下颌骨
CACHE_DIR = Path("/home/yujiannan/Projects/MONAI/data/temp/tooth_jawbone_segmentation_swint_unetr")
LOAD_FROM = Path("/home/yujiannan/Projects/research-contributions/SwinUNETR/Pretrain/logs/4/model_bestValRMSE.pt")
