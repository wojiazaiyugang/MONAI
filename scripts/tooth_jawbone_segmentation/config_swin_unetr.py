from pathlib import Path

import mlflow
from monai.transforms import ScaleIntensityRanged
from scripts import MLFLOW_TRACKING_URI
experiment = mlflow.get_experiment_by_name("颌骨分割")
mlflow.start_run(experiment_id=experiment.experiment_id, run_name="image_size 160 + 数据增强")

work_dir = Path(__file__).parent.resolve().joinpath("logs").joinpath("swin_unetr").joinpath("33")

SPACING = (0.25, 0.25, 0.25)  # 数据预处理
IMAGE_SIZE = (160, 160, 160)  # 数据训练size
mlflow.log_param('image_size', str(IMAGE_SIZE))

scale_intensity_range = ScaleIntensityRanged(
    keys=["image"],
    a_min=0,
    a_max=4000,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)

CLASS_COUNT = 3  # 分类类别，0-背景 1-上颌骨 2-下颌骨
CACHE_DIR = Path("/home/yujiannan/Projects/MONAI/data/temp/tooth_jawbone_segmentation_swin_unetr")
LOAD_FROM = Path("/home/yujiannan/Projects/research-contributions/SwinUNETR/Pretrain/logs/4/model_bestValRMSE.pt")

mlflow.log_artifact(__file__, artifact_path="code")
mlflow.log_artifact("./train_swin_unetr.py", artifact_path="code")