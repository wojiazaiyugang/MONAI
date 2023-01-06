from pathlib import Path
import tempfile
import subprocess

import mlflow
from monai.transforms import ScaleIntensityRanged
from scripts import MLFLOW_TRACKING_URI

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
experiment = mlflow.get_experiment_by_name("颌骨分割")
if not experiment:
    raise ValueError("实验不存在")
run = mlflow.start_run(experiment_id=experiment.experiment_id, run_name="swin_unetr_image_size_96_弹性形变")
mlflow.set_tag("run_id", run.info.run_id)
work_dir = Path(__file__).parent.resolve().joinpath("logs").joinpath("swin_unetr").joinpath("34")

SPACING = (0.25, 0.25, 0.25)  # 数据预处理
IMAGE_SIZE = (96, 96, 96)  # 数据训练size
mlflow.log_param('image_size', str(IMAGE_SIZE))

scale_intensity_range = ScaleIntensityRanged(
    keys=["image"],
    a_min=-500,
    a_max=4000,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)

CLASS_COUNT = 3  # 分类类别，0-背景 1-上颌骨 2-下颌骨
CACHE_DIR = Path("/home/yujiannan/Projects/MONAI/data/temp/jawbone_segmentation_swin_unetr")
LOAD_FROM = Path("/home/yujiannan/Projects/research-contributions/SwinUNETR/Pretrain/logs/4/model_bestValRMSE.pt")

mlflow.log_artifact(__file__, artifact_path="code")
mlflow.log_artifact("./train_swin_unetr.py", artifact_path="code")
with tempfile.TemporaryDirectory() as temp_dir:
    env_file = Path(temp_dir).joinpath("env.txt")
    subprocess.run(f"/home/yujiannan/Projects/MONAI/venv/bin/python -m pip list >> {env_file}", shell=True, check=True)
    subprocess.run(f"""echo -e "\n" >> {env_file}""", shell=True, check=True)
    subprocess.run(f""" /home/yujiannan/Projects/MONAI/venv/bin/python -c "import monai; monai.config.print_config()" >> {env_file} """, shell=True, check=True)
    mlflow.log_artifact(str(env_file), artifact_path="code")