import tempfile
import subprocess
from pathlib import Path

import monai
import mlflow
from monai.transforms import ScaleIntensityRanged
from scripts import MLFLOW_TRACKING_URI


work_dir = Path(__file__).parent.resolve().joinpath("logs").joinpath("unet++").joinpath("3")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
experiment = mlflow.get_experiment_by_name("颌骨分割")
if not experiment:
    raise ValueError("实验不存在")
run = mlflow.start_run(experiment_id=experiment.experiment_id, run_name="unet_plus_plus_image_size_128_翻转旋转")
mlflow.set_tag("run_id", run.info.run_id)
IMAGE_SIZE = (128, 128, 128)  # 数据训练size
mlflow.log_param('image_size', str(IMAGE_SIZE))

scale_intensity_range = ScaleIntensityRanged(
    keys=["image"],
    a_min=0,
    a_max=4000,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)

CLASS_COUNT = 3  # 分类类别，0-背景 1 2 颌骨 3 4 牙齿
CACHE_DIR = Path("/home/yujiannan/Projects/MONAI/data/temp/unet++")

mlflow.log_artifact(__file__, artifact_path="code")
mlflow.log_artifact("./train_unetplusplus.py", artifact_path="code")
with tempfile.TemporaryDirectory() as temp_dir:
    env_file = Path(temp_dir).joinpath("env.txt")
    subprocess.run(f"/home/yujiannan/Projects/MONAI/venv/bin/python -m pip list >> {env_file}", shell=True, check=True)
    subprocess.run(f"""echo -e "\n" >> {env_file}""", shell=True, check=True)
    subprocess.run(f""" /home/yujiannan/Projects/MONAI/venv/bin/python -c "import monai; monai.config.print_config()" >> {env_file} """, shell=True, check=True)
    mlflow.log_artifact(str(env_file), artifact_path="code")