import json
import shutil
from tqdm import tqdm
from pathlib import Path

import numpy as np
from monai.transforms import Compose, LoadImage

if __name__ == '__main__':
    # 初始化目录结构
    dataset_dir = Path("/home/yujiannan/Projects/nnDetection/data/Task100_Teeth")
    assert not dataset_dir.exists(), "dataset_dir already exists"
    splitted_dir = dataset_dir.joinpath("raw_splitted")
    train_images_dir = splitted_dir.joinpath("imagesTr")
    train_labels_dir = splitted_dir.joinpath("labelsTr")
    for d in [train_labels_dir, train_images_dir]:
        d.mkdir(parents=True, exist_ok=False)
    print(f"数据集目录结构初始化完成，目录结构为：{dataset_dir}")

    # 生成`dataset.json`
    config_file = dataset_dir.joinpath("dataset.json")
    with open(config_file, "w") as f:
        f.write(json.dumps({
            "task": "Task100_Teeth",
            "name": "Task100_Teeth",
            "dim": 3,
            "target_class": 1,
            "test_labels": True,
            "labels": {
                "0": "teeth"
            },
            "modalities": {
                "0": "CBCT"
            }
        }, indent=4))
    print(f"数据集配置文件生成完成，文件路径为：{config_file}")

    # 生成训练数据集
    transform = Compose([
        LoadImage(image_only=True)
    ])
    case_index = 0
    from_dataset_dir = Path("/media/3TB/data/xiaoliutech/relu_teeth_instance")
    from_dataset_images_dir = from_dataset_dir.joinpath("images")
    from_dataset_labels_dir = from_dataset_dir.joinpath("labels")
    for image in tqdm(list(from_dataset_images_dir.iterdir())):
        label = from_dataset_labels_dir.joinpath(image.name)
        shutil.copy(image, train_images_dir.joinpath(f"case{case_index:0>3}_0000.nii.gz"))
        shutil.copy(label, train_labels_dir.joinpath(f"case{case_index:0>3}.nii.gz"))
        label_data = transform(label)
        label_info = {}
        for label_key in np.unique(label_data):
            if label_key != 0:
                label_info[str(int(label_key))] = 0
        with open(train_labels_dir.joinpath(f"case{case_index:0>3}.json"), "w") as f:
            f.write(json.dumps({
                "instances": label_info
            }, indent=4))
        case_index += 1
    print(f"训练数据集生成完成，数据集目录为：{splitted_dir}")