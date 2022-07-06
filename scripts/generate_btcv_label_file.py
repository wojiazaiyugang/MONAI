"""
生成btcv数据集标注文件
"""
import json
from pathlib import Path

data_dir = Path("/media/3TB/data/BTCV")
train_images_dir = data_dir.joinpath("Training").joinpath("img")
train_labels_dir = data_dir.joinpath("Training").joinpath("label")
test_images_dir = data_dir.joinpath("Testing").joinpath("img")

training = []

for train_image in train_images_dir.iterdir():
    train_label = train_labels_dir.joinpath(train_image.name.replace("img", "label"))
    if train_image.name.endswith(".nii.gz") and train_label.exists():
        training.append({
            "image": str(train_image),
            "label": str(train_label)
        })

test = []

for test_image in test_images_dir.iterdir():
    test.append(str(test_image))

label_file = {
    "description": "btcv yucheng",
    "labels": {
        "0": "background",
        "1": "spleen",
        "2": "rkid",
        "3": "lkid",
        "4": "gall",
        "5": "eso",
        "6": "liver",
        "7": "sto",
        "8": "aorta",
        "9": "IVC",
        "10": "veins",
        "11": "pancreas",
        "12": "rad",
        "13": "lad"
    },
    "licence": "yt",
    "modality": {
        "0": "CT"
    },
    "name": "btcv",
    "numTest": len(test),
    "numTraining": len(training),
    "reference": "Vanderbilt University",
    "release": "1.0 06/08/2015",
    "tensorImageSize": "3D",
    "test": test,
    "training": training[:-6],
    "validation": training[-6:]
}
output_file = Path(__file__).with_name("btcv.json")
with open(output_file, "w") as f:
    f.write(json.dumps(label_file, indent=4))
print(f"生成标注文件 {output_file}")
