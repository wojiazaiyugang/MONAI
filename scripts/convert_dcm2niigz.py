# 导入包
from pathlib import Path

from monai.transforms import Compose, LoadImaged, SaveImaged

if __name__ == '__main__':
    images = [Path("/home/yujiannan/下载/625574_meyer.dcm")]
    data_dicts = [{"image": images[i]} for i in range(len(images))]
    transform = Compose([
        LoadImaged(keys=["image"]),
        SaveImaged(keys=["image"], meta_keys="image_meta_dict", output_dir=Path("/home/yujiannan/下载"),
                   output_postfix="", resample=False, separate_folder=False),
    ])
    transform(data_dicts)
