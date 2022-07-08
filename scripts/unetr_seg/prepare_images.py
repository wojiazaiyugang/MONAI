from pathlib import Path
from typing import List, Dict

from monai.data import DataLoader, Dataset
from monai.data.utils import no_collation
from monai.transforms import LoadImaged, Compose, MapLabelValued, EnsureChannelFirstd, Orientationd, Spacingd, \
    SaveImaged, DeleteItemsd
from scripts.transforms import MergeLabelValueD, LogD
from scripts.unetr_seg.config import SPACING

if __name__ == '__main__':
    dataset: List[Dict[str, str]] = []
    from_dataset = Path("/media/3TB/data/xiaoliutech/relu_cbct_respacing")
    to_dataset = Path("/home/yujiannan/Projects/MONAI/data/unetr_seg")

    image_key = "image"
    label_key = "label"
    label_keys = ["Cbct_lower_teeth", "Cbct_upper_teeth",  # 上下牙
                  "Mandible", "Mandibular_canals",  # 下颌骨+下颌骨管
                  "Maxillary_complex", "Left_maxillary_sinus", "Right_maxillary_sinus"]  # 上颌骨+左颧骨+右颧骨
    for data_dir in from_dataset.iterdir():
        image_file = data_dir.joinpath(f"{data_dir.stem}.nii.gz")
        if to_dataset.joinpath(image_file.name).exists():
            print(f"跳过已存在数据 {image_file.name}")
            continue
        if not data_dir.is_dir():
            print(f"跳过非目录 {data_dir.name}")
            continue

        if all(file.exists() for file in [data_dir.joinpath(f"{data_key}.nii.gz") for data_key in label_keys]):
            data = {
                "image": str(data_dir.joinpath(f"{data_dir.stem}.nii.gz"))
            }
            for label_key in label_keys:
                data[label_key] = str(data_dir.joinpath(f"{label_key}.nii.gz"))
            dataset.append(data)
    process_transforms = Compose([
        LoadImaged(keys=[image_key] + label_keys),
        LogD(message="开始处理", meta_data_key="image"),
        EnsureChannelFirstd(keys=[image_key] + label_keys),
        Orientationd(keys=[image_key] + label_keys, axcodes="RAS"),
        Spacingd(keys=[image_key] + label_keys, pixdim=SPACING, padding_mode="border"),
        # LogD(message="合并 上颌骨+左颧骨+右颧骨"),
        MergeLabelValueD(keys=label_keys[4:7], name="up", merge_type="same"),
        MapLabelValued(keys="up", orig_labels=list(range(1, 2)), target_labels=list(range(3, 4))),
        # LogD(message="合并 下颌骨+下颌骨管"),
        MergeLabelValueD(keys=label_keys[2:4], name="down", merge_type="same"),
        MapLabelValued(keys="down", orig_labels=list(range(1, 2)), target_labels=list(range(2, 3))),
        # 合并上下牙
        MergeLabelValueD(keys=label_keys[0:2], name="teeth", merge_type="same"),
        # 合并所有标签
        MergeLabelValueD(keys=["up", "down", "teeth"], name=label_key, merge_type="original"),
        SaveImaged(
            keys=image_key,
            output_dir=to_dataset,
            output_postfix="image",
            resample=False,
            separate_folder=False,
            print_log=False
        ),
        SaveImaged(
            keys=label_key,
            meta_keys=f"{image_key}_meta_dict",
            output_dir=to_dataset,
            output_postfix="label",
            resample=False,
            separate_folder=False,
            print_log=False
        ),
        LogD(message="处理完成", meta_data_key="image"),
        DeleteItemsd(keys=label_keys)
    ])

    process_ds = Dataset(
        data=dataset,
        transform=process_transforms,
    )
    process_loader = DataLoader(
        process_ds,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        collate_fn=no_collation,
    )

    for batch_data in process_loader:
        pass
