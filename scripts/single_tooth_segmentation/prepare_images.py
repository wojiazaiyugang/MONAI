from pathlib import Path
from typing import List, Dict

from monai.data import DataLoader, Dataset
from monai.data.utils import no_collation
from monai.transforms import LoadImaged, Compose, MapLabelValued, EnsureChannelFirstd, Orientationd, Spacingd, \
    SaveImaged, DeleteItemsd
from scripts.transforms import MergeLabelValueD, LogD, ConfirmLabelLessD, FormatLabelD
from scripts.single_tooth_segmentation.config import SPACING

if __name__ == '__main__':
    dataset: List[Dict[str, str]] = []
    from_dataset = Path("/media/3TB/data/xiaoliutech/relu_cbct_respacing")
    to_dataset = Path("/home/yujiannan/Projects/MONAI/data/single_tooth_segmentation")
    to_dataset.mkdir(parents=True, exist_ok=True)

    label_keys = ["Cbct_lower_teeth", "Cbct_upper_teeth"]
    for data_dir in from_dataset.iterdir():
        image_file = data_dir.joinpath(f"{data_dir.stem}_image.nii.gz")
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
        LoadImaged(keys=["image"] + label_keys),
        LogD(message="开始处理", meta_data_key="image"),
        EnsureChannelFirstd(keys=["image"] + label_keys),
        Orientationd(keys=["image"] + label_keys, axcodes="RAS"),
        Spacingd(keys=["image"] + label_keys, pixdim=SPACING, padding_mode="border"),
        ConfirmLabelLessD(keys=label_keys[:1], max_val=200),
        MapLabelValued(keys=label_keys[:1], orig_labels=list(range(1, 200)), target_labels=list(range(201, 400))),
        # 合并上下牙
        MergeLabelValueD(keys=label_keys, name="label", merge_type="original"),
        FormatLabelD(keys=["label"]),
        # 合并所有标签
        SaveImaged(
            keys="image",
            output_dir=to_dataset,
            output_postfix="image",
            resample=False,
            separate_folder=False,
            print_log=False
        ),
        SaveImaged(
            keys="label",
            meta_keys=f"image_key_meta_dict",
            output_dir=to_dataset,
            output_postfix="label",
            resample=False,
            separate_folder=False,
            print_log=False
        ),
        LogD(message="处理完成", meta_data_key="image"),
        DeleteItemsd(keys=label_keys)
    ])

    process_transforms(dataset)
    exit()

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
