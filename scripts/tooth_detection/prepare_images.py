import json
import argparse
from pathlib import Path
from typing import List, Dict

import itk
import torch
from monai.transforms import LoadImaged, Compose, MapLabelValued, EnsureChannelFirstd, EnsureTyped, \
    Orientationd, Spacingd, SaveImaged
from monai.data import ITKReader
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToWorldCoordinated,
    ConvertBoxModed
)
from monai.data.box_utils import CenterSizeMode, StandardMode
from monai.data import DataLoader, Dataset
from monai.data.utils import no_collation
from scripts.transforms import SaveBBoxD, GenerateBBoxD, FormatLabelD, MergeLabelValueD, ConfirmLabelLessD

if __name__ == '__main__':
    output_dir = Path("/home/yujiannan/Projects/MONAI/data/teeth_detection_spacing_0.25_0.25_0.25")
    parser = argparse.ArgumentParser(description="LUNA16 Detection Image Resampling")
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train_luna16_16g.json",
        help="config json file that stores hyper-parameters",
    )
    args = parser.parse_args()

    config_dict = json.load(open(args.config_file, "r"))

    for k, v in config_dict.items():
        setattr(args, k, v)

    dataset: List[Dict[str, str]] = []
    for data_dir in Path("/media/3TB/data/xiaoliutech/relu_cbct_respacing").iterdir():
        if output_dir.joinpath(f"{data_dir.stem}.nii.gz").exists():
            print(f"跳过已存在数据 {data_dir.name}")
            continue
        if not data_dir.is_dir():
            print(f"跳过非目录 {data_dir.name}")
            continue
        dataset.append({
            "image": str(data_dir.joinpath(f"{data_dir.stem}.nii.gz")),
            "Cbct_lower_teeth": str(data_dir.joinpath("Cbct_lower_teeth.nii.gz")),
            "Cbct_upper_teeth": str(data_dir.joinpath("Cbct_upper_teeth.nii.gz"))
        })
    process_transforms = Compose([
        LoadImaged(keys=["image"]),
        LoadImaged(keys=["Cbct_lower_teeth", "Cbct_upper_teeth"],
                   reader=ITKReader(pixel_type=itk.UC)),
        EnsureChannelFirstd(keys=["image", "Cbct_lower_teeth", "Cbct_upper_teeth"]),
        EnsureTyped(keys=["image"], dtype=torch.float16),
        Orientationd(keys=["image", "Cbct_lower_teeth", "Cbct_upper_teeth"], axcodes="RAS"),
        Spacingd(keys=["image", "Cbct_lower_teeth", "Cbct_upper_teeth"],
                 pixdim=args.spacing,
                 padding_mode="border",
                 mode=("bilinear", "nearest", "nearest")),
        ConfirmLabelLessD(keys=["Cbct_lower_teeth"], max_val=150),
        MapLabelValued(keys=["Cbct_lower_teeth"], orig_labels=list(range(1, 150)), target_labels=list(range(151, 300))),
        MergeLabelValueD(keys=["Cbct_lower_teeth", "Cbct_upper_teeth"], name="label", merge_type="original"),
        FormatLabelD(keys=["label"]),
        GenerateBBoxD(keys=["label"], bbox_key="box"),  # xyzxyz 图像坐标系
        AffineBoxToWorldCoordinated(box_keys=["box"], box_ref_image_keys="label"),  # xyzxyz 世界坐标系
        ConvertBoxModed(box_keys=["box"], src_mode=StandardMode, dst_mode=CenterSizeMode)
    ])

    # saved images to Nifti
    post_transforms = Compose(
        [
            SaveImaged(
                keys="image",
                meta_keys="image_meta_dict",
                output_dir=output_dir,
                output_postfix="",
                resample=False,
                separate_folder=False
            ),
            SaveBBoxD(
                keys=["box"],
                output_dir=output_dir,
            )
        ]
    )

    process_ds = Dataset(
        data=dataset,
        transform=process_transforms,
    )
    process_loader = DataLoader(
        process_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
        collate_fn=no_collation,
    )

    for batch_data in process_loader:
        for batch_data_i in batch_data:
            boxes = batch_data_i["box"]
            for box in boxes:
                if max(box[3:]) > 50:
                    print(f"""异常数据，{batch_data_i["image_meta_dict"]["filename_or_obj"]} box: {box}""")
                    break
            else:
                batch_data_i = post_transforms(batch_data_i)
                print(f"""数据处理完成，{batch_data_i["image_meta_dict"]["filename_or_obj"]}""")
