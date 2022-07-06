# 导入包
import shutil
from pathlib import Path
from typing import Mapping, Hashable, Dict

import itk
import numpy as np
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms import Compose, LoadImaged, AddChanneld, Orientationd, MapLabelValued, SaveImaged, MapTransform
from monai.utils.enums import TransformBackends
from monai.data import ITKReader, Dataset, DataLoader
from monai.config import KeysCollection


# 定义转换函数将多个label合并
from scripts.transforms import LogD


class MergeLabelValued(MapTransform):
    """
    Merge labels from multiple items.
    Assume the foreground value in each item is any value larger than 0, the background is zero.

    Merge types:
        same:
            Combine all foreground regions in each item, the final foreground will be 1, the background will be 0.
        different:
            The foreground from first item will be 1, from second item will be 2 and so on.
        original:
            Inpaint the each item's original foregound value onto result one by one.

    Note, the later item's foreground will override previous items' foreground region.

    The return data will have key specicied by name, for example, "merge_label", and will copy meta dict from the first item
    with name as "merge_label_meta_dict".
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: KeysCollection, name: str, merge_type: str = "same",
                 allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.name = name
        assert merge_type in ["same", "different", "original"]
        self.merge_type = merge_type

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        output = []
        data_type = None
        for key in self.key_iterator(d):
            if data_type is None:
                data_type = type(d[key])
            elif not isinstance(d[key], data_type):
                raise TypeError("All items in data must have the same type.")
            output.append(d[key])

        if len(output) == 0:
            return d

        def init_with_zero(a):
            b = a - a
            b[:] = 0
            return b

        # Process label value, the bg is 0, the fg in first item is 1, the fg in second item is 2..
        out_label = init_with_zero(output[0])
        for idx, item in enumerate(output):
            fg = item > 0
            if self.merge_type == "same":
                out_label[fg] = 1
            elif self.merge_type == "different":
                out_label[fg] = idx + 1
            elif self.merge_type == "original":
                out_label[fg] = item[fg]
            else:
                raise RuntimeError("No supported merge type: ", self.merge_type)

        d[self.name] = out_label
        d[self.name + "_meta_dict"] = d[self.first_key(d) + "_meta_dict"]

        return d


class FormatLabelD(MapTransform):
    """
    把label转换为从0开始升序排列
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: KeysCollection) -> None:
        super().__init__(keys, False)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[key]
            unique_labels = np.array(sorted(np.unique(image)))
            for index, label in enumerate(unique_labels):
                image[image == label] = index
        return d


# 测试
class ConfirmLabelLessD(MapTransform):
    """
    确保label小于等于某个值
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: KeysCollection, max_val: int) -> None:
        super().__init__(keys, False)
        self.max_val = max_val

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[key]
            max_label_val = np.max(image)
            assert max_label_val <= self.max_val, f"max label value {max_label_val} is larger than {self.max_val}"
        return d


if __name__ == '__main__':
    # 初始化目标数据集目录结构
    from_dataset = Path("/media/3TB/data/xiaoliutech/relu_cbct_dataset")
    to_dataset = Path("/media/3TB/data/xiaoliutech/relu_teeth_instance")
    if to_dataset.exists():
        shutil.rmtree(to_dataset)
    images_dir, labels_dir = to_dataset.joinpath("images"), to_dataset.joinpath("labels")
    for d in [images_dir, labels_dir]:
        d.mkdir(parents=True, exist_ok=False)

    # 匹配待处理的数据和标签
    all_images = list(from_dataset.joinpath("images").rglob("*.dcm"))
    all_label_dirs = []
    for label_dir in from_dataset.joinpath("labels").iterdir():
        for item in label_dir.iterdir():
            if item.is_dir():
                all_label_dirs.append(item)
    images, label_dirs = [], []
    for image in all_images:
        if image.name in ["631533_no_cbct.dcm", "85073_p_cbct.dcm", "629068_In_cbct.dcm"]:
            # TODO 异常的数据暂时跳过，标签有问题，会导致nnDetection在预处理的时候卡死，需要数据清洗
            continue
        for label_dir in all_label_dirs:
            image_label_name = image.name.replace("_cbct.dcm", "_structures")
            if image_label_name == label_dir.name:
                images.append(image)
                label_dirs.append(label_dir)
                break

    # 数据转换
    need_merge_keys = ["upper_teeth", "lower_teeth"]
    data_dicts = [{"image": images[i],
                   need_merge_keys[0]: label_dirs[i].joinpath("Cbct_upper_teeth.dcm"),
                   need_merge_keys[1]: label_dirs[i].joinpath("Cbct_lower_teeth.dcm")
                   } for i in range(len(images))]
    transform = Compose([
        LoadImaged(keys=["image"]),
        LoadImaged(keys=need_merge_keys, reader=ITKReader(pixel_type=itk.UC)),
        AddChanneld(keys=["image"] + need_merge_keys),
        Orientationd(keys=["image"] + need_merge_keys, axcodes="RAS"),
        ConfirmLabelLessD(keys=need_merge_keys[0], max_val=500),
        # LogD(keys=need_merge_keys),
        MapLabelValued(keys=need_merge_keys[0], orig_labels=list(range(1, 500)), target_labels=list(range(501, 1000))),
        MergeLabelValued(keys=need_merge_keys, name="label", merge_type="original"),
        FormatLabelD(keys=["label"]),
        LogD(keys=["label"]),
        SaveImaged(keys=["image"], meta_keys="image_meta_dict", output_dir=to_dataset.joinpath("images"),
                   output_postfix="", resample=False, separate_folder=False),
        SaveImaged(keys=["label"], meta_keys="image_meta_dict", output_dir=to_dataset.joinpath("labels"),
                   output_postfix="", resample=False, separate_folder=False),
    ])
    # data = transform(data_dicts)
    dataset = Dataset(data=data_dicts, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=4)
    for step, batch in enumerate(loader):
        pass
