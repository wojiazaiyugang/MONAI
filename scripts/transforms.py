import gc
import json
from pathlib import Path
from typing import Mapping, Hashable, Dict, Any, Optional

import numpy as np
import torch

from monai.config import KeysCollection, NdarrayOrTensor
from monai.transforms import MapTransform, generate_spatial_bounding_box, Transform
from monai.utils import TransformBackends, ensure_tuple


class SaveBBoxD(MapTransform):
    """
    保存bbox
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: KeysCollection, output_dir: Path):
        super().__init__(keys, allow_missing_keys=False)
        self.output_dir = output_dir

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        image_name = data["image_meta_dict"]["filename_or_obj"]
        bbox_file_name = self.output_dir.joinpath(Path(image_name).name + ".txt")
        for key in self.key_iterator(d):
            boxes = d[key].tolist()  # xyzxyz 世界坐标系
            with open(bbox_file_name, "w") as f:
                f.write(json.dumps({
                    "bbox": boxes
                }))
        print(f"保存bbox到{bbox_file_name}")
        return d


class GenerateBBoxD(MapTransform):
    """
    生成bbox
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: KeysCollection, bbox_key: str):
        super().__init__(keys, allow_missing_keys=False)
        self.bbox_key = bbox_key

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        d[self.bbox_key] = []
        for key in self.key_iterator(d):
            image = d[key]
            labels = np.unique(image)
            for label in labels:
                if label == 0:
                    continue
                start, end = generate_spatial_bounding_box(img=image, select_fn=lambda x: x == label,
                                                           allow_smaller=False)
                d[self.bbox_key].append(list(start + end))
        d[self.bbox_key] = torch.Tensor(np.array(d[self.bbox_key]))
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


class MergeLabelValueD(MapTransform):
    """
    将多个label合并
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self,
                 keys: KeysCollection,
                 name: str,
                 merge_type: str = "same",
                 allow_missing_keys: bool = False,
                 copy_meta_from: str = "image") -> None:
        super().__init__(keys, allow_missing_keys)
        self.name = name
        assert merge_type in ["same", "different", "original"]
        self.merge_type = merge_type
        self.copy_meta_from = copy_meta_from

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
        d[self.name].meta = d[self.copy_meta_from].meta

        for key in self.key_iterator(d):
            del d[key]
        gc.collect()
        return d


class ConfirmLabelLessD(MapTransform):
    """
    确保label小于等于某个值
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: KeysCollection, max_val: int) -> None:
        super().__init__(keys, allow_missing_keys=False)
        self.max_val = max_val

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        for key in self.key_iterator(data):
            image = data[key]
            max_label_val = np.max(image)
            assert max_label_val <= self.max_val, f"max label value {max_label_val} is larger than {self.max_val}"
        return dict(data)


class LogD(MapTransform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, message: str, meta_data_key: str) -> None:
        super().__init__(["not_exist_key"], allow_missing_keys=True)
        self.message = message
        self.meta_data_key = meta_data_key

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        meta_dict = d[self.meta_data_key]
        print(f"""{meta_dict.meta["filename_or_obj"]} {self.message}""")
        return d


class ReplaceD(MapTransform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: KeysCollection, data: Any) -> None:
        assert len(keys) == 1, "Only support one key"
        super().__init__(keys)
        self.data = data

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(data):
            d[key] = self.data
        return d


class SetAttrd(MapTransform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: str, value: Any, meta_dict_key: str) -> None:
        super().__init__(["not_exist_key"], allow_missing_keys=True)
        self.key = keys
        self.value = value
        self.meta_dict_key = meta_dict_key

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        d[self.key] = self.value
        d[f"{self.key}_meta_dict"] = d[self.meta_dict_key]
        return d


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

    def __init__(self, keys: KeysCollection) -> None:
        super().__init__(keys, False)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        for key in self.key_iterator(d):
            output = torch.zeros_like(d[key][0])
            for index, o in enumerate(d[key][:32]):
                fg = o > 0
                output[fg] = index + 1
            d[key] = output
        return d
