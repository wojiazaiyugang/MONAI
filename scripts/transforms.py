import gc
import json
import random
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import Mapping, Hashable, Dict, Any, Optional, Union, Sequence, List, Tuple

import numpy as np
import torch
from torch.nn.functional import conv3d
import torchio

from monai.config import KeysCollection, NdarrayOrTensor, IndexSelection
from monai.data import MetaTensor
from monai.transforms import MapTransform, generate_spatial_bounding_box, InvertibleTransform, CropForeground, \
    SpatialCrop, BorderPad, RandomizableTransform, RandCropByPosNegLabel
from monai.transforms.utils import map_binary_to_indices, generate_pos_neg_label_crop_centers
from monai.transforms.utils_pytorch_numpy_unification import (
    unravel_index,
)
from monai.utils import NumpyPadMode, PytorchPadMode, ensure_tuple_rep, ensure_tuple, ImageMetaKey as Key
from monai.utils.enums import TraceKeys, TransformBackends


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

    def __call__(self, data: Mapping[Hashable, MetaTensor]) -> Dict[Hashable, MetaTensor]:
        for key in self.key_iterator(data):
            image = data[key]
            max_label_val = np.max(image.numpy())
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


class PreprocessForegroundSamples(MapTransform):
    """
    预处理前景（牙齿）
    1、牙齿腐蚀操作
    2、计算牙齿的bbox
    3、计算牙齿的正样本indexes
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
            self,
            keys: KeysCollection,
            label_key: str,
            fg_labels: Sequence = None,
            bg_label: Union[int, float] = 0,
            to_same_fg_label: Union[int, float] = None,
            channel_indices: Optional[IndexSelection] = None,
            margin: Union[Sequence[int], int] = 0,
            mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = NumpyPadMode.CONSTANT,
            meta_keys: Optional[KeysCollection] = None,
            meta_key_postfix: str = "meta_dict",
            allow_missing_keys: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.label_key = label_key
        self.fg_labels = fg_labels
        self.bg_label = bg_label
        self.to_same_fg_label = to_same_fg_label

        self.channel_indices = channel_indices
        self.margin = margin
        self.mode = ensure_tuple_rep(mode, len(self.keys))

        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> List[Dict[Hashable, NdarrayOrTensor]]:
        def find_foreground_labels(label: NdarrayOrTensor, bg_label: Union[int, float] = 0):
            all_labels = None
            if isinstance(label, np.ndarray):
                all_labels = np.unique(label)
            elif isinstance(label, torch.Tensor):
                all_labels = torch.unique(label)
            else:
                raise RuntimeError("Unsupported label type: ", type(label))

            fg_labels = []
            for i in range(all_labels.shape[0]):
                if all_labels[i] == bg_label:
                    continue
                fg_labels.append(all_labels[i])

            return fg_labels

        def dilation3d(binary_image: torch.Tensor) -> torch.Tensor:
            """
            膨胀3D图像
            :param binary_image:
            :return:
            """
            kernel = torch.ones(size=[1, 1, 5, 5, 5], dtype=torch.float32)
            return torch.clamp(conv3d(binary_image, kernel, padding=(1, 1, 1)), 0, 1)

        d = dict(data)
        bg_label = self.bg_label
        fg_labels = find_foreground_labels(d[self.label_key], bg_label) if self.fg_labels is None else self.fg_labels

        preprocess_data = {}  # 预处理结果

        for i, fg_l in enumerate(fg_labels):
            # label裁剪出来
            label: MetaTensor = deepcopy(d[self.label_key])
            # 清空背景，label置为为1
            label[label != fg_l] = bg_label
            label[label == fg_l] = 1
            # 切换前景背景
            label[label == 0] = 2
            label[label == 1] = 0
            label[label == 2] = 1
            # 膨胀相反的label，相当于腐蚀label
            label = dilation3d(torch.unsqueeze(label, dim=0))[0]
            # 切换前景背景
            # label_patch[label_patch == 0] = 2
            # label_patch[label_patch == 1] = 0
            # label_patch[label_patch == 2] = 1
            label[label == 0] = 2
            label[label == 1] = 0
            label[label == 2] = 1

            fg_indices_, bg_indices_ = map_binary_to_indices(label, None, 0)
            size = fg_indices_.size()[0]
            if size > 0:  # 腐蚀后可能把所有的前景都清空了，这里要排除掉
                preprocess_data[int(fg_l.numpy().tolist())] = {
                    "fg_indices_": fg_indices_.numpy(),  # 腐蚀后的label的前景indexes
                    "label_original_shape": label.size()[1:],  # 原始label的shape
                }
        d["preprocess_data"] = preprocess_data
        return d


class CropForegroundSamples(RandomizableTransform, MapTransform, InvertibleTransform):
    """
    搭配PreprocessForegroundSamples使用，从预处理后的前景中随机选一个点，然后crop固定size的patch出来
    通常用于click_XX_segmentation的训练
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
            self,
            keys: KeysCollection,
            label_key: str,
            image_size: Tuple[int, int, int],
            fg_labels: Sequence = None,
            bg_label: Union[int, float] = 0,
            to_same_fg_label: Union[int, float] = None,
            channel_indices: Optional[IndexSelection] = None,
            margin: Union[Sequence[int], int] = 0,
            mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = NumpyPadMode.CONSTANT,
            meta_keys: Optional[KeysCollection] = None,
            meta_key_postfix: str = "meta_dict",
            allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.label_key = label_key
        self.fg_labels = fg_labels
        self.bg_label = bg_label
        self.to_same_fg_label = to_same_fg_label
        self.image_size = image_size

        self.channel_indices = channel_indices
        self.margin = margin
        self.mode = ensure_tuple_rep(mode, len(self.keys))

        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> List[Dict[Hashable, NdarrayOrTensor]]:
        class LabelSelector:
            def __init__(self, label) -> None:
                self.label = label

            def __call__(self, img) -> Any:
                return img == self.label

        d = dict(data)
        bg_label = self.bg_label

        preprocess_data: dict = d.get("preprocess_data")
        # initialize returned list with shallow copy to preserve key ordering
        results: List[Dict[Hashable, NdarrayOrTensor]] = [dict(d) for _ in range(len(preprocess_data.keys()))]

        for i, fg_l in enumerate(preprocess_data.keys()):
            cropper = CropForeground(
                select_fn=LabelSelector(fg_l), channel_indices=self.channel_indices, margin=self.margin,
            )
            pd = preprocess_data[fg_l]
            new_width, new_height, new_depth = self.image_size
            # label裁剪出来
            centers = generate_pos_neg_label_crop_centers(spatial_size=[new_width, new_height, new_depth],
                                                          num_samples=1,
                                                          pos_ratio=1,  # 比例大于1，确保永远只会取到前景 # 可以，但没必要
                                                          label_spatial_shape=pd.get("label_original_shape"),
                                                          fg_indices=pd.get("fg_indices_"),
                                                          bg_indices=[None],
                                                          rand_state=None,
                                                          allow_smaller=False)
            center = centers[0]
            box_start = np.array([center[0] - new_width // 2, center[1] - new_height // 2, center[2] - new_depth // 2])
            box_end = np.array([center[0] + new_width // 2, center[1] + new_height // 2, center[2] + new_depth // 2])

            # fill in the extra keys with unmodified data
            for key in set(d.keys()).difference(set(self.keys)):
                results[i][key] = deepcopy(d[key])

            for key, m in self.key_iterator(d, self.mode):
                patch = cropper.crop_pad(img=d[key], box_start=box_start, box_end=box_end, mode=m)

                # When crop label volume, clean the other labels in current crop patch
                if key == self.label_key:
                    # Warning, the CropForeground may copy(if padding) or inference original volume as result patch
                    # Warning, we should copy patch first, or it will modify and corrupt original label values
                    patch = deepcopy(patch)
                    # patch[patch != fg_l] = bg_label
                    patch[patch == fg_l] = -1
                    patch[(patch != -1) & (patch != bg_label)] = 2
                    patch[patch == -1] = 1
                    if self.to_same_fg_label is not None:
                        patch[patch == fg_l] = self.to_same_fg_label

                results[i][key] = patch
                orig_size = d[key].shape[1:]
                self.push_transform(results[i], key, extra_info={"box_start": box_start, "box_end": box_end},
                                    orig_size=orig_size)

            # add `patch_index` to the meta data
            for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key not in results[i]:
                    results[i][meta_key] = {}  # type: ignore
                results[i][meta_key][Key.PATCH_INDEX] = i  # type: ignore
                results[i][key].meta[Key.PATCH_INDEX] = i  # 更新兼容meta tensor的patch index
        # print(f"裁剪样本数量: {d['image_meta_dict']['filename_or_obj']} {len(results)}")
        return results

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            orig_size = np.asarray(transform[TraceKeys.ORIG_SIZE])
            cur_size = np.asarray(d[key].shape[1:])
            extra_info = transform[TraceKeys.EXTRA_INFO]
            box_start = np.asarray(extra_info["box_start"])
            box_end = np.asarray(extra_info["box_end"])
            # first crop the padding part
            roi_start = np.maximum(-box_start, 0)
            roi_end = cur_size - np.maximum(box_end - orig_size, 0)

            d[key] = SpatialCrop(roi_start=roi_start, roi_end=roi_end)(d[key])

            # update bounding box to pad
            pad_to_start = np.maximum(box_start, 0)
            pad_to_end = orig_size - np.minimum(box_end, orig_size)
            # interleave mins and maxes
            pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
            # second pad back the original size
            d[key] = BorderPad(pad)(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class CropForegroundSamplesByBBox(MapTransform, InvertibleTransform):
    """
    根据label确定前景，然后根据前景的bbox裁剪出来
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
            self,
            keys: KeysCollection,
            label_key: str,
            fg_labels: Sequence = None,
            bg_label: Union[int, float] = 0,
            to_same_fg_label: Union[int, float] = None,
            channel_indices: Optional[IndexSelection] = None,
            margin: Union[Sequence[int], int] = 0,
            mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = NumpyPadMode.CONSTANT,
            meta_keys: Optional[KeysCollection] = None,
            meta_key_postfix: str = "meta_dict",
            allow_missing_keys: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.label_key = label_key
        self.fg_labels = fg_labels
        self.bg_label = bg_label
        self.to_same_fg_label = to_same_fg_label

        self.channel_indices = channel_indices
        self.margin = margin
        self.mode = ensure_tuple_rep(mode, len(self.keys))

        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> List[Dict[Hashable, NdarrayOrTensor]]:
        def find_foreground_labels(label: NdarrayOrTensor, bg_label: Union[int, float] = 0):
            all_labels = None
            if isinstance(label, np.ndarray):
                all_labels = np.unique(label)
            elif isinstance(label, torch.Tensor):
                all_labels = torch.unique(label)
            else:
                raise RuntimeError("Unsupported label type: ", type(label))

            fg_labels = []
            for i in range(all_labels.shape[0]):
                if all_labels[i] == bg_label:
                    continue
                fg_labels.append(all_labels[i])

            return fg_labels

        class LabelSelector:
            def __init__(self, label) -> None:
                self.label = label

            def __call__(self, img) -> Any:
                return img == self.label

        d = dict(data)
        bg_label = self.bg_label
        fg_labels = find_foreground_labels(d[self.label_key], bg_label) if self.fg_labels is None else self.fg_labels

        # initialize returned list with shallow copy to preserve key ordering
        results: List[Dict[Hashable, NdarrayOrTensor]] = [dict(d) for _ in range(len(fg_labels))]

        for i, fg_l in enumerate(fg_labels):
            cropper = CropForeground(
                select_fn=LabelSelector(fg_l), channel_indices=self.channel_indices, margin=self.margin,
            )
            box_start, box_end = cropper.compute_bounding_box(img=d[self.label_key])

            # fill in the extra keys with unmodified data
            for key in set(d.keys()).difference(set(self.keys)):
                results[i][key] = deepcopy(d[key])

            for key, m in self.key_iterator(d, self.mode):
                patch = cropper.crop_pad(img=d[key], box_start=box_start, box_end=box_end, mode=m)

                # When crop label volume, clean the other labels in current crop patch
                if key == self.label_key:
                    # Warning, the CropForeground may copy(if padding) or inference original volume as result patch
                    # Warning, we should copy patch first, or it will modify and corrupt original label values
                    patch = deepcopy(patch)
                    patch[patch != fg_l] = bg_label
                    if self.to_same_fg_label is not None:
                        patch[patch == fg_l] = self.to_same_fg_label

                results[i][key] = patch
                orig_size = d[key].shape[1:]
                self.push_transform(results[i], key, extra_info={"box_start": box_start, "box_end": box_end},
                                    orig_size=orig_size)

            # add `patch_index` to the meta data
            for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key not in results[i]:
                    results[i][meta_key] = {}  # type: ignore
                results[i][meta_key][Key.PATCH_INDEX] = i  # type: ignore
                results[i][key].meta[Key.PATCH_INDEX] = i  # 更新兼容meta tensor的patch index

        return results

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            orig_size = np.asarray(transform[TraceKeys.ORIG_SIZE])
            cur_size = np.asarray(d[key].shape[1:])
            extra_info = transform[TraceKeys.EXTRA_INFO]
            box_start = np.asarray(extra_info["box_start"])
            box_end = np.asarray(extra_info["box_end"])
            # first crop the padding part
            roi_start = np.maximum(-box_start, 0)
            roi_end = cur_size - np.maximum(box_end - orig_size, 0)

            d[key] = SpatialCrop(roi_start=roi_start, roi_end=roi_end)(d[key])

            # update bounding box to pad
            pad_to_start = np.maximum(box_start, 0)
            pad_to_end = orig_size - np.minimum(box_end, orig_size)
            # interleave mins and maxes
            pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
            # second pad back the original size
            d[key] = BorderPad(pad)(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class CropToothClassificationInstance(MapTransform):
    """
    裁剪牙齿分类数据
    """

    # def __init__(self):
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> List[Dict[Hashable, NdarrayOrTensor]]:
        # info文件
        info: Dict[str, Dict[str, Any]] = json.loads(Path(data['info']).read_bytes())
        result = []
        # 图像是re spacing之后的，bbox也要对应改变
        origin_spacing = data["image"].meta["pixdim"][1:4].tolist()
        ratio = origin_spacing[0] / 0.25, origin_spacing[1] / 0.25, origin_spacing[2] / 0.25
        for tooth_label, tooth_info in info.items():
            bbox = tooth_info["bbox"]
            new_bbox = [int(ratio[0] * bbox[0]), int(ratio[1] * bbox[1]), int(ratio[2] * bbox[2]), int(ratio[0] * bbox[3]), int(ratio[1] * bbox[4]),
                        int(ratio[2] * bbox[5])]
            width = new_bbox[3] - new_bbox[0]
            height = new_bbox[4] - new_bbox[1]
            depth = new_bbox[5] - new_bbox[2]
            croper = SpatialCrop(roi_start=[new_bbox[0], new_bbox[1], new_bbox[2]],
                                 roi_end=[new_bbox[3], new_bbox[4], new_bbox[5]])
            patch = croper(data['image'])
            result.append({
                "image": patch,
                "label": torch.nn.functional.one_hot(torch.tensor(np.array(int(tooth_label) - 1)), num_classes=32).float(),
            })
        return result


class RandomElasticDeformation(RandomizableTransform, MapTransform):
    """
    3D弹性变形
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
            self,
            keys: KeysCollection,
            prob: float = 0.5,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys=False)
        self.transform = torchio.RandomElasticDeformation(max_displacement=50)
        self.prob = prob

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> List[Dict[Hashable, NdarrayOrTensor]]:
        if random.random() < self.prob:
            d = dict(data)
            for key in self.key_iterator(d):
                d[key] = self.transform(d[key])
            return d
        else:
            return data


class MyRandCrop(RandomizableTransform, MapTransform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: KeysCollection):
        MapTransform.__init__(self, keys, False)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> List[Dict[Hashable, NdarrayOrTensor]]:
        d = dict(data)
        rand_margin_1 = [random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)]
        rand_margin_2 = [random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)]
        for key in self.key_iterator(d):
            bbox_start = np.array(rand_margin_1)
            bbox_end = np.array(d[key].shape[1:]) - np.array(rand_margin_2)
            cropper = CropForeground(channel_indices=None)
            d[key] = cropper.crop_pad(img=d[key], box_start=bbox_start, box_end=bbox_end, mode="constant")

        return d


class RandCropForegroundSamplesByBBox(RandomizableTransform, MapTransform, InvertibleTransform):
    """
    根据label确定前景，然后根据前景的bbox裁剪出来，裁剪的时候随机外扩
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
            self,
            keys: KeysCollection,
            label_key: str,
            fg_labels: Sequence = None,
            bg_label: Union[int, float] = 0,
            to_same_fg_label: Union[int, float] = None,
            channel_indices: Optional[IndexSelection] = None,
            margin: Union[Sequence[int], int] = 0,
            mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = NumpyPadMode.CONSTANT,
            meta_keys: Optional[KeysCollection] = None,
            meta_key_postfix: str = "meta_dict",
            allow_missing_keys: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.label_key = label_key
        self.fg_labels = fg_labels
        self.bg_label = bg_label
        self.to_same_fg_label = to_same_fg_label

        self.channel_indices = channel_indices
        self.margin = margin
        self.mode = ensure_tuple_rep(mode, len(self.keys))

        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> List[Dict[Hashable, NdarrayOrTensor]]:
        def find_foreground_labels(label: NdarrayOrTensor, bg_label: Union[int, float] = 0):
            all_labels = None
            if isinstance(label, np.ndarray):
                all_labels = np.unique(label)
            elif isinstance(label, torch.Tensor):
                all_labels = torch.unique(label)
            else:
                raise RuntimeError("Unsupported label type: ", type(label))

            fg_labels = []
            for i in range(all_labels.shape[0]):
                if all_labels[i] == bg_label:
                    continue
                fg_labels.append(all_labels[i])

            return fg_labels

        class LabelSelector:
            def __init__(self, label) -> None:
                self.label = label

            def __call__(self, img) -> Any:
                return img == self.label

        d = dict(data)
        bg_label = self.bg_label
        fg_labels = find_foreground_labels(d[self.label_key], bg_label) if self.fg_labels is None else self.fg_labels

        # initialize returned list with shallow copy to preserve key ordering
        results: List[Dict[Hashable, NdarrayOrTensor]] = [dict(d) for _ in range(len(fg_labels))]

        for i, fg_l in enumerate(fg_labels):
            rand_margin = random.randint(0, self.margin)
            cropper = CropForeground(
                select_fn=LabelSelector(fg_l), channel_indices=self.channel_indices, margin=rand_margin,
            )
            box_start, box_end = cropper.compute_bounding_box(img=d[self.label_key])
            rand_offset1 = [random.randint(-rand_margin, rand_margin), random.randint(-rand_margin, rand_margin), random.randint(-rand_margin, rand_margin)]
            rand_offset2 = [random.randint(-rand_margin, rand_margin), random.randint(-rand_margin, rand_margin), random.randint(-rand_margin, rand_margin)]
            box_start = box_start + rand_offset1
            box_end = box_end + rand_offset2
            # fill in the extra keys with unmodified data
            for key in set(d.keys()).difference(set(self.keys)):
                results[i][key] = deepcopy(d[key])

            for key, m in self.key_iterator(d, self.mode):
                patch = cropper.crop_pad(img=d[key], box_start=box_start, box_end=box_end, mode=m)

                # When crop label volume, clean the other labels in current crop patch
                if key == self.label_key:
                    # Warning, the CropForeground may copy(if padding) or inference original volume as result patch
                    # Warning, we should copy patch first, or it will modify and corrupt original label values
                    patch = deepcopy(patch)
                    patch[patch != fg_l] = bg_label
                    if self.to_same_fg_label is not None:
                        patch[patch == fg_l] = self.to_same_fg_label

                results[i][key] = patch
                orig_size = d[key].shape[1:]
                self.push_transform(results[i], key, extra_info={"box_start": box_start, "box_end": box_end},
                                    orig_size=orig_size)

            # add `patch_index` to the meta data
            for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key not in results[i]:
                    results[i][meta_key] = {}  # type: ignore
                results[i][meta_key][Key.PATCH_INDEX] = i  # type: ignore
                results[i][key].meta[Key.PATCH_INDEX] = i  # 更新兼容meta tensor的patch index

        return results

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            orig_size = np.asarray(transform[TraceKeys.ORIG_SIZE])
            cur_size = np.asarray(d[key].shape[1:])
            extra_info = transform[TraceKeys.EXTRA_INFO]
            box_start = np.asarray(extra_info["box_start"])
            box_end = np.asarray(extra_info["box_end"])
            # first crop the padding part
            roi_start = np.maximum(-box_start, 0)
            roi_end = cur_size - np.maximum(box_end - orig_size, 0)

            d[key] = SpatialCrop(roi_start=roi_start, roi_end=roi_end)(d[key])

            # update bounding box to pad
            pad_to_start = np.maximum(box_start, 0)
            pad_to_end = orig_size - np.minimum(box_end, orig_size)
            # interleave mins and maxes
            pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
            # second pad back the original size
            d[key] = BorderPad(pad)(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d
