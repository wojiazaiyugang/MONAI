from typing import Optional, Tuple, Sequence

import numpy as np


def instances_to_boxes_np(
    seg: np.ndarray,
    dim: int = None,
    instances: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert instance segmentation to bounding boxes (not batched)

    Args
    seg: instance segmentation of individual classes [..., dims]
    dim: number of spatial dimensions to create bounding box for
        (always start from the last dimension). If None, all dimensions are
        used

    Returns
        np.ndarray: bounding boxes
            (x1, y1, x2, y2, (z1, z2)) List[Tensor[N, dim * 2]]
        np.ndarray: tuple with classes for bounding boxes
    """
    if dim is None:
        dim = seg.ndim
    boxes = []
    if instances is None:
        instances = np.unique(seg)
        instances = instances[instances > 0]

    for _idx in instances:
        instance_idx = np.stack(np.nonzero(seg == _idx), axis=1)
        _mins = np.min(instance_idx[:, -dim:], axis=0)
        _maxs = np.max(instance_idx[:, -dim:], axis=0)

        box = [_mins[-dim] - 1, _mins[(-dim) + 1] - 1, _maxs[-dim] + 1, _maxs[(-dim) + 1] + 1]
        if dim > 2:
            box = box + [_mins[(-dim) + 2] - 1, _maxs[(-dim) + 2] + 1]
        boxes.append(np.array(box))

    if boxes:
        boxes = np.stack(boxes)
    else:
        boxes = np.array([[]])
    return boxes, instances

if __name__ == '__main__':
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