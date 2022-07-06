"""
各种数据类型转换
"""
from typing import Optional, Tuple, Sequence

import numpy as np

from scripts import get_scripts_data
from scripts.image import read_image


def instances_to_boxes_np(
        seg: np.ndarray,
        dim: int = None,
        instances: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    f"""
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
    :param seg {np}
    :example:
        >>> seg = np.array([[[0, 0, 0, 0, 0],
        >>>                  [0, 0, 0, 0, 0],
        >>>                  [0, 0, 0, 0, 0],
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
    image = read_image(get_scripts_data("image.dcm"))
    label = read_image(get_scripts_data("label1.dcm"))
    print(instances_to_boxes_np(label.data))
    a = 1