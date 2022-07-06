from dataclasses import dataclass

import numpy as np


@dataclass
class Image3D:
    """
    3D 图像
    """
    data: np.ndarray  # 图像数据
    meta_dict: dict  # 图像元数据

    @property
    def shape(self):
        """
        获取图像的shape
        :return:
        """
        return self.data.shape


@dataclass
class Point3D:
    """
    空间中的点
    """
    x: float
    y: float
    z: float


@dataclass
class BBox3D:
    """
    3D BBOX
    """
    point1: Point3D  # 左下角点，暂时不考虑两个点的位置关系
    point2: Point3D  # 右上角点，暂时不考虑两个点的位置关系


@dataclass
class DetectResult3D:
    """
    3D 检测结果
    """
    bbox: BBox3D
    category: int = 0  # 类别
    label: str = "unknown"  # 标签
