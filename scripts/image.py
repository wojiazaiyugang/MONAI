# 导入包
from pathlib import Path

import numpy as np

from monai.transforms import LoadImage
from scripts.data_class import Image3D


def read_image(file: Path) -> Image3D:
    f"""
    读取dcm、niigz等三维图像文件
    :param file {Path} 
    """
    data, meta_dict = LoadImage()(file)
    return Image3D(data=np.array(data), meta_dict=dict())


if __name__ == '__main__':
    data, meta_dict = LoadImage()(Path("/home/yujiannan/桌面/327233_p_cbct_trans.nii.gz"))
    a = 1