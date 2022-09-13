from pathlib import Path
import numpy as np
from typing import List, Dict


def get_project_dir() -> Path:
    """
    获取项目目录
    :return:
    """
    project_dir = Path(__file__).resolve().parent.parent
    return project_dir


def get_log_dir() -> Path:
    """
    获取日志目录
    :return:
    """
    return get_project_dir().joinpath("logs")


def get_data_dir() -> Path:
    """
    获取数据目录
    :return:
    """
    return get_project_dir().joinpath("data")


def get_cache_data_dir() -> Path:
    return get_data_dir().joinpath("temp")


def get_scripts_dir() -> Path:
    """
    获取脚本目录
    :return:
    """
    scripts_dir = Path(__file__).resolve().parent
    return scripts_dir


def get_scripts_data(file: str) -> Path:
    """
    获取脚本数据
    :param file: 数据名词
    :return:
    """
    scripts_data_dir = get_scripts_dir().joinpath("data")
    return scripts_data_dir.joinpath(file)


def get_scripts_output(file: str) -> Path:
    """
    获取脚本输出
    :param file:
    :return:
    """
    scripts_output_dir = get_scripts_dir().joinpath("output")
    scripts_output_dir.mkdir(exist_ok=True, parents=True)
    return scripts_output_dir.joinpath(file)


def get_model(file: str) -> Path:
    """
    获取模型，用于预训练或者存储之类的
    :param file:
    :return:
    """
    return get_project_dir().joinpath("models").joinpath(file)


def normalize_image_to_uint8(image):
    """
    图片正则化到uint8
    """
    draw_img = image
    if np.amin(draw_img) < 0:
        draw_img -= np.amin(draw_img)
    if np.amax(draw_img) > 1:
        draw_img /= np.amax(draw_img)
    draw_img = (255 * draw_img).astype(np.uint8)
    return draw_img


def load_image_label_pair_dataset(d: Path) -> List[Dict[str, str]]:
    """
    加载数据集
    数据是*image*格式
    标签是*label*格式
    :param d:
    :return:
    """
    dataset = []
    for file in d.iterdir():
        if "image" in file.name:
            label_file = file.parent.joinpath(file.name.replace("image", "label"))
            dataset.append({
                "image": str(file),
                "label": str(label_file)
            })
    dataset = list(sorted(dataset, key=lambda x: x["image"]))
    return dataset


if __name__ == '__main__':
    print(get_scripts_data("image.nii.gz"))
