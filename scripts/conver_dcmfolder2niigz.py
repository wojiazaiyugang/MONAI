"""
DCM数据转换为nii.gz文件，同时读取对应的原始数据用于调整Spacing
"""
from typing import Tuple
from pathlib import Path

import SimpleITK as sitk
from tqdm import tqdm

tuple3float = Tuple[float, float, float]


def get_dcm_folder_spacing_and_size(dcm_directory: Path) -> Tuple[tuple3float, tuple3float]:
    """
    读取dcm数据文件夹，获取dcm的spacing和size
    :param folder:
    :return:
    """
    # dcm_directory = "/media/3TB/data/xiaoliutech/原始dicom/001_mainbrand/s1/310606_Fussen"
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dcm_directory))
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dcm_directory), series_ids[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3D = series_reader.Execute()
    spacing = image3D.GetSpacing()
    size = image3D.GetSize()
    return spacing, size


if __name__ == '__main__':
    output_dataset = Path("/media/3TB/data/xiaoliutech/relu_cbct_respacing")
    from_dataset = Path("/media/3TB/data/xiaoliutech/relu_cbct_dataset")
    to_dataset = Path("/media/3TB/data/xiaoliutech/relu_cbct_respacing")
    original_dataset = Path("/media/3TB/data/xiaoliutech/原始dicom")
    for image in list(from_dataset.joinpath("images").rglob("*.dcm")):
        if to_dataset.joinpath(image.name).exists():
            # print(f"{image.name} already exists")
            continue
        label_dir_name_prefix = image.name.split("_")[0]
        label_dirs = list(from_dataset.joinpath("labels").rglob(f"{label_dir_name_prefix}*"))
        label_dirs = [label_dir for label_dir in label_dirs if label_dir.is_dir()]
        if len(label_dirs) == 0:
            print("标签数据异常", image.name, label_dirs)
            continue
        original_dirs = list(original_dataset.rglob(f"{label_dir_name_prefix}*"))
        original_dirs = [original_dir for original_dir in original_dirs if original_dir.is_dir()]
        if len(original_dirs) != 1:
            print(f"原始数据异常，{image.name}, {original_dirs}")
            continue
        spacing, size = get_dcm_folder_spacing_and_size(original_dirs[0])
        # print(f"写image{image.name}")
        itk_img = sitk.ReadImage(str(image))
        if itk_img.GetSize()[:3] != size[:3]:
            print(f"size不一致，{image.name}, {itk_img.GetSize()}, {size}")
            continue
        output_dir = output_dataset.joinpath(image.name)
        output_dir.mkdir(exist_ok=True, parents=True)
        itk_img.SetSpacing(list(spacing))
        print(f"{image}, gt spacing {spacing}， spacing {itk_img.GetSpacing()}")
        # continue
        sitk.WriteImage(itk_img, str(output_dir.joinpath(f"{image.stem}.nii.gz")))
        for label_file in label_dirs[0].iterdir():
            itk_img = sitk.ReadImage(str(label_file))
            if itk_img.GetSize()[:3] != size[:3]:
                print(f"label size不一致，{label_file.name}, {itk_img.GetSize()}, {size}")
                continue
            itk_img.SetSpacing(list(spacing))
            sitk.WriteImage(itk_img, str(output_dir.joinpath(f"{label_file.with_suffix('.nii.gz').name}")))
