import time

import mcubes
import numpy as np
import torch

from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.transforms import SaveImageD, Compose, LoadImaged, AddChanneld, Orientationd, Spacingd, CropForegroundd, \
    ToNumpyd, FromMetaTensord
from scripts import get_log_dir
from scripts.single_tooth_segmentation.config import work_dir, scale_intensity_range, SPACING
from scripts.single_tooth_segmentation.train import get_model
from scripts.transforms import SetAttrd



def get_inference_transformer() -> Compose:
    """
    获取测试transform
    :return:
    """
    transformer = Compose([
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=SPACING),
        scale_intensity_range,
        CropForegroundd(keys=["image"], source_key="image"),
        FromMetaTensord(keys=["image"]),
        ToNumpyd(keys=["image"]),
        # ToTensord(keys=["image"])
    ])
    return transformer


def get_post_transformer() -> Compose:
    """
    获取测试transform
    :return:
    """
    transformer = Compose([
        SaveImageD(keys="label",
                   output_dir=work_dir.joinpath("result"),
                   separate_folder=False,
                   output_postfix="",
                   print_log=False)
    ])
    return transformer


if __name__ == '__main__':
    data = [
        # {"image": "/media/3TB/data/xiaoliutech/relu_cbct_respacing/326923_n_cbct.dcm/326923_n_cbct.nii.gz"},
        {"image": "/home/yujiannan/Projects/MONAI/data/unetr_seg/saassd_d_cbct_image.nii.gz"}
    ]
    dataset = Dataset(data=data,
                      transform=get_inference_transformer())
    data_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available(), collate_fn=lambda x: x)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    model.load_state_dict(torch.load(get_log_dir().joinpath("unetr_seg").joinpath("1").joinpath("best_metric_model.pth")))
    model.eval()
    post_transformer = get_post_transformer()
    with torch.no_grad():
        for batch_data in data_loader:
            t = time.time()
            inference_data = torch.as_tensor(np.array([data["image"] for data in batch_data])).cuda()
            inference_outputs = sliding_window_inference(
                inference_data, (96, 96, 96), 4, model, overlap=0.8
            )
            print(f"inference time: {time.time() - t}")
            post_inference_output = torch.argmax(inference_outputs, dim=1).detach().cpu().numpy()
            if 1:
                for i in range(1, 4):
                    output = inference_outputs[0][i]
                    xyz = post_inference_output[0].copy()
                    xyz[xyz != i] = 0

                    xyz = mcubes.smooth(xyz)
                    verts, faces = mcubes.marching_cubes(xyz, 0.5)
                    mcubes.export_obj(verts, faces, f"mcubes_{i}.obj")

                    # itk_image = numpy_to_itk_image(output, affine_lps_to_ras=False, meta_data=batch_data[0]["image_meta_dict"])
                    # verts, faces = extract_itk_image_isosurface(itk_image, np.average(output.cpu().numpy()))
                    # mesh = o3d.geometry.TriangleMesh()
                    # mesh.vertices = o3d.utility.Vector3dVector(verts)
                    # mesh.triangles = o3d.utility.Vector3iVector(faces)
                    # o3d.visualization.draw_geometries([mesh])
                    # o3d.io.write_triangle_mesh(f"{i}.ply", mesh, write_ascii=True)
                    # igl.write_triangle_mesh("isosurface.ply", verts, faces, force_ascii=False)
                    # open3d.
                    # open3d.cpu.pybind.io.write_triangle_mesh()
                    # xyz = np.argwhere(post_inference_output[0] == i)
                    # pcd = o3d.geometry.PointCloud()
                    # pcd.points = o3d.utility.Vector3dVector(xyz)
                    # o3d.visualization.draw_geometries([pcd])
                    # o3d.io.write_point_cloud(f"./{i}.ply", pcd, write_ascii=True)
            transformer = Compose([
                SetAttrd(keys="pred", value=post_inference_output, meta_dict_key="image_meta_dict"),
                SaveImageD(keys="pred", output_dir=work_dir.joinpath("result"), separate_folder=False,
                           output_postfix="")
            ])
            transformer(batch_data)
