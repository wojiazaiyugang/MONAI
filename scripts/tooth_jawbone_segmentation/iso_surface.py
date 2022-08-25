
from typing import Optional, Union, Dict, Tuple, List

import vtk
from vtk.util.numpy_support import vtk_to_numpy

from vtkmodules.vtkFiltersCore import (
    vtkFlyingEdges3D,
    vtkMarchingCubes,
vtkWindowedSincPolyDataFilter,
vtkPolyDataNormals
)

from vtkmodules.vtkCommonCore import (
    VTK_VERSION_NUMBER,
    vtkVersion
)

import itk
import torch
import numpy as np
from scripts.teeth_jawbone_segmentation.image_writer import ITKWriter


def numpy_to_itk_image(
        img: Union[torch.Tensor, np.ndarray],
        channel_dim: int = None,
        affine_lps_to_ras: bool = True,
        meta_data: Optional[Dict] = None
):
    writer = ITKWriter(output_dtype=np.float32, affine_lps_to_ras=affine_lps_to_ras)
    writer.set_data_array(img, channel_dim=channel_dim)
    writer.set_metadata(meta_data, resample=False)
    itk_image = writer.create_itk_image()
    return itk_image


def extract_itk_image_isosurface(itk_image, iso_value=0):
    # itk to vtk
    vtk_image = itk.vtk_image_from_image(itk_image)
    poly_data = extract_vtk_image_isosurface(vtk_image, iso_value)
    verts, faces = poly_data_to_numpy(poly_data)
    return verts, faces


def extract_vtk_image_isosurface(vtk_image, iso_value=0):
    """
    Ref: https://kitware.github.io/vtk-examples/site/Python/Modelling/MarchingCubes/
    """
    # vtkFlyingEdges3D was introduced in VTK >= 8.2
    use_flying_edges = vtk_version_ok(8, 2, 0)

    if use_flying_edges:
        try:
            surface = vtkFlyingEdges3D()
        except AttributeError:
            surface = vtkMarchingCubes()
    else:
        surface = vtkMarchingCubes()
    # smooth = vtkWindowedSincPolyDataFilter()
    # smooth.SetInputData(vtk_image)
    # smooth.SetNumberOfIterations(20)
    # smooth.SetPassBand(0.01)
    # smooth.BoundarySmoothingOff()
    # smooth.NonManifoldSmoothingOn()
    # smooth.NormalizeCoordinatesOn()
    # normal_generator =vtkPolyDataNormals()
    # normal_generator.ConsistencyOn()
    # normal_generator.SplittingOff()
    # normal_generator.SetInputData(smooth.GetOutput())

    # surface = vtk.vtkDiscreteMarchingCubes()
    surface.SetInputData(vtk_image)
    surface.ComputeNormalsOn()
    surface.SetValue(0, iso_value)
    surface.Update()

    poly_data = surface.GetOutput()
    return poly_data

def vtk_version_ok(major, minor, build):
    """
    Check the VTK version.

    :param major: Major version.
    :param minor: Minor version.
    :param build: Build version.
    :return: True if the requested VTK version is greater or equal to the actual VTK version.
    """
    needed_version = 10000000000 * int(major) + 100000000 * int(minor) + int(build)
    try:
        vtk_version_number = VTK_VERSION_NUMBER
    except AttributeError:  # as error:
        ver = vtkVersion()
        vtk_version_number = 10000000000 * ver.GetVTKMajorVersion() + 100000000 * ver.GetVTKMinorVersion() \
                             + ver.GetVTKBuildVersion()
    if vtk_version_number >= needed_version:
        return True
    else:
        return False

def poly_data_to_numpy(poly_data):
    """
    Ref: https://stackoverflow.com/questions/51201888/retrieving-facets-and-point-from-vtk-file-in-python
    """

    points = poly_data.GetPoints()
    array = points.GetData()
    verts = vtk_to_numpy(array)

    cells = poly_data.GetPolys()
    n_cells = cells.GetNumberOfCells()
    array = cells.GetData()
    # This holds true if all polys are of the same kind, e.g. triangles.
    assert(array.GetNumberOfValues() % n_cells==0)
    n_cols = array.GetNumberOfValues() // n_cells
    assert n_cols == 4

    faces = vtk_to_numpy(array)
    faces = faces.reshape((-1, n_cols))
    faces = faces[:, 1:]

    return verts, faces

if __name__ == '__main__':
    from monai.transforms import LoadImaged, Compose
    import open3d as o3d
    import mcubes

    transformers = Compose([
        LoadImaged(keys=["label"])
    ])
    data = transformers([{"label": "/home/yujiannan/Projects/MONAI/data/unetr_seg/wangzhiguo_g_cbct_label.nii.gz"}])
    xyz, meta_dict = data[0]["label"], data[0]["label_meta_dict"]
    # xyz[xyz != 3] = 0
    xyz[xyz != 0] = 1

    # itk_image = numpy_to_itk_image(xyz, affine_lps_tmcubes.export_mesh(vertices, triangles, "sphere.dae", "MySphere")o_ras=False, meta_data=meta_dict)
    # verts, faces = extract_itk_image_isosurface(itk_image, 0.5)
    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(verts)
    # mesh.triangles = o3d.utility.Vector3iVector(faces)
    # o3d.visualization.draw_geometries([mesh])
    # o3d.io.write_triangle_mesh(f"test.ply", mesh, write_ascii=True)

    xyz = mcubes.smooth(xyz)
    verts, faces = mcubes.marching_cubes(xyz, 0.5)
    mcubes.export_obj(verts, faces, "test.obj")
