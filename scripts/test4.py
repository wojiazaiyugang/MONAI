import sys
import SimpleITK as sitk
import numpy as np

dcm_directory = "/media/3TB/data/xiaoliutech/原始dicom/001_mainbrand/s1/310606_Fussen"
series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dcm_directory)
series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dcm_directory, series_ids[0])
series_reader = sitk.ImageSeriesReader()
series_reader.SetFileNames(series_file_names)
image3D = series_reader.Execute()
image_array = sitk.GetArrayFromImage(image3D)
spacing = image3D.GetSpacing()
print("spacing:", spacing)
