import SimpleITK as sitk

itk_img = sitk.ReadImage('/home/yujiannan/桌面/325890_l_cbct.dcm')
img = sitk.GetArrayFromImage(itk_img)
print("img shape:", img.shape)

## save
out = sitk.GetImageFromArray(img)
# out.SetSpacing(itk_img.GetSpacing())
out.SetSpacing([0.25, 0.25, 0.25])
sitk.WriteImage(out, '/home/yujiannan/桌面/output.nii.gz')
