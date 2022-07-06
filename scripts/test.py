import pydicom

dicom_file = "/home/yujiannan/桌面/325890_l_cbct.dcm"
output_file = "/home/yujiannan/桌面/output.dcm"
data = pydicom.dcmread(dicom_file)
# data.update({
#     "PixelSpacing"
# })
data.ImagePositionPatient = [0.25,0.25,0.25]
pydicom.dcmwrite(output_file, data)
new_data = pydicom.dcmread(output_file)
print(new_data["PixelSpacing"].value)
