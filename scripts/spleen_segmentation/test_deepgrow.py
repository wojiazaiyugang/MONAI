import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import jit

import monai
from monai.apps.deepgrow.transforms import (
    AddGuidanceFromPointsd,
    AddGuidanceSignald,
    ResizeGuidanced,
    RestoreLabeld,
    SpatialCropGuidanced,
)
from monai.transforms import (
    AsChannelFirstd,
    Spacingd,
    LoadImaged,
    AddChanneld,
    NormalizeIntensityd,
    EnsureTyped,
    ToNumpyd,
    Activationsd,
    AsDiscreted,
    Resized
)

max_epochs = 1


def draw_points(guidance, slice_idx):
    if guidance is None:
        return
    colors = ['r+', 'b+']
    for color, points in zip(colors, guidance):
        for p in points:
            if p[0] != slice_idx:
                continue
            p1 = p[-1]
            p2 = p[-2]
            plt.plot(p1, p2, color, 'MarkerSize', 30)


def show_image(image, label, guidance=None, slice_idx=None):
    plt.figure("check", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(image, cmap="gray")

    if label is not None:
        masked = np.ma.masked_where(label == 0, label)
        plt.imshow(masked, 'jet', interpolation='none', alpha=0.7)

    draw_points(guidance, slice_idx)
    plt.colorbar()

    if label is not None:
        plt.subplot(1, 2, 2)
        plt.title("label")
        plt.imshow(label)
        plt.colorbar()
        # draw_points(guidance, slice_idx)
    plt.show()


def print_data(data):
    for k in data:
        v = data[k]

        d = type(v)
        if type(v) in (int, float, bool, str, dict, tuple):
            d = v
        elif hasattr(v, 'shape'):
            d = v.shape

        if k in ('image_meta_dict', 'label_meta_dict'):
            for m in data[k]:
                print('{} Meta:: {} => {}'.format(k, m, data[k][m]))
        else:
            print('Data key: {} = {}'.format(k, d))

# Download data and model

resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/_image.nii.gz"
dst = "_image.nii.gz"

if not os.path.exists(dst):
    monai.apps.download_url(resource, dst)

resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/deepgrow_3d.ts"
dst = "deepgrow_3d.ts"
if not os.path.exists(dst):
    monai.apps.download_url(resource, dst)

# Pre Processing
roi_size = [256, 256]
model_size = [128, 192, 192]
pixdim = (1.0, 1.0, 1.0)
dimensions = 3

data = {
    'image': '_image.nii.gz',
    'foreground': [[266, 180, 105]],
    'background': [],
}
# for x in range(0, 200, 25):
#     for y in range(0, 200, 25):
#         for z in range(0, 200, 25):
#             data['background'].append([x, y, z])

slice_idx = original_slice_idx = data['foreground'][0][2]

pre_transforms = [
    LoadImaged(keys='image'),
    AsChannelFirstd(keys='image'),
    Spacingd(keys='image', pixdim=pixdim, mode='bilinear'),
    AddGuidanceFromPointsd(ref_image='image', guidance='guidance', foreground='foreground', background='background',
                           dimensions=dimensions),
    AddChanneld(keys='image'),
    SpatialCropGuidanced(keys='image', guidance='guidance', spatial_size=roi_size),
    Resized(keys='image', spatial_size=model_size, mode='area'),
    ResizeGuidanced(guidance='guidance', ref_image='image'),
    NormalizeIntensityd(keys='image', subtrahend=208.0, divisor=388.0),
    AddGuidanceSignald(image='image', guidance='guidance'),
    EnsureTyped(keys='image')
]

original_image = None
for t in pre_transforms:
    tname = type(t).__name__
    data = t(data)
    image = data['image']
    label = data.get('label')
    guidance = data.get('guidance')

    print("{} => image shape: {}".format(tname, image.shape))

    guidance = guidance if guidance else [np.roll(data['foreground'], 1).tolist(), []]
    slice_idx = guidance[0][0][0] if guidance else slice_idx
    print('Guidance: {}; Slice Idx: {}'.format(guidance, slice_idx))
    if tname == 'Resized':
        continue

    image = image[:, :, slice_idx] if tname in ('LoadImaged') else image[slice_idx] if tname in (
        'AsChannelFirstd', 'Spacingd', 'AddGuidanceFromPointsd') else image[0][slice_idx]
    label = None

    # show_image(image, label, guidance, slice_idx)
    if tname == 'LoadImaged':
        original_image = data['image']
    if tname == 'AddChanneld':
        original_image_slice = data['image']
    if tname == 'SpatialCropGuidanced':
        spatial_image = data['image']

image = data['image']
label = data.get('label')
guidance = data.get('guidance')
for i in range(image.shape[1]):
    print('Slice Idx: {}'.format(i))
    # show_image(image[0][i], None, guidance, i)

# Evaluation
model_path = 'deepgrow_3d.ts'
model = jit.load(model_path)
model.cuda()
model.eval()

inputs = data['image'][None].cuda()
with torch.no_grad():
    outputs = model(inputs)
outputs = outputs[0]
data['pred'] = outputs

post_transforms = [
    Activationsd(keys='pred', sigmoid=True),
    AsDiscreted(keys='pred', threshold=0.5),
    ToNumpyd(keys='pred'),
    RestoreLabeld(keys='pred', ref_image='image', mode='nearest'),
]

pred = None
for t in post_transforms:
    tname = type(t).__name__

    data = t(data)
    image = data['image']
    label = data['pred']
    print("{} => image shape: {}, pred shape: {}; slice_idx: {}".format(tname, image.shape, label.shape, slice_idx))

    if tname in 'RestoreLabeld':
        pred = label

        image = original_image[:, :, original_slice_idx]
        label = label[original_slice_idx]
        print("PLOT:: {} => image shape: {}, pred shape: {}; min: {}, max: {}, sum: {}".format(
            tname, image.shape, label.shape, np.min(label), np.max(label), np.sum(label)))
        # show_image(image, label)
    elif tname == 'xToNumpyd':
        for i in range(label.shape[1]):
            img = image[0, i, :, :].detach().cpu().numpy() if torch.is_tensor(image) else image[0][i]
            lab = label[0, i, :, :].detach().cpu().numpy() if torch.is_tensor(label) else label[0][i]
            if np.sum(lab) > 0:
                print("PLOT:: {} => image shape: {}, pred shape: {}; min: {}, max: {}, sum: {}".format(
                    i, img.shape, lab.shape, np.min(lab), np.max(lab), np.sum(lab)))
                # show_image(img, lab)
    else:
        image = image[0, slice_idx, :, :].detach().cpu().numpy() if torch.is_tensor(image) else image[0][slice_idx]
        label = label[0, slice_idx, :, :].detach().cpu().numpy() if torch.is_tensor(label) else label[0][slice_idx]
        print("PLOT:: {} => image shape: {}, pred shape: {}; min: {}, max: {}, sum: {}".format(
            tname, image.shape, label.shape, np.min(label), np.max(label), np.sum(label)))
        # show_image(image, label)

for i in range(pred.shape[0]):
    i = 100
    image = original_image[:, :, i]
    label = pred[i, :, :]
    if np.sum(label) == 0:
        continue

    print("Final PLOT:: {} => image shape: {}, pred shape: {}; min: {}, max: {}, sum: {}".format(
        i, image.shape, label.shape, np.min(label), np.max(label), np.sum(label)))
    show_image(image, label)
    break

pred = data['pred']
meta_data = data['pred_meta_dict']
affine = meta_data.get("affine", None)

pred = np.moveaxis(pred, 0, -1)
print('Prediction NII shape: {}'.format(pred.shape))

# file_name = 'result_label.nii.gz'
# write_nifti(pred, file_name=file_name)
# print('Prediction saved at: {}'.format(file_name))