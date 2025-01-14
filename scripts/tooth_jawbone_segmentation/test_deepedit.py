import os
import matplotlib.pyplot as plt
import numpy as np
import torch

import monai
from monai.config import print_config

from monai.apps.deepedit.transforms import (
    AddGuidanceSignalDeepEditd,
    AddGuidanceFromPointsDeepEditd,
    ResizeGuidanceMultipleLabelDeepEditd,
)

from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRanged,
    SqueezeDimd,
    ToNumpyd,
    ToTensord,
)

from monai.networks.nets import DynUNet
from scripts.tooth_jawbone_segmentation.config_deepedit import scale_intensity_range


def draw_points(guidance, slice_idx):
    if guidance is None:
        return
    for p in guidance:
        p1 = p[1]
        p2 = p[0]
        plt.plot(p1, p2, "r+", 'MarkerSize', 30)


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


# labels
labels = {
    "tooth": 1,
    "down": 2,
    "up": 3,
    # background必须写在最后一个位置，写在前面,NormalizeLabelsInDatasetd处理标签的时候会导致标签不连续，无法训练。我真是服了这个老6
    "background": 0,
}

spatial_size = [128, 128, 128]

model = DynUNet(
    spatial_dims=3,
    in_channels=len(labels) + 1,
    out_channels=len(labels),
    kernel_size=[3, 3, 3, 3, 3, 3],
    strides=[1, 2, 2, 2, 2, [2, 2, 1]],
    upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
    norm_name="instance",
    deep_supervision=False,
    res_block=True,
)

data = {
    'image': '/home/yujiannan/Projects/MONAI/data/tooth_jawbone_segmentation/310606_f_cbct_image.nii.gz',
    'guidance': {
        'down': [
            [186, 180, 80], [186, 180, 100]
        ], 'background': [
            [66, 180, 105], [66, 180, 145]
        ]},
}
# for x in range(0, 400, 10):
#     for y in range(0, 400, 10):
#         for z in range(0, 400, 10):
#             data["guidance"]["background"].append([x, y, z])
slice_idx = original_slice_idx = data['guidance']['down'][0][2]

# Pre Processing

pre_transforms = [
    # Loading the image
    LoadImaged(keys="image"),
    # Ensure channel first
    EnsureChannelFirstd(keys="image"),
    # Change image orientation
    Orientationd(keys="image", axcodes="RAS"),
    # Scaling image intensity - works well for CT images
    scale_intensity_range,
    # DeepEdit Tranforms for Inference #
    # Add guidance (points) in the form of tensors based on the user input
    AddGuidanceFromPointsDeepEditd(ref_image="image", guidance="guidance", label_names=labels),
    # Resize the image
    Resized(keys="image", spatial_size=spatial_size, mode="area"),
    # Resize the guidance based on the image resizing
    ResizeGuidanceMultipleLabelDeepEditd(guidance="guidance", ref_image="image"),
    # Add the guidance to the input image
    AddGuidanceSignalDeepEditd(keys="image", guidance="guidance"),
    # Convert image to tensor
    ToTensord(keys="image"),
]

original_image = None

# Going through each of the pre_transforms

for t in pre_transforms:
    tname = type(t).__name__
    data = t(data)
    image = data['image']
    label = data.get('label')
    guidance = data.get('guidance')

    print("{} => image shape: {}".format(tname, image.shape))

    if tname == 'LoadImaged':
        original_image = data['image']
        label = None
        tmp_image = image[:, :, slice_idx]
        show_image(tmp_image, label, [guidance['down'][0]], slice_idx)

transformed_image = data['image']
guidance = data.get('guidance')
# Evaluation
model_path = '/home/yujiannan/Projects/MONAI/scripts/tooth_jawbone_segmentation/logs/deepedit/1/net_key_metric=0.9217.pt'
model.load_state_dict(torch.load(model_path))
model.cuda()
model.eval()

inputs = data['image'][None].cuda()
with torch.no_grad():
    outputs = model(inputs)
outputs = outputs[0]
data['pred'] = outputs

post_transforms = [
    EnsureTyped(keys="pred"),
    Activationsd(keys="pred", softmax=True),
    AsDiscreted(keys="pred", argmax=True),
    SqueezeDimd(keys="pred", dim=0),
    ToNumpyd(keys="pred"),
]

pred = None
for t in post_transforms:
    tname = type(t).__name__
    data = t(data)
    image = data['image']
    label = data['pred']
    print("{} => image shape: {}, pred shape: {}".format(tname, image.shape, label.shape))

for i in range(10, 110, 10):
    image = transformed_image[0, :, :, i]  # Taking the first channel which is the main image
    label = data['pred'][:, :, i]
    if np.sum(label) == 0:
        continue

    print("Final PLOT:: {} => image shape: {}, pred shape: {}; min: {}, max: {}, sum: {}".format(
        i, image.shape, label.shape, np.min(label), np.max(label), np.sum(label)))
    show_image(image, label)
