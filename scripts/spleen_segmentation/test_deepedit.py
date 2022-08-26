# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
labels = {'spleen': 1,
          'background': 0
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

# Download data and model

resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/_image.nii.gz"
dst = "_image.nii.gz"

if not os.path.exists(dst):
    monai.apps.download_url(resource, dst)

resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/\
download/0.8.1/pretrained_deepedit_dynunet-final.pt"
dst = "pretrained_deepedit_dynunet-final.pt"

if not os.path.exists(dst):
    monai.apps.download_url(resource, dst)

data = {
    'image': '_image.nii.gz',
    'guidance': {'spleen': [[66, 180, 105], [66, 180, 145]], 'background': []},
}

for x in range(0, 500, 25):
    for y in range(0, 500, 25):
        data['guidance']['spleen'].append([x, y, 100])

slice_idx = original_slice_idx = data['guidance']['spleen'][0][2]

# Pre Processing

pre_transforms = [
    # Loading the image
    LoadImaged(keys="image", reader="ITKReader"),
    # Ensure channel first
    EnsureChannelFirstd(keys="image"),
    # Change image orientation
    Orientationd(keys="image", axcodes="RAS"),
    # Scaling image intensity - works well for CT images
    ScaleIntensityRanged(keys="image", a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
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
        show_image(tmp_image, label, [guidance['spleen'][0]], slice_idx)

transformed_image = data['image']
guidance = data.get('guidance')
# Evaluation
model_path = 'pretrained_deepedit_dynunet-final.pt'
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

for i in range(100, 110, 10):
    image = transformed_image[0, :, :, i]  # Taking the first channel which is the main image
    label = data['pred'][:, :, i]
    # if np.sum(label) == 0:
    #     continue

    print("Final PLOT:: {} => image shape: {}, pred shape: {}; min: {}, max: {}, sum: {}".format(
        i, image.shape, label.shape, np.min(label), np.max(label), np.sum(label)))
    show_image(image, label)
