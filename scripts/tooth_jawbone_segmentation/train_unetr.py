import os
import random
from typing import Tuple
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import monai.data
from monai.data import (
    DataLoader,
    CacheDataset,
    decollate_batch,
PersistentDataset
)
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    Spacingd,
    RandRotate90d,
    ToTensord
)
from scripts import get_data_dir
from scripts.tooth_jawbone_segmentation.config_unetr import SPACING, SCALE_INTENSITY_RANGE, IMAGE_SIZE, CLASS_COUNT, \
    WORK_DIR, PRETRAINED_MODEL, CACHE_DIR

torch.multiprocessing.set_sharing_strategy('file_system')

def normalize_image_to_uint8(image):
    """
    Normalize image to uint8
    Args:
        image: numpy array
    """
    draw_img = image
    if np.amin(draw_img) < 0:
        draw_img -= np.amin(draw_img)
    if np.amax(draw_img) > 1:
        draw_img /= np.amax(draw_img)
    draw_img = (255 * draw_img).astype(np.uint8)
    return draw_img


def validation(epoch_iterator_val, global_step):
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, IMAGE_SIZE, 1, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (global_step, 10.0)
            )
            if step == 0:
                slice_id = random.choice([60, 20])
                image = val_inputs[0][0].cpu().numpy()[..., slice_id]
                label = val_labels[0][0].cpu().numpy()[..., slice_id]
                image = normalize_image_to_uint8(image)
                gt_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                pred_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                gt_image[label == 1] = (255, 0, 0)
                gt_image[label == 2] = (0, 255, 0)
                gt_image[label == 3] = (0, 0, 255)
                gt_image[label == 4] = (0, 255, 255)
                for index, pre in enumerate(val_output_convert[0]):
                    if index == 0:
                        # 背景
                        continue
                    pred = pre[..., slice_id]
                    if index == 1:
                        color = (255, 0, 0)
                    elif index == 2:
                        color = (0, 255, 0)
                    elif index == 3:
                        color = (0, 0, 255)
                    elif index == 4:
                        color = (0, 255, 255)
                    pred_image[pred.cpu().numpy() > 0] = color
                log_image = np.hstack((gt_image, pred_image))
                log_image = cv2.cvtColor(log_image, cv2.COLOR_BGR2RGB)
                tensorboard_writer.add_image(tag="val_image",
                                             img_tensor=log_image.transpose([2, 1, 0]),
                                             global_step=global_step)
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        tensorboard_writer.add_scalar("step_loss", loss, global_step)
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val = validation(epoch_iterator_val, global_step)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            tensorboard_writer.add_scalar("dice_val", dice_val, global_step)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), str(WORK_DIR.joinpath("best_metric_model.pth")))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best


def get_model() -> torch.nn.Module:
    """
    加载模型
    :return:
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNETR(
        in_channels=1,
        out_channels=CLASS_COUNT,
        img_size=IMAGE_SIZE,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).to(device)
    if PRETRAINED_MODEL:
        print('加载预训练模型 {}'.format(PRETRAINED_MODEL))
        vit_dict = torch.load(PRETRAINED_MODEL, map_location='cuda:0')
        vit_weights = vit_dict['state_dict']

        # Remove items of vit_weights if they are not in the ViT backbone (this is used in UNETR).
        # For example, some variables names like conv3d_transpose.weight, conv3d_transpose.bias,
        # conv3d_transpose_1.weight and conv3d_transpose_1.bias are used to match dimensions
        # while pretraining with ViTAutoEnc and are not a part of ViT backbone.
        model_dict = model.vit.state_dict()
        vit_weights = {k: v for k, v in vit_weights.items() if k in model_dict}
        model_dict.update(vit_weights)
        model.vit.load_state_dict(model_dict)
        del model_dict, vit_weights, vit_dict
        print('预训练模型加载完成')
    return model


def get_train_val_transform() -> Tuple[monai.transforms.Compose, monai.transforms.Compose]:
    """
    获取训练和验证数据的transform
    :return:
    """
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=SPACING,
                mode=("bilinear", "nearest"),
            ),
            SCALE_INTENSITY_RANGE,
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=IMAGE_SIZE,
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=SPACING,
                mode=("bilinear", "nearest"),
            ),
            SCALE_INTENSITY_RANGE,
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=IMAGE_SIZE,
                pos=1,
                neg=0,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )
    return train_transforms, val_transforms


def get_dataset():
    """
    获取数据集
    :return:
    """
    dataset_dir = Path("/media/3TB/data/xiaoliutech/20221020")
    dataset = []
    for file in dataset_dir.iterdir():
        if "image" in file.name:
            label_file = file.parent.joinpath(file.name.replace("image", "label"))
            dataset.append({
                "image": str(file),
                "label": str(label_file)
            })
    train_count = int(len(dataset) * 0.95)
    train_files, val_files = dataset[:train_count], dataset[train_count:]
    train_transforms, val_transforms = get_train_val_transform()
    train_ds = PersistentDataset(
        data=train_files,
        transform=train_transforms,
        cache_dir=str(CACHE_DIR)
    )
    val_ds = PersistentDataset(
        data=val_files, transform=val_transforms, cache_dir=str(CACHE_DIR)
    )
    return train_ds, val_ds


if __name__ == '__main__':
    tensorboard_writer = SummaryWriter(str(WORK_DIR))
    train_ds, val_ds = get_dataset()

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=False
    )

    model = get_model()

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    max_iterations = 25000
    eval_num = 100
    post_label = AsDiscrete(to_onehot=CLASS_COUNT)
    post_pred = AsDiscrete(argmax=True, to_onehot=CLASS_COUNT)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(
            global_step, train_loader, dice_val_best, global_step_best
        )
    model.load_state_dict(torch.load(str(WORK_DIR.joinpath("best_metric_model.pth"))))

    print(
        f"train completed, best_metric: {dice_val_best:.4f} "
        f"at iteration: {global_step_best}"
    )
