import os
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import monai.data
from monai.data import (
    DataLoader,
    decollate_batch,
    PersistentDataset
)
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR
from monai.transforms import (
    AsDiscrete,
    Compose,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandShiftIntensityd,
    RandRotate90d,
    ResizeWithPadOrCropd,
    MapLabelValued, RandCropByPosNegLabeld,
)
from scripts import get_data_dir
from scripts import normalize_image_to_uint8, load_image_label_pair_dataset
from scripts.dataset import RandomSubItemListDataset
from scripts.tooth_instannce_segmentation.config_unetr import scale_intensity_range, IMAGE_SIZE, CLASS_COUNT, work_dir, CACHE_DIR
from scripts.transforms import CropForegroundSamples, ConfirmLabelLessD

colors = []
for i in range(33):
    colors.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

def validation(epoch_iterator_val, global_step):
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, IMAGE_SIZE, 4, model)
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
                slice_id = 60
                for i in range(0, IMAGE_SIZE[2], 10):
                    if val_labels[0][0][..., i].cpu().unique().size()[0] > 6:
                        slice_id = i
                        break
                image = val_inputs[0][0].cpu().numpy()[..., slice_id]
                label = val_labels[0][0].cpu().numpy()[..., slice_id]
                image = normalize_image_to_uint8(image)
                gt_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                pred_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                for i in range(1, 33):
                    gt_image[label == i] = colors[i]
                    pred_image[torch.argmax(val_outputs.cpu(), dim=1)[0, ..., slice_id] == i] = colors[i]
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
                torch.save(model.state_dict(), str(work_dir.joinpath("best_metric_model.pth")))
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
    # if PRETRAINED_MODEL:
    #     print('加载预训练模型 {}'.format(PRETRAINED_MODEL))
    #     vit_dict = torch.load(PRETRAINED_MODEL)
    #     vit_weights = vit_dict['state_dict']
    #
    #     # Remove items of vit_weights if they are not in the ViT backbone (this is used in UNETR).
    #     # For example, some variables names like conv3d_transpose.weight, conv3d_transpose.bias,
    #     # conv3d_transpose_1.weight and conv3d_transpose_1.bias are used to match dimensions
    #     # while pretraining with ViTAutoEnc and are not a part of ViT backbone.
    #     model_dict = model.vit.state_dict()
    #     vit_weights = {k: v for k, v in vit_weights.items() if k in model_dict}
    #     model_dict.update(vit_weights)
    #     model.vit.load_state_dict(model_dict)
    #     del model_dict, vit_weights, vit_dict
    #     print('预训练模型加载完成')
    return model


def get_train_val_transform() -> Tuple[monai.transforms.Compose, monai.transforms.Compose]:
    """
    获取训练和验证数据的transform
    :return:
    """
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            scale_intensity_range,
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=IMAGE_SIZE,
                pos=1,
                neg=0,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            scale_intensity_range,
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
        ]
    )
    return train_transforms, val_transforms


def get_dataset():
    """
    获取数据集
    :return:
    """
    dataset = load_image_label_pair_dataset(Path("/home/yujiannan/Projects/MONAI/data/tooth_instannce_segmentation"))[:3]
    train_count = int(len(dataset) * 0.5)
    train_files, val_files = dataset[:train_count], dataset[train_count:]
    train_transforms, val_transforms = get_train_val_transform()
    train_ds = PersistentDataset(
        data=train_files,
        transform=train_transforms,
        cache_dir=CACHE_DIR,
    )

    val_ds = PersistentDataset(
        data=val_files,
        transform=val_transforms,
        cache_dir=CACHE_DIR,
    )

    return train_ds, val_ds


if __name__ == '__main__':
    tensorboard_writer = SummaryWriter(str(work_dir))
    train_ds, val_ds = get_dataset()

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    )

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    torch.backends.cudnn.benchmark = True
    model = get_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    max_iterations = 45000
    eval_num = 5
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
    model.load_state_dict(torch.load(str(work_dir.joinpath("best_metric_model.pth"))))

    print(
        f"train completed, best_metric: {dice_val_best:.4f} "
        f"at iteration: {global_step_best}"
    )
