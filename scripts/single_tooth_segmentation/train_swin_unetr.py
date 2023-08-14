import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from monai.data import decollate_batch, PersistentDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    AsDiscrete,
    Compose,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandShiftIntensityd,
    RandRotate90d,
    EnsureTyped, MapLabelValued, ResizeWithPadOrCropd
)
from scripts import normalize_image_to_uint8, load_image_label_pair_dataset
from scripts.dataset import RandomSubItemListDataset
from scripts.single_tooth_segmentation.config_swin_unetr import scale_intensity_range, IMAGE_SIZE, work_dir, \
    CLASS_COUNT, CACHE_DIR, DATASET_DIR
from scripts.transforms import ConfirmLabelLessD, CropForegroundSamplesByBBox, RandCropForegroundSamplesByBBox, MyRandCrop

tensorboard_writer = SummaryWriter(str(work_dir))
num_samples = 4

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

crop_margin = 5
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        scale_intensity_range,
        CropForegroundSamplesByBBox(keys=["image", "label"], label_key="label", margin=crop_margin),

        ConfirmLabelLessD(keys=["label"], max_val=50),
        MapLabelValued(keys=["label"], orig_labels=list(range(1, 50)), target_labels=[1 for _ in range(1, 50)]),

        MyRandCrop(keys=["image", "label"]),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=IMAGE_SIZE),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
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
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        scale_intensity_range,
        CropForegroundSamplesByBBox(keys=["image", "label"], label_key="label", margin=crop_margin),
        ConfirmLabelLessD(keys=["label"], max_val=50),
        MapLabelValued(keys=["label"], orig_labels=list(range(1, 50)), target_labels=[1 for _ in range(1, 50)]),

        MyRandCrop(keys=["image", "label"]),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=IMAGE_SIZE),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ]
)

# dataset = load_image_label_pair_dataset(DATASET_DIR)
train_files, val_files = [], []
for dataset_name in ["20220923", "20221107", "20221118", "20230724_tooth_instance", "20230725_tooth_instance"]:
    dataset = Path("/media/3TB/data/xiaoliutech").joinpath(dataset_name)
    train_datas = dataset.joinpath("train.txt").read_text().splitlines()
    val_datas = dataset.joinpath("val.txt").read_text().splitlines()
    for train_data in train_datas:
        train_files.append({
            "image": f"{dataset.joinpath(train_data)}.image.nii.gz",
            "label": f"{dataset.joinpath(train_data)}.label.nii.gz",
        })
    for val_data in val_datas:
        val_files.append({
            "image": f"{dataset.joinpath(train_data)}.image.nii.gz",
            "label": f"{dataset.joinpath(train_data)}.label.nii.gz",
        })

train_ds = PersistentDataset(
    data=train_files,
    transform=train_transforms,
    cache_dir=CACHE_DIR
)

train_ds = RandomSubItemListDataset(train_ds, max_len=2)
val_ds = PersistentDataset(
    data=val_files,
    transform=val_transforms,
    cache_dir=CACHE_DIR,
)

val_ds = RandomSubItemListDataset(val_ds, max_len=1)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinUNETR(
    img_size=IMAGE_SIZE,
    in_channels=1,
    out_channels=CLASS_COUNT,
    feature_size=48,
    use_checkpoint=True,
).to(device)

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scaler = torch.cuda.amp.GradScaler()


def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, IMAGE_SIZE, 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))
            if step == 0:
                slice_id = 60
                for i in range(0, IMAGE_SIZE[2], 10):
                    label = val_labels[0][0].cpu().numpy()[..., i]
                    if np.sum(label) > 0:
                        slice_id = i
                        break
                image = val_inputs[0][0].cpu().numpy()[..., slice_id]
                label = val_labels[0][0].cpu().numpy()[..., slice_id]
                image = normalize_image_to_uint8(image)
                gt_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                pred_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                gt_image[label == 1] = (255, 0, 0)
                gt_image[label == 2] = (0, 255, 0)
                gt_image[label == 3] = (0, 0, 255)
                for index, pre in enumerate(val_output_convert[0]):
                    if index == 0:
                        # 背景
                        continue
                    pred = pre[..., slice_id]
                    if index == 1:
                        color = (255, 0, 0)
                    pred_image[pred.cpu().numpy() > 0] = color
                log_image = np.hstack((gt_image, pred_image))
                log_image = cv2.cvtColor(log_image, cv2.COLOR_BGR2RGB)
                tensorboard_writer.add_image(tag="val_image", img_tensor=log_image.transpose([2, 1, 0]), global_step=global_step)
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
        with torch.cuda.amp.autocast():
            logit_map = model(x)
            loss = loss_function(logit_map, y)
        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss))
        tensorboard_writer.add_scalar("step_loss", loss, global_step)
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            tensorboard_writer.add_scalar("dice_val", dice_val, global_step)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), str(work_dir.joinpath("best_metric_model.pth")))
                print("Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val))
            else:
                print("Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val))
        global_step += 1
    return global_step, dice_val_best, global_step_best


max_iterations = 200000
eval_num = 500
post_label = AsDiscrete(to_onehot=CLASS_COUNT)
post_pred = AsDiscrete(argmax=True, to_onehot=CLASS_COUNT)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)
print(f"train completed, best_metric: {dice_val_best:.4f} at iteration: {global_step_best}")
