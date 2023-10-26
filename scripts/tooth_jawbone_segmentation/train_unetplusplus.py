import os
import random
import tempfile
import time
from pathlib import Path

import mlflow
import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from monai.data import DataLoader, decollate_batch, PersistentDataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import BasicUNetPlusPlus
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    Orientationd,
    RandFlipd,RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    EnsureTyped,
    EnsureChannelFirstd, RandCropByPosNegLabeld, )
from scripts import normalize_image_to_uint8
from scripts.tooth_jawbone_segmentation.config_unetplusplus import work_dir, CLASS_COUNT, scale_intensity_range, IMAGE_SIZE, CACHE_DIR

torch.multiprocessing.set_sharing_strategy('file_system')
tensorboard_writer = SummaryWriter(str(work_dir))

train_transform = Compose(
    [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        scale_intensity_range,
        EnsureTyped(keys=["image", "label"]),
        # RandSpatialCropd(keys=["image", "label"], roi_size=IMAGE_SIZE, random_size=False),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=IMAGE_SIZE,
            pos=1,
            neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
            allow_smaller=True,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.50,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.50,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.50,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.50,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.5,
        ),
    ]
)
val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        scale_intensity_range,
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=IMAGE_SIZE,
            pos=1,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
            allow_smaller=True,
        ),
EnsureTyped(keys=["image", "label"]),
    ]
)

dataset_dir = Path("/media/3TB/data/xiaoliutech/20221124")
mlflow.log_param("dataset", "20221124")
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
train_ds = PersistentDataset(
    data=train_files,
    transform=train_transform,
    cache_dir=CACHE_DIR,
)

val_ds = PersistentDataset(
    data=val_files,
    transform=val_transform,
    cache_dir=CACHE_DIR,
)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

max_epochs = 500
val_interval = 1
VAL_AMP = True

# standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")
model = BasicUNetPlusPlus(
    spatial_dims=3,
    in_channels=1,
    out_channels=CLASS_COUNT,
).to(device)
mlflow.log_param("model", "UnetPlusPlus")

loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose(
    [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)


# define inference method
def inference(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=IMAGE_SIZE,
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)


# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]
epoch_loss_values = []
metric_values = []
metric_values_tc = []
metric_values_wt = []
metric_values_et = []
global_step = 0

total_start = time.time()
for epoch in range(max_epochs):
    epoch_start = time.time()
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step_start = time.time()
        step += 1
        global_step = global_step + 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)[0]
            labels = torch.stack([AsDiscrete(to_onehot=CLASS_COUNT)(l) for l in labels])  # type: ignore
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}"
            f", train_loss: {loss.item():.4f}"
            f", step time: {(time.time() - step_start):.4f}"
        )
        tensorboard_writer.add_scalar("step_loss", loss.item(), global_step)
        mlflow.log_metric("step_loss", loss.item(), global_step)
    lr_scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():

            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_outputs = inference(val_inputs)[0]
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                new_val_labels = torch.stack([AsDiscrete(to_onehot=CLASS_COUNT)(l) for l in val_labels])  # type: ignore
                dice_metric(y_pred=val_outputs, y=new_val_labels)
                dice_metric_batch(y_pred=val_outputs, y=new_val_labels)

            slice_id = random.randint(30, 90)
            image = val_inputs[0][0].cpu().numpy()[..., slice_id]
            label = val_labels[0][0].cpu().numpy()[..., slice_id]
            image = normalize_image_to_uint8(image)
            gt_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            pred_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            pred_label = torch.argmax(val_outputs[0].cpu().detach(), dim=0)[..., slice_id]
            for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]):
                gt_image[label == i + 1] = color
                pred_image[pred_label == i + 1] = color
            log_image = np.hstack((gt_image, pred_image))
            log_image = cv2.cvtColor(log_image, cv2.COLOR_BGR2RGB)
            with tempfile.TemporaryDirectory() as tempdir:
                image_file = Path(tempdir).joinpath(f"step_{global_step}.png")
                cv2.imwrite(str(image_file), log_image)
                mlflow.log_artifact(str(image_file), artifact_path="val_images")
            tensorboard_writer.add_image(tag="val_image",
                                         img_tensor=log_image.transpose([2, 1, 0]),
                                         global_step=global_step)

            metric = dice_metric.aggregate().item()
            metric_values.append(metric)
            metric_batch = dice_metric_batch.aggregate()
            metric_tc = metric_batch[0].item()
            metric_values_tc.append(metric_tc)
            metric_wt = metric_batch[1].item()
            metric_values_wt.append(metric_wt)
            metric_et = metric_batch[2].item()
            metric_values_et.append(metric_et)
            dice_metric.reset()
            dice_metric_batch.reset()
            tensorboard_writer.add_scalar("metric", metric, global_step)
            mlflow.log_metric("Dice", metric, global_step)

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                save_file = work_dir.joinpath(f"jawbone_seg_unetplusplus_20221124_Dice_{round(metric, 4)}_2022XXXX.pth")
                torch.save(
                    model.state_dict(),
                    str(save_file),
                )
                mlflow.log_artifact(str(save_file), artifact_path="checkpoints")
                mlflow.set_tag("best_dice", metric)
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")