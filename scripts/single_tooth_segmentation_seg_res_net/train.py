import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from monai.data import (
    decollate_batch,
    PersistentDataset, DataLoader,
    Dataset
)
import cv2
import numpy as np
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureChannelFirstd, MapLabelValued,
)
from monai.transforms import (
    AsDiscrete,
    Compose,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandShiftIntensityd,
)
from monai.utils import set_determinism
from scripts import get_data_dir
from scripts.dataset import RandomSubItemListDataset
from scripts.single_tooth_segmentation_seg_res_net.config import work_dir, CLASS_COUNT
from scripts.transforms import ConfirmLabelLessD


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


tensorboard_writer = SummaryWriter(str(work_dir))

set_determinism(seed=0)
train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ConfirmLabelLessD(keys=["label"], max_val=50),
        MapLabelValued(keys=["label"], orig_labels=list(range(1, 50)), target_labels=[1 for _ in range(1, 50)]),
        RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]
)
val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ConfirmLabelLessD(keys=["label"], max_val=50),
        MapLabelValued(keys=["label"], orig_labels=list(range(1, 50)), target_labels=[1 for _ in range(1, 50)]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)

dataset_dir = get_data_dir().joinpath("single_tooth_segmentation")
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
    cache_dir="/home/yujiannan/Projects/MONAI/data/temp/train3",
)

# train_ds = RandomSubItemListDataset(train_ds, max_len=4)
val_ds = PersistentDataset(
    data=val_files,
    transform=val_transform,
    cache_dir="/home/yujiannan/Projects/MONAI/data/temp/val3",
)

# val_ds = RandomSubItemListDataset(val_ds, max_len=3)
train_loader = DataLoader(
    train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=False
)
val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
)

max_epochs = 300
val_interval = 1
VAL_AMP = True

# standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")
model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=1,
    out_channels=CLASS_COUNT,
    dropout_prob=0.2,
).to(device)
loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
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
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
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
post_pred = AsDiscrete(argmax=True, to_onehot=2)
best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]
epoch_loss_values = []
metric_values = []
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
            outputs = model(inputs)
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
                val_outputs = inference(val_inputs)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

            slice_id = 200
            image = val_inputs[0][0].cpu().numpy()[..., slice_id]
            label = val_labels[0][0].cpu().numpy()[..., slice_id]
            image = normalize_image_to_uint8(image)
            gt_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            pred_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            gt_image[label == 1] = (255, 0, 0)
            pred_image[val_outputs[0][0][..., slice_id].cpu().numpy() > 0.5] = (255, 0, 0)
            log_image = np.hstack((gt_image, pred_image))
            log_image = cv2.cvtColor(log_image, cv2.COLOR_BGR2RGB)
            tensorboard_writer.add_image(tag="val_image",
                                         img_tensor=log_image.transpose([2, 1, 0]),
                                         global_step=global_step)

            metric = dice_metric.aggregate().item()
            metric_values.append(metric)
            metric_batch = dice_metric_batch.aggregate()
            dice_metric.reset()
            dice_metric_batch.reset()
            tensorboard_writer.add_scalar("metric", metric, global_step)

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                torch.save(
                    model.state_dict(),
                    os.path.join(work_dir, "best_metric_model.pth"),
                )
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")
