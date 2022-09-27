import logging
import os
from pathlib import Path
import random
import sys
import shutil
import tempfile

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset, Dataset, PersistentDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose, Orientationd,
    RandRotate90,
    Resize, Spacingd,
    ScaleIntensity, EnsureChannelFirstd, Resized, RandRotate90d, LoadImaged, EnsureTyped, ResizeWithPadOrCropd, RandShiftIntensityd
)
from scripts.dataset import RandomSubItemListDataset
from scripts.transforms import CropToothClassificationInstance
from scripts import normalize_image_to_uint8

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

datas = []
for image_file in Path("/media/3TB/data/xiaoliutech/20220923").glob("*.image.nii.gz"):
    datas.append({
        "image": str(image_file),
        "info": str(image_file.parent.joinpath(image_file.name.replace(".image.nii.gz", ".info.txt"))),
    })
# datas = datas[:3]
# labels =
train_transforms = Compose([
    LoadImaged(keys="image"),
    EnsureChannelFirstd(keys="image"),
    Orientationd(keys="image", axcodes="RAS"),
    Spacingd(keys="image", pixdim=(0.25, 0.25, 0.25), mode="bilinear"),
    CropToothClassificationInstance(keys="image"),
    ResizeWithPadOrCropd(keys="image", spatial_size=(120, 120, 120)),
    # RandShiftIntensityd(
    #     keys=["image"],
    #     offsets=0.10,
    #     prob=0.10,
    # ),
])
val_transforms = Compose([
    LoadImaged(keys="image"),
    EnsureChannelFirstd(keys="image"),
    Orientationd(keys="image", axcodes="RAS"),
    Spacingd(keys="image", pixdim=(0.25, 0.25, 0.25), mode="bilinear"),
    CropToothClassificationInstance(keys="image"),
    ResizeWithPadOrCropd(keys="image", spatial_size=(120, 120, 120))
])

CACHE_DIR = Path("/home/yujiannan/Projects/MONAI/data/temp/tooth_clssification")
train_count = len(datas) - 5
train_ds = PersistentDataset(data=datas[:train_count], transform=train_transforms, cache_dir=str(CACHE_DIR))
train_ds = RandomSubItemListDataset(train_ds, max_len=5)
val_ds = PersistentDataset(data=datas[train_count:], transform=val_transforms, cache_dir=str(CACHE_DIR))
val_ds = RandomSubItemListDataset(val_ds, max_len=2)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=pin_memory)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, pin_memory=pin_memory)
model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=32).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
# start a typical PyTorch training
val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
writer = SummaryWriter("./logs/1")
max_epochs = 300

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()

        num_correct = 0.0
        metric_count = 0
        for val_data in val_loader:
            val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            with torch.no_grad():
                val_outputs = model(val_images)
                value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
                metric_count += len(value)
                num_correct += value.sum().item()

        metric = num_correct / metric_count
        metric_values.append(metric)

        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
            print("saved new best metric model")

        print(f"Current epoch: {epoch + 1} current accuracy: {metric:.4f} ")
        print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
        writer.add_scalar("val_accuracy", metric, epoch + 1)

print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()
