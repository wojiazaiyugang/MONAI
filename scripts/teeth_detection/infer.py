import json
import json
import time
from pathlib import Path

import numpy as np
import torch
from typer import Typer

from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToWorldCoordinated,
    ClipBoxToImaged,
    ConvertBoxModed,
    BoxToMaskd
)
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.data import DataLoader, Dataset
from monai.data.utils import no_collation
from monai.transforms import Compose, ScaleIntensityRanged
from monai.transforms import (
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Spacingd,
    SaveImaged, DeleteItemsd
)
from scripts.transforms import LogD, MergeLabelValued

app = Typer()


def generate_detection_inference_transform(image_key, pred_box_key, pred_label_key, pred_score_key, gt_box_mode,
                                           intensity_transform, affine_lps_to_ras=False, amp=True):
    """
    Generate validation transform for detection.

    Args:
        image_key: the key to represent images in the input json files
        pred_box_key: the key to represent predicted boxes
        pred_label_key: the key to represent predicted box labels
        pred_score_key: the key to represent predicted classification scores
        gt_box_mode: ground truth box mode in the input json files
        intensity_transform: transform to scale image intensities,
            usually ScaleIntensityRanged for CT images, and NormalizeIntensityd for MR images.
        affine_lps_to_ras: Usually False.
            Set True only when the original images were read by itkreader with affine_lps_to_ras=True
        amp: whether to use half precision

    Return:
        validation transform for detection
    """
    amp = True
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    test_transforms = Compose(
        [
            LoadImaged(keys=[image_key], meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=[image_key]),
            EnsureTyped(keys=[image_key], dtype=torch.float32),
            Orientationd(keys=[image_key], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=[0.5, 0.5, 0.5], padding_mode="border"),
            intensity_transform,
            EnsureTyped(keys=[image_key], dtype=compute_dtype),
        ]
    )
    post_transforms = Compose(
        [
            LogD(keys=[image_key], message="开始ClipBoxToImaged"),
            ClipBoxToImaged(
                box_keys=[pred_box_key],
                label_keys=[pred_label_key, pred_score_key],
                box_ref_image_keys=image_key,
                remove_empty=True,
            ),

            AffineBoxToWorldCoordinated(
                box_keys=[pred_box_key],
                box_ref_image_keys=image_key,
                image_meta_key_postfix="meta_dict",
                affine_lps_to_ras=True,
            ),
            ConvertBoxModed(box_keys=[pred_box_key], src_mode="xyzxyz", dst_mode=gt_box_mode),

            # # LogD(keys=[image_key], message="开始BoxToMaskd"),
            # BoxToMaskd(box_keys=[pred_box_key], box_ref_image_keys=image_key, label_keys=[pred_label_key],
            #            box_mask_keys=["box_mask"], min_fg_label=0),
            # # LogD(keys=[image_key], message="开始MergeLabelValued"),
            # MergeLabelValued(keys=["box_mask"]),
            # # LogD(keys=[image_key], message="开始保存图像"),
            # SaveImaged(keys=["box_mask"], meta_keys=["image_meta_dict"], output_dir="/home/yujiannan/桌面/",
            #            separate_folder=False),
            # DeleteItemsd(keys=[image_key]),
        ]
    )
    return test_transforms, post_transforms


@app.command()
def main(image: Path):
    inference_data = [
        {"image": str(image)}
    ]
    amp = True

    patch_size = [256, 256, 256]

    # 1. define transform
    intensity_transform = ScaleIntensityRanged(
        keys=["image"],
        a_min=-200,
        a_max=3000,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    )
    inference_transforms, post_transforms = generate_detection_inference_transform(
        "image",
        "pred_box",
        "pred_label",
        "pred_score",
        "cccwhd",
        intensity_transform,
        affine_lps_to_ras=False,
        amp=amp,
    )

    inference_ds = Dataset(
        data=inference_data,
        transform=inference_transforms,
    )
    inference_loader = DataLoader(
        inference_ds,
        batch_size=1,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=no_collation,
    )

    # 3. build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) build anchor generator
    # returned_layers: when target boxes are small, set it smaller
    # base_anchor_shapes: anchor shape for the most high-resolution output,
    #   when target boxes are small, set it smaller
    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2 ** l for l in range(len([1, 2]) + 1)],
        base_anchor_shapes=[[8, 8, 16], [8, 11, 21], [8, 11, 33]],
    )

    # 2) build network
    net = torch.jit.load("/home/yujiannan/Projects/MONAI/models/teeth_detection/10/model_luna16_fold0.pt").to(device)

    # 3) build detector
    detector = RetinaNetDetector(
        network=net, anchor_generator=anchor_generator, debug=False
    )

    # set inference components
    detector.set_box_selector_parameters(
        score_thresh=0.1,
        topk_candidates_per_level=1000,
        nms_thresh=0.22,
        detections_per_img=100,
    )
    detector.set_sliding_window_inferer(
        roi_size=patch_size,
        overlap=0.25,
        sw_batch_size=1,
        mode="gaussian",
        device="cpu",
    )

    # 4. apply trained model
    results_dict = {"validation": []}
    detector.eval()

    with torch.no_grad():
        start_time = time.time()
        for inference_data in inference_loader:
            inference_img_filenames = [
                inference_data_i["image_meta_dict"]["filename_or_obj"]
                for inference_data_i in inference_data
            ]
            print(inference_img_filenames)
            use_inferer = not all(
                [
                    inference_data_i["image"][0, ...].numel() < np.prod(patch_size)
                    for inference_data_i in inference_data
                ]
            )
            inference_inputs = [inference_data_i["image"].to(device) for inference_data_i in inference_data]

            if amp:
                with torch.cuda.amp.autocast():
                    inference_outputs = detector(inference_inputs, use_inferer=use_inferer)
            else:
                inference_outputs = detector(inference_inputs, use_inferer=use_inferer)
            del inference_inputs

            # update inference_data for post transform
            for i in range(len(inference_outputs)):
                inference_data_i, inference_pred_i = inference_data[i], inference_outputs[i]
                inference_data_i["pred_box"] = inference_pred_i[detector.target_box_key].to(
                    torch.float32
                )
                inference_data_i["pred_label"] = inference_pred_i[detector.target_label_key]
                inference_data_i["pred_score"] = inference_pred_i[detector.pred_score_key].to(
                    torch.float32
                )
                inference_data[i] = post_transforms(inference_data_i)

            for inference_img_filename, inference_pred_i in zip(inference_img_filenames, inference_data):
                result = {
                    "label": inference_pred_i["pred_label"].cpu().detach().numpy().tolist(),
                    "box": inference_pred_i["pred_box"].cpu().detach().numpy().tolist(),
                    "score": inference_pred_i["pred_score"].cpu().detach().numpy().tolist(),
                }
                result.update({"image": inference_img_filename})
                results_dict["validation"].append(result)

    with open("/home/yujiannan/桌面/result.json", "w") as outfile:
        json.dump(results_dict, outfile, indent=4)

    end_time = time.time()
    print("Testing time: ", end_time - start_time)


if __name__ == "__main__":
    app()
