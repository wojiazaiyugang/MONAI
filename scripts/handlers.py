import cv2
import numpy as np
from ignite.engine import Engine
from ignite.engine import Events

from torch.utils.tensorboard import SummaryWriter

from scripts import normalize_image_to_uint8


class ShowValResultHandler:
    """
    显示验证过程的图片
    monai的handler太难用了，自己写一个
    """

    def __init__(self, writer: SummaryWriter):
        self.writer = writer

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        engine.add_event_handler(Events.EPOCH_COMPLETED(every=1), self)

    def __call__(self, engine: Engine) -> None:
        data = engine.state.batch
        slice = 90
        image = data[0]["image"][0][..., slice]
        label = data[0]["label"][0][..., slice]
        pred = data[0]["pred"][0][..., slice]
        image = normalize_image_to_uint8(image)
        gt_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        pred_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gt_image[label == 1] = (255, 0, 0)
        gt_image[label == 2] = (0, 255, 0)
        gt_image[label == 3] = (0, 0, 255)
        pred_image[pred == 1] = (255, 0, 0)
        pred_image[pred == 2] = (0, 255, 0)
        pred_image[pred == 3] = (0, 0, 255)
        log_image = np.hstack((gt_image, pred_image))
        log_image = cv2.cvtColor(log_image, cv2.COLOR_BGR2RGB)
        self.writer.add_image(tag="val_image",
                              img_tensor=log_image.transpose([2, 1, 0]),
                              global_step=engine.state.epoch)
