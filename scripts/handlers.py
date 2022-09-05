import cv2
import numpy as np
import random
from ignite.engine import Engine
from ignite.engine import Events

from torch.utils.tensorboard import SummaryWriter

from scripts import normalize_image_to_uint8

colors = []
for i in range(33):
    colors.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])


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
        slice = None
        for slice in range(0, 160, 10):
            label = data[0]["label"][0][..., slice]
            if int(label.unique().size()[0]) > 12:
                break
        image = data[0]["image"][0][..., slice]
        label = data[0]["label"][0][..., slice]
        pred = data[0]["pred"][0][..., slice]
        image = normalize_image_to_uint8(image)
        gt_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        pred_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for label_index in range(1, 33):
            gt_image[label == label_index] = colors[label_index]
            pred_image[pred == label_index] = colors[label_index]
        log_image = np.hstack((gt_image, pred_image))
        log_image = cv2.cvtColor(log_image, cv2.COLOR_BGR2RGB)
        self.writer.add_image(tag="val_image",
                              img_tensor=log_image.transpose([2, 1, 0]),
                              global_step=engine.state.epoch)
