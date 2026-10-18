"""ResNet9 model for the VMRD grid detector."""

import torch
from torch import nn


NUM_CLASSES = 10


class ResNet9Detector(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, **kwargs):
        """Initialize the ResNet style detector model.

        Args:
            num_classes: the number of object classes (10)
            **kwargs: optional keyword arguments, use it to
                     pass in any additional settings you want
                     to use in your model, such as `use_skip`
                     to control whether the residual connections
                     are used.
        """
        super().__init__()

        ####################################
        # Your implementation starts here. #
        ####################################
        self.use_skip = kwargs.get("use_skip", True)

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2)

        self.head = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes + 5, kernel_size=1, stride=1)
        )



        ##################################
        # Your implementation ends here. #
        ##################################

    def forward(self, images):
        """Forward pass of the model.

        args:
            images: a batch of images of shape (B, 3, 256, 256)
        outputs:
            logits: a batch of logits of shape (B, num_classes + 5, 16, 16)
        """
        logits = None
        ####################################
        # Your implementation starts here. #
        ####################################
        x = self.block1(images)

        r = self.res1(x)
        x = x + r if self.use_skip else r

        x = self.block2(x)

        r = self.res2(x)
        x = x + r if self.use_skip else r

        x = self.pool(x)
        logits = self.head(x)
        ##################################
        # Your implementation ends here. #
        ##################################

        return logits


def logits_to_pred(logits):
    """Split the model output into objectness, class, and box predictions.

    args:
        logits: shape (B, C + 5, grid_h, grid_w), the raw output of the model
    outputs:
        objectness_logits: shape (B, grid_h, grid_w)
        class_logits: shape (B, C, grid_h, grid_w)
        pred_boxes: shape (B, 4, grid_h, grid_w), the predicted box coordinates in [0, 1]
    """
    ####################################
    # Your implementation starts here. #
    ####################################
    objectness_logits = logits[:, 0, :, :]
    class_logits = logits[:, 1:NUM_CLASSES + 1, :, :]
    pred_boxes = torch.sigmoid(logits[:, NUM_CLASSES + 1:, :, :])
    ##################################
    # Your implementation ends here. #
    ##################################
    return objectness_logits, class_logits, pred_boxes
