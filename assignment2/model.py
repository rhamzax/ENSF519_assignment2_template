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
        pass
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
        pass
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
    pass
    ##################################
    # Your implementation ends here. #
    ##################################
    return objectness_logits, class_logits, pred_boxes
