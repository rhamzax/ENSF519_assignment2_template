"""Detection loss, training loop, and evaluation loop for the VMRD grid detector."""

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import generalized_box_iou_loss

from .box_utils import grid_boxes_to_xyxy
from .evaluation import calculate_detection_accuracy
from .model import logits_to_pred


def select_object_cells(grid_values, object_cells):
    """Keep only the grid cells that contain an object.

    You do not need to modify this function.

    args:
        grid_values: shape (B, K, grid_h, grid_w), K values for every grid cell
        object_cells: shape (B, grid_h, grid_w), True for cells with an object
    outputs:
        object_values: shape (N, K), where N is the number of object cells
    """
    return grid_values.permute(0, 2, 3, 1)[object_cells]


def calculate_objectness_loss(objectness_logits, gt_objectness):
    """Return the focal objectness loss over all grid cells.

    You do not need to modify this function.

    args:
        objectness_logits: shape (B, grid_h, grid_w)
        gt_objectness: shape (B, grid_h, grid_w), with 1 for objects and 0 for background
    outputs:
        objectness_loss: a scalar
    """
    bce_loss = F.binary_cross_entropy_with_logits(
        objectness_logits, gt_objectness, reduction="none"
    )
    object_probability = torch.sigmoid(objectness_logits)

    # Probability assigned to the correct target: p for an object and 1-p
    # for background. Focal loss gives less weight to easy cells.
    correct_probability = torch.where(
        gt_objectness.bool(), object_probability, 1 - object_probability
    )
    alpha = torch.where(
        gt_objectness.bool(),
        torch.full_like(object_probability, 0.75),
        torch.full_like(object_probability, 0.25),
    )
    focal_weight = alpha * (1 - correct_probability).square()

    num_objects = gt_objectness.sum().clamp(min=1)
    return (focal_weight * bce_loss).sum() / num_objects


def calculate_box_loss(pred_boxes, gt_boxes, object_cells):
    """Return the box regression loss over object cells.

    You do not need to modify this function.

    args:
        pred_boxes: shape (B, 4, grid_h, grid_w), the predicted box coordinates in [0, 1]
        gt_boxes: shape (B, 4, grid_h, grid_w), the ground truth box coordinates in [0, 1]
        object_cells: shape (B, grid_h, grid_w), True for cells with an object
    outputs:
        box_loss: a scalar, Smooth L1 loss plus generalized IoU (GIoU) loss
    """
    smooth_l1_loss = F.smooth_l1_loss(
        select_object_cells(pred_boxes, object_cells),
        select_object_cells(gt_boxes, object_cells),
    )

    pred_xyxy = grid_boxes_to_xyxy(pred_boxes)[object_cells]
    gt_xyxy = grid_boxes_to_xyxy(gt_boxes)[object_cells]
    giou_loss = generalized_box_iou_loss(
        pred_xyxy, gt_xyxy, reduction="mean"
    )
    return smooth_l1_loss + giou_loss


def detection_loss(logits, gt_objectness, gt_class, gt_boxes,
                   box_loss_weight=2.0):
    """Return the losses

    args:
        logits: shape (B, C + 5, grid_h, grid_w), the raw output of the model
        gt_objectness: shape (B, grid_h, grid_w), with 1 for objects and 0 for background
        gt_class: shape (B, grid_h, grid_w), with 0..C-1 for object classes
        gt_boxes: shape (B, 4, grid_h, grid_w), the ground truth box coordinates in [0, 1]
        box_loss_weight: the weight for the box regression loss
    outputs:
        total_loss: the total loss, a scalar
        objectness_loss: the objectness loss, a scalar
        class_loss: the classification loss, a scalar
        box_loss: the box regression loss, a scalar"""
    objectness_logits, class_logits, pred_boxes = logits_to_pred(logits)
    object_cells = gt_objectness.bool()

    objectness_loss = calculate_objectness_loss(objectness_logits, gt_objectness)
    box_loss = calculate_box_loss(pred_boxes, gt_boxes, object_cells)

    ####################################
    # Your implementation starts here. #
    ####################################
    selected_class_logits = select_object_cells(class_logits, object_cells)
    selected_class_targets = select_object_cells(gt_class.unsqueeze(1), object_cells).squeeze(1)

    class_loss = F.cross_entropy(selected_class_logits, selected_class_targets)

    total_loss = objectness_loss + class_loss + box_loss_weight * box_loss

    ##################################
    # Your implementation ends here. #
    ##################################

    return total_loss, objectness_loss, class_loss, box_loss


def train_detection_model(model, train_loader, num_epochs, device,
                          **kwargs):
    """Train the model and return the loss from every mini-batch.

    args:
        model: nn.Module, the model to train
        train_loader: torch.utils.data.DataLoader, the DataLoader for the training set
        num_epochs: int, the number of epochs to train
        device: torch.device, the device to train on
        **kwargs: include any optional keyword argument that may be helpful, which can include:
            - loss_kwargs: arguments for the loss function
            - optimizer_kwargs: arguments for the optimizer
            - scheduler_kwargs: arguments for the learning rate scheduler
    outputs:
        loss_history: a dictionary mapping each loss name to a list of Python
            floats, with one value for every mini-batch
    """
    loss_history = {
        "total_loss": [],
        "objectness_loss": [],
        "class_loss": [],
        "box_loss": [],
    }

    ####################################
    # Your implementation starts here. #
    ####################################
    optimizer_kwargs = {"lr": 2.5e-3, "weight_decay": 1e-4}
    optimizer_kwargs.update(kwargs.get("optimizer_kwargs", {}))
    scheduler_kwargs = {"max_lr": 2.5e-3, "pct_start": 0.15}
    scheduler_kwargs.update(kwargs.get("scheduler_kwargs", {}))
    loss_kwargs = kwargs.get("loss_kwargs", {})

    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        **scheduler_kwargs,
    )

    model.train()
    for epoch in range(num_epochs):
        for images, gt_objectness, gt_class, gt_boxes in train_loader:
            images = images.to(device)
            gt_objectness = gt_objectness.to(device)
            gt_class = gt_class.to(device)
            gt_boxes = gt_boxes.to(device)

            logits = model(images)
            total_loss, objectness_loss, class_loss, box_loss = detection_loss(
                logits, gt_objectness, gt_class, gt_boxes, **loss_kwargs
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            loss_history["total_loss"].append(total_loss.item())
            loss_history["objectness_loss"].append(objectness_loss.item())
            loss_history["class_loss"].append(class_loss.item())
            loss_history["box_loss"].append(box_loss.item())

    ##################################
    # Your implementation ends here. #
    ##################################

    return loss_history


def evaluate_detection_model(model, data_loader, device, **kwargs):
    """Return average losses and detection accuracy metrics.

    args:
        model: nn.Module, the model to evaluate
        data_loader: torch.utils.data.DataLoader, the validation or test DataLoader
        device: torch.device, the device to evaluate on
        **kwargs: optional loss and detection settings
    outputs:
        average_losses: a dictionary mapping each loss name to its average over
            the complete dataset, as a Python float
        detection_accuracy: a dictionary containing "precision" and "recall"
    """
    average_losses = {
        "total_loss": 0.0,
        "objectness_loss": 0.0,
        "class_loss": 0.0,
        "box_loss": 0.0,
    }
    ####################################
    # Your implementation starts here. #
    ####################################
    loss_kwargs = kwargs.get("loss_kwargs", {})
    all_logits, all_objectness, all_class, all_boxes = [], [], [], []
    num_images = 0

    model.eval()
    with torch.no_grad():
        for images, gt_objectness, gt_class, gt_boxes in data_loader:
            logits = model(images.to(device))
            total_loss, objectness_loss, class_loss, box_loss = detection_loss(
                logits,
                gt_objectness.to(device),
                gt_class.to(device),
                gt_boxes.to(device),
                **loss_kwargs,
            )

            batch_size = images.shape[0]
            average_losses["total_loss"] += total_loss.item() * batch_size
            average_losses["objectness_loss"] += objectness_loss.item() * batch_size
            average_losses["class_loss"] += class_loss.item() * batch_size
            average_losses["box_loss"] += box_loss.item() * batch_size
            num_images += batch_size

            all_logits.append(logits.cpu())
            all_objectness.append(gt_objectness)
            all_class.append(gt_class)
            all_boxes.append(gt_boxes)

    for name in average_losses:
        average_losses[name] /= num_images

    predicted_logits = torch.cat(all_logits).numpy()
    all_gt_objectness = torch.cat(all_objectness).numpy()
    all_gt_class = torch.cat(all_class).numpy()
    all_gt_boxes = torch.cat(all_boxes).numpy()
    ##################################
    # Your implementation ends here. #
    ##################################

    # Calculate detection accuracy here
    detection_accuracy = calculate_detection_accuracy(
        predicted_logits,   # shape (N, C + 5, H, W)
        all_gt_objectness,  # shape (N, H, W)
        all_gt_class,       # shape (N, H, W)
        all_gt_boxes,       # shape (N, 4, H, W)
        score_threshold=0.4,
        nms_iou_threshold=0.5,
        iou_threshold=0.5,
    )

    return average_losses, detection_accuracy
