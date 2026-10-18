"""Provided decoding and evaluation utilities for the grid detector.

You do not need to modify the functions in this file.
"""

import numpy as np
import torch
from torchvision.ops import batched_nms

from .box_utils import aligned_box_iou, grid_boxes_to_xyxy
from .model import NUM_CLASSES, logits_to_pred


@torch.no_grad()
def decode_predictions(
    logits,
    score_threshold=0.4,
    nms_iou_threshold=0.5,
    max_detections=20,
):
    """Turn the grid output into a list of detections for every image.

    The score of a cell is its objectness probability times its highest class
    probability. Cells with a score of at least ``score_threshold`` become
    detections, and non-maximum suppression (NMS) removes duplicate boxes of
    the same class.

    args:
        logits: shape (B, C + 5, grid_h, grid_w), the raw output of the model
        score_threshold: the minimum score of a detection
        nms_iou_threshold: boxes overlapping more than this are duplicates
        max_detections: the maximum number of detections per image
    outputs:
        detections: a list with one dictionary per image, containing
            - "boxes": shape (K, 4), normalized [x1, y1, x2, y2] boxes
            - "scores": shape (K,), sorted from the highest score
            - "labels": shape (K,), classes from 0 to C-1
    """
    objectness_logits, class_logits, pred_boxes = logits_to_pred(logits)
    class_probability, labels = class_logits.softmax(dim=1).max(dim=1)
    scores = objectness_logits.sigmoid() * class_probability  # (B, grid_h, grid_w)
    boxes = grid_boxes_to_xyxy(pred_boxes)                    # (B, grid_h, grid_w, 4)

    detections = []
    for image_boxes, image_scores, image_labels in zip(boxes, scores, labels):
        keep = image_scores >= score_threshold
        image_boxes = image_boxes[keep]
        image_scores = image_scores[keep]
        image_labels = image_labels[keep]

        # NMS returns the kept indices sorted from the highest score.
        keep = batched_nms(
            image_boxes, image_scores, image_labels, nms_iou_threshold
        )[:max_detections]
        detections.append({
            "boxes": image_boxes[keep].cpu(),
            "scores": image_scores[keep].cpu(),
            "labels": image_labels[keep].cpu(),
        })

    return detections


def _count_detection_results(
    detections, gt_objectness, gt_class, gt_boxes, iou_threshold
):
    """Count true positives, false positives, and false negatives.

    A detection is a true positive when it has the same class as a
    ground-truth object that is not matched yet, and their IoU is at least
    ``iou_threshold``. Otherwise, it is a false positive. Ground-truth objects
    that are never matched are false negatives.
    """
    gt_xyxy = grid_boxes_to_xyxy(gt_boxes)
    true_positive = 0
    false_positive = 0
    num_objects = 0

    for detection, image_objectness, image_class, image_boxes in zip(
        detections, gt_objectness, gt_class, gt_xyxy
    ):
        object_cells = image_objectness.bool()
        target_boxes = image_boxes[object_cells]
        target_labels = image_class[object_cells]
        matched = torch.zeros(len(target_boxes), dtype=torch.bool)
        num_objects += len(target_boxes)

        # Detections are sorted by score, so confident ones are matched first.
        for box, label in zip(detection["boxes"], detection["labels"]):
            ious = aligned_box_iou(box, target_boxes)
            # Only unmatched objects of the same class can be matched.
            ious[(target_labels != label) | matched] = 0.0

            if len(ious) > 0 and ious.max() >= iou_threshold:
                matched[ious.argmax()] = True
                true_positive += 1
            else:
                false_positive += 1

    false_negative = num_objects - true_positive
    return true_positive, false_positive, false_negative


def calculate_detection_accuracy(
    predicted_logits,
    gt_objectness,
    gt_class,
    gt_boxes,
    score_threshold=0.4,
    nms_iou_threshold=0.5,
    iou_threshold=0.5,
):
    """Calculate precision and recall over a complete dataset.

    args:
        predicted_logits: NumPy array of shape (N, C + 5, grid_h, grid_w)
        gt_objectness: NumPy array of shape (N, grid_h, grid_w)
        gt_class: NumPy array of shape (N, grid_h, grid_w)
        gt_boxes: NumPy array of shape (N, 4, grid_h, grid_w)
        score_threshold: the minimum score of a detection
        nms_iou_threshold: the IoU threshold used by NMS
        iou_threshold: the minimum IoU for a detection to be correct
    outputs:
        detection_accuracy: a dictionary containing "precision" and "recall"
    """
    names = ["predicted_logits", "gt_objectness", "gt_class", "gt_boxes"]
    arrays = [predicted_logits, gt_objectness, gt_class, gt_boxes]
    num_images, grid_h, grid_w = gt_objectness.shape
    expected_shapes = [
        (num_images, NUM_CLASSES + 5, grid_h, grid_w),
        (num_images, grid_h, grid_w),
        (num_images, grid_h, grid_w),
        (num_images, 4, grid_h, grid_w),
    ]
    for name, array, expected_shape in zip(names, arrays, expected_shapes):
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array")
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} should have shape {expected_shape}, "
                f"but got {array.shape}"
            )

    predicted_logits, gt_objectness, gt_class, gt_boxes = (
        torch.as_tensor(array) for array in arrays
    )
    detections = decode_predictions(
        predicted_logits,
        score_threshold=score_threshold,
        nms_iou_threshold=nms_iou_threshold,
    )
    true_positive, false_positive, false_negative = _count_detection_results(
        detections, gt_objectness, gt_class, gt_boxes, iou_threshold
    )

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "precision": precision,
        "recall": recall,
    }
