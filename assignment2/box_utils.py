"""Provided bounding-box utilities for the grid detector.

You do not need to modify the functions in this file.
"""

import torch


def grid_boxes_to_xyxy(boxes):
    """Convert grid boxes ``[tx, ty, w, h]`` to image boxes ``[x1, y1, x2, y2]``.

    ``tx`` and ``ty`` locate the box center inside its grid cell, while ``w``
    and ``h`` are relative to the whole image. The returned coordinates are
    normalized to ``[0, 1]``.

    args:
        boxes: shape (B, 4, grid_h, grid_w)
    outputs:
        xyxy_boxes: shape (B, grid_h, grid_w, 4)
    """
    _, _, grid_h, grid_w = boxes.shape
    tx, ty, width, height = boxes.unbind(dim=1)  # each (B, grid_h, grid_w)

    # Row and column index of every grid cell.
    rows = torch.arange(grid_h, device=boxes.device).view(grid_h, 1)
    columns = torch.arange(grid_w, device=boxes.device).view(1, grid_w)

    center_x = (columns + tx) / grid_w
    center_y = (rows + ty) / grid_h
    xyxy_boxes = torch.stack([
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    ], dim=-1)
    return xyxy_boxes.clamp(0, 1)


def aligned_box_iou(boxes1, boxes2):
    """Calculate the IoU between corresponding ``[x1, y1, x2, y2]`` boxes.

    args:
        boxes1: shape (..., 4)
        boxes2: shape (..., 4), broadcastable with boxes1
    outputs:
        iou: shape (...)
    """
    top_left = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    bottom_right = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    intersection_size = (bottom_right - top_left).clamp(min=0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]

    size1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp(min=0)
    size2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp(min=0)
    area1 = size1[..., 0] * size1[..., 1]
    area2 = size2[..., 0] * size2[..., 1]
    union = area1 + area2 - intersection
    return intersection / union.clamp(min=1e-8)
