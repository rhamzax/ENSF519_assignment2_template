"""VMRD Dataset, target construction, and DataLoaders for Assignment 2."""

import json
import os

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Calculate these values in 00_dataset_and_tensors.ipynb.
####################################
# Your implementation starts here. #
####################################
IMAGE_MEAN = None
IMAGE_STD = None
##################################
# Your implementation ends here. #
##################################

IMAGE_SIZE = 256
GRID_SIZE = 16


def make_grid_targets(annotations, image_width, image_height):
    """Create objectness, class, and box targets for the grid detector."""
    objectness_targets = torch.zeros(
        GRID_SIZE, GRID_SIZE, dtype=torch.float32
    )
    class_targets = torch.zeros(
        GRID_SIZE, GRID_SIZE, dtype=torch.long
    )
    box_targets = torch.zeros(4, GRID_SIZE, GRID_SIZE, dtype=torch.float32)

    for annotation in annotations:
        x, y, width, height = annotation["bbox"]
        center_x = (x + width / 2) / image_width
        center_y = (y + height / 2) / image_height
        column = min(int(center_x * GRID_SIZE), GRID_SIZE - 1)
        row = min(int(center_y * GRID_SIZE), GRID_SIZE - 1)

        objectness_targets[row, column] = 1.0
        class_targets[row, column] = annotation["category_id"] - 1
        box_targets[:, row, column] = torch.tensor([
            center_x * GRID_SIZE - column,
            center_y * GRID_SIZE - row,
            width / image_width,
            height / image_height,
        ])

    return objectness_targets, class_targets, box_targets


class VMRDDataset(Dataset):
    """Return an image with objectness, class, and box targets."""

    def __init__(self, root, split, transform):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")
        self.root = root
        self.split = split
        self.transform = transform

        annotation_data = json.load(open(os.path.join(
            self.root, "annotations", f"instances_{split}.json"
        )))

        self.images = annotation_data["images"]
        self.annotations_by_image = {}
        for annotation in annotation_data["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in self.annotations_by_image:
                self.annotations_by_image[image_id] = []
            self.annotations_by_image[image_id].append(annotation)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_info = self.images[index]
        image_path = os.path.join(self.root, "images", self.split, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        annotations = self.annotations_by_image[image_info["id"]]
        objectness_targets, class_targets, box_targets = make_grid_targets(
            annotations,
            image_width=image_info["width"],
            image_height=image_info["height"],
        )

        if self.transform is not None:
            image = self.transform(image)

        return image, objectness_targets, class_targets, box_targets


def get_train_loader(root, batch_size):
    """Create the training DataLoader.

    Define suitable training transformations, using ColorJitter as data
    augmentation. Shuffle the training data.
    args:
        root: path to the dataset root directory
        batch_size: number of samples per batch
    outputs:
        train_loader: DataLoader for the training set
    """
    ####################################
    # Your implementation starts here. #
    ####################################
    pass
    ##################################
    # Your implementation ends here. #
    ##################################

    return train_loader


def get_val_loader(root, batch_size):
    """Create the validation DataLoader.

    Define suitable transformations without data augmentation.
    args:
        root: path to the dataset root directory
        batch_size: number of samples per batch
    outputs:
        val_loader: DataLoader for the validation set
    """
    ####################################
    # Your implementation starts here. #
    ####################################
    pass
    ##################################
    # Your implementation ends here. #
    ##################################

    return val_loader


def get_test_loader(root, batch_size):
    """Create the test DataLoader.

    Define suitable transformations without data augmentation.
    args:
        root: path to the dataset root directory
        batch_size: number of samples per batch
    outputs:
        test_loader: DataLoader for the test set
    """
    ####################################
    # Your implementation starts here. #
    ####################################
    pass
    ##################################
    # Your implementation ends here. #
    ##################################

    return test_loader
