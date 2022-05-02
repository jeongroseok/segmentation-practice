import os
from typing import Any, Callable, Optional, Tuple

import torch
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as transform_lib
from torchvision.datasets import VOCSegmentation, VOCDetection

CLASSES = (
    "__background__ ",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


class VOCSegmentationDataModule(LightningDataModule):
    name = "vocsegmentation"

    def __init__(
        self,
        data_dir: Optional[str] = None,
        year: str = "2012",
        num_workers: int = 0,
        normalize: bool = False,
        batch_size: int = 16,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "You want to use VOC dataset loaded from `torchvision` which is not installed yet."
            )

        super().__init__(*args, **kwargs)

        self.year = year
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.num_workers = num_workers
        self.normalize = normalize
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    @property
    def num_classes(self) -> int:
        """
        Return:
            21
        """
        return 21

    def prepare_data(self) -> None:
        """Saves VOCSegmentation files to data_dir."""
        VOCSegmentation(self.data_dir, year=self.year, image_set="train", download=False)
        VOCSegmentation(self.data_dir, year=self.year, image_set="val", download=False)

    def train_dataloader(
        self, image_transforms: Optional[Callable] = None
    ) -> DataLoader:
        """VOCSegmentation train set uses the `train` subset.
        Args:
            image_transforms: custom image-only transforms
        """
        dataset = VOCSegmentation(
            self.data_dir,
            year=self.year,
            image_set="train",
            transform=self.default_transforms(),
            target_transform=self.default_target_transforms(),
        )
        return self._data_loader(dataset, shuffle=self.shuffle)

    def val_dataloader(self, image_transforms: Optional[Callable] = None) -> DataLoader:
        """VOCSegmentation val set uses the `val` subset.
        Args:
            image_transforms: custom image-only transforms
        """
        dataset = VOCSegmentation(
            self.data_dir,
            year=self.year,
            image_set="val",
            transform=self.default_transforms(),
            target_transform=self.default_target_transforms(),
        )
        return self._data_loader(dataset, shuffle=False)

    def default_transforms(self) -> Callable:
        voc_transforms = transform_lib.Compose(
            [
                transform_lib.ToTensor(),
                transform_lib.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transform_lib.CenterCrop(321),
            ]
        )
        return voc_transforms

    def default_target_transforms(self) -> Callable:
        voc_transforms = transform_lib.Compose(
            [
                transform_lib.PILToTensor(),
                transform_lib.CenterCrop(321),
            ]
        )
        return voc_transforms

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
