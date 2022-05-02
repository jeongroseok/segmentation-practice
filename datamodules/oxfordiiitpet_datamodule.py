# type: ignore[override]
import os
from typing import Any, Callable, Optional

import torch
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet


class OxfordIIITPetDataModule(LightningDataModule):

    name = "oxfordiiitpet"

    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: float = 0.2,
        test_split: float = 0.1,
        num_workers: int = 0,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "You want to use `torchvision` which is not installed yet."
            )

        super().__init__(*args, **kwargs)
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        # split into train, val, test
        kitti_dataset = OxfordIIITPet(
            root=self.data_dir,
            transform=self._default_transforms(),
            target_transform=self._default_target_transforms(),
            target_types="segmentation",
        )
        
        val_len = round(val_split * len(kitti_dataset))
        test_len = round(test_split * len(kitti_dataset))
        train_len = len(kitti_dataset) - val_len - test_len

        self.trainset, self.valset, self.testset = random_split(
            kitti_dataset,
            lengths=[train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def _default_transforms(self) -> Callable:
        kitti_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.CenterCrop(321),
            ]
        )
        return kitti_transforms


    def _default_target_transforms(self) -> Callable:
        kitti_transforms = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.CenterCrop(321),
            ]
        )
        return kitti_transforms
