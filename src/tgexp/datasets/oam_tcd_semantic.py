import os
from collections.abc import Callable
from typing import Literal

import kornia.augmentation as K
import lightning
import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset, Subset
from torchgeo.transforms.transforms import _ExtractPatches
from torchvision import tv_tensors

from .utils import dataset_split


def load_train_transforms(patch_size: int) -> T.Compose:
    return T.Compose(
        [
            T.RandomCrop(
                size=(patch_size, patch_size), pad_if_needed=True, antialias=True
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomResizedCrop(
                size=(patch_size, patch_size),
                scale=(0.8, 1.2),
                ratio=(0.75, 1.3333),
                antialias=True,
            ),
            T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01
                    )
                ],
                p=0.2,
            ),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.1),
            T.RandomGrayscale(p=0.1),
            T.ToDtype(torch.float32, scale=True),
        ]
    )


def load_eval_transforms() -> T.Compose:
    return T.Compose([T.ToDtype(torch.float32, scale=True)])


class OAMTCDSemanticSegmentation(Dataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "tests"],
        transforms: Callable | None = None,
    ) -> None:
        assert split in ["train", "test"], f"Invalid split: {split}"
        self.root = root
        self.split = split
        self.transforms = transforms
        self.image_directory = os.path.join(root, "dataset", "holdout", "images")
        self.masks_directory = os.path.join(root, "dataset", "holdout", "masks")
        labels_path = os.path.join(root, "dataset", "holdout", f"{split}.json")
        coco = COCO(labels_path)
        filenames = sorted([v["file_name"] for k, v in coco.imgs.items()])
        self.images = [os.path.join(self.image_directory, f) for f in filenames]
        self.masks = [
            os.path.join(self.masks_directory, f.replace(".tif", ".png"))
            for f in filenames
        ]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_path = self.images[index]
        mask_path = self.masks[index]
        image = np.array(Image.open(image_path).convert("RGB"))
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = tv_tensors.Image(image)
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = torch.from_numpy(mask.squeeze()).clip(0, 1)
        mask = tv_tensors.Mask(mask)
        sample = dict(image=image, mask=mask)
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


class OAMTCDSemanticSegmentationDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int = 8,
        num_workers: int = 4,
        patch_size: int = 1024,
        val_split_pct: float = 0.1,
        eval_batch_size: int = 4,
        train_transforms: Callable | None = None,
        val_transforms: Callable | None = None,
        test_transforms: Callable | None = None,
        predict_transforms: Callable | None = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.val_split_pct = val_split_pct
        self.eval_batch_size = eval_batch_size
        self.train_transforms = (
            load_train_transforms(patch_size)
            if train_transforms is None
            else train_transforms
        )
        self.val_transforms = (
            load_eval_transforms() if val_transforms is None else val_transforms
        )
        self.test_transforms = (
            load_eval_transforms() if test_transforms is None else test_transforms
        )
        self.predict_transforms = (
            load_eval_transforms() if predict_transforms is None else predict_transforms
        )
        self.patchify = K.AugmentationSequential(
            _ExtractPatches(window_size=patch_size), data_keys=None, same_on_batch=True
        )

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit":
            train_dataset = OAMTCDSemanticSegmentation(
                self.root, split="train", transforms=self.train_transforms
            )
            val_dataset = OAMTCDSemanticSegmentation(
                self.root, split="train", transforms=self.val_transforms
            )
            train_indices, val_indices = dataset_split(
                train_dataset, val_pct=self.val_split_pct, test_pct=None
            )
            self.train_dataset = Subset(train_dataset, train_indices)
            self.val_dataset = Subset(val_dataset, val_indices)
        elif stage == "test" or stage is None:
            self.test_dataset = OAMTCDSemanticSegmentation(
                self.root, split="test", transforms=self.test_transforms
            )
        elif stage == "predict":
            self.predict_dataset = OAMTCDSemanticSegmentation(
                self.root, split="test", transforms=self.predict_transforms
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=1,
        )

    def on_after_batch_transfer(
        self, batch: dict[str, torch.Tensor], dataloader_idx: int
    ) -> dict[str, torch.Tensor]:
        if not self.trainer.training:
            batch = self.patchify(batch)

        return batch
