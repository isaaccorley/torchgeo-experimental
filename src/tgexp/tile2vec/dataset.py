import glob
import os
from collections.abc import Callable

import lightning
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from .transforms import ClipAndScale, GetBands, RandomFlipAndRotate


class Tile2VecDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        n_triplets: int | None = None,
        pairs_only: bool = True,
    ) -> None:
        self.root = root
        self.images = sorted(glob.glob(os.path.join(self.root, "*")))
        self.transform = transform
        self.n_triplets = n_triplets
        self.pairs_only = pairs_only

    def __len__(self) -> int:
        if self.n_triplets:
            return self.n_triplets
        else:
            return len(self.images) // 3

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        a = np.load(os.path.join(self.root, f"{idx}anchor.npy"))
        n = np.load(os.path.join(self.root, f"{idx}neighbor.npy"))

        if self.pairs_only:
            name = np.random.choice(["anchor", "neighbor", "distant"])
            d_idx = np.random.randint(0, self.n_triplets)
            d = np.load(os.path.join(self.root, f"{d_idx}{name}.npy"))
        else:
            d = np.load(os.path.join(self.root, f"{idx}distant.npy"))

        a = np.moveaxis(a, -1, 0)
        n = np.moveaxis(n, -1, 0)
        d = np.moveaxis(d, -1, 0)
        sample = {"anchor": a, "neighbor": n, "distant": d}

        if self.transform is not None:
            sample = self.transform(sample)

        sample["anchor"] = torch.from_numpy(sample["anchor"]).to(torch.float)
        sample["neighbor"] = torch.from_numpy(sample["neighbor"]).to(torch.float)
        sample["distant"] = torch.from_numpy(sample["distant"]).to(torch.float)
        return sample


class Tile2VecDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        root: str,
        img_type: str = "naip",
        bands: int = 4,
        batch_size: int = 4,
        num_workers: int = 4,
        n_triplets: int | None = None,
        pairs_only: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        transforms = T.Compose(
            [
                GetBands(self.hparams.bands),
                ClipAndScale(self.hparams.img_type),
                RandomFlipAndRotate(),
            ]
        )
        self.train_dataset = Tile2VecDataset(
            self.hparams.root,
            transform=transforms,
            n_triplets=self.hparams.n_triplets,
            pairs_only=self.hparams.pairs_only,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )
