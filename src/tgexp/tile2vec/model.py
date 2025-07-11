from typing import Any

import lightning
import timm
import torch
import torch.nn.functional as F


class TripletLoss(torch.nn.Module):
    """Triplet loss function for training tile2vec models."""

    def __init__(self, margin: float = 0.1, l2: float = 0.0) -> None:
        super().__init__()
        self.margin = margin
        self.l2 = l2

    def forward(
        self, z_p: torch.Tensor, z_n: torch.Tensor, z_d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        l_n = torch.sqrt(((z_p - z_n) ** 2).sum(dim=1))
        l_d = -torch.sqrt(((z_p - z_d) ** 2).sum(dim=1))
        l_nd = l_n + l_d
        loss = F.relu(l_n + l_d + self.margin)
        l_n = torch.mean(l_n)
        l_d = torch.mean(l_d)
        l_nd = torch.mean(l_n + l_d)
        loss = torch.mean(loss)
        loss += self.l2 * (torch.norm(z_p) + torch.norm(z_n) + torch.norm(z_d))
        return loss, l_n, l_d, l_nd


class Tile2VecModule(lightning.LightningModule):
    def __init__(
        self,
        backbone: str,
        in_channels: int = 4,
        pretrained: bool = True,
        margin: float = 0.1,
        l2: float = 0.0,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, in_chans=in_channels, num_classes=0
        )
        self.criterion = TripletLoss(margin=margin, l2=l2)

    def configure_optimizers(self) -> dict[str, Any]:
        total_steps = self.trainer.estimated_stepping_batches
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        anchor, neighbor, distant = (
            batch["anchor"],
            batch["neighbor"],
            batch["distant"],
        )
        z_p = self(anchor)
        z_n = self(neighbor)
        z_d = self(distant)
        loss, l_n, l_d, l_nd = self.criterion(z_p, z_n, z_d)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_l_n", l_n, prog_bar=True, on_epoch=True)
        self.log("train_l_d", l_d, prog_bar=True, on_epoch=True)
        self.log("train_l_nd", l_nd, prog_bar=True, on_epoch=True)
        return loss
