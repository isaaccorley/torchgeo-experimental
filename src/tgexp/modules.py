from typing import Any

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchgeo.trainers import SemanticSegmentationTask


class SemanticSegmentationModule(SemanticSegmentationTask):
    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = AdamW(self.parameters(), lr=self.hparams["lr"])
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }
