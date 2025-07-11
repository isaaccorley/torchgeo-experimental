import lightning
import torch
from lightning.pytorch.callbacks import Callback


class WandBSegmentationCallback(Callback):
    def __init__(self, num_images: int = 25, threshold: float = 0.2) -> None:
        super().__init__()
        self.count = 0
        self.num_images = num_images
        self.threshold = threshold

    @torch.inference_mode()
    def on_train_epoch_end(
        self, trainer: lightning.Trainer, module: lightning.LightningModule
    ) -> None:
        class_labels = {0: "road"}
        current_epoch = trainer.current_epoch
        global_step = trainer.global_step
        dataloader = trainer.datamodule.val_dataloader()
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if i >= self.num_images:
                break
            batch = module.transfer_batch_to_device(batch, module.device, 0)
            loss, recon, target = self.reconstruct(module, batch)
            recon[recon > self.threshold] = 1.0
            recon[recon <= self.threshold] = 0.0
            image = (
                (batch["image"].squeeze() * 255)
                .permute(1, 2, 0)
                .to(torch.uint8)
                .detach()
                .cpu()
                .numpy()
            )
            trainer.logger.log_image(
                key="test/reconstruction",
                step=self.count,
                caption=[
                    f"Epoch {current_epoch}, Step {global_step}, Loss: {loss:.2f}"
                ],
                images=[image],
                masks=[
                    {
                        "predictions": dict(
                            mask_data=recon.to(torch.uint8).numpy(),
                            class_labels=class_labels,
                        ),
                        "ground_truth": dict(
                            mask_data=target.to(torch.uint8).numpy(),
                            class_labels=class_labels,
                        ),
                    }
                ],
            )
