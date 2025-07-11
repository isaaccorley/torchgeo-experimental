import argparse

import lightning  # noqa: F401
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

import tgexp  # noqa: F401

torch.set_float32_matmul_precision("medium")


def main(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.config)
    model = instantiate(cfg.module)
    datamodule = instantiate(cfg.datamodule)
    logger = instantiate(cfg.logger)
    trainer = instantiate(cfg.trainer, logger=logger)
    trainer.fit(model, datamodule=datamodule)
    trainer.predict(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    main(args)
