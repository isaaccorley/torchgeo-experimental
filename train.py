import argparse

import lightning  # noqa: F401
import torch
from hydra.utils import instantiate

import tgexp  # noqa: F401

torch.set_float32_matmul_precision("medium")


def main(args: argparse.Namespace) -> None:
    model = instantiate(args.module)
    datamodule = instantiate(args.datamodule)
    logger = instantiate(args.logger)
    trainer = instantiate(args.trainer, logger=logger)
    trainer.fit(model, datamodule=datamodule)
    trainer.predict(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    main(args)
