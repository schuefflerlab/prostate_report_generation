import argparse

import torch
from lightning import Trainer

from classification.data.data_module import BiopsyDataModule
from classification.model.biopsy_mil_model import MulticlassMILModel
from text.utils.config import load_config

torch.set_float32_matmul_precision('high')


def predict(args):
    if args.checkpoint is None:
        raise ValueError("Checkpoint not specified")

    biopsy_data_module = BiopsyDataModule(feature_path=args.feature_path,
                                          output=args.output,
                                          slide_path=args.slide_path,
                                          filename_template=getattr(args, "filename_template", None))

    model = MulticlassMILModel.load_from_checkpoint(checkpoint_path=args.checkpoint, dim=args.feature_dim,
                                                    output_size=6, output=args.output, strict=False,
                                                    slide_id_pattern=getattr(args, "slide_id_pattern", None),
                                                    slide_id_template=getattr(args, "slide_id_template", "{case}_{specimen}"))

    trainer = Trainer(accelerator="gpu", precision="bf16-mixed", logger=False)
    trainer.predict(model, biopsy_data_module)


def main(args):
    config = load_config(args.config)
    for k, v in config.items():
        # Overwrite CLI arguments
        setattr(args, k, v)

    predict(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, required=False)
    parser.add_argument("--feature-dim", type=int, default=1536)
    parser.add_argument("--output", type=str, required=False)
    parser.add_argument("--slide-path", type=str, required=False)
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()
    main(args)
