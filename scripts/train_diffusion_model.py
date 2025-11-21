"""Diffusion model training/testing entry point.

Usage:
    python scripts/train_diffusion_model.py --config configs/diffusion/medsegdiff.yaml
    python scripts/train_diffusion_model.py --config configs/diffusion/berdiff.yaml

Script (nohup bash):
nohup bash -c 'source .venv/bin/activate && uv run python scripts/train_diffusion_model.py --config configs/diffusion/medsegdiff.yaml' > scripts/logs/medsegdiff_train.log 2>&1 &
nohup bash -c 'source .venv/bin/activate && uv run python scripts/train_diffusion_model.py --config configs/diffusion/berdiff.yaml' > scripts/logs/berdiff_train.log 2>&1 &
"""

import torch
import autorootcwd  # noqa: F401
from lightning.pytorch.cli import LightningCLI

from src.archs.diffusion_model import DiffusionModel
from scripts.lightning_utils import (
    setup_environment,
    ensure_subcommand,
    process_arch_name_arg,
    get_config_path_from_argv,
    prepare_config_for_lightning,
    get_datamodule_class,
    find_best_checkpoint,
)


def main():
    setup_environment()
    torch.set_float32_matmul_precision('medium')
    
    subcommand = ensure_subcommand()
    process_arch_name_arg()
    
    config_path = get_config_path_from_argv()
    if not config_path:
        print("Error: --config required")
        return
    
    prepare_config_for_lightning(config_path)
    DataModuleClass = get_datamodule_class(config_path)
    
    cli = LightningCLI(
        DiffusionModel,
        DataModuleClass,
        save_config_kwargs={'overwrite': True},
    )
    
    if subcommand == 'fit':
        ckpt_path = find_best_checkpoint(cli.trainer) or 'last'
        print(f"\n✅ Training finished. Running test with checkpoint: {ckpt_path}")
        cli.trainer.test(
            cli.model,
            datamodule=cli.datamodule,
            ckpt_path=ckpt_path,
        )


if __name__ == "__main__":
    main()

