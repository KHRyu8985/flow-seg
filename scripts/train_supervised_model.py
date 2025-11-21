"""Supervised training script for segmentation models.

Usage:
    python scripts/train_supervised_model.py --config configs/supervised/csnet.yaml
    python scripts/train_supervised_model.py --config configs/supervised/csnet.yaml --arch_name dscnet
    python scripts/train_supervised_model.py --config configs/supervised/csnet.yaml fit  # explicit subcommand

Script (nohup bash):
nohup bash -c 'source .venv/bin/activate && uv run python scripts/train_supervised_model.py --config configs/supervised/csnet.yaml' > scripts/logs/csnet_train.log 2>&1 &
nohup bash -c 'source .venv/bin/activate && uv run python scripts/train_supervised_model.py --config configs/supervised/csnet.yaml' > scripts/logs/csnet_train.log 2>&1 &
"""

import torch
import autorootcwd  # noqa: F401 - ensure project root on sys.path
from lightning.pytorch.cli import LightningCLI

from src.archs.supervised_model import SupervisedModel
from scripts.lightning_utils import (
    setup_environment,
    get_datamodule_class,
    prepare_config_for_lightning,
    process_arch_name_arg,
    ensure_subcommand,
    get_config_path_from_argv,
    find_best_checkpoint,
)


def main():
    """Main training function."""
    setup_environment()
    torch.set_float32_matmul_precision('medium')
    
    subcommand = ensure_subcommand()
    
    process_arch_name_arg()
    
    # Get config path
    config_path = get_config_path_from_argv()
    
    if not config_path:
        print("Error: --config required")
        return
    
    # Prepare config and get DataModule
    prepare_config_for_lightning(config_path)
    DataModuleClass = get_datamodule_class(config_path)
    
    # Run training
    cli = LightningCLI(
        SupervisedModel,
        DataModuleClass,
        save_config_kwargs={'overwrite': True},
    )
    
    # After fitting, automatically run test using best checkpoint
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
