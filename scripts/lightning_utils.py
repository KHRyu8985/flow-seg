"""Common utilities for Lightning training scripts."""
import autorootcwd  # noqa: F401 - ensure project root on sys.path
import os
import sys
import tempfile
from pathlib import Path

import yaml
from lightning.pytorch.callbacks import ModelCheckpoint

from src.utils.registry import DATASET_REGISTRY


def setup_environment():
    """Setup environment variables for training."""
    os.environ['NCCL_P2P_DISABLE'] = '1'


def parse_config(config_path: str) -> dict:
    """Parse YAML config file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary containing parsed config data
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_data_name(config: dict) -> str:
    """Extract data name from config.
    
    Supports both formats:
    - data_name: xca (top-level)
    - data.name: xca (nested)
    
    Args:
        config: Parsed config dictionary
        
    Returns:
        Data name string (e.g., 'xca')
        
    Raises:
        ValueError: If data name is not found in config
    """
    # Try nested format first (data.name)
    data_name = config.get('data', {}).get('name')
    
    # Try top-level format (data_name)
    if data_name is None:
        data_name = config.get('data_name')
    
    if data_name is None:
        raise ValueError("'data_name' or 'data.name' not found in config file.")
    return data_name


def get_datamodule_class(config_path: str):
    """Get DataModule class from registry based on config.
    
    Args:
        config_path: Path to config file
        
    Returns:
        DataModule class from registry
    """
    # Import to ensure registration happens
    from src.data.xca import XCA_DataModule  # noqa: F401
    
    config = parse_config(config_path)
    data_name = get_data_name(config)
    
    # Check if name exists directly in registry
    if data_name in DATASET_REGISTRY:
        DataModuleClass = DATASET_REGISTRY._obj_map[data_name]
    else:
        # Try with registry's get method (handles suffix)
        try:
            DataModuleClass = DATASET_REGISTRY.get(data_name)
        except KeyError:
            available = list(DATASET_REGISTRY.keys())
            raise ValueError(f"Dataset '{data_name}' not found. Available: {available}")
    
    return DataModuleClass


def prepare_config_for_lightning(config_path: str) -> str:
    """Remove 'data.name' from config for LightningCLI compatibility.
    
    Args:
        config_path: Path to original config file
        
    Returns:
        Path to prepared config file (temp file if modified)
    """
    config = parse_config(config_path)
    
    if 'data' in config and 'name' in config['data']:
        del config['data']['name']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(config, tmp)
            temp_path = tmp.name
        
        # Update sys.argv if --config is present
        if '--config' in sys.argv:
            idx = sys.argv.index('--config')
            if idx + 1 < len(sys.argv):
                sys.argv[idx + 1] = temp_path
        
        return temp_path
    
    return config_path


def process_arch_name_arg():
    """Convert --arch_name to --model.arch_name format."""
    if '--arch_name' not in sys.argv:
        return
    
    idx = sys.argv.index('--arch_name')
    if idx + 1 >= len(sys.argv):
        return
    
    arch_name = sys.argv[idx + 1]
    sys.argv.pop(idx)  # Remove --arch_name
    sys.argv.pop(idx)  # Remove value
    
    # Add LightningCLI format
    sys.argv.extend(['--model.arch_name', arch_name])


def ensure_subcommand(default: str = 'fit') -> str:
    """Ensure LightningCLI subcommand exists by inserting default if missing."""
    subcommands = ['fit', 'validate', 'test', 'predict']
    if len(sys.argv) > 1 and sys.argv[1] in subcommands:
        return sys.argv[1]
    sys.argv.insert(1, default)
    return default


def get_config_path_from_argv() -> str | None:
    """Return config path from CLI arguments if present."""
    if '--config' in sys.argv:
        idx = sys.argv.index('--config')
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return None


def find_best_checkpoint(trainer) -> str | None:
    """Find best (or last) checkpoint from ModelCheckpoint callbacks."""
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint) and cb.best_model_path:
            return cb.best_model_path
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint):
            ckpt_dir = Path(cb.dirpath)
            last_ckpt = ckpt_dir / "last.ckpt"
            if last_ckpt.exists():
                return str(last_ckpt)
    return None

