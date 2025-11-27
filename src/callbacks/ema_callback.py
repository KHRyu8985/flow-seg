"""EMA (Exponential Moving Average) callback for Lightning using timm's ModelEmaV2."""
import logging
from lightning.pytorch.callbacks import Callback
from timm.utils.model import get_state_dict, unwrap_model
from timm.utils.model_ema import ModelEmaV2

_logger = logging.getLogger(__name__)


class EMAWeightAveraging(Callback):
    """
    Model Exponential Moving Average callback.
    
    Empirically it has been found that using the moving average of the trained parameters
    of a deep network is better than using its trained parameters directly.
    
    Uses timm's ModelEmaV2 for efficient EMA implementation.
    
    If `use_ema_weights`, then the ema parameters of the network is set after training end.
    """
    
    def __init__(self, decay: float = 0.99, use_ema_weights: bool = True):
        """Initialize EMA callback.
        
        Args:
            decay: EMA decay factor (typically 0.9 to 0.9999)
            use_ema_weights: If True, replace model weights with EMA weights after training
        """
        super().__init__()
        self.decay = decay
        self.use_ema_weights = use_ema_weights
        self.ema = None
        self.collected_params = None
    
    def on_fit_start(self, trainer, pl_module):
        """Initialize ModelEmaV2 from timm to keep a copy of the moving average of the weights."""
        self.ema = ModelEmaV2(pl_module, decay=self.decay, device=None)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Update the stored parameters using a moving average."""
        if self.ema is not None:
            self.ema.update(pl_module)
    
    def on_validation_epoch_start(self, trainer, pl_module):
        """Do validation using the stored EMA parameters."""
        if self.ema is None:
            return
        
        # Save original parameters before replacing with EMA version
        self.store(pl_module.parameters())
        
        # Update the LightningModule with the EMA weights
        self.copy_to(self.ema.module.parameters(), pl_module.parameters())
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Restore original parameters to resume training later."""
        if self.collected_params is None:
            return
        
        self.restore(pl_module.parameters())
    
    def on_test_start(self, trainer, pl_module):
        """Do test using the stored EMA parameters."""
        if self.ema is None:
            return
        
        # Save original parameters before replacing with EMA version
        self.store(pl_module.parameters())
        
        # Update the LightningModule with the EMA weights
        self.copy_to(self.ema.module.parameters(), pl_module.parameters())
    
    def on_test_end(self, trainer, pl_module):
        """Restore original parameters after test."""
        if self.collected_params is None:
            return
        
        self.restore(pl_module.parameters())
    
    def on_train_end(self, trainer, pl_module):
        """Update the LightningModule with the EMA weights after training."""
        if self.ema is None or not self.use_ema_weights:
            return
        
        self.copy_to(self.ema.module.parameters(), pl_module.parameters())
        _logger.info("Model weights replaced with the EMA version.")
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Save EMA state dict to checkpoint."""
        if self.ema is not None:
            return {"state_dict_ema": get_state_dict(self.ema, unwrap_model)}
        return None
    
    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """Load EMA state dict from checkpoint."""
        callback_state = checkpoint.get("callbacks", {}).get(self.__class__.__name__)
        if callback_state and self.ema is not None and "state_dict_ema" in callback_state:
            self.ema.module.load_state_dict(callback_state["state_dict_ema"])
    
    def store(self, parameters):
        """Save the current parameters for restoring later."""
        self.collected_params = [param.clone() for param in parameters]
    
    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        
        Useful to validate the model with EMA parameters without affecting the
        original optimization process.
        """
        if self.collected_params is None:
            return
        
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)
        
        self.collected_params = None
    
    def copy_to(self, shadow_parameters, parameters):
        """Copy current parameters into given collection of parameters."""
        for s_param, param in zip(shadow_parameters, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

