"""Diffusion models for vessel segmentation."""
import torch
import lightning.pytorch as L
from monai.inferers import SlidingWindowInferer
from torchmetrics import MetricCollection
from copy import deepcopy

from src.archs.components.gaussian_diffusion import create_medsegdiff
from src.archs.components.binomial_diffusion import create_berdiff

from src.metrics.general_metrics import Dice, Precision, Recall, Specificity, JaccardIndex
from src.metrics.vessel_metrics import clDice, Betti0Error, Betti1Error


MODEL_REGISTRY = {
    'medsegdiff': create_medsegdiff,
    'berdiff': create_berdiff,
}


class DiffusionModel(L.LightningModule):
    """Diffusion segmentation model with sliding window inference."""
    
    def __init__(
        self,
        arch_name: str = 'medsegdiff',
        image_size: int = 224,
        dim: int = 64,
        timesteps: int = 50,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-5,
        num_classes: int = 2,
        experiment_name: str = None,
        data_name: str = 'octa500_3m',
        num_ensemble: int = 1,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        log_image_enabled: bool = True,
        log_image_names: list = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Set experiment name
        if experiment_name is None:
            experiment_name = f"{data_name}/{arch_name}"
        self.experiment_name = experiment_name
        self.data_name = data_name
        
        # Create diffusion model
        if arch_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown architecture: {arch_name}. Choose from {list(MODEL_REGISTRY.keys())}")
        
        create_fn = MODEL_REGISTRY[arch_name]
        self.diffusion_model = create_fn(image_size=image_size, dim=dim, timesteps=timesteps)
        
        # EMA model for stable inference
        self.use_ema = use_ema
        if use_ema:
            self.ema_model = deepcopy(self.diffusion_model)
            self.ema_model.eval()
            self.ema_model.requires_grad_(False)
            self.ema_decay = ema_decay
        else:
            self.ema_model = None
        
        # Sliding window inferer for validation
        self.inferer = SlidingWindowInferer(
            roi_size=(image_size, image_size),
            sw_batch_size=4,
            overlap=0.25,
            mode='gaussian',
        )
        
        # Metrics
        self.val_metrics = MetricCollection({
            'dice': Dice(num_classes=num_classes, average='macro'),
            'precision': Precision(num_classes=num_classes, average='macro'),
            'recall': Recall(num_classes=num_classes, average='macro'),
            'specificity': Specificity(num_classes=num_classes, average='macro'),
            'iou': JaccardIndex(num_classes=num_classes, average='macro'),
        })
        
        self.vessel_metrics = MetricCollection({
            'cldice': clDice(),
            'betti_0_error': Betti0Error(),
            'betti_1_error': Betti1Error(),
        })
        
        self.log_image_enabled = log_image_enabled
        self.log_image_names = log_image_names if log_image_names is not None else ['00036.png']
    
    def forward(self, img: torch.Tensor, cond_img: torch.Tensor) -> torch.Tensor:
        """Forward pass - returns loss during training.
        
        Args:
            img: Ground truth segmentation mask
            cond_img: Conditional image
        """
        return self.diffusion_model(img, cond_img)
    
    def update_ema(self):
        """Update EMA model parameters."""
        if not self.use_ema:
            return
        
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.diffusion_model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)
    
    def sample(self, cond_img: torch.Tensor, save_steps: list = None) -> torch.Tensor:
        """Sample from diffusion model (inference).
        
        This function is called by sliding window inferer for each patch.
        Each patch goes through the full diffusion sampling process.
        
        Uses EMA model if available for more stable inference.
        
        Args:
            cond_img: Conditional image
            save_steps: List of timesteps to save for visualization
        """
        # Use EMA model for inference if available
        model = self.ema_model if (self.use_ema and self.ema_model is not None) else self.diffusion_model
        return model.sample(cond_img, save_steps)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        
        # Convert to float [0, 1] if needed
        if labels.dtype != torch.float32:
            labels = labels.float()
        if labels.max() > 1:
            labels = labels / 255.0
        
        # Compute diffusion loss
        loss = self(labels, images)
        
        # Log
        self.log('train/loss', loss, prog_bar=True)
        
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA model after each training batch."""
        if self.use_ema:
            self.update_ema()
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])
        
        # Convert labels for metrics
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        labels = (labels > 0.5).long()
        
        # Ensemble: multiple sampling and averaging
        if self.hparams.num_ensemble > 1:
            pred_masks_list = []
            for _ in range(self.hparams.num_ensemble):
                pred_result = self.inferer(images, self.sample)
                # Handle dict output from v2 model
                if isinstance(pred_result, dict):
                    pred_masks = pred_result['mask']
                else:
                    pred_masks = pred_result
                pred_masks_list.append(pred_masks)
            # Average predictions
            pred_masks = torch.stack(pred_masks_list).mean(dim=0)
        else:
            # Single sampling
            pred_result = self.inferer(images, self.sample)
            # Handle dict output from v2 model
            if isinstance(pred_result, dict):
                pred_masks = pred_result['mask']
            else:
                pred_masks = pred_result
        
        # Convert predictions to class indices
        if pred_masks.dim() == 4 and pred_masks.shape[1] == 1:
            pred_masks = pred_masks.squeeze(1)
        preds = (pred_masks > 0.5).long()
        
        # Compute metrics
        general_metrics = self.val_metrics(preds, labels)
        vessel_metrics = self.vessel_metrics(preds, labels)
        
        # Log
        self.log_dict({'val/' + k: v for k, v in general_metrics.items()}, prog_bar=True)
        self.log_dict({'val/' + k: v for k, v in vessel_metrics.items()}, prog_bar=False)
        
        self._log_images(sample_names, images, labels, preds, tag_prefix='val')
        
        return general_metrics['dice']
    
    def test_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        
        # Get sample names if available
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])
        
        # Convert labels for metrics
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        labels = (labels > 0.5).long()
        
        # Ensemble: multiple sampling and averaging
        if self.hparams.num_ensemble > 1:
            pred_masks_list = []
            for _ in range(self.hparams.num_ensemble):
                pred_result = self.inferer(images, self.sample)
                # Handle dict output from v2 model
                if isinstance(pred_result, dict):
                    pred_masks = pred_result['mask']
                else:
                    pred_masks = pred_result
                pred_masks_list.append(pred_masks)
            # Average predictions
            pred_masks = torch.stack(pred_masks_list).mean(dim=0)
        else:
            # Single sampling
            pred_result = self.inferer(images, self.sample)
            # Handle dict output from v2 model
            if isinstance(pred_result, dict):
                pred_masks = pred_result['mask']
            else:
                pred_masks = pred_result
        
        # Convert predictions to class indices
        if pred_masks.dim() == 4 and pred_masks.shape[1] == 1:
            pred_masks = pred_masks.squeeze(1)
        preds = (pred_masks > 0.5).long()
        
        # Compute metrics
        general_metrics = self.val_metrics(preds, labels)
        vessel_metrics = self.vessel_metrics(preds, labels)
        
        # Log
        self.log_dict({'test/' + k: v for k, v in general_metrics.items()})
        self.log_dict({'test/' + k: v for k, v in vessel_metrics.items()})
        
        self._log_images(sample_names, images, labels, preds, tag_prefix='test')
        
        # Store predictions for logging
        if hasattr(self.trainer, 'logger') and hasattr(self.trainer.logger, 'save_predictions'):
            pred_masks_binary = (preds > 0).float()
            label_masks = (labels > 0).float()
            
            # Prepare metrics for each sample
            sample_metrics = []
            for i in range(images.shape[0]):
                sample_metric = {}
                sample_metric.update({k: v.item() if torch.is_tensor(v) else v for k, v in general_metrics.items()})
                sample_metric.update({k: v.item() if torch.is_tensor(v) else v for k, v in vessel_metrics.items()})
                sample_metrics.append(sample_metric)
            
            # Save predictions
            self.trainer.logger.save_predictions(
                sample_names, images, pred_masks_binary, label_masks, sample_metrics
            )
        
        return general_metrics['dice']
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=20,
            factor=0.5,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train/loss',
                'interval': 'epoch',
            }
        }

    def _log_images(self, sample_names, images, labels, preds, tag_prefix: str):
        """Log images to TensorBoard similar to supervised model."""
        if not self.log_image_enabled or not hasattr(self.logger, 'experiment'):
            return
        for i, name in enumerate(sample_names):
            if not any(pattern in name for pattern in self.log_image_names):
                continue
            img = (images[i] + 1) / 2
            pred = preds[i].float().unsqueeze(0)
            label = labels[i].float().unsqueeze(0)
            vis_row = torch.cat([img, label, pred], dim=-1)
            image_tag = name.split('/')[-1]
            self.logger.experiment.add_image(
                tag=f'{tag_prefix}/{image_tag}',
                img_tensor=vis_row,
                global_step=self.global_step,
            )
