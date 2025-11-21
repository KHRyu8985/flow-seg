"""Supervised learning model for OCT segmentation."""

import torch
import torch.nn as nn
import lightning.pytorch as L
from monai.inferers import SlidingWindowInferer
from torchmetrics import MetricCollection

from src.archs.components.csnet import (
    CSNet
)
from src.archs.components.dscnet import (
    DSCNet
)
from src.metrics.general_metrics import Dice, Precision, Recall, Specificity, JaccardIndex
from src.metrics.vessel_metrics import clDice, Betti0Error, Betti1Error


MODEL_REGISTRY = {
    'csnet': CSNet,
    'dscnet': DSCNet,
}


class SupervisedModel(L.LightningModule):
    """Supervised segmentation model with sliding window inference."""
    
    def __init__(
        self,
        arch_name: str = 'csnet',
        in_channels: int = 1,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-5,
        experiment_name: str = None,
        data_name: str = 'octa500_3m',
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
        
        # 모델 생성 (binary segmentation: num_classes=2)
        if arch_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown architecture: {arch_name}. Choose from {list(MODEL_REGISTRY.keys())}")
        
        model_cls = MODEL_REGISTRY[arch_name]
        self.model = model_cls(in_channels=in_channels, num_classes=2)
        
        # Sliding window inferer for validation (128x128 patches)
        self.inferer = SlidingWindowInferer(
            roi_size=(224, 224),
            sw_batch_size=4,
            overlap=0.25,
            mode='gaussian',
        )
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Metrics (binary segmentation: num_classes=2)
        self.val_metrics = MetricCollection({
            'dice': Dice(num_classes=2, average='macro'),
            'precision': Precision(num_classes=2, average='macro'),
            'recall': Recall(num_classes=2, average='macro'),
            'specificity': Specificity(num_classes=2, average='macro'),
            'iou': JaccardIndex(num_classes=2, average='macro'),
        })
        
        self.vessel_metrics = MetricCollection({
            'cldice': clDice(),
            'betti_0_error': Betti0Error(),
            'betti_1_error': Betti1Error(),
        })
        
        # Image logging config
        self.log_image_enabled = log_image_enabled
        self.log_image_names = log_image_names if log_image_names is not None else ['00036.png']
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        
        # Binary labels: (B, 1, H, W) -> (B, H, W)
        labels = labels.squeeze(1).long()
        
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        
        self.log('train/loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        sample_names = batch.get('name', [])
        
        # Binary labels: (B, 1, H, W) -> (B, H, W)
        labels = labels.squeeze(1).long()
        
        logits = self.inferer(images, self.model)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        
        general_metrics = self.val_metrics(preds, labels)
        vessel_metrics = self.vessel_metrics(preds, labels)
        
        self.log('val/loss', loss, prog_bar=True)
        self.log_dict({'val/' + k: v for k, v in general_metrics.items()}, prog_bar=True)
        self.log_dict({'val/' + k: v for k, v in vessel_metrics.items()}, prog_bar=False)
        
        # Log images to TensorBoard if enabled
        if self.log_image_enabled:
            for i, name in enumerate(sample_names):
                if any(pattern in name for pattern in self.log_image_names):
                    # Normalize images to [0, 1] for visualization
                    img = (images[i] + 1) / 2  # [-1, 1] -> [0, 1]
                    pred = preds[i].float().unsqueeze(0)  # (H, W) -> (1, H, W)
                    label = labels[i].float().unsqueeze(0)  # (H, W) -> (1, H, W)
                    
                    # Stack horizontally: image (left), GT (middle), prediction (right)
                    vis_row = torch.cat([img, label, pred], dim=-1)  # (1, H, 3*W)
                    
                    # Log each sample individually
                    # Use image name (without path) as tag
                    image_tag = name.split('/')[-1]  # e.g., "data/.../00036.png" -> "00036.png"
                    self.logger.experiment.add_image(
                        tag=f'val/{image_tag}',
                        img_tensor=vis_row,
                        global_step=self.global_step,
                    )
        
        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])
        
        # Binary labels: (B, 1, H, W) -> (B, H, W)
        labels = labels.squeeze(1).long()
        
        logits = self.inferer(images, self.model)
        preds = torch.argmax(logits, dim=1)
        
        general_metrics = self.val_metrics(preds, labels)
        vessel_metrics = self.vessel_metrics(preds, labels)

        self.log_dict({'test/' + k: v for k, v in general_metrics.items()})
        self.log_dict({'test/' + k: v for k, v in vessel_metrics.items()})
        
        # Store predictions for logging
        if hasattr(self.trainer, 'logger') and hasattr(self.trainer.logger, 'save_predictions'):
            pred_masks = (preds > 0).float()
            label_masks = (labels > 0).float()
            
            sample_metrics = []
            for i in range(images.shape[0]):
                sample_metric = {}
                sample_metric.update({k: v.item() if torch.is_tensor(v) else v for k, v in general_metrics.items()})
                sample_metric.update({k: v.item() if torch.is_tensor(v) else v for k, v in vessel_metrics.items()})
                sample_metrics.append(sample_metric)
            
            self.trainer.logger.save_predictions(
                sample_names, images, pred_masks, label_masks, sample_metrics
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
