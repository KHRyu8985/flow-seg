"""Flow Matching model for vessel segmentation.
Based on FlowSDF trainer.py implementation.
"""
import torch
import lightning.pytorch as L
from monai.inferers import SlidingWindowInferer
from torchmetrics import MetricCollection
from copy import deepcopy
from torchdiffeq import odeint

from src.archs.components.diffusion_unet import MedSegDiffUNet
from src.archs.components.vessel_geometry import to_geometry, from_geometry
from src.metrics.general_metrics import Dice, Precision, Recall, Specificity, JaccardIndex
from src.metrics.vessel_metrics import clDice, Betti0Error, Betti1Error
from monai.losses import PerceptualLoss


class FlowSDFModel(L.LightningModule):
    """Flow Matching model for segmentation based on FlowSDF."""
    
    def __init__(
        self,
        image_size: int = 224,
        dim: int = 64,
        sigma_min: float = 0.00001,
        ode_steps: int = 4,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-5,
        num_classes: int = 2,
        experiment_name: str = None,
        data_name: str = 'octa500_3m',
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        thresh: float = 0.0,
        log_image_enabled: bool = True,
        log_image_names: list = None,
        num_ensemble: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Set experiment name
        if experiment_name is None:
            experiment_name = f"{data_name}/flowsdf"
        self.experiment_name = experiment_name
        self.data_name = data_name
        
        # Flow matching parameters
        self.sigma_min = sigma_min
        self.ode_steps = ode_steps
        self.thresh = thresh
        
        # Create UNet network (MedSegDiff)
        self.net = MedSegDiffUNet(
            dim=dim,
            image_size=image_size,
            mask_channels=1,
            input_img_channels=1,
            apply_tanh=True,
        )
        
        # EMA model for stable inference
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        if use_ema:
            self.ema_model = deepcopy(self.net)
            self.ema_model.eval()
            self.ema_model.requires_grad_(False)
        else:
            self.ema_model = None
        
        # Sliding window inferer for validation
        self.inferer = SlidingWindowInferer(
            roi_size=(image_size, image_size),
            sw_batch_size=8,  # Increased from 4 for faster inference
            overlap=0.25,  # Reduced from 0.25 for faster inference
            mode='gaussian',
        )
        
        # Losses: L2 + Perceptual
        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="resnet50")
        
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
    
    def update_ema(self):
        """Update EMA model parameters."""
        if not self.use_ema:
            return
        
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.net.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)
    
    def forward(self, mt: torch.Tensor, t: torch.Tensor, img_cond: torch.Tensor) -> torch.Tensor:
        """Forward pass of flow network.
        
        Args:
            mt: Noisy mask [B, 1, H, W]
            t: Time [B] or [B, 1]
            img_cond: Conditional image [B, 1, H, W]
        
        Returns:
            Velocity field v [B, 1, H, W]
        """
        # Ensure t is [B]
        if t.dim() > 1:
            t = t.squeeze(-1)
        
        return self.net(mt, t, img_cond)
    
    def training_step(self, batch, batch_idx):
        """Training step following FlowSDF trainer.py."""
        images, labels = batch['image'], batch['label']
        
        # Convert to soft mask using geometry
        soft_labels = to_geometry(labels)  # [B, 1, H, W] in [-1, 1]
        
        # Ensure soft_labels are [B, 1, H, W]
        if soft_labels.dim() == 3:
            soft_labels = soft_labels.unsqueeze(1)
        
        n = soft_labels.shape[0]
        device = soft_labels.device
        
        # Sample random time t ~ U(0, 1)
        t = torch.rand(n, device=device).float()
        
        # Sample noise eta ~ N(0, I)
        eta = torch.randn_like(soft_labels)
        
        # Compute sigma_t and mu_t
        # sigma_t = 1 - (1 - sigma_min) * t
        sigma_t = 1 - (1 - self.sigma_min) * t
        # mu_t = t * m (ground truth soft mask)
        mu_t = t[:, None, None, None] * soft_labels
        
        # Noisy mask: mt = mu_t + sigma_t * eta
        mt = mu_t + sigma_t[:, None, None, None] * eta
        
        # Target velocity: u = (m - (1 - sigma_min) * mt) / (1 - (1 - sigma_min) * t)
        denominator = 1 - (1 - self.sigma_min) * t[:, None, None, None]
        u = (soft_labels - (1 - self.sigma_min) * mt) / denominator
        
        # Ensure images are [B, 1, H, W]
        if images.dim() == 3:
            images = images.unsqueeze(1)
        
        # Predict velocity: v = net(mt, t, img_cond)
        v = self.net(mt, t, images)
        
        # Loss: L2 + Perceptual Loss
        loss1 = torch.abs(v - u).mean()  # L1 loss
        loss2 = self.perceptual_loss(v, u)  # Perceptual loss
        loss = loss1 + 0.1 * loss2
        
        # Log
        self.log('train/loss', loss, prog_bar=True)
        
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA model after each training batch."""
        if self.use_ema:
            self.update_ema()
    
    def sample(self, img_cond: torch.Tensor, save_steps: list = None) -> torch.Tensor:
        """Sample from flow model using ODE solver.
        
        Based on FlowSDF sampler.py implementation.
        
        Args:
            img_cond: Conditional image [B, 1, H, W]
            save_steps: List of timesteps to save (not used in current implementation)
        
        Returns:
            Generated mask [B, 1, H, W]
        """
        # Use EMA model for inference if available
        model = self.ema_model if (self.use_ema and self.ema_model is not None) else self.net
        model.eval()
        
        device = img_cond.device
        
        # Initialize from noise: m0 ~ N(0, I)
        m0 = torch.randn_like(img_cond)
        
        # ODE function: dm/dt = v(m, t, img_cond)
        # Following FlowSDF sampler.py: func_conditional(t, m)
        def func_conditional(t, m):
            # t is scalar, m is [B, 1, H, W]
            t_curr = torch.ones(m.shape[0], device=m.device) * t
            # Interpolate: m_ipt = (1 - (1 - sigma_min) * t) * m0 + t * m
            # This matches FlowSDF sampler.py exactly
            m_ipt = (1 - (1 - self.sigma_min) * t) * m0 + t * m
            # Predict velocity (no_grad not needed here, already in inference_mode)
            v = model(m_ipt, t_curr, img_cond)
            return v
        
        # Solve ODE from t=0 to t=1 using torchdiffeq
        # Using torch.inference_mode() for faster inference
        with torch.inference_mode():
            traj = odeint(
                func_conditional,
                y0=m0,
                t=torch.linspace(0, 1, self.ode_steps, device=device))
        
        # Return final sample (last timestep)
        return traj[-1]
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, labels = batch['image'], batch['label']
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])
        
        # Convert labels for metrics
        if labels.dim() == 4:
            labels = labels.squeeze(1)
        labels = labels.long()
        
        if self.hparams.num_ensemble > 1:
            soft_preds_list = []
            for _ in range(self.hparams.num_ensemble):
                pred_result = self.inferer(images, self.sample)
                if isinstance(pred_result, dict):
                    soft_preds_list.append(pred_result['mask'])
                else:
                    soft_preds_list.append(pred_result)
            soft_preds = torch.stack(soft_preds_list).mean(dim=0)
        else:
            pred_result = self.inferer(images, self.sample)
            soft_preds = pred_result['mask'] if isinstance(pred_result, dict) else pred_result
        
        # Convert soft mask to hard prediction
        hard_preds = from_geometry(soft_preds)
        soft_to_visualize = soft_preds
        if hard_preds.dim() == 4 and hard_preds.size(1) == 1:
            hard_preds = hard_preds.squeeze(1)
        hard_preds = hard_preds.long()
        
        # Compute metrics
        general_metrics = self.val_metrics(hard_preds, labels)
        vessel_metrics = self.vessel_metrics(hard_preds, labels)
        
        # Log
        self.log_dict({'val/' + k: v for k, v in general_metrics.items()}, prog_bar=True)
        self.log_dict({'val/' + k: v for k, v in vessel_metrics.items()}, prog_bar=False)
        
        self._log_images(sample_names, images, labels, hard_preds, soft_to_visualize, tag_prefix='val')
        
        return general_metrics['dice']
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        images, labels = batch['image'], batch['label']
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])
        
        # Convert labels for metrics
        if labels.dim() == 4:
            labels = labels.squeeze(1)
        labels = labels.long()
        
        if self.hparams.num_ensemble > 1:
            soft_preds_list = []
            for _ in range(self.hparams.num_ensemble):
                pred_result = self.inferer(images, self.sample)
                if isinstance(pred_result, dict):
                    soft_preds_list.append(pred_result['mask'])
                else:
                    soft_preds_list.append(pred_result)
            soft_preds = torch.stack(soft_preds_list).mean(dim=0)
        else:
            pred_result = self.inferer(images, self.sample)
            soft_preds = pred_result['mask'] if isinstance(pred_result, dict) else pred_result
        
        # Convert soft mask to hard prediction
        hard_preds = from_geometry(soft_preds)
        if hard_preds.dim() == 4 and hard_preds.size(1) == 1:
            hard_preds = hard_preds.squeeze(1)
        hard_preds = hard_preds.long()
        
        # Compute metrics
        general_metrics = self.val_metrics(hard_preds, labels)
        vessel_metrics = self.vessel_metrics(hard_preds, labels)
        
        # Log
        self.log_dict({'test/' + k: v for k, v in general_metrics.items()})
        self.log_dict({'test/' + k: v for k, v in vessel_metrics.items()})
        
        self._log_images(sample_names, images, labels, hard_preds, soft_preds, tag_prefix='test')
        
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

    def _log_images(self, sample_names, images, labels, hard_preds, soft_preds, tag_prefix: str):
        """Log images to TensorBoard similar to supervised/diffusion models."""
        if not self.log_image_enabled or not hasattr(self.logger, 'experiment'):
            return
        for i, name in enumerate(sample_names):
            if not any(pattern in name for pattern in self.log_image_names):
                continue
            img = (images[i] + 1) / 2
            label = labels[i].float().unsqueeze(0) if labels.dim() == 3 else labels[i].float()
            if label.dim() == 2:
                label = label.unsqueeze(0)
            soft = soft_preds[i].float()
            if soft.dim() == 2:
                soft = soft.unsqueeze(0)
            hard = hard_preds[i].float()
            if hard.dim() == 2:
                hard = hard.unsqueeze(0)
            vis_row = torch.cat([img, label, soft, hard], dim=-1)
            image_tag = name.split('/')[-1]
            self.logger.experiment.add_image(
                tag=f'{tag_prefix}/{image_tag}',
                img_tensor=vis_row,
                global_step=self.global_step,
            )

