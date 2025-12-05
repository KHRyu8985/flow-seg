"""Flow matching models for vessel segmentation."""
import autorootcwd  # noqa: F401
import torch
import lightning.pytorch as L
from torchmetrics import MetricCollection
from torchdiffeq import odeint

from src.utils.registry import ARCHS_REGISTRY
from src.archs.components import unet  # noqa: F401 - Register architectures
from src.archs.components.conditional_flow import SchrodingerBridgeConditionalFlowMatcher
from src.metrics.general_metrics import Dice, Precision, Recall, Specificity, JaccardIndex
from src.metrics.vessel_metrics import clDice, Betti0Error, Betti1Error
from src.archs.components.utils import random_patch_batch, select_patch_params

class FlowCoordModel(L.LightningModule):
    """Lightning module for coordinate-aware flow matching producing binary masks."""
    
    def __init__(
        self,
        arch_name: str = 'dhariwal_unet_4channel',
        image_size: int = 512,
        patch_plan: list = [(320, 6), (384, 4), (416, 3), (512, 1)],
        dim: int = 32,
        timesteps: int = 15,
        sigma: float = 0.25,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-5,
        num_classes: int = 2,
        experiment_name: str = None,
        num_ensemble: int = 1,
        data_name: str = 'xca',
        log_image_enabled: bool = False,
        log_image_names: list = None,
        # UNet architecture parameters
        model_channels: int = 32,
        channel_mult: list = [1, 2, 4, 8],
        channel_mult_emb: int = 4,
        num_blocks: int = 3,
        attn_resolutions: list = [16, 16, 8, 8],
        dropout: float = 0.0,
        label_dim: int = 0,
        augment_dim: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Set experiment name
        if experiment_name is None:
            experiment_name = f"{data_name}/{arch_name}"

        self.experiment_name = experiment_name
        self.data_name = data_name
        
        # Get model class from registry
        if arch_name not in ARCHS_REGISTRY:
            raise ValueError(f"Unknown architecture: {arch_name}. Available: {list(ARCHS_REGISTRY.keys())}")
        
        arch_class = ARCHS_REGISTRY.get(arch_name)
        
        self.unet = arch_class(
            img_resolution=image_size,
            model_channels=model_channels,
            channel_mult=channel_mult,
            channel_mult_emb=channel_mult_emb,
            num_blocks=num_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            label_dim=label_dim,
            augment_dim=augment_dim,
        )

        self.flow_matcher = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)
        
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

    def training_step(self, batch, batch_idx):
        images = batch['image']  # condition
        labels = batch['label']
        geometry = batch['geometry']  # target
        
        patch_size, num_patches = select_patch_params(self.hparams.patch_plan)
        
        # Prepare noise (x0) and target geometry (x1)
        noise = torch.randn_like(geometry)
        
        # Random patch extraction
        noise, geometry, images = random_patch_batch(
            [noise, geometry, images], patch_size, num_patches
        )
        
        # Flow matching: noise (x0) -> geometry (x1)
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(noise, geometry)
        
        # UNet forward: (x=xt, time=t, cond=images)
        # dhariwal_concat_unet expects: forward(x, time, cond)
        v = self.unet(xt, t, images)  # xt: noisy geometry, images: condition
        
        # Compute loss
        loss = torch.abs(v - ut).mean()
        
        # Log
        self.log('train/loss', loss, prog_bar=True)
        
        return loss

    def sample(self, input_4ch, return_intermediate: bool = False, save_steps: list = None):
        """Sample from flow matching model (inference).
        
        Args:
            input_4ch: (B, 4, H, W) = [image, noise, coordx, coordy]
            return_intermediate: If True, return intermediate trajectory
            save_steps: List of step indices to save when return_intermediate=True
        
        Returns:
            If return_intermediate=False: output_geometry (B, 1, H, W)
            If return_intermediate=True, save_steps=None: (traj, output_geometry)
            If return_intermediate=True, save_steps=[...]: (saved_steps dict, output_geometry)
        """
        if save_steps is None:
            save_steps = [2, 4, 6, 8, 10, 12, 14]
        
        traj = odeint(
            self.ode_func,
            input_4ch,
            torch.linspace(0, 1, 15, device=input_4ch.device),
            atol=1e-4,
            rtol=1e-4,
            method="dopri5"
        )
        
        output_4ch = traj[-1]
        output_geometry = output_4ch[:, 1:2, :, :]
        
        if return_intermediate:
            if save_steps is not None:
                saved_steps = {t: traj[t][:, 1:2] for t in save_steps}
                return saved_steps, output_geometry
            return traj, output_geometry
        
        return output_geometry

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']
        geometry = batch['geometry']
        coordinate = batch['coordinate']
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])
        
        # Convert labels for metrics
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        labels = (labels > 0.5).long()
        
        # Prepare input
        coordx = coordinate[:, 0:1]
        coordy = coordinate[:, 1:2]
        
        # Ensemble: multiple sampling and averaging
        if self.hparams.num_ensemble > 1:
            output_geometry_list = []
            for _ in range(self.hparams.num_ensemble):
                noise = torch.randn_like(images)
                input_4ch = torch.cat([images, noise, coordx, coordy], dim=1)
                saved_steps, output_geometry = self.sample(input_4ch, return_intermediate=True)
                output_geometry_list.append(output_geometry)
            # Average predictions
            output_geometry = torch.stack(output_geometry_list).mean(dim=0)
        else:
            # Single sampling
            noise = torch.randn_like(images)
            input_4ch = torch.cat([images, noise, coordx, coordy], dim=1)
            saved_steps, output_geometry = self.sample(input_4ch, return_intermediate=True)
        
        # Compute loss
        loss = torch.abs(output_geometry - geometry).mean()
        self.log('val/loss', loss, prog_bar=True)
        
        # Convert predictions to class indices
        if output_geometry.dim() == 4 and output_geometry.shape[1] == 1:
            output_geometry = output_geometry.squeeze(1)
        preds = (output_geometry > 0.0).long()
        
        # Convert geometry for logging (ensure same dimensions as output_geometry)
        if geometry.dim() == 4 and geometry.shape[1] == 1:
            geometry = geometry.squeeze(1)
        
        # Compute metrics
        general_metrics = self.val_metrics(preds, labels)
        vessel_metrics = self.vessel_metrics(preds, labels)
        
        # Log
        self.log_dict({'val/' + k: v for k, v in general_metrics.items()}, prog_bar=True)
        self.log_dict({'val/' + k: v for k, v in vessel_metrics.items()}, prog_bar=False)
        
        self._log_images(sample_names, images, labels, preds, tag_prefix='val')
        self._log_images(sample_names, images, geometry, output_geometry, tag_prefix='val_geometry')
        
        return general_metrics['dice']

    def test_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']
        coordinate = batch['coordinate']
        
        # Get sample names if available
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])
        
        # Convert labels for metrics
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        labels = (labels > 0.5).long()
        
        # Prepare input
        coordx = coordinate[:, 0:1]
        coordy = coordinate[:, 1:2]
        
        # Ensemble: multiple sampling and averaging
        if self.hparams.num_ensemble > 1:
            output_geometry_list = []
            for _ in range(self.hparams.num_ensemble):
                noise = torch.randn_like(images)
                input_4ch = torch.cat([images, noise, coordx, coordy], dim=1)
                output_geometry = self.sample(input_4ch)
                output_geometry_list.append(output_geometry)
            # Average predictions
            output_geometry = torch.stack(output_geometry_list).mean(dim=0)
        else:
            # Single sampling
            noise = torch.randn_like(images)
            input_4ch = torch.cat([images, noise, coordx, coordy], dim=1)
            output_geometry = self.sample(input_4ch)
        
        # Convert predictions to class indices
        if output_geometry.dim() == 4 and output_geometry.shape[1] == 1:
            output_geometry = output_geometry.squeeze(1)
        preds = (output_geometry > 0.0).long()
        
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

    @torch.no_grad()
    def ode_func(self, t, x):
        """ODE function for flow matching.
        
        Args:
            t: Time step (scalar or tensor)
            x: State tensor (B, 4, H, W)
        
        Returns:
            Velocity field from UNet
        """
        if isinstance(t, torch.Tensor):
            t = t.expand(x.shape[0])
        else:
            t = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        return self.unet(x, t, class_labels=None)

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


class FlowModel(L.LightningModule):
    """Lightning module for flow matching producing binary masks."""
    
    def __init__(
        self,
        arch_name: str = 'dhariwal_concat_unet',
        image_size: int = 512,
        patch_plan: list = [(320, 6), (384, 4), (416, 3), (512, 1)],
        dim: int = 32,
        timesteps: int = 15,
        sigma: float = 0.25,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-5,
        num_classes: int = 2,
        experiment_name: str = None,
        num_ensemble: int = 1,
        data_name: str = 'xca',
        log_image_enabled: bool = False,
        log_image_names: list = None,
        # UNet architecture parameters
        model_channels: int = 32,
        channel_mult: list = [1, 2, 4, 8],
        channel_mult_emb: int = 4,
        num_blocks: int = 3,
        attn_resolutions: list = [16, 16, 8, 8],
        dropout: float = 0.0,
        label_dim: int = 0,
        augment_dim: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Set experiment name
        if experiment_name is None:
            experiment_name = f"{data_name}/{arch_name}"
        self.experiment_name = experiment_name
        self.data_name = data_name
        
        # Get model class from registry
        if arch_name not in ARCHS_REGISTRY:
            raise ValueError(f"Unknown architecture: {arch_name}. Available: {list(ARCHS_REGISTRY.keys())}")
        
        arch_class = ARCHS_REGISTRY.get(arch_name)
        
        # For dhariwal_concat_unet, need mask_channels and input_img_channels
        if arch_name == 'dhariwal_concat_unet':
            self.unet = arch_class(
                img_resolution=image_size,
                mask_channels=1,  # geometry output
                input_img_channels=1,  # image condition
                model_channels=model_channels,
                channel_mult=channel_mult,
                channel_mult_emb=channel_mult_emb,
                num_blocks=num_blocks,
                attn_resolutions=attn_resolutions,
                dropout=dropout,
                label_dim=label_dim,
                augment_dim=augment_dim,
            )
        else:
            self.unet = arch_class(
                img_resolution=image_size,
                model_channels=model_channels,
                channel_mult=channel_mult,
                channel_mult_emb=channel_mult_emb,
                num_blocks=num_blocks,
                attn_resolutions=attn_resolutions,
                dropout=dropout,
                label_dim=label_dim,
                augment_dim=augment_dim,
            )

        self.flow_matcher = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)
        
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

    def training_step(self, batch, batch_idx):
        images = batch['image']  # condition
        geometry = batch['geometry']  # target
        
        patch_size, num_patches = select_patch_params(self.hparams.patch_plan)
        
        # Prepare noise (x0) and target geometry
        noise = torch.randn_like(geometry)
        
        # Random patch extraction
        noise, geometry, images = random_patch_batch(
            [noise, geometry, images], patch_size, num_patches
        )
        
        # Flow matching: x (noise) -> geometry
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(noise, geometry)
        
        # UNet forward: (x=xt, time=t, cond=images)
        v = self.unet(xt, t, images)
        
        # Compute loss
        loss = torch.abs(v - ut).mean()
        
        # Log
        self.log('train/loss', loss, prog_bar=True)
        
        return loss

    def sample(self, noise, images, return_intermediate: bool = False, save_steps: list = None):
        """Sample from flow matching model (inference).
        
        Args:
            noise: (B, 1, H, W) - initial noise
            images: (B, 1, H, W) - condition images
            return_intermediate: If True, return intermediate trajectory
            save_steps: List of step indices to save when return_intermediate=True
        
        Returns:
            If return_intermediate=False: output_geometry (B, 1, H, W)
            If return_intermediate=True, save_steps=None: (traj, output_geometry)
            If return_intermediate=True, save_steps=[...]: (saved_steps dict, output_geometry)
        """
        if save_steps is None:
            save_steps = [2, 4, 6, 8, 10, 12, 14]
        
        # Store images for ode_func
        self._sample_images = images
        
        traj = odeint(
            self.ode_func,
            noise,
            torch.linspace(0, 1, 15, device=noise.device),
            atol=1e-4,
            rtol=1e-4,
            method="dopri5"
        )
        
        output_geometry = traj[-1]
        
        if return_intermediate:
            if save_steps is not None:
                saved_steps = {t: traj[t] for t in save_steps}
                return saved_steps, output_geometry
            return traj, output_geometry
        
        return output_geometry

    def validation_step(self, batch, batch_idx):
        images = batch['image']  # condition
        labels = batch['label']
        geometry = batch['geometry']  # target
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])
        
        # Convert labels for metrics
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        labels = (labels > 0.5).long()
        
        # Ensemble: multiple sampling and averaging
        if self.hparams.num_ensemble > 1:
            output_geometry_list = []
            for _ in range(self.hparams.num_ensemble):
                noise = torch.randn_like(geometry)
                saved_steps, output_geometry = self.sample(noise, images, return_intermediate=True)
                output_geometry_list.append(output_geometry)
            # Average predictions
            output_geometry = torch.stack(output_geometry_list).mean(dim=0)
        else:
            # Single sampling
            noise = torch.randn_like(geometry)
            saved_steps, output_geometry = self.sample(noise, images, return_intermediate=True)
        
        # Compute loss
        loss = torch.abs(output_geometry - geometry).mean()
        self.log('val/loss', loss, prog_bar=True)
        
        # Convert predictions to class indices
        if output_geometry.dim() == 4 and output_geometry.shape[1] == 1:
            output_geometry = output_geometry.squeeze(1)
        preds = (output_geometry > 0.0).long()
        
        # Convert geometry for logging (ensure same dimensions as output_geometry)
        if geometry.dim() == 4 and geometry.shape[1] == 1:
            geometry = geometry.squeeze(1)
        
        # Compute metrics
        general_metrics = self.val_metrics(preds, labels)
        vessel_metrics = self.vessel_metrics(preds, labels)
        
        # Log
        self.log_dict({'val/' + k: v for k, v in general_metrics.items()}, prog_bar=True)
        self.log_dict({'val/' + k: v for k, v in vessel_metrics.items()}, prog_bar=False)
        
        self._log_images(sample_names, images, labels, preds, tag_prefix='val')
        self._log_images(sample_names, images, geometry, output_geometry, tag_prefix='val_geometry')
        
        return general_metrics['dice']

    def test_step(self, batch, batch_idx):
        images = batch['image']  # condition
        labels = batch['label']
        
        # Get sample names if available
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])
        
        # Convert labels for metrics
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        labels = (labels > 0.5).long()
        
        # Prepare noise (same shape as geometry would be)
        noise = torch.randn(images.shape[0], 1, images.shape[2], images.shape[3], 
                          device=images.device, dtype=images.dtype)
        
        # Ensemble: multiple sampling and averaging
        if self.hparams.num_ensemble > 1:
            output_geometry_list = []
            for _ in range(self.hparams.num_ensemble):
                noise = torch.randn_like(noise)
                output_geometry = self.sample(noise, images)
                output_geometry_list.append(output_geometry)
            # Average predictions
            output_geometry = torch.stack(output_geometry_list).mean(dim=0)
        else:
            # Single sampling
            output_geometry = self.sample(noise, images)
        
        # Convert predictions to class indices
        if output_geometry.dim() == 4 and output_geometry.shape[1] == 1:
            output_geometry = output_geometry.squeeze(1)
        preds = (output_geometry > 0.0).long()
        
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

    @torch.no_grad()
    def ode_func(self, t, x):
        """ODE function for flow matching.
        
        Args:
            t: Time step (scalar or tensor)
            x: State tensor (B, 1, H, W) - noise/geometry at time t
        
        Returns:
            Velocity field from UNet
        """
        if isinstance(t, torch.Tensor):
            t = t.expand(x.shape[0])
        else:
            t = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        
        # Get condition images (stored during sample)
        images = self._sample_images
        
        # UNet forward: (x=noise, time=t, cond=images)
        return self.unet(x, t, images)

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
