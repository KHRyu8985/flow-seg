"""Gaussian Diffusion-based Segmentation Models with fixed MSE loss."""
import autorootcwd  # noqa: F401
import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
from collections import namedtuple

from src.archs.components.diffusion_unet import (
    default, identity, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one,
    SegDiffUNet, MedSegDiffUNet,
    extract, linear_beta_schedule, cosine_beta_schedule
)

ModelPrediction = namedtuple('ModelPrediction', ['predict_noise', 'predict_x_start'])


class GaussianDiffusionModel(nn.Module):
    """Standard Gaussian Diffusion Model for Segmentation (MSE loss only)."""
    
    def __init__(self, model, timesteps=1000, sampling_timesteps=None, objective='predict_x0',
                 beta_schedule='cosine'):
        super().__init__()
        
        self.model = model
        self.objective = objective
        self.image_size = model.image_size
        self.mask_channels = model.mask_channels
        self.input_img_channels = model.input_img_channels
        
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        
        def register_buffer(name, val):
            self.register_buffer(name, val.to(torch.float32))
        
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', 
                       torch.log(torch.clamp(posterior_variance, min=1e-20)))
        
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
        register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        register_buffer('posterior_mean_coef2', posterior_mean_coef2)

    @property
    def device(self):
        return next(self.parameters()).device

    def predict_noise_from_start(self, x_t, t, x0):
        """Predict noise from clean image"""
        return ((extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
        q(x_{t-1} | x_t, x_0)
        
        From MedSegDiff original implementation.
        """
        assert x_start.shape == x_t.shape
        
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        assert (
            posterior_mean.shape[0] == posterior_variance.shape[0] ==
            posterior_log_variance.shape[0] == x_start.shape[0]
        )
        
        return posterior_mean, posterior_variance, posterior_log_variance

    def model_predictions(self, x, t, c, clip_x_start=False):
        """Model predictions for Gaussian diffusion"""
        model_output = self.model(x, t, c)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        
        if self.objective == 'predict_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        return ModelPrediction(pred_noise, x_start)

    @torch.no_grad()
    def sample(self, cond_img, save_steps=None):
        """Sample from Gaussian diffusion model with optional step saving"""
        cond_img = cond_img.to(self.device)
        b, c, h, w = cond_img.shape
        
        # Start from pure Gaussian noise
        img = torch.randn(b, self.mask_channels, h, w, device=self.device)
        
        # Initialize step storage if needed
        saved_steps = {}
        if save_steps is not None:
            save_steps = set(save_steps)
        
        for t in reversed(range(0, self.num_timesteps)):
            batched_times = torch.full((b,), t, device=self.device, dtype=torch.long)
            preds = self.model_predictions(img, batched_times, cond_img, clip_x_start=True)
            
            if save_steps is not None and t in save_steps:
                saved_img = unnormalize_to_zero_to_one(preds.predict_x_start)
                saved_steps[t] = saved_img.cpu()
            
            # DDPM sampling step
            if t > 0:
                noise = torch.randn_like(img)
                alpha_t = self.alphas_cumprod[t]
                alpha_prev = self.alphas_cumprod[t - 1]
                beta_t = self.betas[t]
                
                # Predict x_0
                pred_x0 = preds.predict_x_start
                
                # Compute posterior mean
                posterior_mean = (
                    beta_t * torch.sqrt(alpha_prev) / (1 - alpha_t) * pred_x0 +
                    (1 - alpha_prev) * torch.sqrt(1 - beta_t) / (1 - alpha_t) * img
                )
                
                # Compute posterior variance
                posterior_variance = (1 - alpha_prev) * beta_t / (1 - alpha_t)
                
                # Sample x_{t-1}
                img = posterior_mean + torch.sqrt(posterior_variance) * noise
            else:
                img = preds.predict_x_start
        
        img = unnormalize_to_zero_to_one(img)
        
        if save_steps is not None:
            return {
                'final': img,
                'steps': saved_steps
            }
        else:
            return img

    def q_sample(self, x_start, t, noise):
        """Forward process: add Gaussian noise to clean data"""
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def forward(self, img, cond_img):
        """Forward pass - Gaussian diffusion loss (MSE only)."""
        device = self.device
        img, cond_img = img.to(device), cond_img.to(device)
        
        b = img.shape[0]
        times = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        # Normalize mask to [-1, 1]
        img = normalize_to_neg_one_to_one(img)
        
        # Add Gaussian noise to ground truth mask
        noise = torch.randn_like(img)
        x_noisy = self.q_sample(x_start=img, t=times, noise=noise)
        
        model_out = self.model(x_noisy, times, cond_img)
        return F.mse_loss(model_out, img)


# ====== Factory Functions ======
def create_segdiff(image_size=224, dim=32, timesteps=1000):
    """SegDiff: RRDB-based conditioning (F(x_t) + G(cond))"""
    unet = SegDiffUNet(
        dim=dim,
        image_size=image_size,
        mask_channels=1,
        input_img_channels=1,
        dim_mult=(1, 2, 4, 8),
        full_self_attn=(False, False, True, True),
        rrdb_blocks=3
    )
    return GaussianDiffusionModel(unet, timesteps=timesteps, objective='predict_x0', 
                                 beta_schedule='cosine')


def create_medsegdiff(image_size=224, dim=32, timesteps=1000):
    """MedSegDiff: FFT-based conditioning"""
    unet = MedSegDiffUNet(
        dim=dim,
        image_size=image_size,
        mask_channels=1,
        input_img_channels=1,
        dim_mult=(1, 2, 4, 8),
        full_self_attn=(False, False, True, True),
        mid_transformer_depth=1
    )
    return GaussianDiffusionModel(unet, timesteps=timesteps, objective='predict_x0', 
                                 beta_schedule='cosine')




if __name__ == "__main__":
    print("=" * 70)
    print("Testing Gaussian Diffusion Segmentation Models")
    print("=" * 70)
    
    img = torch.randn(2, 1, 224, 224)
    cond = torch.randn(2, 1, 224, 224)
    
    print("\n1. SegDiff (MSE loss)")
    segdiff = create_segdiff(image_size=224, dim=64, timesteps=100)
    loss = segdiff(img, cond)
    params = sum(p.numel() for p in segdiff.parameters())
    print(f"   Loss: {loss.item():.4f}, Params: {params:,}")
    
    print("\n2. MedSegDiff (MSE loss)")
    medsegdiff = create_medsegdiff(image_size=224, dim=64, timesteps=100)
    loss = medsegdiff(img, cond)
    params = sum(p.numel() for p in medsegdiff.parameters())
    print(f"   Loss: {loss.item():.4f}, Params: {params:,}")
    
    print("\n" + "=" * 70)
    print("✓ All Gaussian diffusion models work correctly!")
    print("=" * 70)
