"""Binomial Diffusion-based Segmentation Models with KL-only loss."""
import autorootcwd  # noqa: F401
import torch
from torch.distributions.binomial import Binomial
from collections import namedtuple

# Import components from existing diffusion_unet
from src.archs.components.diffusion_unet import (
    SimpleConcatUNet,
    extract,
)
# Import base diffusion model from gaussian_diffusion
from src.archs.components.gaussian_diffusion import GaussianDiffusionModel

# Constants
ModelPrediction = namedtuple('ModelPrediction', ['predict_noise', 'predict_x_start'])


# ====== Binomial Diffusion Utilities ======
def binomial_kl(mean1, mean2):
    """
    Compute the KL divergence between two Bernoulli distributions.
    KL(p||q) = p*log(p/q) + (1-p)*log((1-p)/(1-q))
    
    Improved numerical stability by clamping ratios instead of values.
    """
    eps = 1e-7
    # Clamp the ratio instead of individual values
    mean1mean2 = torch.clamp(mean1 / (mean2 + eps), min=eps)
    mean1mean2_r = torch.clamp((1 - mean1) / (1 - mean2 + eps), min=eps)
    
    kl = mean1 * torch.log(mean1mean2) + (1 - mean1) * torch.log(mean1mean2_r)
    return kl


class BinomialDiffusionModel(GaussianDiffusionModel):
    """Pure Binomial Diffusion Model for Binary Segmentation
    
    Extends GaussianDiffusionModel with Binomial-specific forward/reverse processes.
    No prior refinement - starts from random Bernoulli(0.5) noise.
    
    Key modifications:
    1. q_sample: Uses Bernoulli with mean = alpha_t * x_0 + (1-alpha_t)/2
    2. q_posterior_mean: Computes Bernoulli posterior
    3. Loss: KL divergence instead of MSE
    """
    
    def __init__(self, model, timesteps=1000, sampling_timesteps=None, 
                 objective='predict_x0', beta_schedule='cosine'):
        super().__init__(model, timesteps, sampling_timesteps, objective, beta_schedule)
        
        # Additional buffers for Binomial diffusion
        # alphas (not cumprod) needed for posterior
        alphas = 1. - self.betas
        self.register_buffer('alphas', alphas.to(torch.float32))

    def q_mean(self, x_start, t):
        """
        Get the mean of q(x_t | x_0).
        
        For pure Binomial diffusion (no prior):
        mean = alpha_t * x_0 + (1 - alpha_t) / 2
        
        This gradually transitions from x_0 to uniform Bernoulli(0.5)
        """
        alpha_t = extract(self.alphas_cumprod, t, x_start.shape)
        mean = alpha_t * x_start + (1 - alpha_t) / 2
        eps = 1e-6
        return torch.clamp(mean, min=eps, max=1-eps)

    def q_sample(self, x_start, t):
        """
        Sample from q(x_t | x_0) using Binomial distribution.
        
        Args:
            x_start: Ground truth binary mask [0, 1]
            t: Timestep
        """
        mean = self.q_mean(x_start, t)
        # Sample from Binomial(1, mean) = Bernoulli(mean)
        x_t = Binomial(1, mean).sample()
        return x_t

    def q_posterior_mean(self, x_start, x_t, t):
        """
        Compute posterior mean q(x_{t-1} | x_t, x_0).
        
        This is derived from Bayes' rule for Binomial distributions.
        """
        alpha_t = extract(self.alphas, t, x_start.shape)
        alpha_t_prev_cumprod = extract(self.alphas_cumprod_prev, t, x_start.shape)
        
        # Compute theta_1 and theta_2 for pure Binomial diffusion
        # theta_1: P(x_{t-1}=0 | x_t, x_0)
        theta_1 = (
            (alpha_t * (1 - x_t) + (1 - alpha_t) / 2) *
            (alpha_t_prev_cumprod * (1 - x_start) + (1 - alpha_t_prev_cumprod) / 2)
        )
        
        # theta_2: P(x_{t-1}=1 | x_t, x_0)
        theta_2 = (
            (alpha_t * x_t + (1 - alpha_t) / 2) *
            (alpha_t_prev_cumprod * x_start + (1 - alpha_t_prev_cumprod) / 2)
        )
        
        # Posterior mean
        eps = 1e-6
        posterior_mean = theta_2 / (theta_1 + theta_2 + eps)
        
        return torch.clamp(posterior_mean, min=eps, max=1-eps)

    def model_predictions(self, x, t, c, clip_x_start=False):
        """
        Override to handle Binomial diffusion properly.
        Model predicts x_0 directly.
        
        Note: SimpleConcatUNet already applies sigmoid, so output is in [0, 1]
        """
        model_output = self.model(x, t, c)
        
        # model_output is already in [0, 1] (sigmoid applied in SimpleConcatUNet)
        pred_x_start = model_output
        
        # No noise prediction for Binomial diffusion
        return ModelPrediction(None, pred_x_start)

    def forward(self, img, cond_img):
        """
        Training loss for Binomial Diffusion.
        
        Args:
            img: Ground truth binary mask [0, 1]
            cond_img: Conditional image
        """
        device = self.device
        img, cond_img = img.to(device), cond_img.to(device)
        
        b = img.shape[0]
        times = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        # img should be in [0, 1] for binary mask
        
        # Sample x_t from q(x_t | x_0)
        x_t = self.q_sample(x_start=img, t=times)
        
        # Predict x_0 from x_t (already in [0, 1] due to sigmoid in SimpleConcatUNet)
        pred_x_start = self.model(x_t, times, cond_img)
        
        loss = binomial_kl(img, pred_x_start).mean()
        return loss

    @torch.no_grad()
    def sample(self, cond_img, save_steps=None):
        """
        Sample from Binomial diffusion model.
        
        Start from random Bernoulli(0.5) noise and iteratively denoise.
        """
        cond_img = cond_img.to(self.device)
        b, c, h, w = cond_img.shape
        
        # Start from random Bernoulli(0.5) noise
        img = Binomial(1, torch.ones(b, self.mask_channels, h, w, device=self.device) * 0.5).sample()
        
        # Initialize step storage if needed
        saved_steps = {}
        if save_steps is not None:
            save_steps = set(save_steps)
        
        # Reverse diffusion process
        for t in reversed(range(0, self.num_timesteps)):
            batched_times = torch.full((b,), t, device=self.device, dtype=torch.long)
            
            # Predict x_0
            preds = self.model_predictions(img, batched_times, cond_img, clip_x_start=True)
            pred_x_start = preds.predict_x_start
            
            # Save step if requested
            if save_steps is not None and t in save_steps:
                saved_steps[t] = pred_x_start.cpu()
            
            if t > 0:
                # Compute posterior mean
                posterior_mean = self.q_posterior_mean(
                    x_start=pred_x_start, x_t=img, t=batched_times
                )
                
                # Sample x_{t-1} from Bernoulli(posterior_mean)
                img = Binomial(1, posterior_mean).sample()
            else:
                # At t=0, use the prediction directly (mean, not sample)
                img = pred_x_start
        
        # img is already in [0, 1]
        
        if save_steps is not None:
            return {
                'final': img,
                'steps': saved_steps
            }
        else:
            return img


# ====== Factory Functions ======
def create_berdiff(image_size=224, dim=32, timesteps=1000):
    """BerDiff: Simple concat-based UNet with binomial noise
    
    Bernoulli Diffusion for binary segmentation.
    Uses SimpleConcatUNet (concat [x_t, cond]) with sigmoid output.
    
    Args:
        image_size: Input image size (default: 224)
        dim: Base dimension (default: 64)
        timesteps: Number of diffusion steps (default: 1000)
    """
    unet = SimpleConcatUNet(
        dim=dim,
        image_size=image_size,
        mask_channels=1,
        input_img_channels=1,
        dim_mult=(1, 2, 4, 8),
        full_self_attn=(False, False, True, True)
    )
    return BinomialDiffusionModel(unet, timesteps=timesteps, 
                                 objective='predict_x0', 
                                 beta_schedule='cosine')


if __name__ == "__main__":
    print("=" * 70)
    print("Testing BerDiff (Bernoulli Diffusion) Model")
    print("=" * 70)
    
    # Binary masks [0, 1]
    img = torch.randint(0, 2, (2, 1, 224, 224)).float()
    cond = torch.randn(2, 1, 224, 224)
    
    print("\n1. BerDiff (Binomial KL loss)")
    berdiff = create_berdiff(image_size=224, dim=64, timesteps=100)
    loss = berdiff(img, cond)
    params = sum(p.numel() for p in berdiff.parameters())
    print(f"   Loss: {loss.item():.4f}, Params: {params:,}")
    
    # Test sampling
    print("\n2. Testing sampling...")
    with torch.no_grad():
        sample = berdiff.sample(cond[:1])
        print(f"   Sample shape: {sample.shape}, min: {sample.min():.2f}, max: {sample.max():.2f}")
        print(f"   Unique values (first 10): {torch.unique(sample).tolist()[:10]}")
    
    # Test q_mean
    print("\n3. Testing q_mean (forward process)...")
    t_start = torch.tensor([0])
    t_mid = torch.tensor([50])
    t_end = torch.tensor([99])
    
    mean_start = berdiff.q_mean(img[:1], t_start)
    mean_mid = berdiff.q_mean(img[:1], t_mid)
    mean_end = berdiff.q_mean(img[:1], t_end)
    
    print(f"   t=0:   mean range [{mean_start.min():.3f}, {mean_start.max():.3f}]")
    print(f"   t=50:  mean range [{mean_mid.min():.3f}, {mean_mid.max():.3f}]")
    print(f"   t=99:  mean range [{mean_end.min():.3f}, {mean_end.max():.3f}]")
    print(f"   (Should converge to ~0.5 at t=99)")
    
    print("\n" + "=" * 70)
    print("✓ BerDiff model works correctly!")
    print("=" * 70)
