import autorootcwd  # noqa: F401
import os
import random
import torch
import torch.nn.functional as F
from src.utils.registry import DATASET_REGISTRY
from src.data.xca_flow import XCAFlow_DataModule  # noqa: F401
from src.archs.unet.unet import DhariwalUNet
from src.archs.components.conditional_flow import SchrodingerBridgeConditionalFlowMatcher
import torchdiffeq
import matplotlib.pyplot as plt

class DhariwalUNet4Channel(torch.nn.Module):
    """4 channel input/output UNet for image inpainting + geometry prediction.
    
    Input: image, noise, coordx, coordy (4 channels)
    Output: image*mask, geometry, coordx, coordy (4 channels)
    - image*mask: 원본 이미지에 마스크를 곱한 결과 (지워야 할 부분)
    - geometry: geometry 출력
    - coordx, coordy: coordinate passthrough
    """
    def __init__(self,
        img_resolution,
        label_dim=0,
        augment_dim=0,
        model_channels=64,
        channel_mult=[1, 2, 3, 4],
        num_blocks=2,
        attn_resolutions=[32, 16, 8],
        dropout=0.0,
    ):
        super().__init__()
        # 4 channel input (image, noise, coordx, coordy)
        # 2 channel output (image*mask, geometry) + 2 channel passthrough (coordx, coordy)
        self.base_unet = DhariwalUNet(
            img_resolution=img_resolution,
            in_channels=4,  # image, noise, coordx, coordy
            out_channels=2,  # image*mask, geometry
            label_dim=label_dim,
            augment_dim=augment_dim,
            model_channels=model_channels,
            channel_mult=channel_mult,
            channel_mult_emb=4,
            num_blocks=num_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            label_dropout=0,
        )

    def forward(self, x, noise_labels, class_labels=None, augment_labels=None):
        """
        Args:
            x: (B, 4, H, W) = [image, noise, coordx, coordy]
            noise_labels: (B,)
        
        Returns:
            (B, 4, H, W) = [image*mask, geometry, coordx, coordy]
        """
        # UNet 처리: image*mask, geometry 출력
        output = self.base_unet(x, noise_labels, class_labels, augment_labels)  # (B, 2, H, W)
        
        # coordinate passthrough
        coordx = x[:, 2:3, :, :]  # (B, 1, H, W)
        coordy = x[:, 3:4, :, :]  # (B, 1, H, W)
        
        # Concat: [image*mask, geometry, coordx, coordy]
        return torch.cat([output, coordx, coordy], dim=1)  # (B, 4, H, W)

def main():
    os.makedirs("results", exist_ok=True)
    
    print("Creating XCA Flow DataModule...")
    dm = DATASET_REGISTRY.get("xca_flow")(train_bs=2)
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    image = batch['image']
    coordinate = batch['coordinate']
    
    print(f"Initial shapes - image: {image.shape}, coordinate: {coordinate.shape}")
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    NEPOCHS = 500
    
    # 모델 생성 - 4 channel input/output
    max_resolution = 512
    model = DhariwalUNet4Channel(
        img_resolution=max_resolution,
        label_dim=0,
        augment_dim=0,
        model_channels=32,
        channel_mult=[1, 2, 2, 3, 3],
        num_blocks=2,
        attn_resolutions=[32, 16, 8, 8, 8],
        dropout=0.0,
    ).to(device)

    # Optimizer: AdamW
    learning_rate = 2e-4
    weight_decay = 1e-5
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    FM = SchrodingerBridgeConditionalFlowMatcher(sigma=0.25)

    patch_plan = [
        (320, 6),
        (384, 4),
        (416, 3),
        (512, 1),
    ]
    
    def select_patch_params(epoch):
        if epoch >= 400:
            return (512, 1)
        return random.choice(patch_plan)

    def random_patch_batch(tensors, patch_size, num_patches):
        if patch_size is None or num_patches <= 0:
            return tensors
        ref = tensors[0]
        batch, _, height, width = ref.shape
        if patch_size > height or patch_size > width:
            return tensors
        total = batch * num_patches
        device = ref.device
        batch_indices = torch.arange(batch, device=device).repeat_interleave(num_patches)
        max_top = height - patch_size
        max_left = width - patch_size
        if max_top < 0 or max_left < 0:
            return tensors
        top = torch.randint(0, max_top + 1, (total,), device=device)
        left = torch.randint(0, max_left + 1, (total,), device=device)
        base_y = torch.arange(patch_size, device=device, dtype=torch.float32)
        base_x = torch.arange(patch_size, device=device, dtype=torch.float32)
        grid_y = top.unsqueeze(1) + base_y.unsqueeze(0)
        grid_x = left.unsqueeze(1) + base_x.unsqueeze(0)
        grid_y = grid_y.unsqueeze(2).expand(-1, -1, patch_size)
        grid_x = grid_x.unsqueeze(1).expand(-1, patch_size, -1)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        if width > 1:
            grid[..., 0] = (grid[..., 0] / (width - 1)) * 2 - 1
        else:
            grid[..., 0] = 0
        if height > 1:
            grid[..., 1] = (grid[..., 1] / (height - 1)) * 2 - 1
        else:
            grid[..., 1] = 0
        patches = []
        for tensor in tensors:
            expanded = tensor[batch_indices]
            needs_float = not torch.is_floating_point(expanded)
            expanded_f = expanded.float() if needs_float else expanded
            patch = F.grid_sample(expanded_f, grid, mode="bilinear", align_corners=True)
            if needs_float:
                patch = patch.to(expanded.dtype)
            patches.append(patch)
        return patches
    
    for epoch in range(NEPOCHS):
        
        for batch in dm.train_dataloader():
            patch_size, num_patches = select_patch_params(epoch)
            with torch.no_grad():
                image = batch['image'].to(device)
                mask = batch['label'].to(device)
                geometry = batch['geometry'].to(device)
                coordinate = batch['coordinate'].to(device)

                # Input: image, noise, coordx, coordy (4 channel)
                coordx = coordinate[:, 0:1, :, :]  # (B, 1, H, W)
                coordy = coordinate[:, 1:2, :, :]  # (B, 1, H, W)
                noise = torch.randn_like(image)
                input_4ch = torch.cat([image, noise, coordx, coordy], dim=1)  # (B, 4, H, W)
                image_masked = image * mask  # (B, 1, H, W)
                target_4ch = torch.cat([image_masked, geometry, coordx, coordy], dim=1)  # (B, 4, H, W)
                input_4ch, target_4ch = random_patch_batch(
                    [input_4ch, target_4ch],
                    patch_size=patch_size,
                    num_patches=num_patches,
                )

            optimizer.zero_grad()
            
            t, xt, ut = FM.sample_location_and_conditional_flow(input_4ch, target_4ch)
            
            # UNet forward
            v = model(xt, t, class_labels=None)  # (B, 4, H, W)
            
            # Loss: image*mask와 geometry만 사용
            v_img_mask = v[:, 0:1, :, :]  # (B, 1, H, W)
            v_geometry = v[:, 1:2, :, :]  # (B, 1, H, W)
            ut_img_mask = ut[:, 0:1, :, :]  # (B, 1, H, W)
            ut_geometry = ut[:, 1:2, :, :]  # (B, 1, H, W)
            
            loss_img = torch.abs(v_img_mask - ut_img_mask).mean()
            loss_geom = torch.abs(v_geometry - ut_geometry).mean()
            loss = 0.1 * loss_img + 0.9 * loss_geom # enforce geometry
            
            loss.backward()
            optimizer.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch}, Loss: {loss.item():.6f} (img: {loss_img.item():.6f}, "
                f"geom: {loss_geom.item():.6f}), LR: {current_lr:.2e}, "
                f"patch={patch_size}, n_patch={num_patches}"
            )

        if epoch % 50 == 0:
            with torch.no_grad():
                model.eval()
                
                def ode_func(t, x):
                    if isinstance(t, torch.Tensor):
                        t = t.expand(x.shape[0])
                    else:
                        t = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
                    return model(x, t, class_labels=None)
                
                eval_batch = next(iter(dm.train_dataloader()))
                eval_image = eval_batch['image'].to(device)
                eval_coordinate = eval_batch['coordinate'].to(device)
                eval_mask = eval_batch['label'].to(device)
                eval_geometry = eval_batch['geometry'].to(device)
                eval_coordx = eval_coordinate[:, 0:1, :, :]
                eval_coordy = eval_coordinate[:, 1:2, :, :]
                eval_noise = torch.randn_like(eval_image)
                eval_input_4ch = torch.cat([eval_image, eval_noise, eval_coordx, eval_coordy], dim=1)
                
                traj = torchdiffeq.odeint(
                    ode_func,
                    eval_input_4ch,
                    torch.linspace(0, 1, 15, device=device),
                    atol=1e-4,
                    rtol=1e-4,
                    method="dopri5"
                )
                output_4ch = traj[-1]  # (B, 4, H, W)
                output_img_mask = output_4ch[:, 0:1, :, :]
                output_geometry = output_4ch[:, 1:2, :, :]
                
                loss_img = torch.abs(output_img_mask - (eval_image * eval_mask)).mean()
                loss_geom = torch.abs(output_geometry - eval_geometry).mean()
                val_loss = loss_img + loss_geom
                print(f"Epoch {epoch}, VAL Loss: {val_loss.item():.6f} (img: {loss_img.item():.6f}, geom: {loss_geom.item():.6f})")
                
                batch_size = eval_image.shape[0]
                fig, axes = plt.subplots(batch_size, 4, figsize=(16, 4 * batch_size))
                
                if batch_size == 1:
                    axes = axes.reshape(1, -1)
                
                for i in range(batch_size):
                    img_np = eval_image[i, 0, :, :].detach().cpu().numpy()
                    geom_np = eval_geometry[i, 0, :, :].detach().cpu().numpy()
                    out_img_mask_np = output_img_mask[i, 0, :, :].detach().cpu().numpy()
                    out_geom_np = output_geometry[i, 0, :, :].detach().cpu().numpy()
                    
                    axes[i, 0].imshow(img_np, cmap='gray', vmin=-1, vmax=1)
                    axes[i, 0].set_title(f'Image {i+1}')
                    axes[i, 0].axis('off')
                    
                    axes[i, 1].imshow(geom_np, cmap='gray', vmin=-1, vmax=1)
                    axes[i, 1].set_title(f'Geometry {i+1}')
                    axes[i, 1].axis('off')
                    
                    axes[i, 2].imshow(out_img_mask_np, cmap='gray', vmin=-1, vmax=1)
                    axes[i, 2].set_title(f'Output img*mask {i+1}')
                    axes[i, 2].axis('off')
                    
                    axes[i, 3].imshow(out_geom_np, cmap='gray', vmin=-1, vmax=1)
                    axes[i, 3].set_title(f'Output geometry {i+1}')
                    axes[i, 3].axis('off')
                
                plt.tight_layout()
                plt.savefig(f"results/cfm_coord_v3_epoch_{epoch}_output.png")
                plt.close()
                
                model.train()

if __name__ == "__main__":
    main()

