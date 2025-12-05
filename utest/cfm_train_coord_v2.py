import autorootcwd
import os
import torch
import torch.nn as nn
from copy import deepcopy
from src.utils.registry import DATASET_REGISTRY
from src.data.xca_flow import XCAFlow_DataModule  # noqa: F401
from src.archs.unet.unet import DhariwalUNetWithCoordinate
from src.archs.components.conditional_flow import SchrodingerBridgeConditionalFlowMatcher
import torchdiffeq
import matplotlib.pyplot as plt

def main():
    os.makedirs("results", exist_ok=True)
    
    print("Creating XCA Flow DataModule...")
    dm = DATASET_REGISTRY.get("xca_flow")()
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    image = batch['image']
    coordinate = batch['coordinate']
    
    print(f"Initial shapes - image: {image.shape}, coordinate: {coordinate.shape}")
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    NEPOCHS = 500
    
    # Epoch 구간별 crop_size 설정
    epoch_1_4 = NEPOCHS // 4
    epoch_2_4 = 2 * NEPOCHS // 4
    epoch_3_4 = 3 * NEPOCHS // 4
    crop_sizes = [160, 240, 320, 448]
    
    # 초기 crop_size 설정
    current_crop_size = crop_sizes[0]
    dm.set_crop_size(current_crop_size)
    dm.setup()
    
    # 모델 생성
    max_resolution = 448
    model_channels = 64
    channel_mult = [1, 2, 3, 4]
    attn_resolutions = [32, 16, 8]
    model = DhariwalUNetWithCoordinate(
        img_resolution=max_resolution,
        label_dim=0,
        augment_dim=0,
        model_channels=model_channels,
        channel_mult=channel_mult,
        num_blocks=2,
        attn_resolutions=attn_resolutions,
        dropout=0.0,
    ).to(device)

    # EMA 모델 생성
    ema_decay = 0.9999
    use_ema = True
    if use_ema:
        ema_model = deepcopy(model)
        ema_model.eval()
        ema_model.requires_grad_(False)
        print("EMA model initialized with decay:", ema_decay)
    
    # Optimizer: AdamW with weight decay
    learning_rate = 2e-4
    weight_decay = 1e-5
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Scheduler: CosineAnnealingLR with warmup
    warmup_epochs = 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NEPOCHS - warmup_epochs,
        eta_min=1e-6
    )
    
    # Warmup scheduler
    def warmup_lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=warmup_lr_lambda
    )
    
    # Gradient clipping
    max_grad_norm = 1.0
    
    FM = SchrodingerBridgeConditionalFlowMatcher(sigma=0.5)
    
    def update_ema():
        """EMA 모델 업데이트"""
        if not use_ema:
            return
        with torch.no_grad():
            for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(ema_decay).add_(model_param.data, alpha=1 - ema_decay)

    for epoch in range(NEPOCHS):
        # Epoch 구간에 따라 crop_size 변경
        if epoch == epoch_1_4:
            current_crop_size = crop_sizes[1]
            print(f"Epoch {epoch}: Changing crop_size to {current_crop_size}")
            dm.set_crop_size(current_crop_size)
            dm.setup()
        elif epoch == epoch_2_4:
            current_crop_size = crop_sizes[2]
            print(f"Epoch {epoch}: Changing crop_size to {current_crop_size}")
            dm.set_crop_size(current_crop_size)
            dm.setup()
        elif epoch == epoch_3_4:
            current_crop_size = crop_sizes[3]
            print(f"Epoch {epoch}: Changing crop_size to {current_crop_size}")
            dm.set_crop_size(current_crop_size)
            dm.setup()
        
        for batch in dm.train_dataloader():
            image = batch['image'].to(device)
            label = batch['label'].to(device)
            geometry = batch['geometry'].to(device)
            coordinate = batch['coordinate'].to(device)
            
            # Patch Diffusion style: image와 coordinate를 3 channel로 concat (flow matching용)
            # image: (B, 1, H, W), coordinate: (B, 2, H, W)
            input_3ch = torch.cat([image, coordinate], dim=1)  # (B, 3, H, W)
            geometry_3ch = torch.cat([geometry, coordinate], dim=1)  # (B, 3, H, W)

            optimizer.zero_grad()
            
            t, xt, ut = FM.sample_location_and_conditional_flow(input_3ch, geometry_3ch)

            # xt에서 image와 coordinate 분리
            xt_img = xt[:, 0:1, :, :]  # (B, 1, H, W)
            xt_coord = xt[:, 1:3, :, :]  # (B, 2, H, W)
            # ut도 분리
            ut_img = ut[:, 0:1, :, :]  # (B, 1, H, W)
            ut_coord = ut[:, 1:3, :, :]  # (B, 2, H, W)
            
            # UNet: unet(input, coordinate, t) 형태로 호출
            v = model(xt_img, xt_coord, t, class_labels=None)  # (B, 3, H, W)
            v_img = v[:, 0:1, :, :]  # (B, 1, H, W)
            v_coord = v[:, 1:3, :, :]  # (B, 2, H, W)
            
            # Loss: image만 사용 (coordinate는 passthrough이므로 ut_coord와 동일해야 함)
            loss = torch.abs(v_img - ut_img).mean()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            update_ema()  # EMA 업데이트
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, LR: {current_lr:.2e}, Crop size: {current_crop_size}")

        # Learning rate scheduling
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        if epoch % 10 == 0:
            with torch.no_grad():
                # 평가 시 EMA 모델 사용
                eval_model = ema_model if use_ema else model
                eval_model.eval()
                
                def ode_func(t, x):
                    # x는 3 channel (image + coordinate)
                    if isinstance(t, torch.Tensor):
                        t = t.expand(x.shape[0])
                    else:
                        t = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
                    x_img = x[:, 0:1, :, :]  # (B, 1, H, W)
                    x_coord = x[:, 1:3, :, :]  # (B, 2, H, W)
                    v = eval_model(x_img, x_coord, t, class_labels=None)  # (B, 3, H, W)
                    return v
                
                # 평가용 배치 가져오기
                eval_batch = next(iter(dm.train_dataloader()))
                eval_image = eval_batch['image'].to(device)
                eval_coordinate = eval_batch['coordinate'].to(device)
                eval_geometry = eval_batch['geometry'].to(device)
                eval_input_3ch = torch.cat([eval_image, eval_coordinate], dim=1)  # (B, 3, H, W)
                
                traj = torchdiffeq.odeint(
                    ode_func,
                    eval_input_3ch,
                    torch.linspace(0, 1, 15, device=device),
                    atol=1e-4,
                    rtol=1e-4,
                    method="dopri5"
                )
                output_3ch = traj[-1]  # (B, 3, H, W)
                output = output_3ch[:, 0:1, :, :]  # (B, 1, H, W) - image만 사용
                loss = torch.abs(output - eval_geometry).mean()
                print(f"Epoch {epoch}, VAL Loss: {loss.item()}")
                
                batch_size = eval_image.shape[0]
                fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
                
                if batch_size == 1:
                    axes = axes.reshape(1, -1)
                
                for i in range(batch_size):
                    img_np = eval_image[i, 0, :, :].detach().cpu().numpy()
                    geom_np = eval_geometry[i, 0, :, :].detach().cpu().numpy()
                    out_np = output[i, 0, :, :].detach().cpu().numpy()
                    
                    axes[i, 0].imshow(img_np, cmap='gray', vmin=-1, vmax=1)
                    axes[i, 0].set_title(f'Image {i+1}')
                    axes[i, 0].axis('off')
                    
                    axes[i, 1].imshow(geom_np, cmap='gray', vmin=-1, vmax=1)
                    axes[i, 1].set_title(f'Geometry {i+1}')
                    axes[i, 1].axis('off')
                    
                    axes[i, 2].imshow(out_np, cmap='gray', vmin=-1, vmax=1)
                    axes[i, 2].set_title(f'Output {i+1}')
                    axes[i, 2].axis('off')
                
                plt.tight_layout()
                plt.savefig(f"results/cfm_coord_epoch_{epoch}_output.png")
                plt.close()
                
                # 모델을 다시 train 모드로
                model.train()

if __name__ == "__main__":
    main()

