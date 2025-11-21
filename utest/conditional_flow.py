import autorootcwd
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.utils.registry import DATASET_REGISTRY
from src.utils.visualize_dataloader import visualize_flow_dataset
from src.data.xca_flow import XCAFlow_DataModule  # noqa: F401
from src.archs.components.conditional_flow import (
    ConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher
)

def plot_trajectory(flow_matcher, x0, x1, num_steps=10, name="trajectory"):
    """
    Flow matching trajectory를 시각화
    
    Args:
        flow_matcher: Flow matcher instance
        x0: Source tensor (B, C, H, W)
        x1: Target tensor (B, C, H, W)
        num_steps: Number of time steps
        name: Output filename
    """
    # 첫 번째 샘플만 사용
    x0_sample = x0[0:1]  # (1, C, H, W)
    x1_sample = x1[0:1]  # (1, C, H, W)
    
    # t 값들을 생성 (0부터 1까지)
    t_values = torch.linspace(0, 1, num_steps)
    
    # 각 t에 대해 xt 계산
    trajectories = []
    for t in t_values:
        t_batch = t.unsqueeze(0)  # (1,)
        epsilon = flow_matcher.sample_noise_like(x0_sample)
        xt = flow_matcher.sample_xt(x0_sample, x1_sample, t_batch, epsilon)
        trajectories.append(xt.squeeze(0).squeeze(0))  # (H, W)
    
    # 시각화: x0, trajectory steps, x1
    fig, axes = plt.subplots(1, num_steps + 2, figsize=(3 * (num_steps + 2), 3))
    
    # x0
    x0_vis = x0_sample.squeeze(0).squeeze(0).detach().cpu().numpy()
    axes[0].imshow(x0_vis, cmap='gray', vmin=-1.0, vmax=1.0)
    axes[0].set_title('x0 (image)', fontsize=10)
    axes[0].axis('off')
    
    # Trajectory steps
    for i, xt in enumerate(trajectories):
        xt_vis = xt.detach().cpu().numpy()
        axes[i + 1].imshow(xt_vis, cmap='gray', vmin=-1.0, vmax=1.0)
        axes[i + 1].set_title(f't={t_values[i]:.2f}', fontsize=10)
        axes[i + 1].axis('off')
    
    # x1
    x1_vis = x1_sample.squeeze(0).squeeze(0).detach().cpu().numpy()
    axes[-1].imshow(x1_vis, cmap='RdBu', vmin=-1.0, vmax=1.0)
    axes[-1].set_title('x1 (geometry)', fontsize=10)
    axes[-1].axis('off')
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{name}.png", dpi=300, bbox_inches="tight")
    print(f"Trajectory saved to results/{name}.png")
    plt.close()

def main():
    # results 폴더 생성
    os.makedirs("results", exist_ok=True)
    
    # 데이터 모듈 생성 및 설정
    print("Creating XCA Flow DataModule...")
    dm = DATASET_REGISTRY.get("xca_flow")()
    dm.setup()
    
    batch = next(iter(dm.train_dataloader()))
    image = batch['image']  # (B, 1, H, W)
    label = batch['label']
    geometry = batch['geometry']  # (B, 1, H, W)
    coordinate = batch['coordinate']

    # 여러 flow matcher 테스트
    print("\nPlotting Conditional Flow Matcher trajectory...")
    flow_matcher = ConditionalFlowMatcher(sigma=0.2)
    plot_trajectory(flow_matcher, image, geometry, num_steps=20, name="cfm_trajectory")
    
    print("\nPlotting Variance Preserving Flow Matcher trajectory...")
    vp_flow_matcher = VariancePreservingConditionalFlowMatcher(sigma=0.5)
    plot_trajectory(vp_flow_matcher, image, geometry, num_steps=20, name="vp_cfm_trajectory")
    
    print("\nPlotting Schrodinger Bridge Flow Matcher trajectory...")
    sb_flow_matcher = SchrodingerBridgeConditionalFlowMatcher(sigma=0.5)
    plot_trajectory(sb_flow_matcher, image, geometry, num_steps=20, name="sb_cfm_trajectory")
    
    print("\n✓ All trajectories plotted!")

if __name__ == "__main__":
    main()

