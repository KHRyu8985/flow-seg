import autorootcwd
import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_image(*img, name=None):
    """
    여러 이미지를 좌우로 배치하고 위아래로 스택하여 플롯.
    
    Args:
        *img: 각각 Nx1xHxW 형태의 텐서들 (torch.Tensor 또는 np.ndarray)
        name: 저장할 파일 이름 (optional). 제공되면 results/{name}.png로 저장
    
    Returns:
        fig, ax: matplotlib figure와 axes 객체
    """
    if len(img) == 0:
        raise ValueError("At least one image must be provided")
    
    # 텐서를 numpy로 변환하고 처리
    processed_images = []
    for im in img:
        # torch.Tensor면 numpy로 변환
        if isinstance(im, torch.Tensor):
            im = im.detach().cpu().numpy()
        
        # Nx1xHxW 형태 확인 및 처리
        if im.ndim == 4:
            N, C, H, W = im.shape
            if C != 1:
                raise ValueError(f"Expected channel=1, got {C}")
            # Nx1xHxW -> NxHxW
            im = im.squeeze(1)
        elif im.ndim == 3:
            N, H, W = im.shape
        else:
            raise ValueError(f"Expected shape Nx1xHxW or NxHxW, got {im.shape}")
        
        # N개 이미지를 좌우로 concat: NxHxW -> Hx(N*W)
        im_concat = np.concatenate([im[i] for i in range(N)], axis=1)  # H x (N*W)
        processed_images.append(im_concat)
    
    # 여러 이미지 세트를 위아래로 스택
    stacked = np.concatenate(processed_images, axis=0)  # (num_imgs*H) x (N*W)
    
    # 플롯
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    im_plot = ax.imshow(stacked, cmap='gray', vmin=-1.0, vmax=1.0)
    ax.axis('off')
    plt.colorbar(im_plot, ax=ax, fraction=0.046, pad=0.04)
    
    # 이름이 제공되면 제목 설정 및 저장
    if name is not None:
        fig.suptitle(name, fontsize=16, fontweight="bold")
        import os
        os.makedirs("results", exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"results/{name}.png", dpi=300, bbox_inches="tight")
    
    return fig, ax 

def visualize_dataset(loader, dataset_name='octa', num_samples=5):
    # 첫 번째 batch에서 키 확인
    first_item = next(iter(loader))
    has_label_prob = "label_prob" in first_item
    has_label_sauna = "label_sauna" in first_item

    num_cols = 2
    if has_label_prob:
        num_cols += 1
    if has_label_sauna:
        num_cols += 1

    fig_width = 3 * num_cols
    fig_height = 4 * num_samples

    if num_samples == 1:
        fig, axes = plt.subplots(1, num_cols, figsize=(fig_width, fig_height))
        axes = axes.reshape(1, -1)
    else:
        fig, axes = plt.subplots(num_samples, num_cols, figsize=(fig_width, fig_height))

    plotted = 0
    for batch in loader:
        batch_size = batch["image"].shape[0]
        for b in range(batch_size):
            if plotted == num_samples:
                break

            # 각 배치에서 개별 샘플 추출
            image = batch["image"][b].squeeze(0)
            label = batch["label"][b].squeeze(0)

            print(f"Sample {plotted+1} - {batch['name'][b]}:")
            print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
            print(f"  Label range: [{label.min():.3f}, {label.max():.3f}]")
            if has_label_prob:
                label_prob = batch["label_prob"][b].squeeze(0)
                print(f"  Label_prob range: [{label_prob.min():.3f}, {label_prob.max():.3f}]")
            if has_label_sauna:
                label_sauna = batch["label_sauna"][b].squeeze(0)
                print(f"  Label_sauna range: [{label_sauna.min():.3f}, {label_sauna.max():.3f}]")

            if len(image.shape) == 2:
                cmap = "gray"
            elif len(image.shape) == 3:
                image = image.permute(1, 2, 0)
                cmap = None
            else:
                raise ValueError(f"Invalid image shape: {image.shape}")

            col_idx = 0
            im1 = axes[plotted, col_idx].imshow(image, cmap=cmap, vmin=-1.0, vmax=1.0)
            axes[plotted, col_idx].set_title(f"Image {plotted+1}", fontsize=12, fontweight="bold")
            axes[plotted, col_idx].text(
                0.5,
                -0.1,
                f"file:{batch['name'][b]}",
                fontsize=8,
                ha="center",
                va="top",
                transform=axes[plotted, col_idx].transAxes,
            )
            axes[plotted, col_idx].axis("off")
            if cmap:
                plt.colorbar(im1, ax=axes[plotted, col_idx], fraction=0.046, pad=0.04)
            col_idx += 1

            im2 = axes[plotted, col_idx].imshow(label, cmap="gray")
            axes[plotted, col_idx].set_title(f"Label {plotted+1}", fontsize=12, fontweight="bold")
            axes[plotted, col_idx].axis("off")
            plt.colorbar(im2, ax=axes[plotted, col_idx], fraction=0.046, pad=0.04)
            col_idx += 1

            if has_label_prob:
                im3 = axes[plotted, col_idx].imshow(label_prob, cmap="viridis")
                axes[plotted, col_idx].set_title(f"Label_prob {plotted+1}", fontsize=12, fontweight="bold")
                axes[plotted, col_idx].axis("off")
                plt.colorbar(im3, ax=axes[plotted, col_idx], fraction=0.046, pad=0.04)
                col_idx += 1

            if has_label_sauna:
                im4 = axes[plotted, col_idx].imshow(label_sauna, cmap="plasma")
                axes[plotted, col_idx].set_title(f"Label_sauna {plotted+1}", fontsize=12, fontweight="bold")
                axes[plotted, col_idx].axis("off")
                plt.colorbar(im4, ax=axes[plotted, col_idx], fraction=0.046, pad=0.04)

            plotted += 1
        if plotted == num_samples:
            break

    fig.suptitle(dataset_name, fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"results/{dataset_name}.png", dpi=300, bbox_inches="tight")
    plt.show()

def visualize_flow_dataset(loader, dataset_name='xca_flow', num_samples=5):
    """
    Flow dataset용 시각화: image, label, geometry, coordinate (x, y) 확인
    """
    first_item = next(iter(loader))
    has_geometry = "geometry" in first_item
    has_coordinate = "coordinate" in first_item

    num_cols = 2  # image, label
    if has_geometry:
        num_cols += 1
    if has_coordinate:
        num_cols += 2  # coordinate x, y

    fig_width = 3 * num_cols
    fig_height = 4 * num_samples

    if num_samples == 1:
        fig, axes = plt.subplots(1, num_cols, figsize=(fig_width, fig_height))
        axes = axes.reshape(1, -1)
    else:
        fig, axes = plt.subplots(num_samples, num_cols, figsize=(fig_width, fig_height))

    plotted = 0
    for batch in loader:
        batch_size = batch["image"].shape[0]
        for b in range(batch_size):
            if plotted == num_samples:
                break

            # 각 배치에서 개별 샘플 추출
            image = batch["image"][b].squeeze(0)  # (1, H, W) -> (H, W)
            label = batch["label"][b].squeeze(0)  # (1, H, W) -> (H, W)

            print(f"Sample {plotted+1} - {batch['name'][b]}:")
            print(f"  Image shape: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")
            print(f"  Label shape: {label.shape}, range: [{label.min():.3f}, {label.max():.3f}]")
            
            if has_geometry:
                geometry = batch["geometry"][b].squeeze(0)  # (1, H, W) -> (H, W)
                print(f"  Geometry shape: {geometry.shape}, range: [{geometry.min():.3f}, {geometry.max():.3f}]")
            
            if has_coordinate:
                coordinate = batch["coordinate"][b]  # (2, H, W)
                coord_x = coordinate[0]  # (H, W)
                coord_y = coordinate[1]  # (H, W)
                print(f"  Coordinate shape: {coordinate.shape}, X range: [{coord_x.min():.3f}, {coord_x.max():.3f}], Y range: [{coord_y.min():.3f}, {coord_y.max():.3f}]")

            col_idx = 0

            # Image
            im1 = axes[plotted, col_idx].imshow(image, cmap="gray", vmin=-1.0, vmax=1.0)
            axes[plotted, col_idx].set_title(f"Image {plotted+1}", fontsize=12, fontweight="bold")
            axes[plotted, col_idx].text(
                0.5, -0.1, f"file:{batch['name'][b]}",
                fontsize=8, ha="center", va="top",
                transform=axes[plotted, col_idx].transAxes,
            )
            axes[plotted, col_idx].axis("off")
            plt.colorbar(im1, ax=axes[plotted, col_idx], fraction=0.046, pad=0.04)
            col_idx += 1

            # Label
            im2 = axes[plotted, col_idx].imshow(label, cmap="gray", vmin=0.0, vmax=1.0)
            axes[plotted, col_idx].set_title(f"Label {plotted+1}", fontsize=12, fontweight="bold")
            axes[plotted, col_idx].axis("off")
            plt.colorbar(im2, ax=axes[plotted, col_idx], fraction=0.046, pad=0.04)
            col_idx += 1

            # Geometry
            if has_geometry:
                im3 = axes[plotted, col_idx].imshow(geometry, cmap="RdBu", vmin=-1.0, vmax=1.0)
                axes[plotted, col_idx].set_title(f"Geometry {plotted+1}", fontsize=12, fontweight="bold")
                axes[plotted, col_idx].axis("off")
                plt.colorbar(im3, ax=axes[plotted, col_idx], fraction=0.046, pad=0.04)
                col_idx += 1

            # Coordinate X
            if has_coordinate:
                im4 = axes[plotted, col_idx].imshow(coord_x, cmap="viridis", vmin=-1.0, vmax=1.0)
                axes[plotted, col_idx].set_title(f"Coord X {plotted+1}", fontsize=12, fontweight="bold")
                axes[plotted, col_idx].axis("off")
                plt.colorbar(im4, ax=axes[plotted, col_idx], fraction=0.046, pad=0.04)
                col_idx += 1

                # Coordinate Y
                im5 = axes[plotted, col_idx].imshow(coord_y, cmap="viridis", vmin=-1.0, vmax=1.0)
                axes[plotted, col_idx].set_title(f"Coord Y {plotted+1}", fontsize=12, fontweight="bold")
                axes[plotted, col_idx].axis("off")
                plt.colorbar(im5, ax=axes[plotted, col_idx], fraction=0.046, pad=0.04)
                col_idx += 1

            plotted += 1
        if plotted == num_samples:
            break

    fig.suptitle(dataset_name, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"results/{dataset_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
