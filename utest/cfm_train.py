import autorootcwd
import os, torch
from src.utils.registry import DATASET_REGISTRY
from src.utils.visualize_dataloader import visualize_flow_dataset
from src.data.xca_flow import XCAFlow_DataModule  # noqa: F401
from src.archs.unet.unet import DhariwalUNet
from src.archs.components.conditional_flow import SchrodingerBridgeConditionalFlowMatcher
import torchdiffeq
import matplotlib.pyplot as plt

def main():
    # results 폴더 생성
    os.makedirs("results", exist_ok=True)
    
    # 데이터 모듈 생성 및 설정
    print("Creating XCA Flow DataModule...")
    dm = DATASET_REGISTRY.get("xca_flow")()
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    image = batch['image']
    label = batch['label']
    geometry = batch['geometry']
    coordinate = batch['coordinate']

    print(image.shape, label.shape, geometry.shape, coordinate.shape)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    NEPOCHS=500

    model = DhariwalUNet(
        img_resolution=224,
        in_channels=1,
        out_channels=1,
        label_dim=0,
        augment_dim=0,
        model_channels=64,
        channel_mult=[1, 2, 3, 4],
        num_blocks=2,
        attn_resolutions=[32, 16, 8],
        dropout=0.0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    FM = SchrodingerBridgeConditionalFlowMatcher(sigma=0.5)

    for epoch in range(NEPOCHS):
        for batch in dm.train_dataloader():
            image = batch['image'].to(device)
            label = batch['label'].to(device)
            geometry = batch['geometry'].to(device)

            optimizer.zero_grad()

            t, xt, ut = FM.sample_location_and_conditional_flow(image, geometry)
            v = model(xt, t, class_labels=None)

            loss = torch.abs(v - ut).mean()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")

        if epoch % 10 == 0:
            with torch.no_grad():
                def ode_func(t, x):
                    # t는 스칼라 또는 1D 텐서, 배치 크기에 맞게 브로드캐스팅
                    if isinstance(t, torch.Tensor):
                        t = t.expand(x.shape[0])
                    else:
                        t = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
                    return model(x, t, class_labels=None)
                
                traj = torchdiffeq.odeint(
                    ode_func,
                    image,
                    torch.linspace(0, 1, 15, device=device),
                    atol=1e-4,
                    rtol=1e-4,
                    method="dopri5"
                )
                output = traj[-1]
                loss = torch.abs(output - geometry).mean()
                print(f"Epoch {epoch}, VAL Loss: {loss.item()}")

                
                batch_size = image.shape[0]
                fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
                
                if batch_size == 1:
                    axes = axes.reshape(1, -1)
                
                for i in range(batch_size):
                    img_np = image[i, 0, :, :].detach().cpu().numpy()
                    geom_np = geometry[i, 0, :, :].detach().cpu().numpy()
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
                plt.savefig(f"results/cfm_epoch_{epoch}_output.png")
                plt.close()

if __name__ == "__main__":
    main()