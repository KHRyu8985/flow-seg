import autorootcwd
import os
import torch
import time
from src.utils.registry import DATASET_REGISTRY
from src.archs.components.vessel_geometry import to_geometry, from_geometry
from src.utils.visualize_dataloader import visualize_image
from src.data.xca import XCA_DataModule  # noqa: F401

def main():
    # results 폴더 생성
    os.makedirs("results", exist_ok=True)
    
    # GPU 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터 모듈 생성 및 설정
    print("Creating XCA DataModule...")
    dm = DATASET_REGISTRY.get("xca")()
    dm.setup()
    
    batch = next(iter(dm.train_dataloader()))
    print("Batch keys:", batch.keys())
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: type={type(v)}, shape={v.shape}, device={v.device}")

    images = batch['image'].to(device)
    labels = batch['label'].to(device)
    print(f"\nImages: {images.shape}, device={images.device}")
    print(f"Labels: {labels.shape}, device={labels.device}")
    
    # GPU에서 to_geometry 실행 및 시간 측정
    print("\n=== Running to_geometry ===")
    start = time.time()
    geometry = to_geometry(labels)
    elapsed = time.time() - start
    print(f"to_geometry took {elapsed:.4f} seconds")
    print(f"  Average per image: {elapsed/labels.shape[0]:.4f} seconds")
    print(f"Geometry: {geometry.shape}, device={geometry.device}")
    
    # from_geometry 실행
    hard_preds = from_geometry(geometry)
    print(f"Hard preds: {hard_preds.shape}, device={hard_preds.device}")
    
    # CPU로 옮겨서 시각화
    images_cpu = images.cpu()
    labels_cpu = labels.cpu()
    geometry_cpu = geometry.cpu()
    hard_preds_cpu = hard_preds.cpu()
    
    visualize_image(images_cpu, labels_cpu, geometry_cpu, hard_preds_cpu, name="geometric_check")

if __name__ == "__main__":
    main()

